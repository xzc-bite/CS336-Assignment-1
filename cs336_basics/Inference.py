"""自回归生成：温度缩放 + top-p（核）采样。

对应讲义 §6 "Generating text" 的 deliverables：给一段 prompt 持续采样出后续 token，
直到遇到 <|endoftext|> 或者达到用户指定的最大长度，中间支持温度缩放和 top-p 截断。

三个环节，顺序不能换：

    logits (B, T, V) --取最后位置--> (B, V) --/τ + softmax--> probs (B, V)
    probs --top-p 截断--> 候选集 V(p) --重新归一化--> multinomial 抽一个
    token id --拼到序列尾--> 下一轮

为什么不直接 argmax（贪心）：贪心是确定性的，同一段 prompt 永远生成同一句话，
文本会原地打转；采样引入随机性才能出多样性。但纯按分布采样又会抽到长尾里的离谱词，
所以先用 top-p 把长尾砍掉——这就是"核采样"（nucleus sampling）。

复杂度：top_p_sampling 是 O(V log V)（排序瓶颈）。整段生成 N 个 token 时，
没做 KV cache，每步都要重新前向整条序列，总计约 O(N · T² · d_model)。
"""

import torch


def top_p_sampling(probabilities, top_p=0.9, generator=None):
    """Top-p（核）采样：只在"累积概率刚好够 p"的那一撮候选里抽。

    参数:
        probabilities: 已经归一化过的概率分布（每行和 ≈ 1），形状 (V,) 或 (B, V)。
                       传 logits 进来是错的——阈值是按概率刻度定的，
                       没归一化的数会让 top_p 完全失去意义。
        top_p: float，(0, 1]，默认 0.9。
               =1 时不截断（纯按分布采样）；越小候选集越窄、越保守。
               按讲义定义，即使 p 很小也至少保留概率最大的那一个 token，
               不会出现"整行被清零"的情况。
        generator: torch.Generator 或 None。传进来可以固定采样结果，
                   复现实验时有用；不传就用全局 RNG。

    返回:
        下一个 token 的 id，形状 (B, 1) 或 (1,)，dtype long。
        是**原始词表下标**，不是排序后的排名。

    算法（讲义式 24）:
        V(p) 是满足 Σ_{j∈V(p)} q_j ≥ p 的**最小**下标集合。
        降序排好、累加，取到"累积刚好 ≥ p"为止的那个前缀，就正好是 V(p)。
        等价于：只要"排在它前面的 token 累积还不够 p"，它就留下。

        实现上用的是 `cumsum - p_i < top_p`：
        第 i 个元素前面的累积和 = cumsum[i] - p_i，比先 cumsum 再 shift 更省一步。
        第一个元素的前缀和恒为 0，所以只要 top_p > 0，它一定被保留——
        这就是"至少留一个"的保证，也顺手避开了 0/0。

    复杂度: 时间 O(V log V)，空间 O(V)。
    """
    # 降序排。dim=-1 是必须的：只对最后一维（词表维）排序，
    # 每个 batch 元素才各排各的；写成 dim=0 会把不同样本的概率混在一起。
    # sorted_idx 记的是"排完第 k 位原本是谁"，最后要靠它把排名翻译回 token id。
    sorted_probs, sorted_idx = torch.sort(probabilities, dim=-1, descending=True)

    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # 前缀和（不含自己）< p 就保留 —— 这正是"最小覆盖 p 的前缀"
    keep = (cumulative - sorted_probs) < top_p
    filtered = sorted_probs.masked_fill(~keep, 0.0)

    # 截断后总和不再等于 1，multinomial 要求合法分布，必须重新归一化。
    # keepdim=True 让分母保持 (B, 1)，才能广播回 (B, V)；
    # 少了它，第 i 行会除到第 i+1 行的和上，结果看着像对的但全是错的。
    denom = filtered.sum(dim=-1, keepdim=True)

    # 兜底：top_p <= 0 会让所有 keep 都为 False，分母变 0。
    # 这时退化成贪心（只取概率最高的那个），比抛 NaN 好。
    fallback = torch.zeros_like(filtered)
    fallback[..., 0] = 1.0
    normalized = torch.where(denom > 0, filtered / denom.clamp_min(1e-12), fallback)

    # 抽一个"排名"。返回的 0 表示"排第一的那个词"，不是词表第 0 号
    next_rank = torch.multinomial(normalized, num_samples=1, generator=generator)

    # 用 gather 把排名翻译回 token id。这步不能省，
    # 直接拿 multinomial 的结果当 id 用，模型会一直输出排名靠前的几个 token
    return torch.gather(sorted_idx, dim=-1, index=next_rank)


def temperature_scaling(logits, temperature=1.0):
    """温度缩放：softmax(z / τ)。

    参数:
        logits: 模型原始输出。形状 (B, T, V) 或 (B, V) 都接受——
                (B, T, V) 时只取最后一个位置，因为生成时只关心
                "当前序列末尾之后那个词"的分布，前面的位置已经用完了。
        temperature: float，> 0，默认 1.0。
                     < 1 让分布更尖、更接近贪心；> 1 让分布更平、更随机。
                     讲义里提到 τ→0 时趋近 one-hot，但代码上不能真取 0
                     （除零得 inf/nan），想要贪心请直接用 argmax。
                     负数没有意义，会把整个概率分布翻转过来，这里直接报错。

    返回:
        概率分布，形状 (B, V)。序列维在内部被吃掉了。

    维度变化:
        (B, T, V) --logits[:, -1, :]--> (B, V) --/τ + softmax(--1)--> (B, V)
    """
    if temperature <= 0:
        raise ValueError(f"temperature 必须 > 0，收到 {temperature}；想要贪心请用 argmax")

    # 兼容 2D 输入：已经取过最后位置的话就不重复切，
    # 否则 (B, V) 上的 [:, -1, :] 会索引错维度。
    if logits.dim() == 3:
        logits = logits[:, -1, :]

    return torch.softmax(logits / temperature, dim=-1)


def decode_token(
    input_tokens,
    model,
    max_tokens_to_generate,
    top_p=0.9,
    temperature=1.0,
    eos_token_id=None,
    context_length=None,
):
    """从 prompt 出发自回归采样，返回整条序列（prompt + 新生成的 token）。

    参数:
        input_tokens: list[int] 或 (T,) / (1, T) 的整数张量，prompt 的 token id。
                      不能为空——模型至少要吃一个位置才有 logits。
        model: 已训练好的 TransformerLM，forward 接受 (1, T) long、返回 (1, T, V)。
        max_tokens_to_generate: int，>= 0，最多新增多少个 token。
                      0 表示原样返回 prompt。
        top_p: 见 top_p_sampling，默认 0.9。
        temperature: 见 temperature_scaling，默认 1.0。
        eos_token_id: int 或 None，结束符的 token id（<|endoftext|> 对应的那个）。
                      生成到这个 id 就停下，并把它留在序列末尾。
                      传 None 表示没有停止条件，只会被 max_tokens_to_generate 截断。
                      注意要传 **id** 而不是字符串 "<|endoftext|>"：
                      张量和字符串比较永远返回 False，那样写等于没有停止条件。
        context_length: int 或 None，模型能吃的最长上下文。
                      传 None 时从 model.context_length 读；读不到就不截断。
                      序列超长时只把**最后 context_length 个** token 喂给模型
                      （滑动窗口），而返回的序列仍然是完整长度。

    返回:
        (1, T + N) 的 long 张量，留在模型所在 device 上。
        前 T 个是原 prompt，后面是采样出来的。要文本得自己拿 tokenizer 解码。

    复杂度:
        每生成一个 token 都要把整条序列重新前向一遍（没做 KV cache），
        单步 O(T² · d_model + T · d_model²)，N 步累计 O(N · T² · d)。
        这也是 KV cache 存在的理由。
    """
    # prompt 必须是 (1, T)。用 as_tensor 而不是 tensor：
    # 传进来已经是张量时不会多拷一次。
    ids = torch.as_tensor(input_tokens, dtype=torch.long).view(1, -1)

    # 放到模型所在 device 上。以前这里建的是 CPU 张量，
    # 模型在 cuda 上就直接 device mismatch——生成函数不该要求调用方自己去搬。
    # 取不到参数时（比如拿个假模型来测）就退回 CPU。
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")
    ids = ids.to(device)

    if ids.numel() == 0:
        raise ValueError("prompt 不能为空，模型至少需要一个位置才能预测下一个 token")

    # 上下文窗口：优先用调用方给的，其次读模型自己的属性
    ctx = context_length if context_length is not None else getattr(model, "context_length", None)

    # 记住进来时的模式，函数结束时恢复。不恢复的话，在训练脚本里顺手调一次生成，
    # 模型就被永久留在 eval 状态（dropout 关掉，后续训练静默变慢/效果变差）。
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(max_tokens_to_generate):
                # 滑动窗口：只把最近 ctx 个 token 喂给模型，但 ids 本身继续往后长。
                # 不做这一步的话，序列一旦超过上下文长度，RoPE 位置表就会越界。
                window = ids if ctx is None else ids[:, -ctx:]

                logits = model(window)               # (1, T_win, V)
                probs = temperature_scaling(logits, temperature)   # (1, V)
                next_token = top_p_sampling(probs, top_p)          # (1, 1)

                # 沿序列维拼：(1, T) + (1, 1) -> (1, T+1)。
                # 必须是 dim=-1，拼到 dim=0 会把 batch 维搞错。
                ids = torch.cat([ids, next_token], dim=-1)

                # 停止条件：比较**刚生成的这个 id**，而不是整个张量。
                # batch=1，所以 .item() 拿到的就是唯一那个值。
                if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                    break
    finally:
        # 无论正常结束还是中途抛异常，都把模式还回去
        model.train(was_training)

    return ids
