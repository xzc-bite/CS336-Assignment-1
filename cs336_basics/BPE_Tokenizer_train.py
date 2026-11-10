import os  # 处理文件路径、建目录这类事，比如最后保存词表时要先建文件夹
from collections import defaultdict, Counter  # Counter：专门用来数数的字典；defaultdict：访问不存在的键时不会报错，自动给个默认值 0
import regex as re  # type: ignore  # 第三方正则库（注意不是 Python 自带的 re），因为要支持 \p{L} 这种"匹配所有字母"的写法
import json  # 最后把词表存成 json 文件用


def train_bpe(
    input_path: str | os.PathLike,  # 输入语料文件的路径
    vocab_size: int,             # 目标词表大小（基础字节 + 合并 Token + 特殊 Token）
    special_tokens: list[str],   # 需要保留的特殊 Token 列表
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE (Byte-Pair Encoding) 分词器。

    该函数 BPE 算法的核心流程：
    1. 初始化词表为所有可能的字节 (0-255)。
    2.  读取输入语料，并根据特殊 Token 进行切分，确保特殊 Token 不参与统计。
    3. 使用 GPT-2 的预分词正则将语料库切分成单词，并统计每个单词的频率。
    4. 迭代进行“合并”操作，直到达到目标词表大小。
       - 合并策略：总是选择当前出现频率最高、且在字典序上最大的字节对。
    5. 使用倒排索引优化合并过程中的频率更新，确保速度。
    6. 将合并产生的 Token 加入词表，并最终加入特殊 Token。

    返回:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: 训练好的词汇表，映射 Token ID -> Token 字节序列。
            merges: BPE 合并规则列表，按生成顺序排列。
    """

    # --- 1. 初始化基础词表 ---
    # 一切文本在电脑里最终都是字节。UTF-8 一个字符可能是 1~4 个字节，
    # 所以把 0~255 每个字节都先当作一个 token，这就是词表的"地基"。
    # bytes([i]) 就是把数字 i 变成对应的单字节，比如 bytes([65]) 是 b'A'。
    vocab = {i: bytes([i]) for i in range(256)}

    # 计算需要进行的合并次数。
    # 目标词表大小 = 基础字节数 (256) + 特殊 Token 数 + 需要新生成的 Token 数。
    # 所以反过来：真正要做的合并次数 = 目标大小 - 地基 256 - 特殊 token 占的名额。
    num_merges = vocab_size - 256 - len(special_tokens)

    # --- 2. 读取语料，并按特殊 Token 分割 ---
    # 一次性把整个文件读进内存。对几百 MB 的语料来说可行，但如果文件特别大就要考虑分块读了。
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 如果指定了特殊 Token，我们需要在开始统计之前将它们从语料中“隔离”出来。
    # 这能防止 BPE 规则将特殊 Token（如 <|endoftext|>）拆开或与普通文本混合。

    """
    For special_tokens:
    在训练时，必须保证特殊 Token 不参与频率统计。
    代码逻辑：
        切割语料：在开始统计词频之前，利用正则将语料库在特殊 Token 处切开。
        独立统计：只对切分出来的普通文本片段进行 BPE 统计。
        最后加入：训练结束后，强制将特殊 Token 加入词表（通常放在最后），确保它们有 ID。
    """
    if special_tokens:
        # 把多个特殊 token 用 | 串成一个正则；re.escape 是给内容里的特殊字符加保护，
        # 避免像 <|endoftext|> 里的 <| 这些字符被当成正则语法来解析。
        # 【改】拼之前先按长度从长到短排序。正则的 | 是"最左优先"，不是"最长优先"：
        # 不排序的话，<|endoftext|> 会抢在 <|endoftext|><|endoftext|> 前面命中，
        # 把后者劈成两截；万一劈出来的半截不是特殊 token，它就会漏进词频统计。
        # （Tokenizer.encode 那边本来就按长度排过序，训练和推理的规则得保持一致。）
        special_regex = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        # 用这个正则把整篇文本从特殊 token 的位置切开。
        # 正则外面套一层圆括号 (捕获组) 很关键：这样切出来的列表里会保留特殊 token 本身，
        # 不然它们就直接消失了。
        parts = re.split(f"({special_regex})", text)
        # 只保留普通文本片段，把特殊 token 本身扔掉——它们不参与词频统计。
        # text = "Hello World World<|endoftext|>Hello happy happy<|endoftext|>!"
        # train_segments =  ['Hello World World', 'Hello happy happy', '!']
        train_segments = [p for p in parts if p not in special_tokens]
    else:
        # 没有特殊 token 的话，整篇文本都拿来训练。
        train_segments = [text]

    # --- 3. 预分词（Pre-tokenization）并统计词频 ---
    # 使用 GPT-2 的 BPE 预分词正则表达式。
    # GPT-2 正则表达式的作用是执行“预分词（Pre-tokenization）”。 它的规则是：
    #   (1)不允许跨越类型合并：比如它会把字母和标点符号分开。
    #   (2)保护空格：它通常会把单词前面的空格和单词连在一起，作为一个整体。
    # text = "Hello World test! ..."
    # 分割后 words = ['Hello', ' World', ' test', '!', ' ...']
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # raw_counts: 存储每个“单词”（预分词后的结果）及其出现频率。
    # 单词被表示为字节元组，例如 "hello" -> (b'h', b'e', b'l', b'l', b'o')
    # 之所以要预先数好每个词出现几次，是因为语料里大量重复的词没必要反复处理，
    # 只处理"不重复的词"+ 它们的次数，能省下大量时间。
    raw_counts = Counter()
    for segment in train_segments:
        # findall 会把这段文本里所有匹配正则的"单词"找出来，返回一个字符串列表
        words = gpt2_pat.findall(segment)
        for word in words:
            # 把这个单词编码成 UTF-8 字节，再拆成一个个单字节，打包成元组作为"词"的唯一标识。
            # word.encode("utf-8") 得到 b'Hi'；for b in b'Hi' 遍历出 72 和 105；
            # bytes([b]) 又把数字变回单字节 b'H'、b'i'；最后拼成 (b'H', b'i')。
            # 为什么必须是元组（tuple）？因为 Counter 的键必须是不可变的，list 不能当键。
            """
            举例：
                raw_counts = {
                    (b'H', b'i'): 50,
                    (b' ', b't', b'h', b'e', b'r', b'e'): 100,
                    (b'!'): 50,
                    (b'\xe4',b'\xbd',b'\xa0',b'\xe5',b'\xa5',b'\xbd'):20, # 你好
                }
            """
            raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1

    # --- 构建高效数据结构以支持快速合并 ---
    # 上面 Counter 的键是不可变的 tuple，但 BPE 合并要不停修改词的内部结构，
    # 所以这里把每个词转成可修改的 list，两个列表按下标一一对应。
    words_list = []   # 每个不重复词的"字节序列"，训练过程中会被不断改写
    counts_list = []  # 对应词出现的次数（后面更新词频时要用）
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))  # 转换成 list 以便后面修改
        counts_list.append(freq)

    # defaultdict(int) 是一个“带默认初始值”的字典。当你访问一个字典中不存在的键时，它不会报错，而是自动为这个键创建一个默认值 0，而在使用普通字典进行计数时，你必须先检查键是否存在，否则会触发 KeyError。
    # stats: 存储所有可能的相邻字节对 (pair) 及其全局出现频率。
    # 结构：{(byte_a, byte_b): frequency}
    # 比如 (b't', b'h') 出现了一万次，那它就很值得合并成一个新 token。
    stats = defaultdict(int)

    # indices: 倒排索引。存储 pair -> {包含该 pair 的单词在 words_list 中的下标集合}
    # 这个结构是性能优化的关键，用于快速找到需要更新的单词。
    # 有了它，合并某个 pair 时不用扫全部词，直接查表就知道哪些词要改。
    indices = defaultdict(set)

    # --- 初始化 `stats` 和 `indices` ---
    # 遍历所有唯一的单词
    for idx, word in enumerate(words_list):
        freq = counts_list[idx]  # 获取该单词的出现频率
        # 把词里每一对紧挨着的字节都登记一遍：
        # 频率累加进 stats；这个词的下标登记进 indices，方便以后反查。
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] += freq       # 累加该 pair 的全局频率
            indices[pair].add(idx)    # 将当前单词的索引加入该 pair 的倒排列表中

    merges = []  # 用于存储生成的 BPE 合并规则，按顺序记录

    # --- 4. 迭代合并流程 ---
    # 循环执行 `num_merges` 次，每次找到并应用一个最佳合并规则
    for _ in range(num_merges):
        # 如果 `stats` 为空（所有可能的对都已合并或频率为0），则停止
        if not stats:
            break

        # --- 4a. 寻找最佳 Pair ---
        # 目标：找到当前 `stats` 中频率最高、且在字典序上最大的 Pair
        # `max(stats.items(), key=lambda x: (x[1], x[0]))`
        #   - key 里写成一个元组 (频率, pair本身)：max 先比频率，频率相同再比 pair 的字典序，
        #     这就是课程要求的平局规则（选字典序大的那个）。
        #     Python 比较元组是逐个元素来的，所以 (b' ', b't') 会大于 (b' ', b'a')。
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]

        # 如果最佳 Pair 的频率已经降到 0（可能是在之前的迭代中由于其组成部分被合并了），则停止
        if stats[best_pair] <= 0:
            break

        # 记录这次合并
        merges.append(best_pair)
        # 拼出新 token 的字节内容，比如 (b't', b'h') 合并成 b'th'
        new_token = best_pair[0] + best_pair[1]

        # --- 4b. 获取需要更新的单词 ---
        # 用倒排索引直接查出所有含 best_pair 的词的下标。
        # 必须先复制成 list：后面循环里会往 indices 里添加新条目，
        # 一边遍历一边改集合会报错或者漏改。
        relevant_indices = list(indices[best_pair])

        # --- 4c. 遍历并更新所有受影响的单词、统计信息和倒排索引 ---
        for idx in relevant_indices:
            word = words_list[idx]   # 获取单词（注意：是个可变的 list）
            freq = counts_list[idx]  # 获取单词的频率

            # 从头到尾扫一遍这个词，找出所有 best_pair 出现的位置
            i = 0
            while i < len(word) - 1:
                # 检查当前位置 `i` 和 `i+1` 是否匹配 `best_pair`
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    # --- 匹配到 `best_pair`，执行合并 ---

                    # 1. 先把这次合并"波及"到的旧邻居 pair 的频率减掉
                    #    （因为合并之后这些旧 pair 就不存在了，不減会虚高）：
                    #    - 左邻居：(word[i-1], word[i])
                    if i > 0:
                        prev_pair = (word[i - 1], word[i])
                        stats[prev_pair] -= freq  # 频率减去该单词的频率
                        if stats[prev_pair] == 0:
                            # 如果频率降为 0，从 `stats` 中移除该 pair，避免未来错误选择
                            """
                            stats 字典里依然会存在这个键：{(b'x', b'y'): 0}。
                            当训练快结束，或者剩下的所有对频率都降为 0 时，max 函数依然会扫描这些值为 0 的项。
                            根据平局规则，如果存在多个频率为 0 的项，max 会返回其中字典序最大的那一个，这是错误的
                            """
                            del stats[prev_pair]
                        # 不从 indices 中移除 idx
                        # 因为我们后续会通过检查 `word[i]` 来确定是否真的匹配。
                        # 频繁移除索引反而可能导致性能下降或逻辑错误。

                    #    - 右邻居：(word[i+1], word[i+2])
                    if i < len(word) - 2:
                        next_pair = (word[i + 1], word[i + 2])
                        stats[next_pair] -= freq  # 同理，减掉即将消失的右邻居 pair
                        if stats[next_pair] == 0:
                            del stats[next_pair]

                    # 2. 修改单词结构：将 (word[i], word[i+1]) 替换为 new_token
                    word[i] = new_token  # 将第一个字节替换为新 Token
                    del word[i + 1]      # 删除第二个字节，使单词变短

                    # 3. 添加新产生的邻居 Pair 的频率和索引
                    #    - 新的左邻居：(word[i-1], new_token)
                    if i > 0:
                        new_prev = (word[i - 1], word[i])  # word[i] 现在是 new_token
                        stats[new_prev] += freq            # 合并产生了新的邻居关系，频率要加上
                        indices[new_prev].add(idx)         # 添加到新 pair 的倒排索引

                    #    - 新的右邻居：(new_token, word[i+1]) (注意：word[i+1] 是旧的 word[i+2])
                    if i < len(word) - 1:
                        new_next = (word[i], word[i + 1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)

                    # 合并后，索引 i 指向的是新 Token。
                    # i 不需要移动（i+=1），因为我们刚刚修改了 word[i] 并且删除了 word[i+1]。
                    # 下一轮循环会检查新的 (word[i], word[i+1])，即 (new_token, old_word[i+2])
                    # 这可以处理像 A A A -> X A 这样的情况，正确地更新新的邻居对
                else:
                    # 如果不匹配，正常移动到下一个位置
                    i += 1

        # 4d. 清理：移除已完全合并的 `best_pair`
        # 这个 pair 已经不存在于 `stats` 和 `indices` 中了
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices: del indices[best_pair]

    # --- 5. 构建最终的词表 ---
    # 添加 BPE 合并产生的 Token，ID 从 256 开始，按合并顺序递增
    # 合并顺序很重要：先合并出来的 ID 小，后合并的 ID 大。
    for pair in merges:
        new_id = len(vocab)  # 当前词表多大，新 token 的 ID 就是几，正好接在后面
        vocab[new_id] = pair[0] + pair[1]

    # 添加特殊 Token（放在词表最后，ID 排在所有普通 token 之后）
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")  # 特殊 token 也按 UTF-8 转成字节存
        vocab[len(vocab)] = s_bytes

    return vocab, merges


def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
    """
    # 先挑出"本来就可见"的字节：英文标点、字母、数字区间等
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]  # 这些可见字节直接映射到自己
    n = 0
    for b in range(256):
        if b not in bs:
            # 剩下那些看不见的控制字符、空白字符，映射到 256+n 开始的"不常用"Unicode 字符，
            # 保证每个字节都有一个能显示、能写进文件的替身
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]  # 数字编码转成真正的字符
    return dict(zip(bs, cs))   # 组成 {字节: 字符} 的对照表


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)  # 输出目录不存在就创建，已存在也不报错

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转换为可见字符串
    # 直接存 bytes 进 json 会乱码，所以每个 token 先通过对照表变成"可见字符"再存
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v)
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)  # indent=4 让文件带缩进，方便人看

    # 合并规则保存，每行一条：前半 pair + 空格 + 后半 pair
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")


def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"  # 你的原始文本路径
    vocab_size = 10000  # 作业要求的词表大小
    # input_path = "data/owt_train.txt"
    # input_path = "data/chinese.txt"
    # vocab_size = 1000  # 作业要求的词表大小

    special_tokens = ["<|endoftext|>"]  # 文档结束符，训练时会被隔离、不参与合并
    output_dir = "data/TinyStoriesV2-GPT4-train"  # 训练结果（vocab.json 和 merges.txt）保存到哪

    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    print("这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率。")

    # 调用你之前写好的逻辑
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # 保存结果：vocab.json 和 merges.txt 会写到 output_dir 里。
    # 注意：这行之后没有任何 print，所以训练和保存都完成后屏幕上不会再有输出，
    # 看起来"啥都没有"其实是正常的——文件已经在 output_dir 里了。
    save_tokenizer_files(vocab, merges, output_dir)


if __name__ == "__main__":
    main()  # 只有直接运行这个文件时才执行 main()，被别人 import 时不会自动跑训练
