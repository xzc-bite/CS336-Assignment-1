"""训练入口：解析参数 → 建模型 → 训练循环 → 定期验证/存盘。

数据流：
    memmap(token id) --随机切片--> x, y (B, T) --模型--> logits (B, T, V)
    --交叉熵--> 标量 loss --backward--> 梯度 --全局范数裁剪--> AdamW 更新

为什么这里要自己拼一个模型：
    cs336_basics/Transformer_LM.py 里的 TransformerLM 是"用外部 state_dict 前向"的写法——
    权重挂在普通属性上，且每层用 load_state_dict({"weight": ...}) 拷进子模块，
    这个过程是 detach 的。结果是 model.parameters() 为空、梯度拿不到、
    optimizer.step() 什么也不会改。拿来对拍没问题，拿来训练不行。

    所以下面用同样几个零件（EmbeddingModule / RMSNorm / SwiGLU / 多头注意力）
    重新组织成一个参数都注册成 nn.Parameter 的版本。
    注意力的两个实现本身就是函数式的（权重从 forward 传进来），
    正好可以在外面把四个投影包成 nn.Parameter 再喂进去。
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.AdamW import AdamW
from cs336_basics.Causal_MultiHead_Self_Attention import CausalMultiHeadAttention
from cs336_basics.Causal_MultiHead_Self_Attention_with_rope import CausalMultiHeadAttentionWithRoPE
from cs336_basics.CheckPoint import load_checkpoint, save_checkpoint
from cs336_basics.Cross_Entropy_Loss import CrossEntropyLoss
from cs336_basics.Data_Loader import DataLoader
from cs336_basics.Gradient_Clipping import GradientClip
from cs336_basics.Linear_Embedding_model import EmbeddingModule, LinearModule
from cs336_basics.Lr_Cosine_Schedule import CosineSchedule
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.SwiGLU import SwiGLU

# wandb 是可选依赖：没装也能训练，只是没有可视化。空跑一遍就能验证流程通不通。
try:
    import wandb
except ImportError:
    wandb = None


class _Tee:
    """把 stdout 同时写进一份日志文件。

    训练动辄几小时，日志只打到终端的话，断开 SSH 或者进程被 OOM 杀掉就全没了。
    --log_file 打开时用这个包一层，print 照常写，同时落盘。
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)
        return len(data)

    def flush(self):
        for st in self.streams:
            st.flush()


class MultiHeadSelfAttention(nn.Module):
    """把四个投影矩阵参数化，前向转手给现成的注意力实现。

    形状（B=batch，T=序列长，D=d_model）：
        x         (B, T, D)
        q/k/v     (B, T, D)              x @ W.T
        拆头后     (B, H, T, D/H)
        scores    (B, H, T, T)
        out       (B, T, D)              合头 + 输出投影

    q/k/v/o 四个权重按 (D, D) 存、前向用 x @ W.T，和 LinearModule 的约定一致。
    """

    def __init__(self, d_model, num_heads, max_seq_len, theta, use_rope=True, device=None):
        super().__init__()
        # 初始化标准差沿用 Xavier：让每层输出的方差大致保持不变，
        # 写死成 0.02 也能跑，但深一点的模型容易在前几层就衰减掉。
        std = (2.0 / (2 * d_model)) ** 0.5
        for name in ("q", "k", "v", "o"):
            w = nn.Parameter(torch.empty(d_model, d_model, device=device))
            nn.init.trunc_normal_(w, std=std, a=-3 * std, b=3 * std)
            setattr(self, f"{name}_proj_weight", w)

        # RoPE 的旋转维度是"每个头"的维度，必须偶数，否则拆不成相邻两维一组
        if use_rope:
            assert (d_model // num_heads) % 2 == 0, \
                f"head_dim={d_model // num_heads} 必须是偶数，RoPE 才能两两配对旋转"
            self.impl = CausalMultiHeadAttentionWithRoPE(
                d_model, num_heads, max_seq_len, theta, device
            )
        else:
            self.impl = CausalMultiHeadAttention(d_model, num_heads)
        self.use_rope = use_rope

    def forward(self, x):
        # 两个实现的 forward 签名一致（带 RoPE 的那个 token_positions 有默认值，
        # 不传就按 0..T-1 走），所以这里不用分叉。
        return self.impl(
            x,
            self.q_proj_weight, self.k_proj_weight,
            self.v_proj_weight, self.o_proj_weight,
        )


class SiLUFFN(nn.Module):
    """单支路 FFN：W2 · SiLU(W1 · x)，用来和 SwiGLU 做消融对比。

    和 SwiGLU 的区别是没有门控支路，参数少三分之一（只剩 w1、w2）。
    形状：x (B,T,D) → (B,T,d_ff) → (B,T,D)。
    """

    def __init__(self, d_model, d_ff, device=None):
        super().__init__()
        # LinearModule(in, out) 内部存 (out, in)，所以这里给的是 (输入维, 输出维)
        self.w1 = LinearModule(d_model, d_ff, device=device)   # W1: (d_ff, d_model)
        self.w2 = LinearModule(d_ff, d_model, device=device)   # W2: (d_model, d_ff)

    def forward(self, x):
        return self.w2(SwiGLU.silu(self.w1(x)))


class TransformerBlock(nn.Module):
    """pre-norm 或 post-norm 的单个 Block，输入输出形状不变（B, T, D）。

    pre-norm:  x = x + Attn(Norm(x));  x = x + FFN(Norm(x))
    post-norm: x = Norm(x + Attn(x));  x = Norm(x + FFN(x))

    pre-norm 的主干是未经归一化的残差流，深层更稳，所以默认用它；
    post-norm 是原论文的写法，这里的 rms_norm 就是 LayerNorm 的替代品。
    """

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta,
                 norm_mode="pre", use_norm=True, use_rope=True, ffn_type="swiglu", device=None):
        super().__init__()
        # 关掉归一化时用 Identity 占位，forward 里就不用为它分叉了
        make_norm = (lambda: RMSNorm(d_model, eps=1e-5, device=device)) if use_norm else nn.Identity
        self.ln1 = make_norm()
        self.ln2 = make_norm()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, use_rope, device)
        self.ffn = SwiGLU(d_model, d_ff) if ffn_type == "swiglu" else SiLUFFN(d_model, d_ff, device)
        self.norm_mode = norm_mode

    def forward(self, x):
        if self.norm_mode == "pre":
            # 残差加的是归一化前的主干，梯度有一条不过 norm 的直通路
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    """decoder-only 语言模型，所有参数都是 nn.Parameter。

    形状流转：
        idx (B, T) --embedding--> (B, T, D) --N×Block--> (B, T, D)
        --ln_final--> (B, T, D) --lm_head--> (B, T, vocab_size)

    输出停在 logits，不做 softmax：交叉熵内部会做 log-softmax，
    提前归一化反而会让数值不稳定。
    """

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff,
                 rope_theta, norm_mode="pre", use_norm=True, use_rope=True,
                 ffn_type="swiglu", device=None):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = EmbeddingModule(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta,
                             norm_mode, use_norm, use_rope, ffn_type, device)
            for _ in range(num_layers)
        ])
        # pre-norm 的主干没被归一化过，最后一层之后必须补一次，
        # 否则输出量级会随层数漂移，lm_head 那边很难训
        self.ln_final = RMSNorm(d_model, eps=1e-5, device=device) if use_norm else nn.Identity()
        self.lm_head = LinearModule(d_model, vocab_size, device=device)

    def forward(self, in_indices):
        # T 不能超过 context_length，否则 RoPE 位置表会越界
        if in_indices.shape[1] > self.context_length:
            raise ValueError(
                f"序列长度 {in_indices.shape[1]} 超过 context_length {self.context_length}"
            )
        x = self.token_embeddings(in_indices)      # (B, T) -> (B, T, D)
        for layer in self.layers:
            x = layer(x)                           # 每层形状不变
        x = self.ln_final(x)                       # (B, T, D)
        return self.lm_head(x)                     # (B, T, V)


def main():
    parser = argparse.ArgumentParser()
    # --- 模型基础超参数 ---
    # batch_size: 每步喂多少条序列。显存不够时优先砍它，再砍 context_length。
    parser.add_argument("--batch_size", type=int, default=32)
    # context_length: 每条序列的 token 数，必须 <= RoPE 位置表长度（这里两者取同一个值）。
    parser.add_argument("--context_length", type=int, default=256)
    # d_model: 隐藏维，必须能被 num_heads 整除，且 d_model//num_heads 必须是偶数（RoPE 要求）。
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    # d_ff: FFN 中间层宽度，SwiGLU 惯例取 (8/3)·d_model 附近。
    parser.add_argument("--d_ff", type=int, default=2048)
    # vocab_size: 必须 >= 数据里最大的 token id。数据是 uint16，上限 65536。
    parser.add_argument("--vocab_size", type=int, default=10000)

    # --- 实验/消融开关 ---
    # 这几个开关会真的改变模型结构，参数量也会变，别跨配置复用同一个 checkpoint。
    parser.add_argument("--no_rms_norm", action="store_true", help="Disable RMSNorm completely")
    parser.add_argument("--norm_mode", type=str, default="pre", choices=["pre", "post"])
    parser.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Embeddings")
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"])

    # --- 优化器超参数 ---
    # lr: 峰值学习率，warmup 结束时达到；之后按余弦降到 min_lr。
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    # warmup_iters: 线性升温步数，取总步数的 1%~5%，必须 < max_iters。
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    # max_norm: 梯度全局 L2 范数上限，超过就整体等比缩放。
    parser.add_argument("--max_norm", type=float, default=1.0)
    # weight_decay: 解耦权重衰减，LLM 常用 0.1。
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # --- 路径与系统 ---
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # 固定种子，保证同配置能复现（DataLoader 用的是 numpy 全局 RNG，所以两个都要设）
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=100)
    # 每次验证平均多少个 batch。1 批只有 B*T 个 token（默认 8192），
    # val_loss 的采样噪声能到 0.03 量级，曲线上看着像在抖。
    parser.add_argument("--val_batches", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1000)

    # --- WandB 设置 ---
    parser.add_argument("--wandb_project", type=str, default="cs336-pretraining")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="不联网、不记录，纯本地跑")
    parser.add_argument("--log_file", type=str, default=None, help="把 stdout 同时写进这个文件")
    # 冒烟测试时只想验证流程通不通，不想动已有 checkpoint
    parser.add_argument("--no_save_final", action="store_true", help="训练结束不写 ckpt_final.pt")

    args = parser.parse_args()

    # 两个 RNG 都要设：torch 管初始化和采样，numpy 管 get_train_batch_data 的随机起点
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # 不关太可惜：训练日志是唯一的实验记录，/tmp 又经常被系统清理
    log_fp = None
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        log_fp = open(args.log_file, "a", buffering=1, encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, log_fp)

    # 1. 加载数据（memmap 延迟加载，几 GB 的语料也能秒开）
    # dtype 必须和写文件时一致，写错了读出来是乱码 id，没报错但 loss 永远不降。
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")

    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.valid_data_path, dtype=np.uint16, mode="r")
    print(f"训练集大小: {len(train_data)} tokens")
    print(f"验证集大小: {len(val_data)} tokens")

    # 数据太短时 get_train_batch_data 会在 randint 里抛 low >= high，提前给一句人话
    for name, d in (("train", train_data), ("valid", val_data)):
        if len(d) <= args.context_length + 1:
            raise ValueError(
                f"{name} 集只有 {len(d)} 个 token，不足以切出长度 {args.context_length} 的样本"
            )

    # 2. 消融开关 → 模型配置
    use_rms_norm = not args.no_rms_norm
    use_rope = not args.no_rope
    rope_theta = 10000.0 if use_rope else 0.0   # 关掉 RoPE 时这个值不会被用到

    # 3. 建模型
    assert args.d_model % args.num_heads == 0, \
        f"d_model={args.d_model} 必须能被 num_heads={args.num_heads} 整除"

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=rope_theta,
        norm_mode=args.norm_mode,
        use_norm=use_rms_norm,
        use_rope=use_rope,
        ffn_type=args.ffn_type,
        device=args.device,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Config: Norm={args.norm_mode}, UseNorm={use_rms_norm}, "
          f"FFN={args.ffn_type}, RoPE={use_rope}")
    print(f"参数量: {n_params/1e6:.2f} M")

    # 4. 优化器。AdamW 的 signature 里 betas/eps 没有默认值，必须显式给。
    # lr 这里只是占位，循环里每步都会被 schedule 覆盖。
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # 5. 断点续训。模型 + 优化器状态 + 步数三件套缺一不可，
    # 只恢复权重的话 Adam 的 m/v 归零，续训初期更新会明显偏大。
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    final_path = os.path.join(args.out_dir, "ckpt_final.pt")
    if os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resuming from iteration {start_iter}")
    elif os.path.exists(final_path):
        # 只留了 ckpt_final.pt 的场景（节省磁盘时常见）：不自动续训，
        # 否则"想从头再跑一遍"会莫名其妙接着旧模型。给一句提示就好。
        print(f"提示: 发现 {final_path} 但没有 ckpt.pt，本次将从第 0 步开始。")
        print(f"      想接着训的话先执行: cp {final_path} {ckpt_path}")

    # 6. WandB（可选）
    use_wandb = wandb is not None and not args.no_wandb
    if use_wandb:
        try:
            wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        except Exception as e:      # 没登录/没网时降级成纯本地，不要让训练卡在这
            print(f"wandb 初始化失败，改为本地记录: {e}")
            use_wandb = False

    schedule = CosineSchedule(args.lr, args.min_lr, args.warmup_iters, args.max_iters)
    train_loader = DataLoader(train_data, args.batch_size, args.context_length)
    val_loader = DataLoader(val_data, args.batch_size, args.context_length)
    clamp = GradientClip(list(model.parameters()), args.max_norm)   # 传 list，别让生成器被消费掉

    # 验证批次固定下来：每步重新随机取的话 val_loss 会带采样噪声，曲线上下抖。
    # 固定 + 多批取平均：同样是"每轮看同一批数据"，但单批只有 B*T 个 token，
    # 一批估出来的 loss 噪声约 0.03，够把趋势淹掉，取 4 批能压到 0.015 左右。
    val_batches = []
    for _ in range(max(1, args.val_batches)):
        vx, vy = val_loader.get_train_batch_data()
        val_batches.append((vx.to(args.device), vy.to(args.device)))

    # 7. 主训练循环
    # 每步计算量：注意力 O(B·T²·D)，FFN O(B·T·D·d_ff)，显存约为参数量 4 倍（权重+梯度+m+v）。
    for it in range(start_iter, args.max_iters):
        # A. 学习率。余弦周期终点取 max_iters，所以最后一步正好落到 min_lr。
        lr = schedule(it)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # B. 训练步
        model.train()
        x, y = train_loader.get_train_batch_data()      # (B, T) int64，在 CPU
        x, y = x.to(args.device), y.to(args.device)

        logits = model(x)                               # (B, T, V)，显存峰值就在这
        # CrossEntropyLoss 内部已经是 batch 平均，直接拿来用，不用再除 B*T。
        # 拉平成 (B*T, V)：模型对每个位置各给一个词表分布，损失要按位置平均。
        loss = CrossEntropyLoss(
            logits.reshape(-1, args.vocab_size), y.reshape(-1)
        ).forward()

        # 清梯度必须在 backward 之前；忘了的话梯度会跨 batch 累加
        optimizer.zero_grad()
        loss.backward()

        # 裁剪必须放在 step 之前，否则裁的是已经用过的梯度
        clamp()

        optimizer.step()

        # C. 验证与日志
        if it % args.eval_every == 0 or it == args.max_iters - 1:
            model.eval()
            v_sum = 0.0
            with torch.no_grad():
                for vx, vy in val_batches:
                    v_logits = model(vx)
                    v_sum += CrossEntropyLoss(
                        v_logits.reshape(-1, args.vocab_size), vy.reshape(-1)
                    ).forward().item()
            v_loss = v_sum / len(val_batches)      # 已经 .item() 过，这里就是个 float
            # 困惑度 = exp(交叉熵)，讲义要求报的指标。它和 loss 是单调对应的，
            # 但尺度更直观：PPL=20 约等于"在 20 个候选里犹豫"。
            # .item() 会触发 GPU 同步，只在日志里用，别放进训练主干
            print(f"Iter {it}: train_loss {loss.item():.4f}, "
                  f"val_loss {v_loss:.4f}, "
                  f"val_ppl {math.exp(v_loss):.2f}, lr {lr:.2e}")
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "val/loss": v_loss,
                    "val/ppl": math.exp(v_loss),
                    "lr": lr,
                    "iter": it + 1,
                })

        # D. 存检查点
        # 存 it + 1（已完成步数），续训时 range(start_iter, ...) 才不会把这一步重跑一遍
        if args.save_every > 0 and it % args.save_every == 0 and it > 0:
            save_checkpoint(model, optimizer, it + 1, ckpt_path)

    # 最终模型单独存一份，避免下次启动读到"已训完"的 ckpt.pt 又从头到尾空转一轮。
    #
    # 保护：同一个 out_dir 下跑一次 20 步的冒烟测试，会把它覆盖成 20 步的模型，
    # 之前几小时训出来的东西就没了。所以这里用 sidecar 文件记下已经存过的步数，
    # 新的 max_iters 比它小就不覆盖（除非显式 --no_save_final 之外还想强写）。
    if not args.no_save_final:
        iter_marker = final_path + ".iter"
        prev_iter = -1
        if os.path.exists(iter_marker):
            try:
                with open(iter_marker, encoding="utf-8") as f:
                    prev_iter = int(f.read().strip())
            except (OSError, ValueError):
                prev_iter = -1      # 标记文件坏了就当没有，宁可覆盖也别卡住

        if args.max_iters >= prev_iter:
            save_checkpoint(model, optimizer, args.max_iters, final_path)
            with open(iter_marker, "w", encoding="utf-8") as f:
                f.write(str(args.max_iters))
        else:
            print(f"跳过最终保存：{final_path} 里已是第 {prev_iter} 步，本次只有 "
                  f"{args.max_iters} 步。确实想覆盖就先删掉 {final_path} 和 "
                  f"{iter_marker} 再跑。")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
