import torch
import torch.nn as nn


# 记不住公式就看这条流水线：
#
#   x (B, T, d_model)
#     ├── W1 ──> (B, T, d_ff) ── SiLU ──┐
#     │                                  ├── 逐元素相乘 ──> (B, T, d_ff) ── W2 ──> (B, T, d_model)
#     └── W3 ──> (B, T, d_ff) ──────────┘
#
# 两条支路：上面算"内容"，下面算"门控信号"。
# 门控信号的取值范围是 0~1，相当于给每个通道配一个音量旋钮，
# 逐通道去调节内容——这就是 GLU（gated linear unit）这个名字的由来。


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络（Transformer 里 FFN 部分的一种实现）。

    公式（⊙ 是逐元素相乘，不是矩阵乘法）：

        FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )

    三个线性层各管一段：

        W1: d_model → d_ff    把输入拉宽，得到"内容"支路
        W3: d_model → d_ff    同样拉宽，得到"门控"支路
        W2: d_ff → d_model    把相乘后的结果压回原维度，好接回残差

    为什么中间要拉宽：d_ff 通常取 4×d_model（LLaMA 用 8/3 倍），
    相当于给模型一块更大的"草稿纸"，算完再压回原维度，
    保证输出形状和输入完全一致。

    名字拆开看：SiLU 也叫 Swish，"GLU" 指门控结构，
    合起来就是"用 Swish 做门控的 GLU"。LLaMA、PaLM 都用这个结构。

    按讲义不加 bias，可训练参数只有三组权重。
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # nn.Linear(d_in, d_out) 内部存的是 (d_out, d_in) 的权重，前向做 x @ W.T，
        # 等价于数学写法 W x。所以参数顺序是 (输出维, 输入维)：
        # w1 的权重形状 (d_ff, d_model)，正好对应公式里的 W1 ∈ R^{d_ff × d_model}。
        # 把顺序写反了不会报错（维度刚好也能乘），但语义完全不对。
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

        # nn.Linear 默认是 U(-1/√in, 1/√in) 的均匀初始化，和仓库里 LinearModule 的
        # Xavier 截断正态不是一套尺度。同一个模型里两套初始化，深层堆叠时
        # 各子层的输出方差会不一致；这里手动对齐成同样的规则。
        for layer in (self.w1, self.w2, self.w3):
            std = (2.0 / (layer.in_features + layer.out_features)) ** 0.5
            nn.init.trunc_normal_(layer.weight, std=std, a=-3 * std, b=3 * std)

    @staticmethod # 能够在adapter里直接调，不用凭空造一个对象
    def silu(x):
        """
        SiLU（也叫 Swish）：y = x · sigmoid(x)。

        sigmoid(x) = 1 / (1 + e^(-x))，输出永远落在 (0, 1) 之间：

            x = -10  →  约 0.0000
            x =   0  →  0.5
            x =  10  →  约 0.9999

        所以 SiLU 相当于"拿一个 0~1 的开关去乘自己"：

            大正数：开关接近 1，输出约等于输入（放行）
            大负数：开关接近 0，输出约等于 0（压掉）
            0 附近：一条平滑曲线，不像 ReLU 在 0 处有个折角，
                    导数连续，训练时更稳

        两个能记住的数值：silu(1.0) ≈ 0.731，silu(-1.0) ≈ -0.269
        （注意负数侧不是 0，而是压扁成一个小负数）

        标成 staticmethod，是因为它不依赖任何实例状态：
        这样 self.silu(x) 和 SwiGLU.silu(x) 都能调，
        adapter 里测试它时不用凭空造一个对象。
        """
        return x * torch.sigmoid(x)

    def forward(self, x):
        # 第一步：两条支路各自把输入拉宽到 d_ff
        # gate 这条会被 SiLU 压进 0~1，当音量旋钮用；content 这条原样保留
        gate = self.silu(self.w1(x))   # (B, T, d_ff)，每个通道一个 0~1 的旋钮
        content = self.w3(x)           # (B, T, d_ff)，真正要传下去的内容

        # 第二步：逐元素相乘。
        # 这里必须是 * 而不是 @：不是"两个向量做内积"，
        # 而是"第 i 个通道的旋钮 × 第 i 个通道的内容"，通道之间互不干扰，
        # 所以形状保持不变。
        gated = gate * content         # (B, T, d_ff)

        # 第三步：压回 d_model。形状和输入一致，后面才能直接加回残差主干
        return self.w2(gated)          # (B, T, d_model)
