import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    旋转位置编码（RoPE）。

    把每一对相邻维度看成平面上的一个箭头，按 token 所在位置旋转一个角度：
    位置越靠后转得越多。旋转之后两个 token 的 q、k 做点积，结果只跟它们的
    相对距离有关，模型于是能感知"隔得远不远"。

    只作用在 q 和 k 上，不作用在 v 上——v 是要被取走的内容，旋转它没有语义收益。

    Args:
        theta: 频率的底数，常用 10000
        d_k: 每个头的维度，必须是偶数，且等于 q/k 的最后一维
        max_seq_len: 位置表预先算多长；推理序列更长时要扩展
        device: 位置表放在哪个设备，需要和输入一致
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 每个维度对配一个频率：θ_i = theta^(-2i/d)，i 取 0,2,4,... 直到 d-2。
        # i 越小转得越快（周期短），负责区分紧邻的位置；
        # i 越大转得越慢（周期长），负责远距离。
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))

        # 位置 m 转过的角度 = m × θ_i。外积一次算完"所有位置 × 所有频率"，
        # 结果形状 (max_seq_len, d_k // 2)。
        # 注意 device 要一起传：位置表建在 CPU 上、输入在 GPU 上时，查表会直接报错。
        sinusoids = torch.outer(torch.arange(self.max_seq_len, device=device).float(), freqs)

        # 位置表是固定值，不参与训练。注册成 buffer：跟着模块一起搬设备，
        # 但不会进优化器。persistent=False 表示不写进 state_dict。
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False)
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 按位置查表：token_positions 形状 (..., T)，查出来 (..., T, d/2)。
        # 不额外 unsqueeze——乘法靠广播和 x (..., T, d) 对齐；
        # 提前加维度的话，(4,12,64) 配 (4,12) 位置会算出 (1,4,12,64)。
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # 相邻两维配成一对：(0,1)、(2,3)…，每对各旋一个角度
        x_part1 = x[..., 0::2]
        x_part2 = x[..., 1::2]

        output1 = x_part1 * cos - x_part2 * sin  # 第 2k 维：x1·cos − x2·sin
        output2 = x_part1 * sin + x_part2 * cos  # 第 2k+1 维：x1·sin + x2·cos

        # 交替拼回去：stack 得到 (..., d/2, 2)，压平就是 (..., d)，
        # 顺序和前面切分的顺序一致。用 cat 会变成"前半 + 后半"，对不上 [0::2]/[1::2]。
        out = torch.stack([output1, output2], dim=-1)
        return out.flatten(-2)
