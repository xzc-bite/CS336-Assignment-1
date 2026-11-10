import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    ScaledDotProductAttention 是缩放点积注意力，它通过将输入的稠密向量与输入的稠密向量进行点积来得到输出。
    公式是：
    out = softmax(QK^T / sqrt(d_k))V

    Args:
        Q: (..., T_q, d_k) 查询向量
        K: (..., T_k, d_k) 键向量
        V: (..., T_k, d_v) 值向量
        mask: (..., T_q, T_k) 布尔掩码，**True = 允许注意**（和讲义、torch 的 SDPA 一致）
              传 0/1 的浮点掩码也兼容，非 0 视为允许
    output:
        out: (..., T_q, d_v) 输出的稠密向量

    前面的 ... 是随便多少个 batch 维：既支持 (B, T, d)，也支持多头拆开后的 (B, H, T, d)，
    只对最后两维做矩阵乘，其余维度按批次广播。
    """

    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        # 缩放因子取 Q 的实际最后一维（拆头后就是 head_dim），不写死 d_model：
        # 用 d_model 除会多缩一倍，softmax 被压平，注意力失去分辨力。
        d_k = Q.shape[-1]

        # 用 math.sqrt 拿 Python 标量，别写 torch.sqrt(torch.tensor(d_k))：
        # 后者每次前向都新建一个 CPU 张量，GPU 上还会多一次隐式搬运。
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)   # (..., T_q, T_k)

        if mask is not None:
            # 统一成"True = 允许"的布尔量；浮点 0/1 掩码也能吃
            keep = mask if mask.dtype == torch.bool else mask != 0
            # 必须在 softmax 之前填：先归一化再乘 0，会让被屏蔽位照样占分母。
            # 用 -inf 而不是 -1e9：后者只是"足够小的数"，分数本身量级很大时压不住，
            # 会漏出一丁点权重。
            scores = scores.masked_fill(~keep, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)   # 沿 key 维归一化，每行和为 1

        # 边界：整行都被屏蔽时，分母是 0（全是 -inf），softmax 给出 NaN。
        # padding 很长或 mask 写错时真会发生，NaN 一旦进残差就会污染整个模型。
        # 对齐 torch SDPA 的做法：这种行直接给 0。
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        return torch.matmul(attn_weights, V)           # (..., T_q, d_v)
