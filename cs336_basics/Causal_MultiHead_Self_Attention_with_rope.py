import math

import torch
import torch.nn as nn

# RoPE 按"每头维度"实现：多头里每个头各旋自己的 head_dim 维，
# 传进去的是 d_model // n_heads，不是 d_model。
# 依赖 cs336_basics/rope.py 里的 RoPE 类——那个文件建好之前，这行会报 ModuleNotFoundError。
from cs336_basics.run_rope import RoPE


class CausalMultiHeadAttentionWithRoPE(nn.Module):
    """
    带旋转位置编码（RoPE）的因果多头注意力。

    和无位置编码的版本比，只多一步：拆完头、打分之前，把 q 和 k 按位置各旋一下。
    v 不旋——v 是要被取走的内容，旋转它没有语义收益。

    整条流水线：

        x --投影--> q/k/v --拆头--> ★旋 q、k --打分+掩码--> 按权重取 v --合头--> 输出
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, theta: float, device=None):
        super(CausalMultiHeadAttentionWithRoPE, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 头数除不尽 d_model 时，// 会静默给出一个偏小的维度，
        # 报错要拖到后面 view 才发现，在入口拦一下更省事
        assert d_model % n_heads == 0, f"d_model={d_model} 必须能被 n_heads={n_heads} 整除"

        # 每个头分到的维度。RoPE 的旋转维度、注意力里的缩放因子，用的都是它
        self.head_dim = d_model // n_heads

        self.max_seq_len = max_seq_len
        self.theta = theta

        # 位置表在构造时就按 max_seq_len 算好，推理时直接查。
        # 传 head_dim 而不是 d_model：旋转是按头做的，表只需要 d/2 个频率
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device)

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        # 缩放因子取 Q 的最后一维。拆完头之后这里就是 head_dim；
        # 用 d_model 会多缩一倍，softmax 被压平，注意力就失去分辨力了
        d_k = Q.shape[-1]

        # 打分：每个 query 和每个 key 做内积，形状 (..., T_q, T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask 里 True 表示"允许看"（和讲义、F.scaled_dot_product_attention 一致），
            # forward 用 tril 造出来，True 正好是"当前及之前"的位置。
            # 取反再填 -inf：数学上就是"概率为 0"，
            # 而 -1e9 只是"足够小的数"，分数本身很大时不保证压得下去
            scores = scores.masked_fill(~mask, float("-inf"))

        # 沿 key 方向归一化：每个 query 对所有 key 的权重加起来等于 1
        attn_weights = torch.softmax(scores, dim=-1)

        # 按权重把 v 加权求和，得到这一层"取回来的信息"
        return torch.matmul(attn_weights, V)

    def forward(self, x, wq, wk, wv, wo, token_positions=None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # token_positions 不给就默认从 0 数到 seq_len-1（训练时的常规情况）。
        # 这个兜底不能省：直接把它传给 RoPE 的话，cos_cache[None] 会被当成
        # "在最前面插一维"，拿到 (1, max_seq_len, d/2)，再跟 (B, H, T, d/2) 广播，
        # 报出来的是 "size of tensor a (T) must match ... (max_seq_len)"，
        # 完全看不出病根在 None 上。
        # 推理带 KV cache 时仍然要显式传绝对位置，那时不能从 0 重新数。
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        # 三块投影一次算完。权重按 (out, in) 存，乘之前要转置
        q = x @ wq.T  # (B, T, D) @ (D, D) -> (B, T, D)
        k = x @ wk.T
        v = x @ wv.T

        # 拆头：(B, T, D) -> (B, T, H, head_dim)，再换轴成 (B, H, T, head_dim)。
        # view 只改解释方式、不搬数据；transpose 是为了让 (batch, head) 合成一批同时算
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 旋 q、k，位置信息在这里注入。
        # 必须放在拆头之后：每个头只旋自己那 head_dim 维，头与头互不影响。
        # 旋在投影之前是不对的——那时还是输入 x 的坐标，不是 q/k 的坐标。
        # token_positions 给出每个位置的真实编号：训练时是 0..T-1，
        # 推理带 KV cache 时要传绝对位置，不能从 0 重新数起
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # 因果掩码：下三角（含对角线）为 True = 允许看，上三角是"未来"、False = 屏蔽。
        # 对角线必须为 True，保证每个 token 至少能看见自己，否则整行被屏蔽会算出 NaN
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        # 补两个长度为 1 的维度，靠广播对上 (B, H, T, T)，不用真的复制 B×H 份
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = self.attention(q, k, v, mask)

        # 合头：(B, H, T, head_dim) -> (B, T, H, head_dim) -> (B, T, D)
        out = out.transpose(1, 2)
        # transpose 之后内存不连续，直接 view 会报错，先拷成连续的一块
        out = out.contiguous().view(batch_size, seq_len, d_model)

        # 输出投影：把各头的信息混到同一个坐标系里
        out = out @ wo.T
        return out
