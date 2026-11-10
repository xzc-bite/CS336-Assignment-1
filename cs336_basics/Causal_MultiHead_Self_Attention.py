import torch
import torch.nn as nn
import math
import torch.nn.functional as F

"""
因果多头注意力。整个模块只干一件事：

    (B, T, D) --投影--> Q/K/V --分头--> H 份 --各自注意力--> 拼回 --输出投影--> (B, T, D)

拆开看是四步：
  0. 用 W_q / W_k / W_v 把输入投影成三个角色：想找什么的（Q）、能被怎么找到的（K）、
     真正拿得出手的内容（V）。
  1. 把最后一维按头数切成 n_heads 份，每份 d_model / n_heads 维，各头独立计算。
  2. 每个头内部算 softmax(QKᵀ / √d)·V，也就是"按相关度把大家的信息加权取回来"。
  3. 各头结果按最后一维拼回去，再过 W_o 融合——不融合的话，各头等于各说各话。
"""

class CausalMultiHeadAttention(nn.Module):
    """
    因果多头注意力：每个位置只能看自己和左边，看不到未来。

    单个头内部就是标准的缩放点积注意力：
        out = softmax(QKᵀ / √d_k) · V

    Args:
        d_model (int): 输入/输出的特征维度
        n_heads (int): 头数，必须能整除 d_model

    输入:
        x:  (batch_size, seq_len, d_model)  上一层传下来的向量序列
        wq: (d_model, d_model) 查询投影权重
        wk: (d_model, d_model) 键投影权重
        wv: (d_model, d_model) 值投影权重
        wo: (d_model, d_model) 输出投影权重

        注意这四个权重的形状都是 (d_model, d_model)：所有头拼在一起存。
        别按 (d_model, head_dim) 准备数据——切头是 forward 里 view 的事。

    输出:
        out: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model, n_heads):
        super(CausalMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 头数除不尽 d_model 时，// 会悄悄给一个偏小的整数，
        # 真正的报错要等到后面 view 发现元素总数对不上才爆出来，
        # 那时候看 traceback 很难想到是头数配错了。这里提前拦一下。
        assert d_model % n_heads == 0, f"d_model={d_model} 必须能被 n_heads={n_heads} 整除"
        self.head_dim = d_model // n_heads

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        # 缩放因子用的是"每个头"的维度，不是 d_model。
        # 传进来的 Q 已经按头切好了，最后一维只有 head_dim；
        # 如果拿 d_model 来除，等于多缩了一倍，softmax 被压得太平，
        # 注意力就失去了"挑重点"的能力。
        # 这里按 Q 的实际形状取，而不是读 self.head_dim——
        # 调用方万一传了别的头维度（测试里就会这样），也不会算错。
        d_k = Q.shape[-1]

        # QKᵀ 是"每个 query 对每个 key 打的分"，形状 (..., T, T)。
        # 除以 sqrt(d_k) 是因为：q、k 各分量零均值单位方差时，点积的方差会随维度线性增长，
        # 维度一大分数就飙高，softmax 容易饱和成一个尖峰，梯度跟着消失。
        # 除以标准差正好把分数的量级拉回 1 附近。
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask 的约定和讲义、F.scaled_dot_product_attention 一致：True = 允许注意。
            # 所以要取反再填 -inf，被屏蔽的位置 softmax 之后权重精确等于 0。
            # 两个细节：必须在 softmax 之前填（先归一化再乘 0 会让被屏蔽位参与分母），
            # 并且用 -inf 而不是 -1e9——后者只是"足够小的数"，
            # 当分数本身量级很大时不保证压得下去。
            scores = scores.masked_fill(~mask, float("-inf"))

        # 沿最后一维（key 方向）归一化：每个 query 对所有 key 的权重加起来等于 1
        attn_weights = torch.softmax(scores, dim=-1)

        # 用这组权重把 V 加权求和，得到这个头"取回来的信息"
        return torch.matmul(attn_weights, V)

    def forward(self, x, wq, wk, wv, wo)->torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # 一次矩阵乘把三个投影全算完。权重按 (out, in) 存，所以左乘前要转置。
        # 注意不是逐头循环：GPU 上一次性做完，比 n_heads 次小矩阵乘快得多。
        q = x @ wq.T  # (B, T, D) @ (D, D) -> (B, T, D)
        k = x @ wk.T
        v = x @ wv.T

        # 拆头：把最后一维 D 摊成 (n_heads, head_dim)，相当于"每人分一段"。
        # view 只改变怎么解释内存，不搬数据，所以这一步几乎不花时间。
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # 把"头"这一维挪到序列前面，变成 (B, H, T, head_dim)。
        # 这样后面的矩阵乘会把 (batch, head) 当成一批同时算，各头之间互不干扰。
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 因果掩码：下三角（含对角线）为 True = 允许看，上三角是"未来"、False = 屏蔽。
        # 对角线必须为 True——每个 token 至少得能看见自己，
        # 否则整行全被屏蔽，softmax 的分母变成 0，结果直接 NaN。
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # 补两个长度为 1 的维度变成 (1, 1, T, T)，靠广播自动对上 (B, H, T, T)。
        # 不用真的复制 B×H 份，省显存。
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = self.attention(q, k, v, mask)

        # 合头：(B, H, T, head_dim) -> (B, T, H, head_dim) -> (B, T, D)
        out = out.transpose(1, 2)
        # transpose 之后内存不连续，直接 view 会报错，先 contiguous 拷成连续的一块
        out = out.contiguous().view(batch_size, seq_len, d_model)

        # 各头的信息在这里汇合，重新混合到同一个坐标系。
        # 少了这层投影，各头只是被硬拼在一起，互相之间没有交流。
        out = out @ wo.T
        return out
