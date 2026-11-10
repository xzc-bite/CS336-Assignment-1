import torch
from torch import nn

"""
模型最两头的两个零件，一进一出：

    token id ──Embedding──> 一串向量 (B, T, d_model) ── ...中间若干层... ──> Linear ──> 每个词一个分数

两个类都只负责"把数据从一种形状变成另一种形状"，参数怎么更新不归它们管，交给优化器。
"""


class LinearModule(nn.Module):
    """
    线性层：把每个 d_in 维向量变成 d_out 维。

    算法拆到最朴素的样子：输出的每一个数，都是输入所有数乘上对应权重再加起来。

        输入 x = [1, 2]，权重 W = [[3, 4]]
        输出 = 1*3 + 2*4 = 11

    写成矩阵就是 y = x @ W.T（@ 表示矩阵乘法）。
    d_out 有多大，就配多少套这样的权重，每一套负责输出里的一个数。
    这些权重一开始是随机的，训练就是让它们慢慢变成"合适"的值。

    按讲义要求不带 bias（现代 LLM 的通行做法）。
    """

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # 两头的维度都存下来，算初始化标准差时要用
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # 形状 (out, in)：第 i 行是"产出第 i 个数"所用的那套权重。
        # torch.empty 只占内存、不填值，里面是随机的垃圾数据；
        # 紧接着下面会重新初始化，所以这些垃圾值不会留下来。
        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))

        # 初始化为什么要讲究：权重太大会让每层输出被逐层放大，几下就爆成 inf；
        # 太小又会让信号逐层衰减到接近 0，梯度跟着消失。
        # Xavier 的做法是让"输入个数 + 输出个数"共同决定尺度，使前向输出的波动不随层数漂移。
        # 讲义给的结论是方差 2/(in+out)——注意这是方差，标准差要再开一次根号。
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        # 截断正态：按正态分布取值，但超出 ±3σ 的直接丢掉重抽，
        # 免得偶尔一个离群值就把某条通路的激活值顶飞。
        # trunc_normal_ 的 a / b 填的是绝对数值，不是"几倍标准差"。
        torch.nn.init.trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 转置的原因：W 按 (out, in) 存，而 x 的最后一维是 in，两边得对齐才能相乘。
        # 前面的 batch 维、序列维原样保留，只有最后一维从 in 变成 out。
        return x @ self.W.T


class EmbeddingModule(nn.Module):
    """
    嵌入层：把每个 token id 换成一行向量。本质就是查表。

        表里一行一个 token：
            id 0 -> [ 0.11, -0.42, ... ]
            id 1 -> [-0.83,  0.07, ... ]
            id 2 -> [ 0.55,  0.91, ... ]

    整张表形状是 (vocab_size, d_model)。查表就是拿 id 当行号把那一行取出来。

    输入任意形状的整数张量，输出在最后多一维 d_model：
        (2, 3) 的 id  ->  (2, 3, d_model)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings  # 词表大小 vocab_size，也就是表的行数
        self.embedding_dim = embedding_dim  # 每个 token 的向量维度 d_model，也就是表的列数
        self.device = device
        self.dtype = dtype

        # 整张表一次性建好：行数 = 词表大小
        self.embedding_matrix = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype))

        # 起步是随机数，语义是训练出来的：老是出现在相似上下文里的 token，
        # 向量会慢慢靠近，于是"意思相近"就变成了"几何上接近"。
        # 讲义给的初始化：均值 0、标准差 1、截断在 [-3, 3]。
        std = 1
        torch.nn.init.trunc_normal_(self.embedding_matrix, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 用张量做下标，PyTorch 会按元素逐个去取：token_ids 里每个数都当行号。
        # 所以 token_ids 必须是整数（int64），传浮点会直接报错。
        # 输出形状 = token_ids 的形状 + (embedding_dim,)：每个 id 换来的是整整一行。
        # 反传时只有这一批真正取到的行会拿到梯度，没出现过的行保持原样（稀疏更新）。
        return self.embedding_matrix[token_ids]
