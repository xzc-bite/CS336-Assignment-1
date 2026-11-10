import torch
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    softmax：沿 dim 把一组实数压成概率分布，输出非负且沿该维加起来等于 1。
    公式：
    out = exp(x - x_max) / sum(exp(x - x_max))
    先减最大值只是为了数值稳定（exp 不会上溢），数学上和不减完全等价。

    Args:
        x: 任意形状的浮点张量。形状标签写成 (B, T, D) 只是最典型的用法，
           实际对 1D / 2D / 4D 都成立——只对 dim 那一维归一化，其余维度不动。
        dim: 沿哪一维归一化。注意负数下标（-1）比写死 2 更抗形状变化。
    output:
        out: 与 x 形状、dtype 完全相同，沿 dim 求和为 1。
    """
    x_max = x.max(dim=dim, keepdim=True)[0]  # keepdim 保住维度，才能广播回 x；例如 3D 输入时这里是 (B, T, 1)
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)