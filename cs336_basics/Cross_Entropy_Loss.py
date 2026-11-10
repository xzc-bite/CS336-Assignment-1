# 交叉熵损失：把"模型对正确答案给出的概率有多低"量化成一个标量。
#
# 数据流一路看下来是这样的：
#   logits (B, V)  --log_softmax-->  log_probs (B, V)  --按标签取一列-->  log p (B,)  --求和取负-->  scalar
#   V = vocab_size，B = batch_size（也可以理解成"一句话里被拉平的 B*T 个位置"）。
#
# 为什么用 log_softmax 而不是 softmax 再取 log：softmax 在大 logits 下会把
# 非最大值的位置下溢成 0，接下来 log(0) = -inf，整个 loss 变 inf、梯度全 nan。
# log_softmax 内部用 logsumexp 计算，全程留在对数域，不会有这一步。
# 数学上 -log softmax(z)_i = logsumexp(z) - z_i，两者完全等价，只是数值表现不同。

import torch


class CrossEntropyLoss:
    """交叉熵损失的最小实现。

    典型用法（注意它把数据存在 __init__ 里，forward 不带参数）：

        loss_fn = CrossEntropyLoss(logits, targets)
        loss = loss_fn.forward()

    这种"构造时喂数据、调用时算结果"的写法很像 Keras 早期或教材示例，
    和 PyTorch 的 nn.Module 惯例（forward(inputs, targets)）不一样，
    所以它没法直接塞进 nn.Sequential 或 DataParallel，也不能重复用于不同批次。
    """

    def __init__(self, inputs, targets):
        """记录输入并把两个维度缓存下来。

        参数:
            inputs: 模型的原始输出 logits，形状 (B, V)，浮点类型。
                    取值范围是 (-inf, +inf)——不是概率，不需要提前归一化，
                    softmax 会负责这一步。命名上叫 inputs 有点含糊，
                    更贴切的名字是 logits 或 scores，读到后面的 shape[1]
                    才知道它其实是"类别维在最后一维"。
            targets: 每个样本的正确类别下标，形状 (B,)，整型（int64/long）。
                    取值范围 [0, V-1]，越界会在下面高级索引那一步直接报 IndexError。
                    这里不做 one-hot，因为交叉熵只需要"正确类别那一个位置"的概率，
                    做成 one-hot 再乘一遍是纯粹的内存浪费。
        """
        self.inputs = inputs
        self.targets = targets

        # shape[1] 就是类别数 V。这个假设写死了"类别在最后一维"，
        # 所以输入必须是二维的——如果哪天传进 (B, T, V)，shape[1] 会悄悄变成 T，
        # 算出来的 loss 不会报错但完全没意义。要支持 (B, T, V) 得先 reshape(-1, V)。
        self.vocab_size = inputs.shape[1]

        # B。后面构造 range(B) 用它，同理只在二维输入下成立。
        self.batch_size = inputs.shape[0]

    def forward(self):
        """计算损失，返回一个 0 维标量张量（带梯度）。

        返回:
            torch.Tensor，0 维标量，是 **batch 内的平均** loss（与 F.cross_entropy 的
            默认 reduction="mean" 一致）。平均值的好处是不随 batch 大小漂移，
            调学习率时不会因为改了 batch 就跟着改有效步长。
        """
        # (B, V) -> (B, V)：沿类别维在**对数域**归一化。
        # 输出的第 i 行是 log softmax(z_i)，取 exp 之后每行加起来才是 1。
        # dim=1 是写死的，因为上面已经假设类别在最后一维；
        # 用 1 而不是 -1，在这里等价，但换成 -1 会更"抗形状变化"一些。
        # 内部先减最大值再取 exp，所以 logits 大到 1e3 量级也不会溢出。
        log_probs = torch.log_softmax(self.inputs, 1)

        # 高级索引（advanced indexing）：两个索引张量按位置配对，
        # 取出的第 i 个元素是 log_probs[i, targets[i]]，
        # 也就是"第 i 个样本被分给正确类别的对数概率"。
        # 形状从 (B, V) 塌缩成 (B,)——这是整个函数里唯一的降维操作。
        # 这里的 range(B) 是 Python 内建 range，不是 torch.arange：
        # 结果对，但它会在 torch.compile / CUDA graph 捕获时踩坑（要求静态 Python 对象），
        # 工程代码里更稳妥的写法是 torch.arange(B, device=self.inputs.device)。
        log_p = log_probs[range(self.batch_size), self.targets]

        # 取负再求平均，即 -mean(log p_i)。
        # 因为 log_p 已经在对数域里，这一步没有任何 log(0) 的风险——
        # 哪怕正确答案的概率小到 float32 存不下，log_p 也还是个正常的负数。
        # 用 mean 而不是 sum：丢掉的那个 1/B 因子会让梯度整体放大 B 倍，
        # 相当于学习率被隐式乘了 batch size，而且 loss 数值会随 batch 大小漂移，
        # 跨配置的曲线就没法直接比。讲义和 test_cross_entropy 都要求平均值。
        return -log_p.mean()

    # 关于反向传播：这个类没有写 backward，也不需要——
    # log_softmax、索引、mean 全是可微的 torch 算子，autograd 会自动串起来。
    # 手推出来的梯度很简洁：dL/dlogits = softmax(z) - one_hot(targets)，
    # 也就是"预测分布减去正确答案"，这也是它和 softmax 天然绑定的原因。
    #
    # 还有几个这个实现没覆盖、但作业里会碰到的点：
    #   1. ignore_index：padding 位置的标签不该计入 loss，这里没有跳过机制，
    #      要把 mask 乘进去或者提前筛掉对应行。
    #   2. 归约方式：这里固定返回平均，调用方要"总共多少"得自己乘回 batch_size。
    #   3. 和 F.cross_entropy 的关系：默认 reduction="mean"，数值上完全一致。
