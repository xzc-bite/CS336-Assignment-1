"""全局梯度裁剪（gradient clipping by global L2 norm）。

干的事一句话说完：把所有参数的梯度当成一根超长向量，量出它的长度；
超过 max_l2_norm 就整体等比缩回去，让长度正好落在阈值上。

    total = sqrt( Σ_i Σ_j  g_ij² )          所有参数、所有元素一起算
    if total > max_norm:
        c = max_norm / (total + ε)           只放这一次裁剪系数
        所有梯度 *= c                        方向不变，只改长度

为什么需要它：Transformer 训练里偶尔会遇到一批"梯度特别大"的样本，
Adam 的自适应虽然能压一点，但压不住数量级差几百倍的情况，
一步就能把权重推到不可恢复的区域。裁剪是个便宜的保险丝，
代价只是几次全量读写。

"""

import torch


class GradientClip:
    """可调用对象：构造时绑定参数和阈值，每次 step 之后调一次。

        clipper = GradientClip(model.parameters(), max_l2_norm=1.0)
        loss.backward()
        clipper()                 # 原地改 grad，不需要接收返回值
        optimizer.step()

    做成类而不是函数，是为了把 max_l2_norm 和 ε 存下来复用。
    注意它不持有什么状态，除了构造时传进来的那几个引用。
    """

    def __init__(self, parameters, max_l2_norm, epsilon=1e-6):
        """保存参数引用、阈值与数值保护项。

        参数:
            parameters: 可迭代的 Parameter（通常是 model.parameters()）。
                        类型上接受任何 iterable，但**最好传 list/tuple**：
                        传生成器只会被消费一次，见下面 __call__ 的坑。
                        注意这里不拷贝，存的就是原引用——构造之后再往模型里
                        加参数（比如后插 LoRA），这个 clipper 是看不到的。
            max_l2_norm: float，> 0，梯度的全局 L2 范数上限。
                        典型值 1.0；LLM 预训练常见 0.5 ~ 1.0，
                        小模型或微调可以更小。设成 0 或负数会把梯度直接清零
                        （系数 <= 0，见 __call__ 末尾的说明）。
            epsilon:    float，默认 1e-6，加在分母上防止 total≈0 时系数爆掉。
                        只影响 total 极小的退化情况，正常情况下几乎不起作用。

        返回: 无。
        """
        self.parameters = parameters
        self.max_l2_norm = max_l2_norm
        self.epsilon = epsilon

    def __call__(self):
        """计算全局范数，必要时原地缩放所有梯度。

        返回: None。梯度是被**原地修改**的，调用方不需要赋值回去——
              p.grad 指向的还是同一块内存，改动自动生效。
              代价是拿不到"裁了没有、裁了多少"的反馈，
              想监控训练稳定性就自己在外面先算一遍范数。

        维度变化：
            p.grad              各层形状不一，如 (768, 768)、(768,)、(4096, 768)
            grad.flatten()      统一压成 (numel,)
            torch.cat([...])    拼成 (N,)，N = Σ numel（全部参数的元素总数）
            torch.norm(..., 2)  塌缩成 0 维标量
            clip_coeff          0 维标量，靠广播作用到每个梯度上
            grad.mul_()         原地，形状不变
        """
        # 过滤掉没有梯度的参数：冻结层、这次 forward/backward 没走到的分支
        # （MoE 里没被路由到的专家、条件分支的另一半）grad 都是 None。
        # 注意这里只看 grad 是否为 None，不看 requires_grad——
        # 一个参数被冻结但残留着上一次的 grad，仍然会被计入范数并被缩放。
        # 这正是 torch.nn.utils.clip_grad_norm_ 的行为，对拍时别改。
        #
        # 求全局 L2 范数必须把所有元素放到一起算，不能逐层算完再平均：
        # 每层参数量差两个数量级，(768,) 的 norm 和 (4096,768) 的 norm
        # 直接相加毫无意义。
        grads = [p.grad for p in self.parameters if p.grad is not None]  # 我们求l2范数是对所有元素求的，所以要先把所有元素给flatten

        # 这里做了两件事：flatten 把每层的多维梯度压成一维（norm 本身不看形状，
        # 压平只是为了能 cat），cat 把所有层拼成一根长向量。
        #
        # 这行是整个实现里最贵的地方：它分配了一块 N 个元素的新显存，
        # 相当于把所有梯度又拷了一份。1B 参数 fp32 就是额外 4GB。
        # 想省内存的话改成累加平方和（total += (g**2).sum()），额外空间 O(1)，
        # 代价是多次小 kernel 启动，在 GPU 上反而可能更慢——
        # 官方 clip_grad_norm_ 走的是 foreach/fused 那条路。
        all_grads = torch.cat([grad.flatten() for grad in grads])

        # 0 维标量。用 torch.norm 是历史写法，等价于 torch.linalg.vector_norm(all_grads)。
        # 内部就是 sqrt(Σx²)，一遍 O(N) 扫描。
        grad_l2 = torch.norm(all_grads, 2)

        # 只有超阈值才动手。没超就什么都不做，省掉那次 O(N) 的乘法。
        if grad_l2 > self.max_l2_norm:
            # 系数 = 目标长度 / 当前长度，一次除法搞定。
            # ε 放在分母上（不是给 grad_l2 加个下限），和官方实现一致：
            # grad_l2 接近 0 时系数会被压到接近 0 而不是 inf。
            #
            # 边界：max_l2_norm 传 0 或负数时，系数 <= 0，梯度会被清零甚至反向，
            # 这种情况不该发生，但真传错了不会有人拦你——训练会静默地不动或发散。
            clip_coeff = self.max_l2_norm / (grad_l2 + self.epsilon)

            # mul_ 是原地操作，直接改 p.grad 那块内存，不用写回、不用 detach：
            # p.grad 的 requires_grad 本来就是 False，不会被 autograd 记录。
            # （但如果哪天改成对 p.data 做原地操作，就必须套 no_grad 了。）
            for grad in grads:
                grad.mul_(clip_coeff)

        # 两个会直接抛异常的边界，报错信息都不太好懂，提前知道能省不少时间：
        #
        #   1. 所有参数都没有梯度（没 backward 就调、或者整批样本全被 mask 掉）：
        #      grads 是空 list，torch.cat([]) 抛
        #      ValueError: torch.cat(): expected a non-empty list of Tensors
        #      —— 跟"没有梯度"半毛钱关系没有，第一次见很容易误判。
        #
        #   2. parameters 传的是生成器：第一次 __call__ 就把它遍历空了，
        #      第二次调用同样抛上面那个 ValueError。想复用 clipper，
        #      构造时传 list(model.parameters()) 而不是 model.parameters()。
        #
        # 另外两个只在特定场景冒头的：
        #   - 混合精度：若同一批里既有 fp16 又有 fp32 的 grad，cat 会因 dtype
        #     不一致失败；纯 AMP 场景下 grad 类型是统一的，一般碰不到。
        #   - 稀疏梯度：embedding(..., sparse=True) 产生的是 sparse tensor，
        #     .flatten() 不支持，这条路径需要单独处理（官方实现会去 coalesce）。
        #
        # 复杂度小结：时间 O(N)（cat + norm + mul 各扫一遍，约 3 次全量读写），
        # 空间 O(N)（那块拼接出来的副本）。N 是全部参数的元素总数。
