"""学习率调度：线性 warmup + 余弦退火 + 尾部平台。

为什么长这样（三段各自的意图）：

    lr
    max |      /‾‾‾--..__
        |     /           ‾--..__
        |    /                   ‾--..__
    min |   /                          ‾‾‾‾‾‾‾‾‾‾  ← 平台
        +----------------------------------------→ iteration
           warmup    cosine decay      flat

1. warmup（0 → max）：模型刚开始参数是随机的，梯度方向乱且大。一上来就用大 lr，
   更新步长会盖过参数本身的量级，几十步内 loss 就能飙到 nan。线性升上去给模型
   一个"先对齐方向再加速"的机会。
2. 余弦退火（max → min）：前期大步快跑、后期小步精修，形状比阶梯式衰减平滑，
   不会在跳档那一刻抖一下。cos 曲线两端斜率接近 0，所以端点是平滑衔接的。
3. 尾部平台（min）：训练末尾保持一个下限而不是降到 0，避免最后完全不更新。

用法：调用即得 lr，返回值需要自己写回优化器——

    sched = CosineSchedule(1e-3, 1e-4, warmup_iters=100, cosine_cycle_iters=10000)
    for it in range(num_iters):
        for g in opt.param_groups:
            g['lr'] = sched(it)
        ...  # 正常训练一步

整个类全程是 Python 标量运算，不涉及任何张量，所以没有 shape 流转的问题；
"维度"上的唯一注意点见 __call__ 结尾的说明。
"""

import math


class CosineSchedule:
    """一个可调用对象：喂迭代数，返回该步的学习率。

    刻意没继承 torch.optim.lr_scheduler.LRScheduler，也没包成 nn.Module：
    它是个纯函数式的计算器，不持有优化器引用、不自己改参数组。
    好处是可以随便单测、可以和任意训练循环组合；代价是得手动把返回值
    写回 param_groups（见文件头的用法示例）。
    """

    def __init__(self, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
        """

        参数:
            max_learning_rate: 峰值学习率，float，> 0。warmup 结束那一刻达到它。
                               同时它也是"余弦的起点"。典型值 1e-4 ~ 3e-3。
            min_learning_rate: 学习率下限，float，>= 0，必须 <= max_learning_rate。
                               余弦走完后的平台值。设成 0 意味着后期完全冻结参数，
                               一般是想要"训练收敛后不再动"才这么干；多数情况留
                               峰值的一个零头（如 max/10，或 1e-5 这种固定小值）。
            warmup_iters:      warmup 持续多少步，int，>= 0。常见是总步数的 1%~5%，
                               小模型几百步、大模型上千步。设 0 表示跳过 warmup
                               （能跳过，但模型会直接从 max 起步，小模型还好，
                               大模型容易炸）。
            cosine_cycle_iters: 余弦衰减的结束步（绝对步数，不是持续步数），
                               int，必须 >= warmup_iters。也就是说衰减区间是
                               [warmup_iters, cosine_cycle_iters]，长度 = 两者之差。
                               常见坑：把它当成"衰减总步数"来传，结果整个调度被压缩
                               到了错误的区间里。
        """
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def __call__(self, it):
        """
        it: 当前迭代次数，int（传 float 也能跑，会得到 float 结果），
            取值范围理论上是 [0, +inf)。语义是从 0 开始数的"第几步"，
            不是"已完成的步数"。两者差 1，会让整个曲线偏一步，
            对得上官方实现时要注意这个约定。

        返回: float，该步的学习率。取值恒落在 [min_learning_rate, max_learning_rate]
              区间内（余弦本身的值域保证的，不需要额外 clamp），不会返回负数。
        """
        # 防御：负步数没有任何意义，但断点续训读错 step 时会悄悄返回负 lr，
        # 优化器于是反着更新、loss 一路上升。clamp 到 0 比让训练静默跑偏好。
        if it < 0:
            it = 0

        # 防御：warmup 和余弦终点重合时，下面余弦分支的分母是 0（实测 ZeroDivisionError）。
        # 这种配置等价于"跳过衰减"，直接把整个区间压成 min 更符合意图。
        if self.cosine_cycle_iters <= self.warmup_iters:
            return self.min_learning_rate if it >= self.warmup_iters else self.max_learning_rate * it / max(self.warmup_iters, 1)

        # 第一段：线性升。it=0 时返回 0——第一步 lr 为 0，等于原地踏步一次。
        # 这是有意的（有些实现写成 (it+1)/warmup 让第一步就有 lr），
        # 差别微不足道，但对拍测试时会因为这 1 步对不上。
        if it < self.warmup_iters:
            return self.max_learning_rate * it / self.warmup_iters

        # 第三段：平台。用 > 而不是 >=，是为了让 it == cosine_cycle_iters 走下面的
        # 余弦分支——那里 cos(π)= -1，结果正好等于 min_learning_rate，
        # 所以两个分支在这个点上取值一致，边界连续、不会出现跳变。
        elif it > self.cosine_cycle_iters:
            return self.min_learning_rate

        # 第二段：余弦衰减，把 [warmup_iters, cosine_cycle_iters] 映射到 [max, min]。
        # 拆开看它的形状：
        #   (it - warmup) / (cycle - warmup)  →  进度 p ∈ [0, 1]
        #   cos(π·p)                          →  +1 平滑降到 -1
        #   (1 + cos)/2                       →  归一化到 [1, 0]，这就是"余弦"的名字来源
        #   最后映射回 [min, max]
        # 用 (1+cos)/2 而不是直接 cos，是为了保证不管 max/min 是什么量级，
        # 系数都严格在 [0,1] 内，曲线不会冲出上下界。
        else:
            return self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * (1 + math.cos(math.pi * (it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters))) / 2

        # 边界与易错点：
        #   1. it 传负数：已经 clamp 到 0，不会再返回负 lr。
        #   2. warmup_iters >= cosine_cycle_iters：函数开头就拦下来了，返回 min（或 warmup 段），
        #      不会走到除零那一步。
        #   3. warmup_iters == 0 是安全的：it >= 0 时第一段不会被进入，
        #      不会有除零；此时 it=0 由余弦分支给出，值正好是 max_learning_rate。
        #   4. 关于"维度"：这里返回的是一个标量。如果模型分了多个 param_group
        #      （比如 backbone 和 head 用不同 lr），别直接把这个标量塞进去覆盖掉
        #      各组的差异，应该按组做比例缩放，例如 g['lr'] = base * sched(it) / max_lr。
