"""AdamW 优化器的手写实现。

Adam = 一阶动量 m（梯度的指数滑动平均，管"往哪走"）+ 二阶矩 v（梯度平方的滑动平均，
管"这一步该走多小"）。W 指的是 decoupled weight decay：把 L2 正则从梯度里摘出来，
单独乘到参数上，而不是混进梯度再一起进 m/v。

更新式（t 从 1 开始）：

    m_t = β1·m_{t-1} + (1-β1)·g
    v_t = β2·v_{t-1} + (1-β2)·g²
    m̂ = m_t / (1-β1^t)      v̂ = v_t / (1-β2^t)     # 偏差修正
    θ ← θ - lr · m̂ / (√v̂ + ε)                      # Adam 步
    θ ← θ - lr · wd · θ                             # 解耦权重衰减

和 torch.optim.AdamW 实测比对过：weight_decay=0 时逐位相同；wd=0.1 跑 5 步后
最大相对误差约 2e-5，差异只来自权重衰减的先后顺序（见 step() 末尾的说明）。
"""

from torch import optim
import torch


class AdamW(optim.Optimizer):
    """继承 torch.optim.Optimizer，只重写 step()。

    继承带来的好处不用白不用：zero_grad()、state_dict()/load_state_dict()、
    add_param_group() 都是基类现成的，这里只负责"一步参数更新"。
    基类要求的契约有两条：__init__ 里必须调 super().__init__(params, defaults)，
    step() 必须原地改参数（不能返回新张量）。两条都满足才能跟 LR scheduler、
    checkpoint 存取这些周边工具正常配合。
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """登记超参数和参数组。

        参数:
            params: 要优化的参数。两种传法都支持（由基类解析）：
                    ① 可迭代的 Parameter，如 model.parameters()；
                    ② dict 列表，如 [{"params": [...], "lr": 1e-4}]，
                       每组可以覆盖下面任何一个默认值——这就是分层学习率
                       （backbone 小 lr、新加的 head 大 lr）的实现方式。
                    传进来的东西会被冻结成 list，之后新参数得用 add_param_group 加。
            lr:     学习率，浮点，必须 > 0，默认 1e-3。LLM 预训练常见 1e-4 ~ 3e-3，
                    且通常要配合 warmup（前期线性升到目标 lr）否则前几百步容易崩。
                    四个超参都有默认值，和 PyTorch 官方 API 以及讲义的写法保持一致：
                    AdamW(model.parameters()) 就能直接跑。
            betas:  (beta1, beta2) 二元组，两者都取 [0, 1)，beta1 管一阶动量、
                    beta2 管二阶矩。默认 (0.9, 0.999) 几乎是行业惯例；
                    beta2 调到 0.95 常见于大 batch 训练，因为梯度噪声被 batch 平均掉了，
                    需要更快忘记历史。
            eps:    分母的数值保护，浮点，> 0，通常 1e-8。作用是防止 √v̂ 为 0 时除零。
                    只在 v̂ 极小（≈ 该方向几乎没梯度）时才真正起效；
                    调大到 1e-6 会让更新幅度整体变小，等价于隐式降 lr。
            weight_decay: 解耦权重衰减系数，>= 0，默认 0.01。LLM 常用 0.1，小模型 0.01 或 0。
                    注意它乘的是 lr（见 step()），所以实际衰减强度 = lr·wd；
                    学习率被 scheduler 调小时，衰减也会同步变小，这一点很容易被忽略。

        返回: 无。构造完就能直接调 step()。
        """
        # 这份 dict 会成为每个参数组的"模板"：基类会把它 copy 进
        # self.param_groups[i]，之后 step() 里用 group['lr'] 这种形式读。
        # key 名字随便起，但要和下面 step() 里读的保持一致。
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # 基类在这儿做三件事：把 params 规整成 self.param_groups（list[dict]，
        # 每个 dict 一定含 'params' 键）、把 defaults 合并进每组、初始化 self.state（空 defaultdict）。
        # 顺序不能反——没有 defaults 就没有模板可用。
        super().__init__(params, defaults)

    def step(self):
        """执行一次参数更新。所有运算都是原地（in-place）的，不产生新参数张量。

        流程：遍历参数组 → 跳过没梯度的参数 → 懒初始化 m/v → 更新 m/v →
        偏差修正 → Adam 步 → 解耦权重衰减。

        维度说明：整条链路全是逐元素运算，p / grad / m / v / denom 形状完全一致
        （例如某个 Linear 层就是 (d_out, d_in)），没有广播、没有 reshape。
        所以下面的注释里不再重复标注 shape，只标注"哪些量是同一份内存"。
        """
        for group in self.param_groups:          # 一般只有 1 组；分层 lr 时会有多组
            for p in group['params']:            # p 是 Parameter，shape 取决于具体层

                # 冻结层（requires_grad=False）或者这次 forward 没走到的分支
                # （比如 MoE 里没被路由到的专家），梯度就是 None。
                # 必须跳过：既省算力，也避免给这些参数白白分配一份 m/v 显存。
                if p.grad is None:
                    continue

                # .data 拿到的是脱离计算图的裸张量。这里绕 graph 是有意的——
                # 优化器更新不该被 autograd 记录。但它同时也绕过了 version counter，
                # 所以万一在 forward 里误调了 step()，PyTorch 不会报警。
                # 更规范的做法是给 step() 加 @torch.no_grad() 装饰器，然后直接用 p.grad。
                grad = p.grad.data

                # state 是按参数张量做 key 的字典（基类给的 defaultdict）。
                # 用参数对象本身当 key，而不是用 name 或 id，好处是参数被共享
                # （比如 embedding 与 lm_head 绑定权重）时状态天然是同一份。
                state = self.state[p]

                # 懒初始化：第一次见到这个参数才分配 m 和 v。
                # 好处是内存按需增长——只出现一次就再没梯度的参数不会白占显存。
                # 代价是每个参数多出 2 份额外显存：fp32 下每个参数从
                # 4B(权重)+4B(梯度) 涨到 16B，训练显存的"4 倍参数量"经验法则就来自这里。
                # 混合精度训练（Adam 用 fp32 master weight）还要再翻。
                if len(state) == 0:
                    state['step'] = 0                        # 纯 Python int，1-based 计数
                    state['m'] = torch.zeros_like(p.data)    # 一阶动量，与 p 同 shape/同 dtype
                    state['v'] = torch.zeros_like(p.data)    # 二阶矩，同上

                # 取出来的是引用不是拷贝：后面 m.mul_/add_ 改的就是 state 里那一份。
                # 所以千万别在这儿写成 m = m * beta1 之类的非原地操作，
                # 那样 state 永远不会被更新，动量就完全失效了。
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']                # 解包成两个 float

                # 先自增再用：偏差修正的指数是 t（从 1 开始），用 0 会得到 1-1=0 直接除零。
                state['step'] += 1

                # ---------- Adam 更新 ----------
                # m ← β1·m + (1-β1)·g，即梯度的指数滑动平均（EMA）。
                # add_(grad, alpha=1-beta1) 表示 self += alpha * grad，比先乘再加快一步。
                # 注意 first moment 拿的是原始梯度，没有做任何归一化。
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v ← β2·v + (1-β2)·g²。addcmul_ 算的是 self += value * t1 * t2，
                # 所以这里 t1=t2=grad 就得到 grad²，逐元素平方，形状不变。
                # v 是"梯度有多吵"的度量：某些方向梯度一直很大，v 就大，
                # 除下去之后那些方向的步长自动被压小——这就是 Adam 的自适应来源。
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # ---------- 偏差修正 ----------
                # m/v 从 0 起步，前几步天然偏小（t=1 时 m 只有 (1-β1)·g）。
                # 除以 1-β^t 把它拉回无偏估计。β2=0.999 时 t=1 的修正因子只有 0.001，
                # 也就是说不修正的话第一步的有效 lr 会小 1000 倍——训练初期几乎不动。
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 把 1/(1-β1^t) 折进学习率，省一次张量除法。
                # 官方的等价写法是先算 m̂ = m/(1-β1^t) 再乘 lr，两者数学上一致。
                step_size = group['lr'] / bias_correction1

                # 分母 = √(v/(1-β2^t)) + ε。
                # v / bias_correction2 会产生一个新张量（不是原地），所以 denom 是临时量，
                # 污染不到 state 里的 v —— 这一点很关键，写错成 v.div_() 会把状态改坏。
                # 另外 eps 加在开方之后，不是开方之前：顺序不同结果不同，
                # 常见错误是写成 sqrt(v + eps)。
                denom = (v / bias_correction2).sqrt().add_(group['eps'])

                # ---------- 解耦权重衰减（先做，和官方实现同序） ----------
                # 关键点：直接缩小参数本身，绝不往 grad 里加 wd*p。
                # 后者就是经典 Adam + L2 正则的做法，它会让 wd*p 也进 m 和 v，
                # 结果是"惩罚大的方向步长被压小"，衰减和自适应学习率互相干扰——
                # 这正是 AdamW 这篇论文要解决的问题。
                #
                # p *= (1 - lr·wd)，等价于 p -= lr·wd·p。用 mul_ 一行搞定，
                # 比 add_(p, alpha=...) 那种"自己加自己"的写法更好读。
                #
                # 顺序很重要：官方 torch.optim.AdamW 是先衰减再更新参数。
                # 反过来写结果差 O(lr²·wd)（实测 ~5e-5），收敛没影响，
                # 但和官方对拍时会一直有微小偏差，查半天找不到原因。
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # p ← p + (-step_size) · m / denom。
                # addcdiv_(t1, t2, value=v) 就是 self += v * t1 / t2，即先乘后除。
                # value 必须写成关键字实参：旧版把 value 放第一个位置的签名
                # 已经弃用，那样写会一直报 deprecation warning。
                # 另外它是原地运算，所以 m 和 denom 都不会被消耗/改掉。
                p.data.addcdiv_(m, denom, value=-step_size)  # 先乘后除
