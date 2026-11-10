# 学习率调度：Cosine Schedule

> 本文面向零基础读者，结合 CS336 Assignment 1 的 `learning_rate_schedule` 任务和仓库里的 `Lr_Cosine_Schedule.py`，讲清楚 **学习率为什么要变、Cosine Schedule 的三个阶段、常见策略怎么选、代码怎么接进训练循环**。
>
> 读之前最好知道：
>
> - 学习率 $\eta$ 决定每次参数更新走多远；
> - 训练是按 iteration（step）推进的，不是按 epoch；
> - 优化器（如 AdamW）负责“怎么走”，学习率负责“走多大步”。
>
> 读完你应该能回答五个问题：
>
> 1. 固定学习率有什么问题？为什么训练前期和后期需要不同的学习率？
> 2. Cosine Schedule 的预热、余弦衰减、结束后三个阶段分别在做什么？
> 3. 公式里的 $T_w$、$T_c$、$\alpha_{max}$、$\alpha_{min}$ 怎么选？
> 4. Step Decay、Exponential Decay、Cosine Annealing 各有什么优缺点？
> 5. CS336 的 `test_get_lr_cosine_schedule` 在测什么？代码怎么接进训练循环？

## 目录

- [0. 阅读前的准备](#0-阅读前的准备)
- [1. 为什么学习率需要调度](#1-为什么学习率需要调度)
- [2. Cosine Schedule 的三阶段](#2-cosine-schedule-的三阶段)
- [3. 数学细节与边界情况](#3-数学细节与边界情况)
- [4. 常见学习率策略对比](#4-常见学习率策略对比)
- [5. 参考代码与训练集成](#5-参考代码与训练集成)
- [6. 联系 CS336 Assignment 1](#6-联系-cs336-assignment-1)
- [7. 常见误区 FAQ](#7-常见误区-faq)
- [8. 小结与自测](#8-小结与自测)
- [9. 延伸阅读](#9-延伸阅读)

---

## 0. 阅读前的准备

### 0.1 一句话版本

> **Cosine Schedule 就是让学习率先“慢慢热起来”，再沿着半个余弦曲线“平滑冷下去”，最后保持在一个很小的值继续微调。**

三个阶段：

```text
学习率
  ▲
  │        ╭────────╮
  │      ╱            ╲
  │    ╱                ╲
  │  ╱                    ╲────────
  │╱
  └──┬────────┬──────────┬────────► iteration
     0       T_w         T_c
     线性预热   余弦退火     保持最小值
```

一句话记忆：

> **预热：别一上来就猛踩油门；余弦：该松油门时平滑地松；收尾：留一点小油门继续微调。**

### 0.2 符号表

| 符号 | 含义 | 一句话解释 |
| --- | --- | --- |
| $t$ | 当前 iteration | 第几次参数更新，从 0 开始 |
| $\alpha_t$ | 第 $t$ 步的学习率 | 这次更新走多大步 |
| $\alpha_{max}$ | 最大学习率 | 预热结束后的峰值 |
| $\alpha_{min}$ | 最小/最终学习率 | 余弦结束后的下限 |
| $T_w$ | warmup iterations | 预热持续多少步 |
| $T_c$ | cosine cycle iterations | 余弦退火到哪一步结束 |

### 0.3 训练循环里的位置

```text
for it in range(total_iters):
    lr = scheduler(it)                 # 1) 算这一步的学习率
    for group in optimizer.param_groups:
        group["lr"] = lr               # 2) 写给优化器

    optimizer.zero_grad()
    logits = model(inputs)             # 3) 前向
    loss = loss_fn(logits, targets)
    loss.backward()                    # 4) 反向
    optimizer.step()                   # 5) 按当前 lr 更新
```

学习率调度器**不更新参数**，它只回答一个问题：

> **“第 `it` 步，我该用多大的学习率？”**

### 0.4 学习路线

| 你想知道 | 看哪节 |
| --- | --- |
| 为什么不能固定学习率 | §1 |
| 预热/余弦/收尾分别干什么 | §2 |
| 公式和边界怎么处理 | §3 |
| 和其他策略比怎么样 | §4 |
| 代码怎么跑、怎么接训练 | §5 |
| 作业怎么过 | §6 |
| 常见坑 | §7、§8 |

---

## 1. 为什么学习率需要调度

### 1.1 生活化类比：学投篮

想象你学投篮：

- **刚开始**：动作框架都没建立，应该大胆试、大幅度调整；
- **中期**：基本动作有了，需要逐渐减小调整幅度，精雕细琢；
- **后期**：已经很接近目标，只需要很小的微调，否则一用力就又偏了。

如果全程用同一个“调整幅度”：

- 前期太小 → 学得慢，半天找不到方向；
- 后期太大 → 一直在好答案附近来回跳，落不下来。

训练神经网络是一样的：

> **前期需要大学习率快速下降，后期需要小学习率精细收敛。**

### 1.2 固定学习率的问题

固定学习率 $\eta$ 有三个典型问题：

1. **前期太小**：损失下降慢，训练时间长；
2. **后期太大**：在最优点附近震荡，loss 下不去，甚至发散；
3. **对初始化/尺度敏感**：不同阶段的最优步长不一样，一个固定值很难兼顾。

> 类比：开车全程定速巡航。起步时太慢，快到目的地时又冲过头。

### 1.3 学习率调度是什么

学习率调度器就是一个函数：

$$
\alpha_t=f(t)
$$

输入当前 step $t$，输出这一步该用的学习率。

最简单的调度是“常数”：

$$
\alpha_t=\alpha_0
$$

但它没有解决“前期/后期需求不同”的问题。于是有了各种形状的调度：

- Step Decay：到点突然降；
- Exponential Decay：按指数平滑降；
- Cosine Annealing：按余弦曲线平滑降；
- Warmup + Cosine：先线性升，再余弦降。

### 1.4 一个重要的单位问题：iteration 还是 epoch

调度器通常按 **iteration（参数更新次数）** 工作，不是 epoch。

- 一个 epoch = 把训练集过一遍；
- 一个 epoch 包含 `steps_per_epoch` 次参数更新；
- 如果按 epoch 设计调度，要换算：

$$
t=\text{epoch}\times\text{steps\_per\_epoch}+\text{step\_in\_epoch}
$$

> CS336 的测试直接用 `it=0,1,2,...`，所以本文所有 $t$ 都指 iteration。

### 1.5 调度器和优化器的分工

| 组件 | 负责 |
| --- | --- |
| 优化器（AdamW/SGD） | 根据梯度决定“往哪走、怎么走” |
| 学习率调度器 | 决定“这一步走多大” |
| 梯度裁剪 | 防止某一步梯度过大 |
| 损失函数 | 告诉模型“现在错得多离谱” |

> 调度器只改 `param_group["lr"]`，不碰梯度、不碰动量。

---

## 2. Cosine Schedule 的三阶段

CS336 使用的 Cosine Schedule 是 **linear warmup + cosine annealing + post-annealing clamp**。

### 2.1 阶段一：线性预热（$t<T_w$）

公式：

$$
\alpha_t=\frac{t}{T_w}\alpha_{max}
$$

- $t=0$：学习率 0；
- $t=T_w$：学习率 $\alpha_{max}$；
- 中间线性上升。

**为什么需要预热？**

1. **模型刚初始化**：参数随机，梯度可能很大且方向不稳定；
2. **Adam 的 $m$、$v$ 从 0 开始**：早期估计不准，虽然偏差修正有帮助，但步子太大仍可能出事；
3. **大学习率 + 大梯度**：容易一步把参数打飞，loss 变 `nan`；
4. **让训练先“站稳”**：用小步试探，再逐渐加速。

> 类比：运动前先热身，不要一上来就百米冲刺。

**预热多久？**

- 常见是总步数的 1%–5%；
- 小模型可以更短，大模型/大 batch 往往更长；
- CS336 测试用 7/21 ≈ 33%，只是为了测试数值，不代表实际比例。

### 2.2 阶段二：余弦退火（$T_w\le t\le T_c$）

公式：

$$
\alpha_t=\alpha_{min}+\frac{1}{2}\left(1+\cos\left(\pi\frac{t-T_w}{T_c-T_w}\right)\right)(\alpha_{max}-\alpha_{min})
$$

令进度：

$$
p=\frac{t-T_w}{T_c-T_w}\in[0,1]
$$

则：

$$
\alpha_t=\alpha_{min}+\frac{1+\cos(\pi p)}{2}(\alpha_{max}-\alpha_{min})
$$

几个关键点：

| 位置 | $p$ | $\cos(\pi p)$ | 学习率 |
| --- | ---: | ---: | ---: |
| 开始 | 0 | 1 | $\alpha_{max}$ |
| 中点 | 0.5 | 0 | $(\alpha_{max}+\alpha_{min})/2$ |
| 结束 | 1 | -1 | $\alpha_{min}$ |

**形状特点**：

- 开始和结束时曲线是“平的”（导数为 0），中间下降最快；
- 前期在最大值附近停留一段时间，充分探索；
- 后期在最小值附近缓慢收尾，适合精细收敛；
- 没有突然的台阶，训练曲线通常更平滑。

> 类比：开车快到目的地时，不是猛踩刹车，而是平滑地松油门。

### 2.3 阶段三：结束后保持最小值（$t>T_c$）

公式：

$$
\alpha_t=\alpha_{min}
$$

为什么还要训练？

- 余弦结束后，模型通常还没“完全训完”；
- 保持一个很小的学习率，继续微调；
- 如果直接降到 0，后续数据就完全学不进去了。

> CS336 的测试里，$t>21$ 之后学习率一直是 0.1。

### 2.4 三个阶段的衔接

```text
t = 0                  T_w                 T_c
│                       │                   │
│  线性上升              │   余弦下降         │   保持 min
│  α = t/T_w * α_max    │   α: max → min    │   α = α_min
│                       │                   │
└───────────────────────┴───────────────────┴────► t
```

两个边界：

- $t=T_w$：预热结束，余弦开始；两边都等于 $\alpha_{max}$，值连续；
- $t=T_c$：余弦结束，之后保持 $\alpha_{min}$；两边都等于 $\alpha_{min}$，值连续。

> 值连续，但 $t=T_w$ 处斜率不一定连续：预热是直线上升，余弦在起点斜率为 0。实际训练中这通常没问题。

### 2.5 参数怎么选

| 参数 | 建议 |
| --- | --- |
| $\alpha_{max}$ | 通过小规模实验找“能降 loss 但不发散”的最大值 |
| $\alpha_{min}$ | 通常是 $\alpha_{max}$ 的 1/10 左右 |
| $T_w$ | 总步数的 1%–5%；大模型可适当加长 |
| $T_c$ | 通常设为总训练步数；如果想留出“收尾期”，可以略小于总步数 |
| $t$ 的起点 | 作业测试从 0 开始；实际训练要和日志/checkpoint 对齐 |

> 如果 $T_c$ 大于总步数，学习率不会降到 $\alpha_{min}$；如果 $T_c$ 小于总步数，后面会一直保持 $\alpha_{min}$。

---

## 3. 数学细节与边界情况

### 3.1 为什么用“半个余弦”

余弦函数在 $[0,\pi]$ 上从 1 平滑降到 -1：

$$
\cos(0)=1,\qquad \cos(\pi)=-1
$$

映射到 $[0,1]$：

$$
\frac{1+\cos(\pi p)}{2}
$$

- $p=0$：1；
- $p=0.5$：0.5；
- $p=1$：0。

再映射到 $[\alpha_{min},\alpha_{max}]$：

$$
\alpha_t=\alpha_{min}+\text{map}(p)(\alpha_{max}-\alpha_{min})
$$

### 3.2 导数与“平滑”

对 $p$ 求导：

$$
\frac{d}{dp}\left(\frac{1+\cos(\pi p)}{2}\right)
=-\frac{\pi}{2}\sin(\pi p)
$$

- $p=0$：导数为 0（起步平缓）；
- $p=0.5$：导数绝对值最大（中段下降最快）；
- $p=1$：导数为 0（收尾平缓）。

这就是“两端慢、中间快”的来源。

### 3.3 边界情况

| 情况 | 处理 |
| --- | --- |
| $T_w=0$ | 没有预热；$t=0$ 直接取 $\alpha_{max}$；注意不要除以 0 |
| $T_c=T_w$ | 余弦分母为 0；需要单独处理（直接返回 $\alpha_{min}$ 或报错） |
| $t<0$ | 作业不测；实际可 clamp 到 0 |
| $t>T_c$ | 返回 $\alpha_{min}$ |
| $\alpha_{max}<\alpha_{min}$ | 参数写反了；检查配置 |
| `cos` 的角度单位 | Python 的 `math.cos` 用弧度，不是角度 |

### 3.4 一个数值表

参数：$\alpha_{max}=1,\alpha_{min}=0.1,T_w=7,T_c=21$（CS336 测试参数）。

| $t$ | $\alpha_t$ | 阶段 |
| ---: | ---: | --- |
| 0 | 0.0000 | 预热起点 |
| 1 | 0.1429 | 预热 |
| 3 | 0.4286 | 预热 |
| 6 | 0.8571 | 预热 |
| 7 | 1.0000 | 预热结束/余弦起点 |
| 10 | 0.9018 | 余弦 |
| 14 | 0.5500 | 余弦中点 |
| 18 | 0.1982 | 余弦 |
| 21 | 0.1000 | 余弦结束 |
| 22 | 0.1000 | 保持 |
| 24 | 0.1000 | 保持 |

这张表就是 `test_get_lr_cosine_schedule` 的期望值（四舍五入）。

---

## 4. 常见学习率策略对比

### 4.1 Constant（常数）

$$
\alpha_t=\alpha_0
$$

**特点**：最简单，没有调度。

**优点**：不用调 schedule；实现零成本。

**缺点**：前期可能太慢，后期可能震荡；对学习率非常敏感。

**适用**：小模型、短训练、调试 baseline。

### 4.2 Step Decay（阶梯衰减）

$$
\alpha_t=\alpha_0\cdot\gamma^{\lfloor t/s\rfloor}
$$

- 每 $s$ 步，学习率乘 $\gamma$；
- 例如 $\alpha_0=1,\gamma=0.1,s=10$：前 10 步 1，接着 0.1，再接着 0.01。

**优点**：简单、可解释；在固定 epoch 预算下好用。

**缺点**：需要选“在哪一步降”；下降是突变的，可能造成 loss 抖动。

**适用**：传统 CV 训练（ResNet 等）常见。

> 类比：到固定站点就突然降一档。

### 4.3 Exponential Decay（指数衰减）

$$
\alpha_t=\alpha_0\cdot\gamma^t
$$

或：

$$
\alpha_t=\alpha_0e^{-kt}
$$

**特点**：每一步都乘一个小于 1 的系数，平滑地指数下降。

**优点**：实现简单；没有突变。

**缺点**：衰减速度由 $\gamma$ 决定，前期降得太快会“早停”，后期又可能太小；需要调 $\gamma$。

**适用**：一些 NLP/语音任务；作为 baseline 很常见。

> 类比：越走越松油门，但松得快慢全靠一个参数。

### 4.4 Linear Decay（线性衰减）

$$
\alpha_t=\alpha_{max}-(\alpha_{max}-\alpha_{min})\frac{t}{T}
$$

**特点**：匀速下降，到 $T$ 时到最小值。

**优点**：简单、直观。

**缺点**：前期下降可能偏快，后期可能偏慢；不如余弦平滑。

**适用**：一些序列任务、作为对比基线。

### 4.5 Cosine Annealing（余弦退火）

$$
\alpha_t=\alpha_{min}+\frac{1+\cos(\pi p)}{2}(\alpha_{max}-\alpha_{min})
$$

**特点**：两端平缓、中间快；只需 $T_c$、$\alpha_{max}$、$\alpha_{min}$。

**优点**：平滑、超参数少、适合 Transformer/LLM；LLaMA 使用。

**缺点**：需要知道总步数 $T_c$；如果训练长度变化，要重设。

**适用**：大模型预训练、微调。

### 4.6 Warmup + Cosine（本文重点）

先线性预热，再余弦退火，最后保持最小值。

**优点**：

- 预热避免早期发散；
- 余弦平滑收尾；
- 大模型训练的标准组合。

**缺点**：

- 多两个超参数（$T_w$、$T_c$）；
- 需要知道总步数；
- 预热太长会浪费时间，太短又不够稳。

### 4.7 Cosine Restarts（余弦重启）

每隔一段时间把学习率“升回去”，再降下来：

```text
lr
 ▲    ╱╲      ╱╲      ╱╲
 │  ╱    ╲  ╱    ╲  ╱    ╲
 │╱        ╲╱      ╲╱      ╲
 └────────────────────────────► t
```

**优点**：有机会跳出局部最优；适合长训练。

**缺点**：多了“重启周期”超参数；可能破坏收敛。

### 4.8 ReduceLROnPlateau（按指标降）

不是按步数，而是看验证集指标：

- 指标一段时间不降 → 学习率乘 0.1；
- 指标继续降 → 保持。

**优点**：自适应，不需要知道总步数。

**缺点**：依赖验证集；反应有延迟；不适合大规模预训练（验证成本高）。

### 4.9 一张表对比

| 策略 | 形状 | 超参数 | 优点 | 缺点 | 适用 |
| --- | --- | --- | --- | --- | --- |
| Constant | 平的 | $\alpha_0$ | 最简单 | 前后期难兼顾 | baseline |
| Step Decay | 台阶 | $\alpha_0,\gamma,s$ | 简单可解释 | 突变、要选站点 | CV |
| Exponential | 指数下滑 | $\alpha_0,\gamma$ | 平滑 | 衰减速度难调 | NLP/语音 |
| Linear | 直线下滑 | $\alpha_{max},\alpha_{min},T$ | 直观 | 不如余弦平滑 | 对比基线 |
| Cosine | 半余弦 | $\alpha_{max},\alpha_{min},T_c$ | 平滑、超参少 | 要总步数 | Transformer/LLM |
| Warmup + Cosine | 先升后降 | 多 $T_w$ | 稳、标准 | 要总步数 | 大模型预训练 |
| Cosine Restarts | 周期升降 | 周期 | 探索性强 | 难调 | 长训练 |
| Plateau | 按指标降 | 耐心、因子 | 自适应 | 依赖验证 | 微调/小数据 |

### 4.10 怎么选

- **不知道选什么**：Warmup + Cosine，通常不会太差；
- **训练预算固定、模型大**：Warmup + Cosine，$T_c$ = 总步数；
- **小模型/快速实验**：Constant 或 Exponential 先跑 baseline；
- **CV 传统模型**：Step Decay 仍然常见；
- **微调**：小学习率 + Cosine 或 Plateau；
- **长训练/探索**：可以试 Cosine Restarts，但要多调参。

> 没有“绝对最好”的调度。学习率要和模型、batch size、优化器、数据量一起调。

---

## 5. 参考代码与训练集成

### 5.1 你给的 `Lr_Cosine_Schedule.py`

当前代码：

```python
import math

class CosineSchedule:
    def __init__(self, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def __call__(self, it):
        if it < self.warmup_iters:
            return self.max_learning_rate * it / self.warmup_iters
        elif it > self.cosine_cycle_iters:
            return self.min_learning_rate
        else:
            return self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * (
                1 + math.cos(math.pi * (it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters))
            ) / 2
```

这段代码的公式是正确的，和讲义、测试期望一致。

三个可以改进的点：

1. **`cosine_cycle_iters == warmup_iters` 会除以 0**
   加一个保护：
   ```python
   if self.cosine_cycle_iters <= self.warmup_iters:
       return self.min_learning_rate

   if self.warmup_iters <= 0:
       # 没有预热：it=0 直接取 max，后续走余弦
       ...
   ```

2. **测试适配器需要的是函数，不是类**
   `tests/adapters.py` 里的 `run_get_lr_cosine_schedule` 是一个函数；可以这样包一层：
   ```python
   def run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
       return CosineSchedule(
           max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
       )(it)
   ```

3. **`it` 的起点要和训练循环对齐**
   测试从 `it=0` 开始，`it=0` 时学习率为 0；实际训练如果不想第一步空转，可以在调用前 `it += 1`，但日志、checkpoint 和测试要一致。

### 5.2 完整可运行示例

保存为 `lr_schedule_demo.py`：

```python
"""Cosine Schedule 小示例：打印每个 iteration 的学习率。"""
import math


def get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    # 阶段 1：线性预热
    if it < warmup_iters:
        return max_lr * it / warmup_iters

    # 阶段 3：余弦结束后保持最小值
    if it > cosine_cycle_iters:
        return min_lr

    # 阶段 2：余弦退火
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def main():
    max_lr, min_lr = 1.0, 0.1
    warmup_iters, cosine_cycle_iters = 7, 21

    print("it   lr       curve")
    for it in range(25):
        lr = get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        bar = "#" * int(lr * 40)
        print(f"{it:2d}  {lr:.4f}  {bar}")


if __name__ == "__main__":
    main()
```

### 5.3 预期输出

```text
it   lr       curve
 0  0.0000
 1  0.1429  #####
 2  0.2857  ###########
 3  0.4286  #################
 4  0.5714  ######################
 5  0.7143  ############################
 6  0.8571  ##################################
 7  1.0000  ########################################
 8  0.9887  #######################################
 9  0.9554  ######################################
10  0.9018  ####################################
11  0.8306  #################################
12  0.7452  #############################
13  0.6501  ##########################
14  0.5500  ######################
15  0.4499  #################
16  0.3548  ##############
17  0.2694  ##########
18  0.1982  #######
19  0.1446  #####
20  0.1113  ####
21  0.1000  ####
22  0.1000  ####
23  0.1000  ####
24  0.1000  ####
```

看图能直观看到：

- 0–7：线性上升；
- 7–21：先慢后快再慢地下降；
- 21 之后：保持 0.1。

### 5.4 接进训练循环

```python
from cs336_basics.Lr_Cosine_Schedule import CosineSchedule

scheduler = CosineSchedule(
    max_learning_rate=1e-3,
    min_learning_rate=1e-4,
    warmup_iters=100,
    cosine_cycle_iters=10_000,
)

for it in range(total_iters):
    lr = scheduler(it)                    # 1) 算学习率
    for group in optimizer.param_groups:
        group["lr"] = lr                  # 2) 写给优化器

    optimizer.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    # 可选：clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()                      # 3) 用当前 lr 更新
```

三个细节：

1. **先设 lr，再 `step()`**：优化器在 `step()` 里读取 `group["lr"]`；
2. **`it` 是参数更新次数**，不是 epoch；
3. **checkpoint 要保存 `it`**：恢复训练时从同一个 step 继续，否则学习率会跳回早期。

### 5.5 和 AdamW 的配合

```python
for it in range(total_iters):
    for group in optimizer.param_groups:
        group["lr"] = scheduler(it)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

- 学习率调度器只改 `lr`；
- AdamW 的 $m$、$v$、weight decay 都不受影响；
- 顺序：**算 lr → zero_grad → forward/backward → clip → step**。

> 如果你在 `step()` 之后才改 lr，这一步用的还是旧学习率；通常没问题，但要和日志一致。

---

## 6. 联系 CS336 Assignment 1

### 6.1 作业要求（讲义 §4.4）

作业要求实现 **带线性预热的余弦退火学习率调度**：

- 输入：当前迭代 $t$、$\alpha_{max}$、$\alpha_{min}$、$T_w$、$T_c$；
- 返回第 $t$ 步的学习率；
- 实现 `adapters.run_get_lr_cosine_schedule`；
- 运行 `uv run pytest -k test_get_lr_cosine_schedule`；
- 这一题 1 分。

> 讲义里写的是 `get_lr_cosine_schedule`，仓库 adapter 实际叫 `run_get_lr_cosine_schedule`，以代码为准。

### 6.2 接口

```python
def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    ...
```

| 参数 | 含义 |
| --- | --- |
| `it` | 当前迭代次数，从 0 开始 |
| `max_learning_rate` | 峰值学习率 |
| `min_learning_rate` | 最终学习率 |
| `warmup_iters` | 预热步数 $T_w$ |
| `cosine_cycle_iters` | 余弦结束步数 $T_c$ |

### 6.3 测试细节

`test_get_lr_cosine_schedule`：

- `max_learning_rate=1`；
- `min_learning_rate=0.1`；
- `warmup_iters=7`；
- `cosine_cycle_iters=21`；
- 对 `it=0..24` 逐个检查期望值；
- 容差由 `numpy.testing.assert_allclose` 默认值控制（很严格）。

几个关键点：

| $t$ | 期望 lr | 说明 |
| ---: | ---: | --- |
| 0 | 0 | 预热起点 |
| 7 | 1.0 | 预热结束，达到峰值 |
| 14 | 0.55 | 余弦中点，正好是 $(1+0.1)/2$ |
| 21 | 0.1 | 余弦结束 |
| 22–24 | 0.1 | 保持最小值 |

### 6.4 当前代码与测试的关系

`Lr_Cosine_Schedule.py` 的 `CosineSchedule` 用 `__call__` 实现，公式和测试一致。我本地对 `it=0..24` 逐个对比，最大误差约 `4.4e-11`，远小于测试容差。

但 adapter 需要的是一个**函数**，所以需要包一层：

```python
from cs336_basics.Lr_Cosine_Schedule import CosineSchedule


def run_get_lr_cosine_schedule(
    it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
):
    scheduler = CosineSchedule(
        max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )
    return scheduler(it)
```

### 6.5 在完整训练中的位置

CS336 的训练循环大致是：

```text
采样 batch
 → forward
 → cross entropy loss
 → backward
 → gradient clipping
 → 计算 lr（cosine schedule）
 → optimizer.step()
 → 记录日志 / checkpoint
```

学习率调度是训练循环里的一个“小零件”，但它对稳定性影响很大：

- 没有 warmup，早期可能发散；
- 没有 decay，后期可能震荡；
- schedule 和 checkpoint 不对齐，恢复训练后曲线会跳。

### 6.6 常见错误对照表

| 现象 | 可能原因 | 修正 |
| --- | --- | --- |
| `it=0` 学习率不是 0 | 没写预热分支 | 先判断 `it < T_w` |
| 到 $T_w$ 时学习率不是 max | 余弦公式写错 | $p=0$ 时 $\cos=1$ |
| $T_c$ 后还在下降 | 没 clamp 到 min | `it > T_c` 返回 min |
| `ZeroDivisionError` | $T_w=0$ 或 $T_c=T_w$ | 加边界保护 |
| 学习率是负数 | $t<0$ 或参数写反 | clamp / 检查配置 |
| 学习率变化是阶梯 | 用了 step decay | 检查是否用了余弦公式 |
| 学习率一直是常数 | 没把 lr 写回 optimizer | 更新 `group["lr"]` |
| 恢复训练后 lr 跳变 | checkpoint 没存 `it` | 保存/加载迭代数 |
| 测试差一点 | 用了角度制/整数除法 | `math.cos` 用弧度；Python 3 的 `/` 是浮点 |
| 训练前期发散 | 预热太短/没预热 | 加长 $T_w$ 或降低 max lr |
| 训练后期震荡 | 最小学习率太大 | 降低 $\alpha_{min}$ |

### 6.7 实现检查清单

- [ ] `it < T_w` 时返回线性预热值；
- [ ] `T_w <= it <= T_c` 时返回余弦值；
- [ ] `it > T_c` 时返回 $\alpha_{min}$；
- [ ] 边界 $t=T_w$、$t=T_c$ 连续；
- [ ] 处理 $T_w=0$、$T_c=T_w$ 的除零；
- [ ] 用弧度制 `math.cos`；
- [ ] adapter 返回函数结果；
- [ ] 训练循环里把 lr 写回 `param_group`；
- [ ] checkpoint 保存 iteration；
- [ ] 跑过 `pytest -k test_get_lr_cosine_schedule`。

---

## 7. 常见误区 FAQ

**Q1：为什么不能一直用大学习率？**
前期大学习率有助于快速下降，但后期会在最优点附近震荡，甚至发散。

**Q2：为什么不能一直用小学习率？**
训练太慢，容易卡在差的区域；前期需要探索。

**Q3：为什么 warmup 从 0 开始？**
讲义公式在 $t=0$ 时返回 0；实际训练中第一步相当于空更新。如果不想空转，可以在调用前 `it += 1`，但要和测试/日志一致。

**Q4：为什么 warmup 结束后不直接从 max 开始余弦？**
余弦在 $p=0$ 处值就是 max，所以是连续的；预热只是把“从 0 到 max”这段补上。

**Q5：为什么最后保持 $\alpha_{min}$ 而不是 0？**
保持一个小学习率可以继续微调；降到 0 就完全停止学习了。

**Q6：Cosine 和 Cosine Annealing 是一回事吗？**
通常同义；严格说 Cosine Annealing 指“按余弦降到最小值”的过程，Warmup + Cosine 是它的常见组合。

**Q7：$T_c$ 应该设成多少？**
通常设成总训练步数；如果想留出“平台期”，可以略小于总步数。

**Q8：如果训练步数变了，$T_c$ 要改吗？**
要。$T_c$ 是按步数定义的；换训练长度要重新设，否则学习率曲线会被拉长/压缩。

**Q9：调度器应该放在 `step()` 前还是后？**
通常先算 lr、写回 `param_group`，再 `optimizer.step()`，这样这一步用的就是新 lr。

**Q10：AdamW 自带学习率，为什么还要调度？**
AdamW 决定“怎么走”，调度决定“走多大”；两者是不同维度。

**Q11：Cosine 一定比 Step Decay 好吗？**
不一定。Cosine 更平滑、超参少；Step Decay 在固定 epoch 的 CV 任务里仍然很好用。

**Q12：可以中途换调度策略吗？**
可以，但要保证学习率连续，否则 loss 可能抖；实验上要记录清楚。

---

## 8. 小结与自测

### 8.1 一页速查

```text
预热:   t < T_w:  lr = max_lr * t / T_w
余弦:   T_w <= t <= T_c:
        p = (t - T_w) / (T_c - T_w)
        lr = min_lr + 0.5*(1+cos(pi*p))*(max_lr-min_lr)
结束:   t > T_c:  lr = min_lr

关键点:
  t=T_w -> max_lr
  t=T_c -> min_lr
  中点   -> (max_lr+min_lr)/2
  两端平缓，中间下降快
```

### 8.2 自测题

1. 固定学习率在训练前期和后期分别有什么问题？
2. 画出 Warmup + Cosine 的三阶段示意图，标出 $T_w$、$T_c$。
3. 写出预热、余弦、结束三个阶段的公式。
4. 为什么 $t=T_w$ 和 $t=T_c$ 处学习率是连续的？
5. Cosine 曲线为什么“两端慢、中间快”？
6. Step Decay、Exponential Decay、Cosine Annealing 各有什么优缺点？
7. 为什么大模型训练通常要 warmup？
8. 为什么最后要保持一个最小学习率，而不是降到 0？
9. `it` 和 epoch 怎么换算？
10. 调度器应该怎么接进 AdamW 的训练循环？
11. CS336 的测试参数是什么？关键节点的 lr 是多少？
12. 你给的 `Lr_Cosine_Schedule.py` 有哪三个可以改进的点？

> 参考答案在 §1–§6 对应小节；先自己写一遍，再回去核对。

### 8.3 动手清单

- [ ] 手算 $t=7,14,21$ 三个点的学习率；
- [ ] 跑通 §5.2 的示例，观察 ASCII 曲线；
- [ ] 把 `CosineSchedule` 包成 adapter 函数；
- [ ] 跑过 `pytest -k test_get_lr_cosine_schedule`；
- [ ] 在 AdamW 训练循环里接入 scheduler；
- [ ] 保存/加载 iteration，确认恢复训练后 lr 连续。

---

## 9. 延伸阅读

**论文与资料**

- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, 2016 —— 余弦退火与重启。
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023 —— LLaMA 的 Warmup + Cosine。
- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, 2017 —— 大 batch 的 warmup。
- He et al., *Deep Residual Learning for Image Recognition*, 2016 —— Step Decay 的经典使用场景。

**课程与文档**

- CS336 课程主页：<https://cs336.stanford.edu>
- 作业讲义 `cs336_assignment1_basics.pdf` §4.4
- PyTorch 文档：`torch.optim.lr_scheduler`

**同目录文档**

- `AdamW.md`：优化器，和学习率调度配合使用。
- `Cross_Entropy_Loss.md`：损失函数，训练信号。
- `完整语言模型架构的组装.md`：模型输出 logits 的链路。
- `Transformer_Block.md`：模型结构。

> **最后一句**：学习率调度不是“玄学调参”，它是在回答一个很朴素的问题：**训练早期该大胆，训练后期该细心。** Cosine Schedule 只是把这件事做得平滑一点。
