# AdamW：把“方向、步伐、体重”分开管的优化器

> 本文面向零基础读者，结合 CS336 Assignment 1 的 `adamw` 任务，讲清楚优化器到底在做什么、AdamW 是怎么从 SGD 一步步演化来的、权重衰减为什么要“解耦”，以及代码怎么写。
>
> 读之前最好知道：
>
> - 梯度是什么：损失对参数的导数，指向“损失上升最快”的方向；
> - 训练循环的基本顺序：前向 → loss → `backward()` → `optimizer.step()`；
> - 什么是学习率：每次更新参数走多远。
>
> 读完你应该能回答五个问题：
>
> 1. 优化器到底在优化什么？它和梯度、学习率是什么关系？
> 2. SGD、Momentum、RMSProp、Adam 各自解决了什么问题，又留下了什么坑？
> 3. 权重衰减和 L2 正则化不是一回事吗？为什么 AdamW 要拆开？
> 4. AdamW 的每一步计算分别是什么？偏差修正为什么必要？
> 5. CS336 的 `test_adamw` 在测什么？`Adamw.py` 有哪些可以改进的地方？

## 目录

- [0. 阅读前的准备](#0-阅读前的准备)
- [1. 优化器的本质](#1-优化器的本质)
- [2. 历史演变：从 SGD 到 Adam](#2-历史演变从-sgd-到-adam)
- [3. AdamW 的由来：权重衰减 vs L2](#3-adamw-的由来权重衰减-vs-l2)
- [4. AdamW 算法全流程](#4-adamw-算法全流程)
- [5. 参考代码与逐段讲解](#5-参考代码与逐段讲解)
- [6. 联系 CS336 Assignment 1](#6-联系-cs336-assignment-1)
- [7. 常见误区 FAQ](#7-常见误区-faq)
- [8. 小结与自测](#8-小结与自测)
- [9. 延伸阅读](#9-延伸阅读)

---

## 0. 阅读前的准备

### 0.1 一句话版本

> **优化器就是训练时的“下坡策略”：它拿到梯度，决定每个参数往哪走、走多大一步，以及要不要顺手把参数往 0 拉一点。**

AdamW 的三句话：

1. **方向**：用一阶矩 $m$（梯度的滑动平均）记住“最近一直往哪走”；
2. **步伐**：用二阶矩 $v$（梯度平方的滑动平均）给每个参数配一个自适应步长；
3. **体重**：用解耦的权重衰减，单独把参数往 0 缩，不混进梯度里。

### 0.2 符号表

| 符号 | 含义 | 一句话解释 |
| --- | --- | --- |
| $\theta$ | 模型参数 | 要优化的东西 |
| $g$ | 梯度 | 损失对参数的导数 |
| $\eta$ | 学习率 | 每次走多远 |
| $\beta_1$ | 一阶矩衰减率 | 动量记多久的历史，常用 0.9 |
| $\beta_2$ | 二阶矩衰减率 | 梯度平方记多久的历史，常用 0.999 |
| $\epsilon$ | 数值稳定常数 | 防止除以 0，常用 $10^{-8}$ |
| $\lambda$ | 权重衰减系数 | 参数往 0 缩的强度 |
| $m$ | 一阶矩 | 梯度的滑动平均，像“速度” |
| $v$ | 二阶矩 | 梯度平方的滑动平均，像“每个参数的步长表” |
| $t$ | 步数 | 从 1 开始，用于偏差修正 |

### 0.3 训练循环里的位置

```text
for batch in data:
    optimizer.zero_grad()      # 清空上一轮梯度
    logits = model(batch)      # 前向
    loss = loss_fn(logits, y)  # 算损失
    loss.backward()            # 反向，得到每个参数的梯度
    optimizer.step()           # 优化器根据梯度更新参数
```

优化器**不负责**算梯度，也不负责前向；它只负责：

> **拿到 `p.grad`，更新 `p.data`。**

### 0.4 学习路线

| 你想知道 | 看哪节 |
| --- | --- |
| 优化器到底在干什么 | §1 |
| SGD → Momentum → RMSProp → Adam | §2 |
| 为什么要 AdamW | §3 |
| 每一步公式 | §4 |
| 代码怎么写 | §5 |
| 作业怎么过 | §6 |
| 常见坑 | §7、§8 |

---

## 1. 优化器的本质

### 1.1 一个生活化类比：雾里下山

想象你在一座大雾弥漫的山上，要走到山谷最低点：

- **你站的位置**：当前的参数 $\theta$；
- **脚下的坡度**：梯度 $g$；
- **一步走多远**：学习率 $\eta$；
- **你的下山策略**：优化器。

如果只看脚下最陡的方向，每次迈固定一步，就是 **SGD**。

但现实里：

- 坡度可能有噪声（mini-batch 采样）；
- 有的方向很陡，有的方向很平（不同参数的梯度尺度不同）；
- 一直按最陡方向走，可能在峡谷两边来回震荡；
- 走得太快会冲过头，走得太慢又到不了。

优化器就是用来处理这些问题的“下山策略”。

### 1.2 数学形式

几乎所有优化器都可以写成：

$$
\theta_{t+1}=\theta_t+\Delta\theta_t
$$

区别只在于 $\Delta\theta_t$ 怎么算：

| 优化器 | $\Delta\theta_t$ |
| --- | --- |
| SGD | $-\eta g_t$ |
| Momentum | $-\eta v_t$，$v_t$ 是梯度的滑动平均 |
| RMSProp | $-\eta \frac{g_t}{\sqrt{v_t}+\epsilon}$ |
| Adam | $-\eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$ |
| AdamW | 先缩参数，再套 Adam |

### 1.3 优化器不做什么

| 它不做 | 谁来做 |
| --- | --- |
| 算梯度 | `loss.backward()` |
| 定义模型 | `nn.Module` |
| 算损失 | 交叉熵等损失函数 |
| 决定 batch 大小 | 数据加载器 |
| 保证一定收敛 | 没有人能保证；只能靠调参和实验 |

> 优化器是“执行者”，不是“魔法”。它只能在给定梯度的情况下，决定怎么走。

### 1.4 为什么不能只用“梯度下降”

最朴素的 SGD：

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

它有三个明显问题：

1. **所有参数共用一个步长**：有的参数梯度很大，有的很小；
2. **容易被噪声带偏**：mini-batch 梯度只是真实梯度的估计；
3. **峡谷里来回震荡**：陡方向反复横跳，平方向进展缓慢。

后面的 Momentum、RMSProp、Adam 都是在补这三个坑。

---

## 2. 历史演变：从 SGD 到 Adam

### 2.1 SGD：最朴素的下山

**更新**：

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

**直觉**：每次沿着当前梯度反方向走一步。

**优点**：

- 简单、便宜；
- 内存只存参数和梯度；
- 在大数据/大模型上，噪声反而可能帮助泛化。

**局限**：

- 对学习率非常敏感；
- 所有参数同一步长；
- 峡谷地形来回震荡；
- 收敛慢。

> 类比：蒙着眼下山，每次只根据脚下的坡度迈固定一步。

### 2.2 Momentum：加一点惯性

**更新**：

$$
v_t=\beta v_{t-1}+g_t
$$

$$
\theta_{t+1}=\theta_t-\eta v_t
$$

其中 $0<\beta<1$，常用 0.9。

**直觉**：把历史梯度累积成“速度”。

- 如果最近一直往同一个方向走，速度会越来越大；
- 如果方向来回变，正负抵消，震荡被抑制。

**优点**：

- 加速平坦方向；
- 抑制峡谷里的震荡；
- 对噪声更鲁棒。

**局限**：

- 仍然是全局步长；
- 可能冲过头（overshoot）；
- 需要调 $\beta$ 和学习率。

> 类比：下山时推一个小球，小球有惯性；不会因为一脚踩偏就立刻转向。

> 术语说明：不同框架对动量缓冲区的写法略有差异，有的用 $v_t=\beta v_{t-1}+(1-\beta)g_t$（指数滑动平均），有的用 $v_t=\beta v_{t-1}+g_t$。两者只差一个常数缩放，本质都是“历史梯度的加权平均”。

### 2.3 RMSProp：给每个参数配自己的步长

**更新**：

$$
v_t=\beta v_{t-1}+(1-\beta)g_t^2
$$

$$
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{v_t}+\epsilon}
$$

**直觉**：$v_t$ 记录“这个参数最近的梯度有多大”。

- 梯度一直很大的参数：分母大，步长自动变小；
- 梯度一直很小的参数：分母小，步长自动变大。

这就是“自适应步长”（adaptive learning rate）。

**优点**：

- 不同参数可以有不同有效步长；
- 对梯度尺度差异更鲁棒；
- 适合非平稳目标。

**局限**：

- 没有动量；
- 没有偏差修正（$v$ 从 0 开始，早期偏小）；
- 仍然需要调全局学习率。

> 类比：给每个科目配不同的学习计划：数学好的少练，英语差的多练。

### 2.4 Adam：动量 + 自适应步长

Adam = **Ada**ptive **m**oment estimation，把 Momentum 和 RMSProp 合在一起：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

**偏差修正**：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

**更新**：

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

**优点**：

- 有动量，方向更稳；
- 有自适应步长，不同参数不同步长；
- 偏差修正让早期更新不至太小；
- 对超参数相对不敏感，默认值通常能用。

**局限**：

- 每个参数要存 $m$ 和 $v$，内存约为参数的 2 倍；
- 权重衰减和 L2 的等价关系被破坏（见 §3）；
- 有时泛化不如 SGD + Momentum；
- 不是“万能优化器”，仍要调学习率和 schedule。

### 2.5 一张表看懂演化

| 方法 | 核心 | 解决的问题 | 留下的问题 |
| --- | --- | --- | --- |
| SGD | 沿梯度走 | 简单 | 同一步长、震荡、慢 |
| Momentum | 历史梯度累积 | 震荡、平坦方向慢 | 仍是全局步长 |
| RMSProp | 梯度平方归一化 | 不同参数尺度差异 | 无动量、无偏差修正 |
| Adam | Momentum + RMSProp + 偏差修正 | 方向、尺度、早期偏差 | L2 与权重衰减不等价、内存 |
| AdamW | Adam + 解耦权重衰减 | 正则化与自适应步长解耦 | 仍需调参、仍占内存 |

> 历史不是“后面的一定全面碾压前面的”。SGD + Momentum 在很多视觉任务上仍然很强；AdamW 在大模型/Transformer 里更常见。选哪个要看任务、数据、算力和调参预算。

### 2.6 Adam 的“历史遗留问题”

Adam 把 L2 正则的梯度直接加到 $g$ 里：

$$
g_t\leftarrow g_t+\lambda\theta_{t-1}
$$

然后整个梯度被 $m$、$v$ 和自适应分母加工。结果是：

- 每个参数受到的“往 0 拉”的力度，被它的历史梯度尺度改变了；
- 梯度大的参数，衰减被压小；梯度小的参数，衰减被放大；
- 这不是我们想要的“统一体重管理”。

这就是 AdamW 要解决的问题。

---

## 3. AdamW 的由来：权重衰减 vs L2

### 3.1 L2 正则化是什么

在损失里加一项：

$$
\mathcal{L}_{\text{total}}=\mathcal{L}+\frac{\lambda}{2}\|\theta\|^2
$$

梯度变成：

$$
g_{\text{total}}=g+\lambda\theta
$$

对 SGD：

$$
\begin{aligned}
\theta_{t+1}
&=\theta_t-\eta(g_t+\lambda\theta_t)\\
&=(1-\eta\lambda)\theta_t-\eta g_t
\end{aligned}
$$

所以 **在普通 SGD（不含动量）里**：

> “加 L2 正则” 等价于 “每次把参数乘 $(1-\eta\lambda)$，再走梯度步”。

这就是 **weight decay（权重衰减）** 的名字来源：参数每次都被“衰减”一点。

### 3.2 为什么在 Adam 里就不等价了

Adam 的更新不是直接用 $g$，而是：

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

如果把 L2 的 $\lambda\theta$ 加进 $g$：

- 它会进入 $m$ 和 $v$；
- 它会被分母 $\sqrt{\hat v_t}+\epsilon$ 缩放；
- 每个参数的有效衰减力度取决于它的历史梯度尺度。

于是：

> **L2 在 Adam 里不再是“统一乘 $(1-\eta\lambda)$”**，而是被自适应步长“重新加权”了。

这会导致：

- 权重衰减的强度不可控；
- 大梯度参数和小梯度参数受到不同力度；
- 调 $\lambda$ 时，效果和 $\eta$、梯度尺度纠缠在一起。

### 3.3 AdamW 的解耦

AdamW 的做法很简单：

> **不要把权重衰减塞进梯度；把它单独做。**

标准更新：

$$
\theta_t\leftarrow\theta_t-\eta\lambda\theta_t
$$

然后：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\theta_t\leftarrow\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

注意：

- 权重衰减只碰 $\theta$，不碰 $g$；
- $m$、$v$ 只由原始梯度决定；
- 衰减力度和自适应分母无关。

> 类比：
>
> - **L2** 像“把体重写进健身考核”：教练会根据你其他训练项目的强度，折算成不同的减重任务；
> - **AdamW** 像“每天先固定扣掉一点体重，再按训练计划练”：减重就是减重，和训练强度分开。

### 3.4 顺序重要吗

标准 AdamW 先衰减**旧参数**，再做 Adam 更新：

$$
\theta_t\leftarrow(1-\eta\lambda)\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

如果把衰减放在 Adam 更新之后，并且乘的是**更新后的参数**：

$$
\theta_t\leftarrow(1-\eta\lambda)\left(\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\right)
$$

两者差一个 $\eta^2\lambda\cdot\text{AdamUpdate}$ 的项。通常很小，测试容差下可能都过，但**严格实现应该先衰减旧参数**。

### 3.5 L2 vs 权重衰减：一张表

| | L2 正则 | 权重衰减（AdamW） |
| --- | --- | --- |
| 加在哪里 | 损失函数 | 参数更新 |
| 对梯度的影响 | $g\leftarrow g+\lambda\theta$ | 不改 $g$ |
| 普通 SGD 下 | 等价于 $(1-\eta\lambda)\theta$ | 就是 $(1-\eta\lambda)\theta$ |
| Adam 下 | 被自适应分母缩放 | 独立、统一 |
| 调参 | 和梯度尺度纠缠 | $\lambda$ 更独立 |
| 典型用途 | 传统正则化 | 大模型/AdamW |

### 3.6 优势总结

1. **正则化更可控**：$\lambda$ 就是“往 0 缩多少”，不被梯度尺度扭曲；
2. **泛化通常更好**：原论文在多个任务上观察到改善；
3. **和自适应步长解耦**：换 $\eta$、换 $\beta$ 不会悄悄改变衰减强度；
4. **工程上简单**：只是一次乘法，容易融合；
5. **成为 LLM 常用选择**：LLaMA、Mistral 等现代开源大模型普遍采用 AdamW。

> 原始论文：Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 2019。

---

## 4. AdamW 算法全流程

### 4.1 初始化

对每个参数 $\theta$：

$$
m_0=0,\qquad v_0=0,\qquad t=0
$$

- $m$：一阶矩，和参数同形状；
- $v$：二阶矩，和参数同形状；
- $t$：步数，从 0 开始，第一步时变成 1。

> 注意：$m$ 和 $v$ 是**每个参数一份**，不是全局一个标量。

### 4.2 每一步的八个动作

假设当前是第 $t$ 步（$t$ 从 1 开始）：

**第 1 步：拿梯度**

$$
g_t=\nabla_\theta\mathcal{L}(\theta_{t-1})
$$

**第 2 步：步数加一**

$$
t\leftarrow t+1
$$

**第 3 步：解耦权重衰减**

$$
\theta_t\leftarrow\theta_{t-1}-\eta\lambda\theta_{t-1}
=(1-\eta\lambda)\theta_{t-1}
$$

**第 4 步：更新一阶矩**

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

**第 5 步：更新二阶矩**

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

**第 6 步：偏差修正**

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

**第 7 步：算自适应分母**

$$
\text{denom}_t=\sqrt{\hat v_t}+\epsilon
$$

**第 8 步：更新参数**

$$
\theta_t\leftarrow\theta_t-\eta\frac{\hat m_t}{\text{denom}_t}
$$

> 顺序记法：**衰减 → 一阶矩 → 二阶矩 → 修正 → 更新**。

### 4.3 伪代码

```text
for each parameter θ:
    m ← 0
    v ← 0
    t ← 0

for each training step:
    g ← gradient of loss w.r.t. θ
    t ← t + 1

    θ ← θ - lr * weight_decay * θ      # 解耦衰减
    m ← β1 * m + (1 - β1) * g
    v ← β2 * v + (1 - β2) * g²

    m_hat ← m / (1 - β1^t)
    v_hat ← v / (1 - β2^t)

    θ ← θ - lr * m_hat / (sqrt(v_hat) + eps)
```

> **讲义的写法**：CS336 讲义 Algorithm 1 把偏差修正合并到步长里：

$$
\alpha_t=\alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t},\qquad
\theta\leftarrow\theta-\alpha_t\frac{m}{\sqrt v+\epsilon}
$$

> 它和上面的“分别修正 $m$、$v$”是同一套思路，只是在 $\epsilon$ 的位置上有细微差别；$\epsilon$ 很小时数值上几乎一致，测试也接受两种实现。

### 4.4 偏差修正为什么必要

$m$ 和 $v$ 都从 0 开始，早期会“偏小”。

以 $\beta_1=0.9$ 为例，第一步：

$$
m_1=0.9\times0+0.1g=0.1g
$$

如果直接用 $m_1$ 更新，步长只有应有值的 10%。除以 $1-\beta_1^1=0.1$：

$$
\hat m_1=\frac{0.1g}{0.1}=g
$$

就恢复成“真实梯度”。

$v$ 同理：

$$
v_1=(1-\beta_2)g^2
$$

除以 $1-\beta_2^1$ 后：

$$
\hat v_1=g^2
$$

> 类比：新员工第一个月只上了一周班，不能直接拿这一周的业绩当“月均业绩”；按入职时间折算，才是真实水平。

### 4.5 一个标量小例子

设：

- $\theta=1.0$
- $g=1.0$
- $\eta=0.1$
- $\beta_1=0.9,\beta_2=0.999$
- $\epsilon=10^{-8}$
- $\lambda=0.01$

第一步：

| 步骤 | 计算 | 结果 |
| --- | --- | --- |
| 衰减 | $\theta\leftarrow(1-0.1\times0.01)\times1.0$ | 0.999 |
| 一阶矩 | $m=0.9\times0+0.1\times1$ | 0.1 |
| 二阶矩 | $v=0.999\times0+0.001\times1$ | 0.001 |
| 修正 | $\hat m=0.1/(1-0.9)=1$ | 1 |
| 修正 | $\hat v=0.001/(1-0.999)=1$ | 1 |
| 更新 | $\theta\leftarrow0.999-0.1\times1/(1+10^{-8})$ | ≈0.899 |

如果不做偏差修正：

$$
\theta\approx0.999-0.1\times\frac{0.1}{\sqrt{0.001}+10^{-8}}
\approx0.999-0.316=0.683
$$

步子会大很多。偏差修正让第一步回到“梯度为 1 时走 0.1”的合理尺度。

### 4.6 默认超参数

| 参数 | 常用值 | 说明 |
| --- | --- | --- |
| $\eta$ | $10^{-3}$ | 学习率，通常还要配 schedule |
| $\beta_1$ | 0.9 | 一阶矩衰减率 |
| $\beta_2$ | 0.999 | 二阶矩衰减率；大模型有时用 0.95 |
| $\epsilon$ | $10^{-8}$ | 防止除以 0 |
| $\lambda$ | 0.01 或 0.1 | 权重衰减，视模型/数据而定 |

> 讲义提到：LLaMA、GPT-3 等大模型常用 $(0.9,0.95)$ 而不是 $(0.9,0.999)$。

### 4.7 内存与状态

AdamW 是**有状态**优化器：

- 每个参数要存 $m$、$v$，内存约为参数量的 2 倍；
- 还要存步数 $t$（通常每个参数一份，或每个参数组一份）；
- checkpoint 必须保存优化器状态，否则恢复训练后动量/二阶矩丢失，曲线会抖。

> 这也是为什么大模型训练对显存要求高：参数 + 梯度 + $m$ + $v$ + 激活。

---

## 5. 参考代码与逐段讲解

下面是一个完整、可运行的 AdamW 实现（纯 PyTorch，CPU 即可）。它和 PyTorch 的 `torch.optim.AdamW` 在 1000 步小实验里几乎完全一致。

### 5.1 完整代码

```python
"""AdamW 最小实现与自检脚本（纯 PyTorch，CPU 可跑）。"""
import torch
from torch import optim


class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                # 1) 解耦权重衰减：先把参数往 0 缩一点
                p.mul_(1 - lr * weight_decay)

                # 2) 一阶矩 / 二阶矩
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3) 偏差修正
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # 4) Adam 更新
                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss


def optimize(cls):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = cls(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


if __name__ == "__main__":
    ours = optimize(AdamW)
    pytorch = optimize(torch.optim.AdamW)
    print("ours[:2]:", ours[:2])
    print("pytorch[:2]:", pytorch[:2])
    print("max diff:", (ours - pytorch).abs().max().item())
```

### 5.2 预期输出

```text
ours[:2]: tensor([[ 0.8643,  0.8822,  0.2076],
        [ 0.0553, -0.5219, -0.3904]])
pytorch[:2]: tensor([[ 0.8643,  0.8822,  0.2076],
        [ 0.0553, -0.5219, -0.3904]])
max diff: 1.1920928955078125e-07
```

> 误差在 float32 的正常范围内，说明实现与 PyTorch 的 AdamW 一致。

### 5.3 逐段讲解

#### 5.3.1 继承 `optim.Optimizer`

```python
class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
```

- 必须继承 `torch.optim.Optimizer`；
- `defaults` 里放每个参数组的默认超参数；
- `super().__init__` 会建立 `self.param_groups` 和 `self.state`；
- 这样 `state_dict()` / `load_state_dict()` 才能工作。

#### 5.3.2 遍历参数组

```python
for group in self.param_groups:
    lr = group["lr"]
    beta1, beta2 = group["betas"]
    eps = group["eps"]
    weight_decay = group["weight_decay"]

    for p in group["params"]:
        if p.grad is None:
            continue
```

- 一个优化器可以有多个参数组，每组自己的学习率/衰减；
- `p.grad is None` 说明这个参数没参与本次计算，跳过。

#### 5.3.3 初始化状态

```python
state = self.state[p]
if len(state) == 0:
    state["step"] = 0
    state["m"] = torch.zeros_like(p)
    state["v"] = torch.zeros_like(p)
```

- `self.state` 是一个字典，键是参数对象，值是该参数的状态；
- `m`、`v` 和参数同形状；
- `step` 从 0 开始，第一步后变 1。

#### 5.3.4 权重衰减

```python
p.mul_(1 - lr * weight_decay)
```

- 这是**解耦**的权重衰减；
- 直接乘参数，不碰梯度；
- 标准顺序是在 Adam 更新之前。

#### 5.3.5 一阶矩和二阶矩

```python
m.mul_(beta1).add_(grad, alpha=1 - beta1)
v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
```

等价于：

$$
m\leftarrow\beta_1m+(1-\beta_1)g
$$

$$
v\leftarrow\beta_2v+(1-\beta_2)g^2
$$

- `mul_` 是原地乘；
- `add_` / `addcmul_` 是原地加；
- `addcmul_(grad, grad, value=1-beta2)` 就是 $g\cdot g\cdot(1-\beta_2)$。

#### 5.3.6 偏差修正

```python
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
```

- 第一步 $t=1$，分母是 $1-\beta_1$；
- 随 $t$ 增大，分母趋近 1，修正越来越小；
- 这一步让早期更新不被“从 0 开始”拖慢。

#### 5.3.7 参数更新

```python
denom = v_hat.sqrt().add_(eps)
p.addcdiv_(m_hat, denom, value=-lr)
```

等价于：

$$
\theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}
$$

- `addcdiv_(tensor1, tensor2, value=-lr)` 计算 $\theta += -\eta\cdot\frac{\hat m}{\sqrt{\hat v}+\epsilon}$；
- 注意新式写法参数顺序是 `(tensor1, tensor2, value=...)`，不要用旧式的 `addcdiv_(-lr, m, denom)`（会报警告）。

### 5.4 仓库里的 `AdamW.py` 逐段看

当前核心循环（已按标准顺序修正）：

```python
m.mul_(beta1).add_(grad, alpha=1 - beta1)
v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
bias_correction1 = 1 - beta1 ** state['step']
bias_correction2 = 1 - beta2 ** state['step']
step_size = group['lr'] / bias_correction1
denom = (v / bias_correction2).sqrt().add_(group['eps'])

# 1) 先解耦衰减：只动参数，不进 m/v
p.data.mul_(1 - group['lr'] * group['weight_decay'])
# 2) 再做 Adam 更新
p.data.addcdiv_(m, denom, value=-step_size)
```

**做得对的地方**：

- 继承了 `optim.Optimizer`；
- 每个参数维护 `m`、`v`、`step`；
- 一阶矩/二阶矩更新正确；
- 偏差修正正确；
- `step_size = lr / (1 - β1^t)`、`denom = sqrt(v/(1-β2^t)) + ε` 是标准写法；
- 衰减顺序与官方一致，实测与 `torch.optim.AdamW` 的差在 float32 噪声量级（约 `1e-7`）。

**历史版本踩过的两个坑**（都已修）：

1. **`addcdiv_` 的旧重载**
   ```python
   p.data.addcdiv_(-step_size, m, denom)   # 旧写法，会触发 deprecation warning
   ```
   改成 `p.data.addcdiv_(m, denom, value=-step_size)`。

2. **权重衰减顺序**
   早先是“先做 Adam 更新，再把更新后的参数乘 $(1-\eta\lambda)$”，
   整体等价于 $(1-\eta\lambda)(\theta-\text{AdamUpdate})$；标准 AdamW 是先衰减旧参数。
   两者差一个 $\eta^2\lambda\cdot\text{AdamUpdate}$ 的项，数值上只有 `5e-5` 量级，
   测试容差下都能过，但和官方对拍时会一直有微小偏差。

**还可以改进的地方**：

- `.data` 绕过了 autograd。优化器里通常没问题，但换成 `@torch.no_grad()` + 直接用 `p.grad` 更清晰，也不容易误用。

> 结论：核心算法正确，且已经和官方实现逐位对齐。

---

## 6. 联系 CS336 Assignment 1

### 6.1 作业要求（讲义 §4.3）

作业要求实现 AdamW：

- 继承 `torch.optim.Optimizer`；
- `__init__` 接收 `lr`、`betas`、`eps`、`weight_decay`；
- 实现 `step()`，原地更新参数；
- 为每个参数维护一阶矩、二阶矩和步数；
- 实现解耦权重衰减；
- 实现 `adapters.get_adamw_cls`；
- 运行 `uv run pytest -k test_adamw`；
- 这一题 2 分。

### 6.2 接口与超参数

测试里的实例化方式：

```python
opt = opt_class(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

| 参数 | 测试值 | 含义 |
| --- | ---: | --- |
| `lr` | 1e-3 | 学习率 |
| `weight_decay` | 0.01 | 权重衰减 |
| `betas` | (0.9, 0.999) | 一阶/二阶矩衰减率 |
| `eps` | 1e-8 | 数值稳定常数 |

### 6.3 测试细节

`test_adamw`：

- 一个 `Linear(3, 2, bias=False)`；
- 跑 1000 步；
- 每步：`zero_grad` → 前向 → MSE loss → `backward` → `step`；
- 把最终权重和 `torch.optim.AdamW` 或参考快照比较；
- 容差 `atol=1e-4`。

测试注释特别说明：

> 参考实现和 PyTorch AdamW 在权重衰减的浮点顺序上可能略有不同，所以接受两种结果。

这意味着：

- 你的核心公式必须对；
- 权重衰减的微小顺序差异可以容忍；
- 但偏差修正、$m/v$ 更新、分母位置不能错。

### 6.4 与 checkpoint / 训练循环的关系

`test_checkpointing` 也会用 `get_adamw_cls()`：

- 保存模型 + 优化器状态；
- 加载后继续训练；
- 比较参数和优化器状态。

所以：

- `m`、`v`、`step` 必须放在 `self.state[p]` 里；
- 不要把它们存成模块外部的普通变量；
- 这样基类的 `state_dict()` / `load_state_dict()` 才能自动处理。

训练循环里：

```python
optimizer.zero_grad()
loss.backward()
# 可选：梯度裁剪
optimizer.step()
# 可选：更新学习率 schedule
```

### 6.5 与学习率 schedule 的关系

AdamW 的 `step()` 使用 `group["lr"]`；外部 schedule 每个 iteration 修改 `param_group["lr"]` 即可。

```python
# 仓库里这个调度器是个可调用对象，不是函数（tests/adapters.py 的
# run_get_lr_cosine_schedule 才叫那个名字，它内部也是转手调这个类）
from cs336_basics.Lr_Cosine_Schedule import CosineSchedule

schedule = CosineSchedule(max_lr, min_lr, warmup_iters, cosine_cycle_iters)

for it in range(num_iters):
    lr = schedule(it)
    for group in optimizer.param_groups:
        group["lr"] = lr
    ...
    optimizer.step()
```

> schedule 和 AdamW 是分开的：schedule 决定“这次用多大学习率”，AdamW 决定“在这个学习率下怎么更新”。

### 6.6 常见错误对照表

| 现象 | 可能原因 | 修正 |
| --- | --- | --- |
| loss 不下降/发散 | 偏差修正写错 | 检查 $1-\beta^t$ 和 $t$ 是否从 1 开始 |
| 第一步更新特别大 | 没做偏差修正 | 加 `m_hat`、`v_hat` |
| 第一步更新特别小 | 偏差修正方向反了 | 除以 $1-\beta^t$，不是乘 |
| 权重衰减像 L2 | 把 $\lambda\theta$ 加进了梯度 | 解耦，直接乘参数 |
| 训练结果和 PyTorch 差一点 | 权重衰减顺序 | 先衰减旧参数，再 Adam 更新 |
| `addcdiv_` 警告 | 用了旧重载 | `addcdiv_(m, denom, value=-lr)` |
| 某个参数不更新 | `p.grad is None` 没跳过/误跳过 | 检查梯度是否存在 |
| 恢复训练后曲线抖 | 没保存优化器状态 | checkpoint 保存 `optimizer.state_dict()` |
| `v` 为 0 除零 | 没加 `eps` | 分母加 $10^{-8}$ |
| betas 顺序反了 | `(beta2, beta1)` | 确认 `(beta1, beta2)` |
| 对所有参数都衰减 | norm/bias 也被衰减 | 大模型常对 norm/bias 关闭 weight decay（作业测试模型没有 bias，不影响） |
| 用 SGD 的动量写法 | $m$ 没乘 $(1-\beta_1)$ | Adam 用 EMA，注意系数 |

### 6.7 实现检查清单

- [ ] 继承 `torch.optim.Optimizer`，调用 `super().__init__(params, defaults)`；
- [ ] 每个参数状态里有 `step`、`m`、`v`；
- [ ] `step` 从 1 开始参与偏差修正；
- [ ] $m=\beta_1m+(1-\beta_1)g$；
- [ ] $v=\beta_2v+(1-\beta_2)g^2$；
- [ ] $\hat m=m/(1-\beta_1^t)$、$\hat v=v/(1-\beta_2^t)$；
- [ ] 分母是 $\sqrt{\hat v}+\epsilon$；
- [ ] 先解耦权重衰减，再做 Adam 更新；
- [ ] 用 `addcdiv_(m_hat, denom, value=-lr)`；
- [ ] 跑过 `pytest -k test_adamw`；
- [ ] checkpoint 能保存/加载优化器状态。

---

## 7. 常见误区 FAQ

**Q1：AdamW 和 Adam 有什么区别？**
Adam 把 L2 的 $\lambda\theta$ 加进梯度；AdamW 把权重衰减单独作用在参数上。AdamW 的衰减不被自适应分母缩放。

**Q2：权重衰减和 L2 正则化不是一回事吗？**
在 SGD 里等价；在 Adam 里不等价。AdamW 的“解耦”就是为了恢复“统一衰减”的语义。

**Q3：为什么需要偏差修正？**
$m$、$v$ 从 0 开始，早期估计偏小；除以 $1-\beta^t$ 把它们放大到正确尺度。

**Q4：$\epsilon$ 有什么用？**
防止 $\sqrt{\hat v}$ 为 0 时除零；通常放分母，$10^{-8}$。

**Q5：为什么 AdamW 占更多显存？**
每个参数要存 $m$ 和 $v$，约等于两份额外参数。

**Q6：为什么大模型用 AdamW 而不是 SGD？**
Transformer 参数多、尺度差异大，自适应步长 + 解耦衰减更容易训稳；但这不是绝对的。

**Q7：$\beta_2$ 为什么有时用 0.95？**
大模型训练中梯度非平稳，较短的第二矩记忆有时更合适；这是经验选择。

**Q8：所有参数都要 weight decay 吗？**
不一定。LayerNorm/Bias 通常不衰减；作业测试模型只有权重，所以不影响。

**Q9：AdamW 一定比 Adam 好吗？**
在正则化语义和泛化上通常更好；但具体任务仍要实验。

**Q10：优化器能保证收敛吗？**
不能。学习率、数据、模型、初始化都很重要；优化器只是其中一环。

**Q11：为什么 `zero_grad()` 要在 `backward()` 前？**
PyTorch 默认累加梯度；不清零会把上一批梯度加进来。

**Q12：AdamW 的 step 要放在 `backward()` 之后吗？**
是。`backward()` 先算梯度，`step()` 再根据梯度更新。

---

## 8. 小结与自测

### 8.1 一页速查

```text
本质:     优化器根据梯度决定参数怎么走
SGD:      θ ← θ - lr * g
Momentum: v ← βv + g; θ ← θ - lr * v
RMSProp:  v ← βv + (1-β)g²; θ ← θ - lr*g/(sqrt(v)+eps)
Adam:     m,v + 偏差修正 + 自适应步长
AdamW:    先 θ ← (1-lr*wd)θ，再 Adam 更新

AdamW 每步:
  1. g = grad
  2. t += 1
  3. θ *= (1 - lr * wd)
  4. m = β1*m + (1-β1)*g
  5. v = β2*v + (1-β2)*g²
  6. m_hat = m/(1-β1^t), v_hat = v/(1-β2^t)
  7. θ -= lr * m_hat/(sqrt(v_hat)+eps)
```

### 8.2 自测题

1. 优化器在训练循环里负责哪一步？它和 `loss.backward()` 的分工是什么？
2. SGD 的三个主要问题是什么？Momentum 和 RMSProp 分别解决了哪个？
3. 写出 Adam 的 $m$、$v$、偏差修正和更新公式。
4. 为什么 Adam 需要偏差修正？第一步不修正会发生什么？
5. 权重衰减和 L2 在 SGD 下为什么等价？在 Adam 下为什么不等价？
6. AdamW 的“解耦”具体解耦了什么？
7. AdamW 的更新顺序是什么？权重衰减放在哪里？
8. 为什么 AdamW 要存 $m$ 和 $v$？对显存有什么影响？
9. `test_adamw` 跑多少步、用什么容差、和谁比较？
10. 你给的 `Adamw.py` 有哪三处可以改进？
11. 为什么 checkpoint 要保存优化器状态？
12. AdamW 能保证训练一定收敛吗？

> 参考答案在 §1–§6 对应小节；先自己写一遍，再回去核对。

### 8.3 动手清单

- [ ] 手算 §4.5 的标量例子；
- [ ] 跑通 §5 的 AdamW 示例，确认与 PyTorch 的误差；
- [ ] 把 `Adamw.py` 的 `addcdiv_` 改成新式写法；
- [ ] 把权重衰减移到 Adam 更新之前；
- [ ] 跑过 `pytest -k test_adamw`；
- [ ] 保存/加载一次 optimizer state，确认 checkpoint 正常。

---

## 9. 延伸阅读

**论文**

- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 2019 —— AdamW 原论文。
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, 2015 —— Adam 原论文。
- Polyak, *Some methods of speeding up the convergence of iteration methods*, 1964 —— 动量法。
- Tieleman & Hinton, *RMSProp*, 2012 —— RMSProp 课程讲义。
- Ruder, *An overview of gradient descent optimization algorithms*, 2016 —— 优化器综述。

**课程与文档**

- CS336 课程主页：<https://cs336.stanford.edu>
- 作业讲义 `cs336_assignment1_basics.pdf` §4.3
- PyTorch 文档：`torch.optim.Optimizer`、`torch.optim.AdamW`

**同目录文档**

- `Cross_Entropy_Loss.md`：损失函数，优化器的输入。
- `完整语言模型架构的组装.md`：模型输出 logits 到 loss 的链路。
- `Transformer_Block.md`：模型结构。
- `RMSNorm.md`、`SwiGLU.md`、`Attention.md`：模型内部组件。

> **最后一句**：AdamW 不神秘，它只是把三件事分开管：**方向（动量）、步伐（自适应步长）、体重（解耦衰减）**。把这三件事拆开看，公式就顺了。
