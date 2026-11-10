# Gradient Clipping：给梯度装一个“限速器”

> 本文面向零基础读者，结合 CS336 Assignment 1 的 `gradient_clipping` 任务和仓库里的 `Gradient_Clipping.py`，讲清楚 **梯度裁剪和梯度下降有什么区别、它解决什么问题、数学上怎么算、代码怎么接进训练循环**。
>
> 读之前最好知道：
>
> - 梯度是什么：损失对参数的导数；
> - 训练循环：前向 → loss → `backward()` → `optimizer.step()`；
> - 梯度下降/AdamW 是用梯度更新参数。
>
> 读完你应该能回答五个问题：
>
> 1. 梯度裁剪和梯度下降是一回事吗？如果不是，差别在哪？
> 2. 为什么训练中会出现“梯度爆炸”？裁剪如何救场？
> 3. “全局 L2 范数裁剪”的公式是什么？为什么它不改变梯度方向？
> 4. `Gradient_Clipping.py` 的代码在做什么？有哪些可以改进的地方？
> 5. CS336 的 `test_gradient_clipping` 在测什么？训练循环里应该把裁剪放在哪一步？

## 目录

- [0. 阅读前的准备](#0-阅读前的准备)
- [1. 梯度裁剪 vs 梯度下降](#1-梯度裁剪-vs-梯度下降)
- [2. 梯度裁剪的作用与使用场景](#2-梯度裁剪的作用与使用场景)
- [3. 数学原理与直觉](#3-数学原理与直觉)
- [4. CS336 示例代码逐段讲解](#4-cs336-示例代码逐段讲解)
- [5. 常见误区 FAQ](#5-常见误区-faq)
- [6. 小结与自测](#6-小结与自测)
- [7. 延伸阅读](#7-延伸阅读)

---

## 0. 阅读前的准备

### 0.1 一句话版本

> **梯度裁剪不是“另一种梯度下降”，而是在更新参数之前，给梯度向量装一个限速器：总长度超过上限，就按比例缩短；没超过，就原样放行。**

一句话记忆：

> **梯度下降决定“往哪走”，梯度裁剪决定“这一步最多走多猛”。**

### 0.2 符号表

| 符号 | 含义 | 一句话解释 |
| --- | --- | --- |
| $\theta$ | 模型参数 | 要更新的东西 |
| $g$ | 梯度 | 所有参数的梯度拼成的大向量 |
| $\|g\|_2$ | 全局 L2 范数 | 所有梯度元素的平方和再开根 |
| $M$ | `max_l2_norm` | 允许的最大梯度范数 |
| $\epsilon$ | 数值稳定常数 | 通常 $10^{-6}$，防止除以 0 |
| $c$ | 裁剪系数 | 缩放梯度用的正数 |
| $\tilde g$ | 裁剪后的梯度 | 交给优化器的梯度 |

### 0.3 训练循环里的位置

```text
for batch in data:
    optimizer.zero_grad()
    logits = model(batch)
    loss = loss_fn(logits, targets)
    loss.backward()                    # 1) 得到每个参数的梯度

    clip_grad_norm_(model.parameters(), max_norm)  # 2) 裁剪梯度

    optimizer.step()                   # 3) 用裁剪后的梯度更新参数
```

三个关键点：

1. **在 `backward()` 之后**：这时才有梯度；
2. **在 `optimizer.step()` 之前**：这样优化器拿到的才是裁剪后的梯度；
3. **它不更新参数**：只改 `p.grad`。

### 0.4 学习路线

| 你想知道 | 看哪节 |
| --- | --- |
| 它和梯度下降的区别 | §1 |
| 它解决什么问题、什么时候用 | §2 |
| 公式和直觉 | §3 |
| 作业代码怎么讲 | §4 |
| 常见坑 | §5、§6 |

---

## 1. 梯度裁剪 vs 梯度下降

### 1.1 梯度下降在做什么

最朴素的梯度下降：

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

- $g_t$：当前梯度；
- $\eta$：学习率；
- 含义：沿着负梯度方向走一步。

AdamW 等优化器也是同一个框架，只是把 $g_t$ 换成了更聪明的更新量：

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

**梯度下降/优化器负责“更新参数”。**

### 1.2 梯度裁剪在做什么

梯度裁剪只改梯度：

$$
g_t\leftarrow \tilde g_t
$$

其中：

$$
\tilde g_t=
\begin{cases}
g_t, & \|g_t\|_2\le M\\
g_t\cdot\dfrac{M}{\|g_t\|_2+\epsilon}, & \|g_t\|_2>M
\end{cases}
$$

**它不更新参数，也不决定方向**；它只是把梯度的“长度”限制住。

> 类比：
>
> - **梯度下降**是司机，决定往左打方向盘还是往右；
> - **梯度裁剪**是限速器，只在速度超过上限时把油门收一点，不改变方向。

### 1.3 一张表看清区别

| | 梯度下降 / 优化器 | 梯度裁剪 |
| --- | --- | --- |
| 作用对象 | 参数 $\theta$ | 梯度 $g$ |
| 公式 | $\theta\leftarrow\theta-\eta g$ | $g\leftarrow g\cdot c$ |
| 何时 | `optimizer.step()` | `backward()` 之后、`step()` 之前 |
| 是否改变方向 | 沿负梯度方向 | 全局范数裁剪**不改变方向** |
| 是否有参数 | 优化器有状态（如 AdamW 的 $m,v$） | 没有可学习参数 |
| 目的 | 真正更新参数 | 限制单步更新幅度 |
| 能否单独训练 | 能 | **不能**，必须配合优化器 |

一句话：

> **梯度裁剪不是优化器的替代品，而是优化器前面的“安全阀”。**

### 1.4 和“调小学习率”有什么区别

| | 调小学习率 | 梯度裁剪 |
| --- | --- | --- |
| 何时生效 | 每一步都缩小 | 只在梯度超过阈值时缩小 |
| 影响范围 | 所有更新 | 只影响异常大的那一步 |
| 是否改变方向 | 不改变 | 全局范数裁剪不改变 |
| 主要目的 | 控制整体步长 | 防止偶发爆炸 |
| 副作用 | 训练可能变慢 | 阈值太小也会变慢 |

> 调小学习率像“全程开慢车”；梯度裁剪像“平时正常开，只在超速时自动降速”。

### 1.5 和“权重衰减/正则化”有什么区别

| | 权重衰减 | 梯度裁剪 |
| --- | --- | --- |
| 作用 | 把参数往 0 拉 | 把梯度长度压小 |
| 是否改变梯度方向 | 不一定 | 全局范数裁剪不改变 |
| 目的 | 正则化、防止过拟合 | 稳定训练、防止爆炸 |
| 是否影响参数值 | 是 | 不直接；通过更新间接影响 |

> 裁剪不是正则化：它不会让参数“更小”，只会让某一步“别走太猛”。

---

## 2. 梯度裁剪的作用与使用场景

### 2.1 梯度爆炸是什么

训练时，梯度是所有层、所有样本贡献的叠加。有时候会遇到：

- 某个 batch 特别“难”，梯度特别大；
- 长序列/注意力里某些位置梯度爆炸；
- 混合精度下数值范围有限，容易溢出；
- RNN 里梯度沿时间步连乘，指数级放大；
- 学习率偏大，更新一步走太远。

表现：

| 现象 | 说明 |
| --- | --- |
| loss 突然飙高 | 一步更新把参数打飞 |
| loss 变成 `nan` / `inf` | 数值溢出 |
| 梯度范数突然很大 | 通常是爆炸的前兆 |
| 参数值异常变大 | 更新过猛 |
| 训练曲线剧烈抖动 | 大梯度反复冲击 |

> 类比：水管里突然来了一股高压水，不装减压阀，管子可能爆。

### 2.2 裁剪怎么救场

梯度裁剪在**更新之前**把梯度的总长度压到上限以内：

- 正常步骤：梯度不大，原样通过；
- 异常步骤：梯度太大，按比例缩小；
- 方向不变：只是“同样的方向，走短一点”。

这样：

- 参数不会被一步打飞；
- loss 不容易突然爆炸；
- 训练更平滑；
- 可以用稍大的学习率/更激进的配置。

### 2.3 适用场景

| 场景 | 是否常用 | 说明 |
| --- | --- | --- |
| Transformer / LLM 训练 | **非常常用** | 长序列、注意力、混合精度都可能有梯度尖峰 |
| RNN / LSTM | **非常常用** | 梯度沿时间步连乘，容易爆炸 |
| 强化学习 | **常用** | 回报方差大，梯度不稳定 |
| GAN | **常用** | 对抗训练容易震荡 |
| 深层网络 | **常用** | 深层连乘，数值更敏感 |
| 小模型/简单任务 | 不一定 | 梯度稳定时可以不裁 |
| 梯度消失 | **不适用** | 裁剪只封顶，不抬底 |

> 典型阈值：LLM 训练常用 `max_norm=1.0`；具体值要看模型、优化器、batch size 和梯度范数分布。CS336 测试用 `1e-2` 只是为了让裁剪一定触发。

### 2.4 三种常见裁剪方式

| 方式 | 做法 | 特点 |
| --- | --- | --- |
| 按值裁剪（clip by value） | 每个元素 clamp 到 $[-M,M]$ | 简单，但可能改变方向 |
| 按范数裁剪（clip by norm） | 整个梯度向量按比例缩放 | 保持方向，本文/作业采用 |
| 按参数逐个裁剪 | 每个参数自己的范数单独裁 | 可能总范数仍超限 |

> CS336 要求的是**全局 L2 范数裁剪**：把所有参数的梯度拼成一个大向量，算总范数，再统一缩放。

### 2.5 代价与注意点

| 点 | 说明 |
| --- | --- |
| 阈值太小 | 每步都被压，训练变慢，可能欠拟合 |
| 阈值太大 | 几乎不触发，等于没裁 |
| 只裁一部分参数 | 要清楚自己在干什么；通常裁全部 |
| 不处理 NaN/Inf | 裁剪救不了已经变成 `nan` 的梯度，需要检查数据/学习率 |
| 不能解决消失 | 小梯度不会被放大；要靠残差/归一化/初始化 |

---

## 3. 数学原理与直觉

### 3.1 把梯度拼成一个大向量

假设模型有多个参数张量，每个都有自己的梯度。把它们全部展平、拼起来：

$$
g=\operatorname{concat}\bigl(\operatorname{flatten}(g_1),\operatorname{flatten}(g_2),\dots\bigr)
$$

总元素个数记为 $n$。

**全局 L2 范数**：

$$
\|g\|_2=\sqrt{\sum_{i=1}^{n}g_i^2}
$$

> 注意：不是“每个参数范数相加”，而是“所有元素的平方和再开根”。

### 3.2 裁剪公式

给定上限 $M$：

$$
\tilde g=
\begin{cases}
g, & \|g\|_2\le M\\[4pt]
g\cdot\dfrac{M}{\|g\|_2+\epsilon}, & \|g\|_2>M
\end{cases}
$$

也可以写成：

$$
\tilde g=g\cdot c,\qquad
c=\min\left(1,\frac{M}{\|g\|_2+\epsilon}\right)
$$

- 范数没超：$c=1$，原样；
- 范数超了：$c<1$，整体缩短。

### 3.3 裁剪后范数是多少

当 $\|g\|>M$：

$$
\|\tilde g\|_2
=\left\|\frac{M}{\|g\|_2+\epsilon}g\right\|_2
=\frac{M}{\|g\|_2+\epsilon}\|g\|_2
<M
$$

所以裁剪后的范数**刚好略小于 $M$**。

> 为什么不是刚好等于 $M$？因为分母加了 $\epsilon$；$\epsilon$ 很小，几乎等于 $M$。

### 3.4 为什么“不改变方向”

当 $\|g\|>M$ 时：

$$
\tilde g=cg,\qquad c=\frac{M}{\|g\|+\epsilon}>0
$$

乘以一个**正数**只会改变长度，不会改变方向。

二维直觉：

```text
       g
      ↗
     ↗
    ↗
   ●────────► 原来的方向
    ＼
     ＼ 裁剪后：同方向，短一点
      ↘
```

> 类比：把一支箭按比例剪短，箭头指的方向不变。

### 3.5 一个具体例子

两个参数，梯度分别是：

```text
g1 = [3, 4]     → ||g1|| = 5
g2 = [0, 12]    → ||g2|| = 12
```

拼起来：

```text
g = [3, 4, 0, 12]
||g|| = sqrt(9 + 16 + 0 + 144) = sqrt(169) = 13
```

设上限 $M=6.5$：

$$
c=\frac{6.5}{13+10^{-6}}\approx0.5
$$

裁剪后：

```text
g1 = [1.5, 2]    → ||g1|| = 2.5
g2 = [0, 6]      → ||g2|| = 6
||g|| = sqrt(2.25 + 4 + 0 + 36) = 6.5
```

方向：

```text
[3,4]  → [1.5,2]   方向相同，长度减半
[0,12] → [0,6]     方向相同，长度减半
```

| 量 | 裁剪前 | 裁剪后 |
| --- | ---: | ---: |
| $g_1$ | [3, 4] | [1.5, 2] |
| $g_2$ | [0, 12] | [0, 6] |
| 全局范数 | 13 | 6.5 |
| 方向 | 不变 | 不变 |

### 3.6 全局裁剪 vs 逐参数裁剪

同样是上面的例子，$M=6.5$：

**全局裁剪**（本文/作业）：

- 总范数 13 > 6.5；
- 系数 0.5；
- 两个参数都乘 0.5；
- 结果总范数 6.5。

**逐参数裁剪**：

- $g_1$ 范数 5 ≤ 6.5，不变；
- $g_2$ 范数 12 > 6.5，乘 $6.5/12≈0.5417$；
- 结果 $g_2=[0,6.5]$；
- 总范数 $\sqrt{25+42.25}≈8.2$，**仍然超过 6.5**。

| 方式 | $g_1$ | $g_2$ | 总范数 |
| --- | --- | --- | ---: |
| 原始 | [3,4] | [0,12] | 13 |
| 全局裁剪 | [1.5,2] | [0,6] | 6.5 |
| 逐参数裁剪 | [3,4] | [0,6.5] | ≈8.2 |

> 结论：如果你想要“总力度不超过 $M$”，就要用**全局范数**；逐参数裁剪只能保证“每个参数不超过 $M$”。

### 3.7 什么时候不裁剪

如果 $\|g\|_2\le M$：

$$
\tilde g=g
$$

不做任何修改。

> 裁剪不是“每次都缩小”，而是“只在超速时缩小”。正常步骤应该原样通过。

### 3.8 为什么加 $\epsilon$

如果 $\|g\|$ 很小，分母可能接近 0；虽然“不裁剪”分支不会除法，但为了统一公式和数值稳定，通常加一个很小的 $\epsilon$（如 $10^{-6}$）。

> 在裁剪分支里，$\|g\|>M\ge0$，分母一般不会为 0；$\epsilon$ 更多是工程习惯和防极端情况。

---

## 4. CS336 示例代码逐段讲解

### 4.1 作业要求（讲义 §4.5）

作业要求实现梯度裁剪：

- 输入一组参数和一个最大 L2 范数 `max_l2_norm`；
- 计算所有参数梯度的**全局 L2 范数**；
- 如果超过上限，按比例缩放每个梯度；
- **原地修改**每个 `p.grad`；
- 使用 $\epsilon=10^{-6}$；
- 实现 `adapters.run_gradient_clipping`；
- 运行 `uv run pytest -k test_gradient_clipping`；
- 这一题 1 分。

### 4.2 仓库里的 `Gradient_Clipping.py`

核心逻辑（省略了注释和空参检查）：

```python
import torch

class GradientClip:
    def __init__(self, parameters, max_l2_norm, epsilon=1e-6):
        self.parameters = parameters
        self.max_l2_norm = max_l2_norm
        self.epsilon = epsilon

    def __call__(self):
        grads = [p.grad for p in self.parameters if p.grad is not None]
        all_grads = torch.cat([grad.flatten() for grad in grads])
        grad_l2 = torch.norm(all_grads, 2)
        if grad_l2 > self.max_l2_norm:
            clip_coeff = self.max_l2_norm / (grad_l2 + self.epsilon)
            for grad in grads:
                grad.mul_(clip_coeff)
```

这段代码的**核心逻辑是对的**：

1. 收集所有非空梯度；
2. 拼成一个大向量；
3. 算全局 L2 范数；
4. 超过上限就乘系数；
5. 原地修改梯度。

我用随机梯度对比过 `torch.nn.utils.clip_grad_norm_`，结果完全一致（差值为 0）。

### 4.3 逐行解释

```python
grads = [p.grad for p in self.parameters if p.grad is not None]
```

- 只处理有梯度的参数；
- `requires_grad=False` 的参数梯度是 `None`，自动跳过；
- 冻结参数不会被裁剪。

```python
all_grads = torch.cat([grad.flatten() for grad in grads])
```

- 每个梯度展平成一维；
- 拼成一个大向量；
- 这是为了算**全局**范数，而不是每个参数单独算。

```python
grad_l2 = torch.norm(all_grads, 2)
```

- 计算 L2 范数；
- 等价于 `sqrt(sum(all_grads ** 2))`。

```python
if grad_l2 > self.max_l2_norm:
    clip_coeff = self.max_l2_norm / (grad_l2 + self.epsilon)
    for grad in grads:
        grad.mul_(clip_coeff)
```

- 只有超限才裁剪；
- 系数是 $M/(\|g\|+\epsilon)$；
- `mul_` 原地修改，优化器之后会读到新梯度。

### 4.4 可以改进的地方

#### 4.4.2 梯度为空时会报错

如果没有参数有梯度：

```python
torch.cat([])   # RuntimeError: expected a non-empty list
```

建议加：

```python
if not grads:
    return
```

#### 4.4.3 传生成器时，第二次调用会“没参数”

`model.parameters()` 返回的是生成器。当前 `__init__` 直接存了生成器：

```python
self.parameters = parameters
```

第一次 `__call__` 里的列表推导会把它**消费掉**；第二次再调用时，生成器已经空了。

建议在 `__init__` 里转成列表：

```python
self.parameters = list(parameters)
```

> 测试只调用一次，所以当前代码能过；但训练循环里通常会反复调用，这个坑要避开。

#### 4.4.4 `torch.cat` 会复制一份完整梯度

把所有梯度拼起来会多占一份显存。可以改成“平方和”：

```python
total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
```

效果一样，但不需要拼大向量。

#### 4.4.5 `nan` / `inf` 不处理

如果梯度已经是 `nan`/`inf`，裁剪可能得到 `nan` 系数，救不回来。可以在裁剪前检查：

```python
if not torch.isfinite(total_norm):
    # 跳过、报错或记录日志
    ...
```

作业测试不涉及这个，但真实训练值得注意。

### 4.5 一个更干净的版本

```python
import torch


class GradientClip:
    def __init__(self, parameters, max_l2_norm, epsilon=1e-6):
        self.parameters = list(parameters)   # 避免生成器被消费
        self.max_l2_norm = max_l2_norm
        self.epsilon = epsilon

    def __call__(self):
        grads = [p.grad for p in self.parameters if p.grad is not None]
        if not grads:
            return

        # 全局 L2 范数：所有梯度元素的平方和再开根
        total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

        # 只在超限时裁剪；不变时原样放行
        if total_norm > self.max_l2_norm:
            clip_coeff = self.max_l2_norm / (total_norm + self.epsilon)
            for g in grads:
                g.mul_(clip_coeff)
```

和原版相比：

- 参数转成 list；
- 空梯度保护；
- 用平方和代替 `cat`；
- 拼写修正；
- 逻辑不变。

### 4.6 完整可运行示例

保存为 `gradient_clipping_demo.py`：

```python
"""Gradient Clipping 小示例：全局 L2 范数裁剪。"""
import torch
from torch import nn


class GradientClip:
    def __init__(self, parameters, max_l2_norm, epsilon=1e-6):
        self.parameters = list(parameters)
        self.max_l2_norm = max_l2_norm
        self.epsilon = epsilon

    def __call__(self):
        grads = [p.grad for p in self.parameters if p.grad is not None]
        if not grads:
            return

        total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
        print(f"before: total_norm={total_norm.item():.6f}")

        if total_norm > self.max_l2_norm:
            coef = self.max_l2_norm / (total_norm + self.epsilon)
            print(f"clip coefficient={coef.item():.6f}")
            for g in grads:
                g.mul_(coef)

        total_norm_after = torch.sqrt(sum(g.pow(2).sum() for g in grads))
        print(f"after : total_norm={total_norm_after.item():.6f}")


def main():
    p1 = nn.Parameter(torch.zeros(2))
    p2 = nn.Parameter(torch.zeros(2))
    p1.grad = torch.tensor([3.0, 4.0])    # norm 5
    p2.grad = torch.tensor([0.0, 12.0])   # norm 12
    params = [p1, p2]

    print("original grads:", p1.grad, p2.grad)
    GradientClip(params, max_l2_norm=6.5)()
    print("clipped grads :", p1.grad, p2.grad)


if __name__ == "__main__":
    main()
```

预期输出：

```text
original grads: tensor([3., 4.]) tensor([ 0., 12.])
before: total_norm=13.000000
clip coefficient=0.500000
after : total_norm=6.500000
clipped grads : tensor([1.5000, 2.0000]) tensor([0.0000, 6.0000])
```

### 4.7 接进作业 adapter

```python
def run_gradient_clipping(parameters, max_l2_norm):
    GradientClip(parameters, max_l2_norm)()
```

或者不用类，直接写函数：

```python
def run_gradient_clipping(parameters, max_l2_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total_norm > max_l2_norm:
        coef = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(coef)
```

### 4.8 测试细节

`test_gradient_clipping`：

- 6 个 5×5 的随机张量；
- 最后一个 `requires_grad=False`，用来验证冻结参数被跳过；
- `max_norm=1e-2`，故意让裁剪触发；
- 用 `torch.nn.utils.clip_grad_norm_` 作为参考；
- 容差 `atol=1e-5`；
- 比较每个参数的梯度。

当前 `Gradient_Clipping.py` 的实现与参考一致（我本地对比差值为 0）。

### 4.9 训练循环里的正确位置

```python
for batch in data:
    optimizer.zero_grad()
    logits = model(batch)
    loss = loss_fn(logits, targets)
    loss.backward()

    # 关键：backward 之后，step 之前
    GradientClip(model.parameters(), max_l2_norm=1.0)()

    optimizer.step()
```

顺序错误会怎样：

| 位置 | 结果 |
| --- | --- |
| `backward()` 之前 | 梯度还是 `None` 或旧值，裁剪无效 |
| `step()` 之后 | 参数已经更新，裁剪太晚 |
| 在 `zero_grad()` 之前 | 裁的是上一轮残留梯度，容易错 |
| 在 `backward()` 之后、`step()` 之前 | 正确 |

### 4.10 常见错误对照表

| 现象 | 可能原因 | 修正 |
| --- | --- | --- |
| 裁剪没效果 | 阈值太大 / 顺序错 | 检查范数是否真的超限、放在 step 前 |
| 训练变慢/欠拟合 | 阈值太小 | 调大 `max_norm` |
| `RuntimeError: expected a non-empty list` | 没有梯度 | 加空列表保护 |
| 第二次调用报错 | 传了生成器 | `list(parameters)` |
| 总范数仍超限 | 逐参数裁剪 | 用全局范数 |
| 梯度方向变了 | 用了按值裁剪 | 改用按范数裁剪 |
| `nan` 仍在 | 梯度已经 NaN | 检查数据/学习率；裁剪救不了 |
| 冻结参数被改 | 没过滤 `p.grad is None` | 只处理非空梯度 |
| 参考值对不上 | 用了 sum 而不是 sqrt(sum squares) | 用全局 L2 范数 |
| 忘记 epsilon | 分母可能为 0 | 加 $10^{-6}$ |

### 4.11 实现检查清单

- [ ] 收集所有 `p.grad is not None` 的梯度；
- [ ] 计算全局 L2 范数（平方和再开根）；
- [ ] 只在 `norm > max_norm` 时裁剪；
- [ ] 系数是 `max_norm / (norm + 1e-6)`；
- [ ] 对所有梯度乘同一个系数；
- [ ] 原地修改 `p.grad`；
- [ ] 不改变梯度方向；
- [ ] 处理空梯度、生成器；
- [ ] 放在 `backward()` 之后、`step()` 之前；
- [ ] 跑过 `pytest -k test_gradient_clipping`。

---

## 5. 常见误区 FAQ

**Q1：梯度裁剪是另一种梯度下降吗？**
不是。梯度下降/优化器负责更新参数；裁剪只改梯度长度，不更新参数。

**Q2：裁剪会改变梯度方向吗？**
全局范数裁剪不会：$\tilde g=cg$，$c>0$。按值裁剪可能改变方向。

**Q3：裁剪能解决梯度消失吗？**
不能。它只给梯度设上限，不会把很小的梯度放大。消失要靠残差、归一化、初始化等。

**Q4：裁剪能替代学习率调度吗？**
不能。裁剪是异常保护，学习率调度是整体步长策略；两者可以一起用。

**Q5：裁剪和权重衰减是一回事吗？**
不是。权重衰减把参数往 0 拉；裁剪限制梯度长度。

**Q6：为什么用全局范数，不逐个参数裁？**
全局范数保证“总力度”不超过 $M$；逐个裁可能总范数仍超限。

**Q7：阈值设多少？**
常见 `1.0`（LLM），也可以看梯度范数分布选；太小会拖慢训练，太大等于没裁。CS336 测试用 `1e-2` 是为了触发裁剪。

**Q8：梯度已经是 `nan` 怎么办？**
裁剪救不了。要检查数据、学习率、混合精度；可以跳过该 batch 或报错。

**Q9：裁剪后还要 `zero_grad()` 吗？**
要。`zero_grad()` 是清空梯度，和裁剪无关；通常在下一轮 `backward()` 前调用。

**Q10：裁剪会影响 AdamW 的 $m$、$v$ 吗？**
会间接影响：优化器用裁剪后的梯度更新 $m$、$v$；裁剪本身不改优化器状态。

**Q11：所有参数都要裁吗？**
通常全部一起裁；如果只想裁某些层，传对应参数子集，但要清楚总范数的含义。

**Q12：裁剪会让训练一定稳定吗？**
不能保证。它只是降低“梯度爆炸”的风险；学习率、数据、模型结构仍然重要。

---

## 6. 小结与自测

### 6.1 一页速查

```text
作用:     限制梯度的全局 L2 范数
公式:     norm = sqrt(sum(g_i^2))
          if norm > M: g *= M/(norm + eps)
          else: 不变
方向:     全局范数裁剪不改变方向
位置:     backward() 之后，optimizer.step() 之前
范围:     所有参数一起算总范数
参数:     M（max_norm），eps=1e-6
不是:     不是梯度下降、不是权重衰减、不解决梯度消失
```

### 6.2 自测题

1. 梯度裁剪和梯度下降分别作用在什么对象上？
2. 写出全局 L2 范数裁剪的公式。
3. 为什么裁剪后的范数“略小于” $M$？
4. 为什么全局范数裁剪不改变梯度方向？
5. 全局裁剪和逐参数裁剪有什么区别？各有什么风险？
6. 梯度爆炸有哪些表现？裁剪如何缓解？
7. 裁剪能解决梯度消失吗？为什么？
8. 裁剪应该放在训练循环的哪一步？
9. `Gradient_Clipping.py` 有哪几处可以改进？
10. CS336 的测试为什么把 `max_norm` 设成 `1e-2`？
11. 阈值太小会怎样？太大呢？
12. 裁剪和 AdamW 的 $m$、$v$ 是什么关系？

> 参考答案在 §1–§4 对应小节；先自己写一遍，再回去核对。

### 6.3 动手清单

- [ ] 手算 §3.5 的例子；
- [ ] 跑通 §4.6 的示例，观察范数从 13 变 6.5；
- [ ] 把 `GradientClip` 包成 adapter；
- [ ] 跑过 `pytest -k test_gradient_clipping`；
- [ ] 在 AdamW 训练循环里接入裁剪；
- [ ] 试着把 `max_norm` 调大/调小，观察训练变化。

---

## 7. 延伸阅读

**论文与资料**

- Pascanu, Mikolov & Bengio, *On the difficulty of training recurrent neural networks*, 2013 —— 梯度裁剪的经典讨论。
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016 —— 梯度爆炸/消失与裁剪。
- PyTorch 文档：`torch.nn.utils.clip_grad_norm_`、`torch.nn.utils.clip_grad_value_`。

**课程与文档**

- CS336 课程主页：<https://cs336.stanford.edu>
- 作业讲义 `cs336_assignment1_basics.pdf` §4.5
- 同目录 `AdamW.md`：裁剪后的梯度交给优化器。
- 同目录 `Cross_Entropy_Loss.md`：梯度从哪来。
- 同目录 `完整语言模型架构的组装.md`：完整训练链路。

> **最后一句**：梯度裁剪不是“让模型学得更好”的魔法，而是“别让某一步把模型打飞”的安全带。方向交给梯度，力度交给裁剪，更新交给优化器。
