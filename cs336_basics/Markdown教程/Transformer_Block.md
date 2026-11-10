# Transformer Block：从残差、Pre-Norm 到 CS336 的实现

> 本文是 CS336 Assignment 1（Building a Transformer LM）的配套概念笔记，对应讲义 §3.4–§3.5 的 **pre-norm Transformer Block**。它面向零基础读者，把 Block **由什么组成、残差为什么重要、网络退化与梯度消失怎么来的、Pre-Norm 和 Post-Norm 差在哪、作业怎么接线** 讲清楚。
>
> 读之前最好已经了解：
>
> - `RMSNorm.md`：每个子层前做归一化；
> - `SwiGLU.md`：前馈网络 FFN；
> - `Attention.md`：因果多头注意力（Q/K/V、掩码）。
>
> 读完你应该能回答五个问题：
>
> 1. 一个 Transformer Block 到底包含哪两个子层？为什么形状不变？
> 2. 残差连接 `y = x + F(x)` 为什么能缓解深层网络的退化与梯度消失？
> 3. 网络退化和梯度消失是一回事吗？各自的表现是什么？
> 4. Pre-Norm 和 Post-Norm 只是“先后顺序”不同吗？为什么现代大模型偏爱 Pre-Norm？
> 5. 在 CS336 的 `run_transformer_block` 里，权重键、形状、测试容差分别是什么？

## 目录

- [0. 阅读前的准备](#0-阅读前的准备)
- [1. Transformer Block 的定义与作用](#1-transformer-block-的定义与作用)
- [2. 残差连接：给深层网络修一条高速公路](#2-残差连接给深层网络修一条高速公路)
- [3. 网络退化与梯度消失](#3-网络退化与梯度消失)
- [4. Pre-Norm 与 Post-Norm](#4-pre-norm-与-post-norm)
- [5. Pre-Norm 的工作原理与逻辑](#5-pre-norm-的工作原理与逻辑)
- [6. 示例项目代码与逐段讲解](#6-示例项目代码与逐段讲解)
- [7. 联系 CS336 Assignment 1](#7-联系-cs336-assignment-1)
- [8. 常见误区 FAQ](#8-常见误区-faq)
- [9. 小结与自测](#9-小结与自测)
- [10. 延伸阅读](#10-延伸阅读)

---

## 0. 阅读前的准备

### 0.1 一句话版本

> **Transformer Block 是 Transformer 里重复堆叠的那一层“小工厂”：它先让 token 之间通过注意力交换信息，再让每个 token 通过前馈网络独立加工，两个子层都用残差连接把结果加回主干。**

用公式写：

$$
y=x+\operatorname{MHA}(\operatorname{RMSNorm}(x))
$$

$$
z=y+\operatorname{SwiGLU}(\operatorname{RMSNorm}(y))
$$

输入 $(B,T,D)$，输出还是 $(B,T,D)$。把这样的 Block 重复 $N$ 次，再接一个最终归一化和 LM head，就是 decoder-only Transformer。

### 0.2 符号表

| 符号 | 含义 | 一句话解释 |
| --- | --- | --- |
| $B$ | batch size | 一次处理多少条序列 |
| $T$ | sequence length | 一条序列有多少个 token |
| $D$ 或 $d_{model}$ | 隐藏维度 | 每个 token 用多少个数表示 |
| $H$ | num_heads | 注意力头数 |
| $d_{ff}$ | FFN 中间维度 | SwiGLU 内部先升到多宽 |
| $x$ | Block 的输入 | 形状 $(B,T,D)$ |
| $F$ | 子层函数 | 注意力或 FFN |
| $\operatorname{Norm}$ | 归一化 | 本作业用 RMSNorm |
| $\operatorname{MHA}$ | 多头自注意力 | 跨 token 交换信息 |
| $\operatorname{SwiGLU}$ | 前馈网络 | 逐 token 非线性加工 |

### 0.3 你需要的前置知识

1. **线性层**：$y=xW^\top$，输入输出最后一维可以不同；
2. **残差加法**：两个同形状张量逐元素相加；
3. **归一化**：RMSNorm 把每个 token 的尺度拉回 1 附近；
4. **注意力**：每个 token 按相关度加权读取其他 token；
5. **反向传播**：梯度沿计算图从后往前乘。

不需要会从头推注意力；只要知道“注意力输出和输入形状相同”就够了。

### 0.4 Block 在整张图里的位置

```text
token IDs  (B, T)
   │
   ▼
Embedding
   │
   ▼
x  (B, T, D)
   │
   ├──► Transformer Block × N
   │        │
   │        ├── RMSNorm → 注意力 → (+)
   │        └── RMSNorm → SwiGLU → (+)
   │
   ▼
Final RMSNorm
   │
   ▼
LM head (D → V)
   │
   ▼
logits  (B, T, V)
```

三个关键事实：

- **Block 不改变形状**：$(B,T,D)\to(B,T,D)$，所以能一层层叠；
- **Block 不改变序列长度**：它只改每个位置的特征；
- **Block 内部有两条子层**：注意力负责“跨位置”，FFN 负责“逐位置”。

### 0.5 学习路径

| 你想知道 | 看哪节 |
| --- | --- |
| Block 是什么、由什么组成 | §1 |
| 残差为什么重要 | §2 |
| 退化/梯度消失怎么来的 | §3 |
| Pre-Norm 和 Post-Norm 的区别 | §4 |
| Pre-Norm 为什么更稳 | §5 |
| 直接跑一个可运行项目 | §6 |
| 作业接口、权重键、测试 | §7 |
| 常见坑 | §8、§9 |

---

## 1. Transformer Block 的定义与作用

### 1.1 定义

**Transformer Block** 是 decoder-only Transformer 中重复堆叠的基本单元。它接收一个三维张量：

$$
x\in\mathbb{R}^{B\times T\times D}
$$

经过两个子层，输出同样形状的张量：

$$
\operatorname{Block}(x)\in\mathbb{R}^{B\times T\times D}
$$

两个子层是：

1. **多头自注意力（Multi-Head Self-Attention，MHA）**：让每个 token 看其他 token，交换信息；
2. **逐位置前馈网络（Position-wise Feed-Forward Network，FFN）**：对每个 token 独立做非线性变换。

每个子层都包着：

- 一个 **RMSNorm**（归一化）；
- 一个 **残差连接**（把子层输出加回输入）。

### 1.2 两个子层的分工

| 子层 | 作用轴 | 是否看其他 token | 类比 |
| --- | --- | --- | --- |
| 多头自注意力 | 跨位置（token mixing） | 是 | 开会时听别人发言 |
| 前馈网络 FFN | 跨通道（channel mixing） | 否 | 散会后自己写笔记 |

一句话：

> **注意力负责“信息交换”，FFN 负责“就地思考”。**

只有注意力，模型能搬运信息但缺少逐位置的非线性加工；只有 FFN，模型无法利用上下文。两者缺一不可。

### 1.3 为什么形状必须不变

残差连接要求：

$$
x+F(x)
$$

两边形状必须完全一致。因此：

- 注意力输出必须是 $(B,T,D)$；
- FFN 输出必须是 $(B,T,D)$；
- 归一化不改变形状；
- Block 的输出自然也是 $(B,T,D)$。

这也是为什么注意力的输出投影 $W_O$ 要把多头拼回来的维度压回 $D$，以及 SwiGLU 的 $W_2$ 要把 $d_{ff}$ 压回 $D$。

### 1.4 堆叠起来是什么样

一个 $N$ 层的 decoder-only Transformer：

```text
Embedding
   │
Block 1
   │
Block 2
   │
  ...
   │
Block N
   │
Final RMSNorm
   │
LM head
```

每个 Block 结构相同、参数不同。第 $l$ 层的输出可以写成：

$$
x_{l+1}=\operatorname{Block}_l(x_l)
$$

所有 Block 共享同一个残差流（residual stream）：一条从输入一直通到输出的 $(B,T,D)$ 主路径。

### 1.5 一个生活化类比

把 Block 想成一次“小组讨论 + 个人笔记”：

1. **注意力**：每个人听其他人的发言，把和自己相关的信息记下来；
2. **残差**：不是把原来的想法丢掉，而是在原想法上“补充”；
3. **FFN**：散会后，每个人独自整理笔记，形成自己的新理解；
4. **再残差**：把整理后的心得加回原来的想法；
5. 下一轮讨论，大家带着更新后的想法继续。

Block 重复 $N$ 次，就是“讨论—整理—再讨论—再整理”循环 $N$ 轮。

### 1.6 Block 不是什么

| 说法 | 纠正 |
| --- | --- |
| “Block 是一个注意力层” | 它包含注意力 **和** FFN 两个子层，还有归一化与残差 |
| “Block 会改变序列长度” | 不会。输入输出都是 $(B,T,D)$ |
| “Block 里的 FFN 会看其他 token” | 不会。FFN 逐位置作用 |
| “残差连接是一个可学习层” | 它没有参数，只是一个加法 |

---

## 2. 残差连接：给深层网络修一条高速公路

### 2.1 公式

对一个子层 $F$（注意力或 FFN）：

$$
y=x+F(x)
$$

- $x$：子层输入；
- $F(x)$：子层输出（也叫“残差分支”“delta”）；
- $y$：加回主干后的结果。

在 Block 里用两次：

$$
y=x+\operatorname{MHA}(\operatorname{RMSNorm}(x))
$$

$$
z=y+\operatorname{SwiGLU}(\operatorname{RMSNorm}(y))
$$

注意：**残差加的是子层输出，不是归一化输出**。归一化只在分支内部。

### 2.2 直觉：原画 + 批注

把 $x$ 想成一幅原画，$F(x)$ 想成一张透明批注：

- 批注可以很精彩，但原画始终还在；
- 就算这一层什么都没学到（$F(x)\approx0$），输出也约等于输入；
- 模型不需要“重新画一遍”，只需要学“在原画上补什么”。

这正是残差最重要的性质：**恒等映射（identity）是默认选项**。

### 2.3 梯度视角：为什么能缓解梯度消失

对 $y=x+F(x)$ 求导：

$$
\frac{dy}{dx}=I+\frac{\partial F}{\partial x}
$$

其中 $I$ 是单位矩阵。反向传播时，梯度可以沿着两条路走：

1. **恒等路径**：直接穿过加号，梯度原封不动；
2. **分支路径**：穿过 $F$，会被 $F$ 的雅可比矩阵变换。

即使 $F$ 的梯度很小，恒等路径仍然能传梯度。深层堆叠时：

$$
\frac{\partial x_L}{\partial x_0}
=\prod_{l=1}^{L}\left(I+\frac{\partial F_l}{\partial x_{l-1}}\right)
$$

把乘积展开后，始终包含一条“每层都走恒等路径”的项；它不会被连乘吃掉，因此梯度至少有一条不衰减的直路。

> 严格地说，残差不能保证梯度绝对不消失（多个矩阵仍可能相互抵消），但它提供了一条**不经过任何变换的直路**，这是深层网络可训的关键。

### 2.4 残差流：一条共享的主干

把所有 Block 的残差加法连起来看：

$$
x_{l+1}=x_l+F_l(\operatorname{Norm}(x_l))
$$

$x_l$ 就是**残差流**：

- 每个 Block 从残差流“读”一份输入；
- 子层算出一个“修改量”；
- 再把修改量“写回”残差流。

```text
x0 ──┬──► Block 1 ──► (+) ──┬──► Block 2 ──► (+) ──► ...
     │                       │
     └───────────────────────┘   ← 残差流一直保留
```

这个视角很重要：

- 残差流是**共享内存**，每个 Block 往里加东西；
- 归一化只发生在“读出来进子层”的支路上；
- 主干本身不被归一化（pre-norm 的核心）。

### 2.5 残差连接的三个作用

1. **让深层网络至少不退化**：如果新加的层学成恒等，深网络可以匹配浅网络；
2. **给梯度一条直路**：反向传播时有 $I$ 项，缓解梯度消失；
3. **保护原始信息**：子层只需要学“修改量”，不用从头重建表示。

### 2.6 常见误解

| 误解 | 纠正 |
| --- | --- |
| “残差就是把两个张量拼起来” | 不是拼接，是**逐元素相加**，形状不变 |
| “残差连接有参数” | 没有参数，只是一个加法 |
| “加了残差就一定不会梯度消失” | 只是提供直路；极深网络仍可能需要归一化、初始化、学习率调度 |
| “残差可以替代归一化” | 不能。残差管“加法直路”，归一化管“数值尺度”，两者配合 |

---

## 3. 网络退化与梯度消失

### 3.1 网络退化（degradation）

**现象**：把网络加深，训练误差反而变大——注意，不是验证误差，而是**训练误差**也变差。这不是过拟合，而是优化困难。

一个经典观察（ResNet 论文，He 等，2016）：

- 20 层普通网络 vs 56 层普通网络；
- 56 层的**训练误差**反而更高；
- 说明深层网络并不是“容量不够”，而是“优化器找不到好的解”。

**直觉**：如果新加的层都能学成恒等映射，深网络至少应该和浅网络一样好。但让一堆非线性层精确学出恒等映射并不容易；残差连接把“恒等”变成默认选项：

$$
y=x+F(x)
$$

只要 $F(x)\to0$，这一层就是恒等。于是“加深”至少不会破坏已有能力。

### 3.2 梯度消失与梯度爆炸

反向传播是链式法则的连乘：

$$
\frac{\partial L}{\partial x_0}
=\frac{\partial L}{\partial x_L}
\prod_{l=1}^{L}\frac{\partial x_l}{\partial x_{l-1}}
$$

如果每个雅可比矩阵的“平均放大倍数”小于 1，连乘会指数级衰减：

$$
0.5^{20}\approx9.5\times10^{-7}
$$

如果大于 1，则会指数级放大：

$$
2^{20}\approx1.05\times10^{6}
$$

前者叫**梯度消失**，后者叫**梯度爆炸**。

常见原因：

- 权重矩阵的奇异值偏小/偏大；
- 激活函数饱和（sigmoid/tanh 的导数远小于 1）；
- 层数太深，连乘次数太多；
- 归一化位置不当，让梯度反复穿过“压缩型”的变换。

### 3.3 表现：怎么判断出现了问题

| 现象 | 可能原因 |
| --- | --- |
| loss 很快变成常数，降不下去 | 梯度消失，浅层学不动 |
| 浅层参数几乎不更新 | 梯度在深层被压扁 |
| loss 突然变成 `nan` / `inf` | 梯度爆炸，数值溢出 |
| 加深网络后训练误差反而变高 | 网络退化 + 优化困难 |
| 必须用很小的学习率才能训 | 梯度尺度不稳定 |
| 需要很长时间 warmup | 初期梯度/尺度不稳定 |

### 3.4 残差如何同时缓解两件事

| 问题 | 残差的作用 |
| --- | --- |
| 网络退化 | 让恒等映射成为默认，深网络至少能匹配浅网络 |
| 梯度消失 | 反向传播有 $I$ 项，提供不经过子层的直路 |
| 梯度爆炸 | 主干上的加法不会放大梯度；子层内部的爆炸仍要靠归一化/裁剪 |
| 信息丢失 | 原始输入一直保留在残差流里 |

### 3.5 归一化的作用与位置

归一化解决的是**尺度**问题：

- 子层输入太大 → 激活饱和、梯度变小；
- 子层输入太小 → 信号弱、梯度小；
- 归一化把尺度拉回稳定区间。

但归一化放在哪里，会改变梯度路径：

- **Post-Norm**：归一化在残差加法之后，主干上的梯度要穿过归一化；
- **Pre-Norm**：归一化在子层之前，主干上的梯度不经过归一化。

这就是 §4 的核心。

### 3.6 退化 vs 梯度消失：不是一回事

| | 网络退化 | 梯度消失 |
| --- | --- | --- |
| 核心 | 深层网络训练误差变差 | 梯度幅度指数级变小 |
| 原因 | 恒等映射难学、优化困难 | 连乘的雅可比范数 < 1 |
| 表现 | 加深反而更差 | 浅层学不动、loss 停滞 |
| 解决 | 残差、更好的初始化/归一化 | 残差、ReLU/归一化、梯度裁剪、门控 |
| 关系 | 梯度消失是导致退化的原因之一，但退化不只由它造成 |

> 一句话：**退化是“深网络训不好”的现象，梯度消失是“梯度传不回去”的机制。** 残差对两者都有帮助。

---

## 4. Pre-Norm 与 Post-Norm

### 4.1 两种顺序

**Post-Norm**（原版 Transformer，Vaswani 等，2017）：

$$
y=\operatorname{Norm}\bigl(x+\operatorname{MHA}(x)\bigr)
$$

$$
z=\operatorname{Norm}\bigl(y+\operatorname{SwiGLU}(y)\bigr)
$$

顺序是：**子层 → 残差加法 → 归一化**。

**Pre-Norm**（本作业采用）：

$$
y=x+\operatorname{MHA}\bigl(\operatorname{Norm}(x)\bigr)
$$

$$
z=y+\operatorname{SwiGLU}\bigl(\operatorname{Norm}(y)\bigr)
$$

顺序是：**归一化 → 子层 → 残差加法**。

> 说明：原版 Post-Norm 用的是 LayerNorm；本作业的 Pre-Norm 用 RMSNorm。§6 的对比为了控制变量，Post-Norm 也用 RMSNorm。

### 4.2 结构对比图

Post-Norm：

```text
x ──┬──► 注意力 ──► (+) ──► Norm ──► y ──┬──► FFN ──► (+) ──► Norm ──► z
    │               ▲                     │             ▲
    └───────────────┘                     └─────────────┘
```

Pre-Norm：

```text
x ──┬──► Norm ──► 注意力 ──► (+) ──► y ──┬──► Norm ──► FFN ──► (+) ──► z
    │                        ▲            │                    ▲
    └────────────────────────┘            └────────────────────┘
```

一张图记住：

- **Post-Norm**：残差流每次都要“过一遍归一化”；
- **Pre-Norm**：归一化只在分支里，残差流本身是干净的。

### 4.3 对比表

| | Post-Norm | Pre-Norm |
| --- | --- | --- |
| 公式 | $y=\operatorname{Norm}(x+F(x))$ | $y=x+F(\operatorname{Norm}(x))$ |
| 归一化位置 | 残差加法之后 | 子层之前 |
| 残差主干 | 每个 Block 都被归一化 | 不被归一化 |
| 梯度路径 | 要穿过归一化的雅可比 | 有干净的 $I$ 直路 |
| 训练稳定性 | 对初始化/学习率更敏感 | 更稳，深层更容易训 |
| 是否需要 warmup | 通常更需要 | 也常用，但没那么敏感 |
| 最终归一化 | 不一定需要（最后已是归一化输出） | **需要**（残差流可能变大） |
| 代表模型 | 原版 Transformer | GPT-3、LLaMA、PaLM 等现代模型 |

### 4.4 为什么 Post-Norm 更难训

对 Post-Norm 的一个子层：

$$
x_{l+1}=\operatorname{Norm}\bigl(x_l+F_l(x_l)\bigr)
$$

求雅可比：

$$
\frac{\partial x_{l+1}}{\partial x_l}
=J_{\operatorname{Norm}}\left(I+\frac{\partial F_l}{\partial x_l}\right)
$$

每个 Block 都在左边乘一个 $J_{\operatorname{Norm}}$。如果归一化的雅可比范数小于 1，堆叠 $L$ 层就会指数级压缩梯度。也就是说：

> **Post-Norm 的残差直路被归一化“截断”了**：加法还在，但加法之后马上被归一化，梯度不能原封不动地传回去。

此外，Post-Norm 的残差流每一层都被强制拉回固定尺度，深层的信息需要不断“重新缩放”，优化更困难。

### 4.5 为什么 Pre-Norm 更稳

对 Pre-Norm：

$$
x_{l+1}=x_l+F_l\bigl(\operatorname{Norm}(x_l)\bigr)
$$

求雅可比：

$$
\frac{\partial x_{l+1}}{\partial x_l}
=I+\frac{\partial F_l}{\partial u}\frac{\partial \operatorname{Norm}}{\partial x_l}
$$

其中 $u=\operatorname{Norm}(x_l)$。注意第一项是**纯单位矩阵**：

- 梯度可以绕过 $F_l$ 和归一化，直接回到前一层；
- 归一化的雅可比只出现在分支里，不会把主干的直路“截断”；
- 深层堆叠时，梯度至少有一条不衰减的路径。

### 4.6 为什么 Pre-Norm 还需要 Final Norm

Pre-Norm 的主干不做归一化，所以残差流的范数通常会随层数增长：每一层都往里加一个修改量，深度越深，累积越大。如果直接把残差流送进 LM head：

- logits 的尺度可能过大；
- softmax 可能过度尖锐；
- 训练不稳定。

所以在所有 Block 之后、LM head 之前，要再加一次 **Final RMSNorm**：

```text
Block N → Final RMSNorm → LM head
```

> 注意：Final Norm 属于**模型级**组件，不属于单个 Block。作业的 `run_transformer_block` 不需要它；`run_transformer_lm` 才需要。

### 4.7 变体：不是只有两种

除了标准 Pre-Norm / Post-Norm，还有一些工程变体：

| 变体 | 归一化位置 | 直觉 |
| --- | --- | --- |
| Post-Norm | 残差加法之后 | 原版 Transformer |
| Pre-Norm | 子层之前 | 现代 LLM 默认 |
| Sandwich Norm | 子层前后都归一化 | 更稳，但多一次归一化 |
| DeepNorm | Post-Norm + 残差缩放 | 用缩放因子让极深网络可训 |

> 对新手来说，先把 Pre-Norm 和 Post-Norm 的区别搞清楚就够了；变体是工程优化，不是理解 Block 的前提。

### 4.8 一个诚实的边界

Pre-Norm 不是“绝对更好”：

- 它让深层网络更容易训，但残差流会变大，需要 final norm；
- Post-Norm 在合适的初始化、warmup、学习率下也能训得很好；
- 有些模型用 Post-Norm + DeepNorm 训练极深网络；
- “Pre-Norm 是现代默认”是经验事实，不是数学定理。

---

## 5. Pre-Norm 的工作原理与逻辑

### 5.1 第一步：把子层输入拉回稳定尺度

Pre-Norm 在每个子层前做：

$$
u=\operatorname{RMSNorm}(x)
$$

RMSNorm 让 $u$ 的均方根接近 1。于是：

- 注意力/FFN 的输入不会过大或过小；
- 权重初始化更容易设；
- 激活函数不容易饱和；
- 梯度尺度更稳定。

这一步解决的是“子层内部”的数值问题。

### 5.2 第二步：让子层只学“修改量”

子层输出 $F(u)$ 不是新的完整表示，而是**加在残差流上的修改量**：

$$
x_{\text{new}}=x+F(u)
$$

这有三个好处：

1. **恒等映射容易**：$F(u)\to0$ 就是恒等；
2. **信息不丢**：原始 $x$ 一直在主干上；
3. **每层任务更轻**：只学“补什么”，不用重建整个表示。

### 5.3 第三步：保持残差流干净

Pre-Norm 的主干：

$$
x_{l+1}=x_l+F_l(\operatorname{Norm}(x_l))
$$

归一化只发生在 $F_l$ 的输入上，主干本身没有 Norm。于是：

- 反向传播时，梯度可以沿 $x_l$ 直接回到更早的层；
- 前向传播时，信息可以沿主干一路保留；
- 这就是讲义说的 **clean residual stream**（干净的残差流）。

### 5.4 第四步：用 Final Norm 收尾

因为主干没有归一化，残差流范数可能随深度增长。Final Norm 在最后把尺度拉回稳定范围，再送进 LM head。

这就是 Pre-Norm 的完整逻辑：

```text
每个子层：Norm → 子层 → 加回主干
整个模型：Embedding → N × Block → Final Norm → LM head
```

### 5.5 梯度视角的总结

| 结构 | 主干梯度 | 结果 |
| --- | --- | --- |
| 无残差 | $\prod J_F$ | 容易指数级消失/爆炸 |
| Post-Norm | $\prod J_{\text{Norm}}(I+J_F)$ | 残差直路被 Norm 截断 |
| Pre-Norm | $\prod (I+J_F J_{\text{Norm}})$ | 有干净的 $I$ 直路 |

这就是为什么 Pre-Norm 在深层 Transformer 里更受欢迎。

### 5.6 一个数值直觉

假设每个 Block 的“有效放大倍数”是 0.8：

- 12 层：$0.8^{12}\approx0.069$；
- 24 层：$0.8^{24}\approx0.0047$；
- 48 层：$0.8^{48}\approx2.2\times10^{-5}$。

如果是 1.2：

- 12 层：$1.2^{12}\approx8.9$；
- 24 层：$1.2^{24}\approx79$；
- 48 层：$1.2^{48}\approx6.3\times10^{3}$。

深层网络的梯度对“每层放大倍数”极其敏感。残差和 Pre-Norm 的作用，就是把每层的有效放大倍数拉回 1 附近。

### 5.7 代价与注意点

| 点 | 说明 |
| --- | --- |
| 残差流范数增长 | Pre-Norm 不限制主干尺度，需要 final norm |
| 额外归一化 | 每个子层前一次，最后再一次 |
| 不是万能 | 极深网络仍需要好的初始化、学习率、warmup、梯度裁剪 |
| 实现细节 | 归一化位置、残差加法顺序、final norm 都不能写错 |

---

## 6. 示例项目代码与逐段讲解

下面是一个**完整可运行**的最小项目（纯 PyTorch，CPU 即可）。它实现了：

- RMSNorm；
- SwiGLU FFN；
- 简化版因果多头注意力（省略 RoPE，聚焦 Block 接线）；
- 可切换 Pre-Norm / Post-Norm 的 Transformer Block；
- 12 层堆叠的梯度对比实验。

> **关于 RoPE**：为了让代码聚焦 Block 本身，示例的注意力省略了 RoPE。CS336 中只需在拆头之后、打分之前，对 q/k 做旋转；Block 的接线完全不变。RoPE 的细节见 `Attention.md` 和作业讲义 §3.4.4。
>
> 示例的 Block 构造函数只有 `d_model/num_heads/d_ff`；CS336 里还要把 `max_seq_len` 和 `theta` 传给 RoPE 注意力。

### 6.1 运行方式

把下面的代码保存为 `transformer_block_demo.py`，然后：

```bash
python transformer_block_demo.py
```

不需要 GPU，也不需要安装额外依赖（只需要 PyTorch）。

### 6.2 完整代码

```python
"""最小 Transformer Block 示例：pre-norm / post-norm 对比。

纯 PyTorch，CPU 可跑。为了聚焦 Block 的接线，示例的注意力省略了 RoPE；
在 CS336 中只需在 q/k 上多做一次旋转，Block 的结构完全不变。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)


class RMSNorm(nn.Module):
    """逐 token 的 RMSNorm，gain 初始化为 1。"""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)


class SwiGLU(nn.Module):
    """FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )。"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """简化版多头因果自注意力（省略 RoPE）。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        # 1) 投影并拆头：(B,T,D) -> (B,H,T,head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 2) 打分 + 缩放
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 3) 因果掩码：上三角（不含对角线）设为 -inf，不能看未来
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        # 4) softmax + 加权求和
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        # 5) 合头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """一个 pre-norm（或 post-norm）Transformer Block。"""

    def __init__(self, d_model, num_heads, d_ff, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        if self.pre_norm:
            # pre-norm: 先归一化，再进子层，最后加残差
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            # post-norm: 先子层 + 残差，再归一化
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        return x


def grad_stats(pre_norm, n_layers=12, seed=0):
    """堆叠 n_layers 个 Block，返回输入梯度、首层参数梯度、输出范数。"""
    torch.manual_seed(seed)
    D, H, F_ = 64, 4, 128
    blocks = nn.Sequential(*[TransformerBlock(D, H, F_, pre_norm) for _ in range(n_layers)])
    x = torch.randn(2, 8, D, requires_grad=True)
    y = blocks(x)
    loss = y.pow(2).mean()
    loss.backward()

    first = blocks[0]
    grads = [p.grad.norm().item() for p in first.parameters() if p.grad is not None]
    return x.grad.norm().item(), sum(grads) / len(grads), y.norm().item()


def main():
    D, H, F_ = 64, 4, 128
    block = TransformerBlock(D, H, F_, pre_norm=True)
    x = torch.randn(2, 8, D)

    y = block(x)
    print("输入形状:", tuple(x.shape))
    print("输出形状:", tuple(y.shape))
    print("Block 参数量:", sum(p.numel() for p in block.parameters()))

    pre = grad_stats(pre_norm=True, n_layers=12, seed=0)
    post = grad_stats(pre_norm=False, n_layers=12, seed=0)
    print("\n12 层堆叠的梯度/范数：")
    print("  pre-norm : 输入梯度 %.4e，首层参数梯度均值 %.4e，输出范数 %.2f" % pre)
    print("  post-norm: 输入梯度 %.4e，首层参数梯度均值 %.4e，输出范数 %.2f" % post)


if __name__ == "__main__":
    main()
```

### 6.3 预期输出

```text
输入形状: (2, 8, 64)
输出形状: (2, 8, 64)
Block 参数量: 41088

12 层堆叠的梯度/范数：
  pre-norm : 输入梯度 1.6091e-01，首层参数梯度均值 2.8517e-01，输出范数 41.25
  post-norm: 输入梯度 1.3626e-07，首层参数梯度均值 2.3871e-07，输出范数 32.00
```

> 这是**随机初始化**下的演示，不是严格 benchmark。它想说明的是：在这个 12 层小模型里，post-norm 的浅层梯度比 pre-norm 小了约 6 个数量级；pre-norm 的干净残差流让梯度更容易传回去。

### 6.4 逐段讲解

#### 6.4.1 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)
```

- `weight` 是逐通道增益，初始化为 1；
- 先升 `float32`，避免平方溢出；
- `mean(dim=-1, keepdim=True)` 只在通道维求平均，每个 token 独立；
- 最后还原输入 dtype。

> 详细推导见 `RMSNorm.md`。

#### 6.4.2 SwiGLU

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

- 两个升维矩阵 `w1`、`w3`，一个降维矩阵 `w2`；
- `w1` 支路做 SiLU，`w3` 支路不做激活；
- 两条支路逐元素相乘，再降维回 $D$；
- 没有 bias。

> 详细推导见 `SwiGLU.md`。

#### 6.4.3 因果多头注意力

```python
def forward(self, x):
    B, T, D = x.shape
    q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    out = out.transpose(1, 2).contiguous().view(B, T, D)
    return self.out_proj(out)
```

逐行：

1. `q/k/v` 投影后拆成 `(B,H,T,head_dim)`，每个头独立算；
2. `q @ k.T` 得到每个 query 对每个 key 的分数；
3. 除以 $\sqrt{d_{head}}$，防止分数过大导致 softmax 过尖；
4. 上三角（不含对角线）是未来，填 `-inf`；
5. softmax 得到权重，再对 `v` 加权求和；
6. 合头、`contiguous()`、输出投影回 $D$。

> **CS336 的差异**：作业要求把 RoPE 加在 q/k 上（拆头之后、打分之前），并且注意力 forward 可能要显式接收权重。Block 的接线不变。

#### 6.4.4 Block 本体

```python
def forward(self, x):
    if self.pre_norm:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
    else:
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
    return x
```

Pre-Norm 版：

1. `self.ln1(x)` 归一化；
2. `self.attn(...)` 注意力；
3. `x + ...` 残差；
4. `self.ln2(x)` 再归一化；
5. `self.ffn(...)` 前馈；
6. `x + ...` 再残差。

Post-Norm 版：

1. 先算 `self.attn(x)`；
2. 加残差 `x + ...`；
3. 再归一化 `self.ln1(...)`；
4. FFN 同理。

> 作业要求的是 **Pre-Norm** 版本。Post-Norm 版本放在这里是为了对比和理解，不是作业的默认实现。

#### 6.4.5 梯度实验

```python
def grad_stats(pre_norm, n_layers=12, seed=0):
    torch.manual_seed(seed)
    D, H, F_ = 64, 4, 128
    blocks = nn.Sequential(*[TransformerBlock(D, H, F_, pre_norm) for _ in range(n_layers)])
    x = torch.randn(2, 8, D, requires_grad=True)
    y = blocks(x)
    loss = y.pow(2).mean()
    loss.backward()

    first = blocks[0]
    grads = [p.grad.norm().item() for p in first.parameters() if p.grad is not None]
    return x.grad.norm().item(), sum(grads) / len(grads), y.norm().item()
```

- 堆 12 个 Block，模拟“深层”；
- 用同一个 seed 初始化 pre/post 两版，尽量控制变量；
- 损失用 `y.pow(2).mean()`，简单且尺度可控；
- 反传后看：
  - `x.grad.norm()`：梯度能传回输入多少；
  - 首层参数梯度：浅层还能不能学；
  - `y.norm()`：残差流的尺度。

实验结论：

- Pre-Norm 的输入梯度约 $1.6\times10^{-1}$；
- Post-Norm 的输入梯度约 $1.4\times10^{-7}$；
- 相差约 6 个数量级。

这不是说 Post-Norm 一定训不了（真实训练有 warmup、初始化、LayerNorm 等），而是直观展示了**残差主干是否被归一化截断**对梯度的影响。

### 6.5 代码与概念对照表

| 代码 | 对应概念 |
| --- | --- |
| `self.ln1`、`self.ln2` | 每个子层前的归一化 |
| `self.attn` | 跨 token 信息交换 |
| `self.ffn` | 逐 token 非线性加工 |
| `x + ...` | 残差连接 |
| `pre_norm=True` | 归一化在子层前 |
| `pre_norm=False` | 归一化在残差后 |
| `x.grad.norm()` | 梯度能否传回浅层 |
| `y.norm()` | 残差流的尺度 |

### 6.6 小实验

**实验 1：把残差去掉**

```python
def forward(self, x):
    x = self.attn(self.ln1(x))
    x = self.ffn(self.ln2(x))
    return x
```

再跑一次梯度实验，观察浅层梯度是否明显变小、输出范数是否失控。

**实验 2：加深到 24 层 / 48 层**

把 `grad_stats` 的 `n_layers` 改成 24、48，比较 pre/post 的输入梯度。层数越深，差距通常越明显。

**实验 3：换成 Post-Norm 但加 warmup**

Post-Norm 不是“绝对不能训”。在真实训练里，warmup、较小的学习率、合适的初始化都能帮助它。这个实验可以让你体会“结构 + 训练技巧”的关系。

**实验 4：换归一化**

把 RMSNorm 换成 LayerNorm，比较 pre/post 的梯度。结论可能不同，但“主干是否被归一化”仍然是关键。

---

## 7. 联系 CS336 Assignment 1

### 7.1 作业要求（讲义 §3.5）

作业要求你实现 **pre-norm Transformer block**：

- 两个子层：多头自注意力、SwiGLU FFN；
- 每个子层：先 RMSNorm，再主子层，最后残差；
- 输入输出都是 `(batch_size, sequence_length, d_model)`；
- 实现测试适配器 `adapters.run_transformer_block`；
- 运行 `uv run pytest -k test_transformer_block`；
- 这一题 3 分。

讲义给出的第一个子层公式：

$$
y=x+\operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x))
$$

第二个子层同理：

$$
z=y+\operatorname{SwiGLU}(\operatorname{RMSNorm}(y))
$$

### 7.2 接口与权重键

`run_transformer_block` 的签名：

```python
def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    ...
```

| 参数 | 含义 |
| --- | --- |
| `d_model` | 隐藏维度 |
| `num_heads` | 注意力头数，必须整除 `d_model` |
| `d_ff` | FFN 中间维度 |
| `max_seq_len` | RoPE 预计算的最大长度 |
| `theta` | RoPE 的频率参数 |
| `weights` | 参考实现的 state dict |
| `in_features` | 输入 `(B,T,D)` |

`weights` 的键（去掉 `layers.0.` 前缀后）：

| 键 | 形状 | 对应模块 |
| --- | --- | --- |
| `attn.q_proj.weight` | `(D, D)` | 查询投影 |
| `attn.k_proj.weight` | `(D, D)` | 键投影 |
| `attn.v_proj.weight` | `(D, D)` | 值投影 |
| `attn.output_proj.weight` | `(D, D)` | 输出投影 |
| `ln1.weight` | `(D,)` | 第一个 RMSNorm 的增益 |
| `ffn.w1.weight` | `(d_ff, D)` | SwiGLU 升维（门控） |
| `ffn.w2.weight` | `(D, d_ff)` | SwiGLU 降维 |
| `ffn.w3.weight` | `(d_ff, D)` | SwiGLU 升维（内容） |
| `ln2.weight` | `(D,)` | 第二个 RMSNorm 的增益 |

没有 `bias` 键。所有线性层都不带 bias。

### 7.3 学习路径：Block 是“组装”

CS336 的作业顺序大致是：

```text
BPE 分词 → Embedding → Linear
                    → RMSNorm
                    → SiLU / SwiGLU
                    → RoPE / 因果多头注意力
                    → Transformer Block   ← 本文
                    → 完整 Transformer LM
                    → 训练 / 评估 / 消融
```

Block 本身几乎没有新数学，主要是**接线**：

1. 先把 RMSNorm、注意力、SwiGLU 分别写对；
2. 再把它们按 `Norm → 子层 → 残差` 拼起来；
3. 最后用测试确认权重键、形状、数值都对。

如果你前面的模块都通过了各自的测试，Block 通常只是“组合正确性”的问题。

### 7.4 一个适配器骨架

如果你的模块把权重注册为 `nn.Parameter`，并且命名和参考实现一致：

```python
import torch

# 注意模块名是大驼峰：仓库里的文件叫 Transformer_Block.py，
# Linux 区分大小写，写成 transformer_block 会直接 ModuleNotFoundError
from cs336_basics.Transformer_Block import TransformerBlock


def run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features):
    # 仓库这套 Block 的权重是构造函数传进去的（不是先建空模块再 load_state_dict），
    # 9 组权重的顺序必须和 __init__ 的签名一致
    block = TransformerBlock(
        d_model, num_heads, d_ff, max_seq_len, theta,
        weights["attn.q_proj.weight"], weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"], weights["attn.output_proj.weight"],
        weights["ln1.weight"], weights["ln2.weight"],
        weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"],
    )
    return block(in_features)
```

如果你的注意力模块 forward 需要显式传入权重（像仓库里的 `CausalMultiHeadAttentionWithRoPE` 那样），就在 adapter 里把 `weights` 拆开（下面是示意，变量名按你的实现调整）：

```python
attn_out = attn(
    ln1_out,
    weights["attn.q_proj.weight"],
    weights["attn.k_proj.weight"],
    weights["attn.v_proj.weight"],
    weights["attn.output_proj.weight"],
    token_positions,
)
```

两种方式都可以，关键是把**同一个权重**放到**同一个位置**。

### 7.5 测试细节

`test_transformer_block` 会：

- 从完整的 state dict 中取 `layers.0.*`，去掉前缀，得到单个 Block 的权重；
- 传入 `d_model=64`、`num_heads=4`、`d_ff=128`、`max_seq_len=16`、`theta=10000.0`；
- 输入 `in_embeddings`，形状 `(4,12,64)`；
- 容差 `atol=1e-4`。

这意味着：

- 你的 Block 必须使用 **pre-norm** 顺序；
- 必须使用 **RMSNorm**，不是 LayerNorm；
- 必须把 **RoPE 作用在 q/k** 上；
- 残差不能漏；
- 权重键必须对得上；
- 数值误差要小（float32 计算）。

### 7.6 常见错误对照表

| 现象 | 可能原因 | 修正 |
| --- | --- | --- |
| 形状不对 | 注意力/FFN 输出没压回 $D$ | 检查 `output_proj` 和 `w2` |
| 数值和参考差很多 | 写成了 post-norm | 改成 `x + F(Norm(x))` |
| 输出完全不对 | 忘了残差，或残差加错对象 | 加的是子层输出，不是 Norm 输出 |
| 浅层梯度异常 | 归一化位置错 | 作业要求 pre-norm |
| 测试报权重键缺失 | state dict 前缀没去掉 | 用 `layers.0.*` 并去掉 `layers.0.` |
| `d_model` 不能被 `num_heads` 整除 | 头维度算错 | 入口加断言 |
| 注意力看到未来 | 忘了因果掩码 | 上三角填 `-inf` |
| 位置信息没生效 | RoPE 没加在 q/k 上 | 拆头后、打分前旋转 |
| 数值 `nan` | softmax 整行 `-inf` | 掩码不要遮住对角线 |
| 参数比预期多 | 加了 bias | 删除 bias |
| 测试误差略大 | 用了半精度 | 用 float32 |
| 把 final norm 放进 Block | 多了一次归一化 | final norm 属于 LM，不属于 Block |

### 7.7 实现检查清单

- [ ] 两个子层：注意力在前，FFN 在后；
- [ ] 每个子层前都有 RMSNorm；
- [ ] 每个子层输出都加回输入（残差）；
- [ ] 使用 pre-norm：`x + F(Norm(x))`；
- [ ] 注意力带因果掩码和 RoPE；
- [ ] SwiGLU 无 bias；
- [ ] 输入输出都是 `(B,T,D)`；
- [ ] 权重键与参考实现一致；
- [ ] 跑过 `pytest -k test_transformer_block`。

### 7.8 作业里的消融

配置文件里有这些开关：

```json
{
  "remove_rmsnorm": false,
  "use_post_norm": false,
  "ffn_type": null
}
```

它们用于后续实验：

- `use_post_norm=true`：把 Block 改成 post-norm，比较训练稳定性；
- `remove_rmsnorm=true`：去掉归一化，观察退化/梯度问题；
- `ffn_type`：切换 FFN 类型（如 SiLU vs SwiGLU）。

> 必做的 `transformer_block` 是 **pre-norm + RMSNorm + SwiGLU**；这些开关是给你做对比实验用的，不要让它们影响默认实现。

---

## 8. 常见误区 FAQ

**Q1：Block 和 Layer 是一回事吗？**
通常同义。Transformer 里说“一层”一般就是一个 Block；但 Block 内部有两个子层，不要把“层”和“子层”混为一谈。

**Q2：残差连接和跳跃连接（skip connection）是一回事吗？**
是。Residual connection、skip connection 通常指同一个东西：$y=x+F(x)$。

**Q3：为什么 Pre-Norm 还需要 Final Norm？**
因为主干没有归一化，残差流范数会随深度增长；Final Norm 在 LM head 前把尺度拉回稳定范围。

**Q4：Post-Norm 是不是完全不能用？**
不是。原版 Transformer 就是 Post-Norm；配合 warmup、合适的学习率和初始化，它也能训。只是在极深模型里，Pre-Norm 更稳、更省心。

**Q5：残差能彻底解决梯度消失吗？**
不能。它提供了一条直路，但极深网络、糟糕的初始化、过大的学习率仍可能导致问题。残差要和归一化、初始化、学习率调度一起用。

**Q6：为什么 Block 的输出形状必须和输入一样？**
因为要做残差加法。如果形状不同，`x + F(x)` 无法逐元素相加。

**Q7：Block 会混合不同 token 吗？**
注意力会；FFN 不会。Block 整体既做跨 token 交换，也做逐 token 加工。

**Q8：可以把 FFN 放在注意力前面吗？**
可以，但标准 Transformer 是“先注意力、后 FFN”。顺序会影响信息流；作业按标准顺序。

**Q9：为什么 Block 里用 RMSNorm 而不是 LayerNorm？**
作业和现代 LLaMA 系模型用 RMSNorm，因为它更简单、更快；概念上 LayerNorm 也能用。详见 `RMSNorm.md`。

**Q10：为什么没有 bias？**
现代 LLM 通常省略线性层和归一化的 bias，减少参数和计算；作业也要求无 bias。

**Q11：Block 里有两个残差，能不能只用一个？**
不行。每个子层都需要自己的残差，否则子层输出会替换主干，深层信息容易丢失。

**Q12：Pre-Norm 的“干净残差流”到底是什么意思？**
从 Embedding 到最后一个 Block 的输出，主路径上没有归一化；归一化只发生在“读出来进子层”的支路上。这样梯度可以沿主路径直接回传。

---

## 9. 小结与自测

### 9.1 一页速查

```text
Block:      两个子层 = 注意力 + FFN
Pre-Norm:   y = x + Attn(Norm(x))
            z = y + FFN(Norm(y))
Post-Norm:  y = Norm(x + Attn(x))
            z = Norm(y + FFN(y))
残差:       y = x + F(x)，形状不变，无参数
梯度:       dy/dx = I + dF/dx，恒等路径防止梯度消失
退化:       加深后训练误差变差，不是过拟合
Pre-Norm:   归一化在分支，残差流干净；最后要 Final Norm
CS336:      pre-norm + RMSNorm + RoPE 注意力 + SwiGLU；
            run_transformer_block；pytest -k test_transformer_block
```

### 9.2 自测题

1. 画出 Pre-Norm Block 和 Post-Norm Block 的结构图，标出归一化、子层、残差的位置。
2. 写出 Pre-Norm Block 的两行公式，并说明每个符号的含义。
3. 残差连接为什么能缓解梯度消失？从雅可比矩阵的角度解释。
4. 网络退化和梯度消失有什么区别？各自的表现是什么？
5. 为什么 Post-Norm 的残差直路会被“截断”？
6. 为什么 Pre-Norm 需要在所有 Block 之后加 Final Norm？
7. Block 的输入输出为什么必须同形状？
8. 如果去掉 Block 里的残差，会发生什么？怎么用实验验证？
9. 在 CS336 的 `run_transformer_block` 里，`weights` 有哪些键？分别对应什么？
10. 为什么注意力要加因果掩码？掩码加在哪里？
11. Pre-Norm 和 Post-Norm 哪个更适合极深模型？为什么？
12. 如果 Block 测试通过但完整 LM 测试失败，可能是什么原因？

> 参考答案在 §1–§7 对应小节；先自己写一遍，再回去核对。

### 9.3 动手清单

- [ ] 跑通 §6 的示例，确认输出形状和梯度对比；
- [ ] 把 `pre_norm` 改成 `False`，观察浅层梯度变化；
- [ ] 去掉残差，观察梯度和输出范数；
- [ ] 把层数加深到 24/48，比较 pre/post；
- [ ] 把示例接上 RoPE，改成 CS336 的注意力；
- [ ] 实现 `run_transformer_block`，跑过测试；
- [ ] 用 `use_post_norm` 做一次小消融。

---

## 10. 延伸阅读

**论文**

- Vaswani et al., *Attention Is All You Need*, 2017 —— Transformer 与 Post-Norm。
- He et al., *Deep Residual Learning for Image Recognition*, 2016 —— 残差连接与网络退化。
- Ba, Kiros & Hinton, *Layer Normalization*, 2016 —— LayerNorm。
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, 2020 —— Pre-Norm 与训练稳定性。
- Nguyen & Salazar, *Transformers without Tears: Improving the Normalization of Self-Attention*, 2019 —— Pre-Norm 的早期讨论。
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019 —— RMSNorm。
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023 —— LLaMA 的 Pre-Norm + RMSNorm。

**课程与文档**

- CS336 课程主页：<https://cs336.stanford.edu>
- 作业讲义 `cs336_assignment1_basics.pdf` §3.4–§3.5
- PyTorch 文档：`torch.nn.Module`、`torch.nn.Linear`、`torch.softmax`

**同目录文档**

- `RMSNorm.md`：Block 里每个子层前的归一化。
- `SwiGLU.md`：Block 里的前馈网络。
- `Attention.md`：Q/K/V、因果掩码与多头注意力。
- `Transformer 模块.md`：从文本到下一个词预测的完整链路。
- `Embedding_Linear.md`：线性层与嵌入层的基础。

> **最后一句**：Block 的代码很短，但它把“信息交换、逐位置加工、数值稳定、梯度直路”四件事装进了一个可重复的单元。理解这四件事，比记住代码更重要。
