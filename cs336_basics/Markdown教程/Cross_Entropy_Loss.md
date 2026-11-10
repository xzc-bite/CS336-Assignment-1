# 交叉熵损失：从“惊讶程度”到 Log-Sum-Exp

> 本文面向零基础读者，结合 CS336 Assignment 1 的 `cross_entropy` 任务，讲清楚交叉熵损失 **在比较什么、怎么对齐、公式怎么来、为什么会溢出、代码怎么写**。
>
> 读之前最好知道：
>
> - softmax 能把一组分数变成概率；
> - `log` 和 `exp` 互为逆运算；
> - 语言模型输出的是 logits，不是概率。
>
> 读完你应该能回答五个问题：
>
> 1. 交叉熵到底在比较哪两个分布？
> 2. 预测值和标签怎么逐样本、逐类别对齐？
> 3. 为什么工程实现不直接算 `-log(softmax(z)[y])`？
> 4. Log-Sum-Exp 技巧为什么能避免溢出/下溢？
> 5. CS336 的 `run_cross_entropy` 要什么形状、返回什么、测试怎么过？

## 目录

- [0. 阅读前的准备](#0-阅读前的准备)
- [1. 核心思想：两个分布的差距](#1-核心思想两个分布的差距)
- [2. 数据对齐：谁和谁比](#2-数据对齐谁和谁比)
- [3. 概率定义与工程公式](#3-概率定义与工程公式)
- [4. 数值稳定性：Log-Sum-Exp](#4-数值稳定性log-sum-exp)
- [5. 小示例：一步一步算](#5-小示例一步一步算)
- [6. 与 CS336 Assignment 1 的关联](#6-与-cs336-assignment-1-的关联)
- [7. 常见误区 FAQ](#7-常见误区-faq)
- [8. 小结与自测](#8-小结与自测)
- [9. 延伸阅读](#9-延伸阅读)

---

## 0. 阅读前的准备

### 0.1 一句话版本

> **交叉熵损失衡量的是：模型给“正确答案”分配的概率有多低。模型越自信地押中正确答案，损失越小；越自信地押错，损失越大。**

公式：

$$
\mathcal{L}=-\log p_y
$$

其中：

- $p$ 是模型预测的概率分布；
- $y$ 是正确答案的下标；
- $p_y$ 是模型给正确答案的那个概率。

一句话记忆：

> **交叉熵 = “正确答案被预测的概率”的负对数。**

### 0.2 符号表

| 符号 | 含义 | 一句话解释 |
| --- | --- | --- |
| $N$ | 样本数 | 一次算多少个预测 |
| $V$ | 类别数 / 词表大小 | 每个样本有多少个候选 |
| $z$ | logits | 模型输出的未归一化分数 |
| $p$ | 预测概率分布 | softmax 之后的概率 |
| $q$ | 真实标签分布 | 通常是 one-hot |
| $y$ | 正确类别下标 | $0\le y<V$ |
| $p_y$ | 正确类别的预测概率 | 我们最关心的那个数 |
| $\mathcal{L}$ | 损失 | 标量，越小越好 |

### 0.3 你需要的前置知识

1. **softmax**：把 $V$ 个分数变成 $V$ 个概率，加起来等于 1；
2. **one-hot**：正确类别位置是 1，其他位置是 0；
3. **对数**：$-\log p$ 在 $p$ 越接近 1 时越小，在 $p$ 越接近 0 时越大；
4. **张量形状**：`(N,V)` 表示 $N$ 个样本、每个样本 $V$ 个分数；
5. **平均**：损失通常对所有样本取平均，而不是求和。

### 0.4 它在训练流程里的位置

```text
输入 token IDs
   │
   ▼
Transformer LM
   │
   ▼
logits (B, T, V)
   │
   ▼
交叉熵损失（和错开一位的标签比）
   │
   ▼
loss（标量）
   │
   ▼
loss.backward() → 优化器更新参数
```

交叉熵是**训练信号**：它告诉模型“你给正确答案的概率是高了还是低了”，反向传播再根据这个信号调整参数。

### 0.5 阅读路线

| 你想知道 | 看哪节 |
| --- | --- |
| 交叉熵在比什么 | §1 |
| 预测和标签怎么对齐 | §2 |
| 公式怎么推 | §3 |
| 为什么会溢出、怎么修 | §4 |
| 具体数值怎么算 | §5 |
| 作业怎么过 | §6 |
| 常见坑 | §7、§8 |

---

## 1. 核心思想：两个分布的差距

### 1.1 分类问题在做什么

假设词表有 $V=5$ 个 token，模型对某个位置输出 5 个 logits：

```text
logits = [2.0, 0.5, 0.1, -1.0, 0.3]
```

经过 softmax，变成 5 个概率：

```text
p = [0.6230, 0.1390, 0.0932, 0.0310, 0.1138]   （加起来等于 1）
```

真实标签是第 0 个 token，所以真实分布是 one-hot：

```text
q = [1, 0, 0, 0, 0]
```

这个例子里，模型给正确答案的概率是 $p_0\approx0.623$，所以损失约 $-\log0.623\approx0.474$。

交叉熵比较的就是这两个分布：

- **预测分布** $p$：模型认为每个 token 有多可能；
- **真实分布** $q$：正确答案的 one-hot 分布。

### 1.2 交叉熵的定义

对离散分布 $q$（真实）和 $p$（预测）：

$$
H(q,p)=-\sum_{j=1}^{V}q_j\log p_j
$$

因为 $q$ 是 one-hot：

$$
q_j=
\begin{cases}
1, & j=y\\
0, & j\neq y
\end{cases}
$$

所以求和里只有 $j=y$ 这一项留下：

$$
H(q,p)=-\log p_y
$$

这就是交叉熵在分类任务里的形式，也叫**负对数似然**（negative log-likelihood，NLL）。

> **一句话**：交叉熵只关心“模型给正确答案分配了多少概率”，其他类别的概率通过 softmax 的分母间接影响它。

### 1.3 直观类比：惊讶程度

把 $p_y$ 想成“模型对正确答案的自信程度”：

| 模型给正确答案的概率 $p_y$ | 损失 $-\log p_y$ | 直觉 |
| ---: | ---: | --- |
| 0.99 | 0.010 | 几乎猜中，几乎不惊讶 |
| 0.90 | 0.105 | 比较自信，损失很小 |
| 0.50 | 0.693 | 一半一半，中等惊讶 |
| 0.10 | 2.303 | 不太看好正确答案，比较惊讶 |
| 0.01 | 4.605 | 几乎押错，非常惊讶 |
| 0.00 | $+\infty$ | 完全不给正确答案机会 |

所以：

> **交叉熵 = 模型看到正确答案时的“惊讶程度”。**

### 1.4 几个重要性质

1. **最小值为 0**：当 $p_y=1$ 时，$-\log 1=0$；
2. **没有上界**：$p_y$ 越接近 0，损失越大；
3. **均匀分布时**：$p_y=1/V$，损失是 $\log V$；
4. **只关心正确类**：其他类的分数通过 softmax 分母影响 $p_y$；
5. **可微**：可以对 logits 求梯度，所以能反向传播。

> 交叉熵不是“准确率”。准确率只看 argmax 对不对；交叉熵还看“错得有多自信”。一个模型可能全猜对，但对正确答案只给 0.4 的概率，交叉熵仍然不低。

### 1.5 为什么要取平均

一次训练通常有 $N$ 个样本（或者 $B\times T$ 个位置）。每个样本都有自己的损失：

$$
\mathcal{L}_i=-\log p_{i,y_i}
$$

总损失通常取平均：

$$
\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_i
$$

这样：

- 损失的大小和 batch 大小无关；
- 学习率更容易设置；
- 不同 batch 之间可以比较。

> **求和还是平均？** 作业要求平均。求和会让梯度随 batch 大小线性放大，训练会变得难调。

---

## 2. 数据对齐：谁和谁比

### 2.1 输入和标签的形状

在分类任务里，最常见的形式是：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `inputs` / `logits` | `(N, V)` | 每个样本对每个类别的分数 |
| `targets` | `(N,)` | 每个样本的正确类别下标 |

例如 $N=2,V=5$：

```text
logits =
[[2.0, 0.5, 0.1, -1.0, 0.3],    ← 样本 0 的 5 个类别分数
 [0.1, 0.5, 2.0, -0.5, 0.0]]    ← 样本 1 的 5 个类别分数

targets = [0, 2]
           ↑  ↑
        样本0 样本1 的正确类别
```

对齐关系：

```text
样本 0：logits[0] 与 targets[0]=0 比 → 取 logits[0, 0]
样本 1：logits[1] 与 targets[1]=2 比 → 取 logits[1, 2]
```

### 2.2 用代码取“正确类别的分数”

最直观的写法：

```python
# probs: (N, V)
correct_probs = probs[torch.arange(N), targets]   # (N,)
```

为什么这样写？

- `torch.arange(N)` 是 `[0, 1, ..., N-1]`，表示“第几个样本”；
- `targets` 是 `[y0, y1, ..., yN-1]`，表示“该样本的正确类别”；
- 两个索引一起用，取到 `probs[0, y0]`、`probs[1, y1]`……

> 这就是“逐样本、逐类别”的对齐：第 $i$ 行只和第 $i$ 个标签比，不能跨行。

### 2.3 更通用的写法：`gather`

如果 `inputs` 不是二维，而是 `(B,T,V)`，可以先展平：

```python
logits = logits.reshape(-1, V)     # (B*T, V)
targets = targets.reshape(-1)      # (B*T,)
```

或者直接用 `gather`，保留任意前导维度：

```python
# log_probs: (..., V), targets: (...)
correct_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
```

形状变化：

```text
log_probs:            (..., V)
targets.unsqueeze(-1):(..., 1)
gather(-1, ...):      (..., 1)
squeeze(-1):          (...)
```

> `gather(-1, index)` 的意思是：在最后一维上，按 `index` 指定的位置取值。`unsqueeze(-1)` 是为了让索引和 `log_probs` 的维度对齐。

### 2.4 语言模型里的对齐：还要错开一位

语言模型训练时，位置 $t$ 的 logits 预测的是**下一个 token** $t+1$：

```text
输入:  [t0] [t1] [t2] [t3]
标签:      [t1] [t2] [t3] [t4]
```

所以：

```python
logits = model(ids)                  # (B, T, V)
loss = cross_entropy(
    logits[:, :-1],                  # 前 T-1 个位置的预测
    ids[:, 1:],                      # 后 T-1 个位置的真实 token
)
```

如果你先展平再错位，或者只错位一边，就会发生“张冠李戴”：

```python
# 错误：logits 错位了，但 labels 没有错位
cross_entropy(logits[:, :-1].reshape(-1, V), ids.reshape(-1))
```

这样每个位置的预测都在和“自己”比，而不是和“下一个 token”比，loss 可能仍然是个有限数，但训练信号完全错了。

### 2.5 对齐错误的几种常见形态

| 写法 | 结果 | 问题 |
| --- | --- | --- |
| `probs[targets]` | 取“第 targets[i] 个样本”的整行 | 把类别下标当成了样本下标 |
| `probs[:, targets]` | 得到 `(N,N)` 的矩阵 | 每个样本都取了所有标签列，形状错 |
| `probs[range(N), targets]`（2D） | 正确 | 但只适用于二维、CPU 索引 |
| `gather(-1, targets.unsqueeze(-1))` | 正确 | 支持任意前导维度 |
| logits 展平、targets 没展平 | 形状不匹配 | 无法逐样本对齐 |
| logits 错位、targets 没错位 | 形状可能对，语义错 | 预测和标签不是同一个位置 |

> **对齐方式决定了损失计算是否正确**：形状对不代表语义对。一定要问自己：“这个 logits 向量对应的是哪个标签？”

---

## 3. 概率定义与工程公式

### 3.1 概率定义：先 softmax，再取正确类

模型输出 logits $z\in\mathbb{R}^{V}$。softmax 把分数变成概率：

$$
p_j=\frac{e^{z_j}}{\sum_{a=1}^{V}e^{z_a}}
$$

正确类别的概率：

$$
p_y=\frac{e^{z_y}}{\sum_{a=1}^{V}e^{z_a}}
$$

交叉熵：

$$
\mathcal{L}=-\log p_y
=-\log\frac{e^{z_y}}{\sum_{a=1}^{V}e^{z_a}}
$$

这是最直观的定义：

> **先算概率，再看正确答案的概率，最后取负对数。**

### 3.2 工程公式：把 log 和 exp 抵消

对上面的式子做变形：

$$
\begin{aligned}
\mathcal{L}
&=-\log\frac{e^{z_y}}{\sum_a e^{z_a}}\\
&=-\left(\log e^{z_y}-\log\sum_a e^{z_a}\right)\\
&=\log\sum_a e^{z_a}-z_y
\end{aligned}
$$

所以：

$$
\boxed{
\mathcal{L}=\log\sum_{a=1}^{V}e^{z_a}-z_y
}
$$

这就是工程实现里的标准公式。

它和“先 softmax 再取负对数”完全等价，但有一个巨大的好处：

> **不需要真的算出概率 $p$，直接对 logits 做 log-sum-exp，再做减法。**

### 3.3 批量平均

对 $N$ 个样本：

$$
\mathcal{L}
=\frac{1}{N}\sum_{i=1}^{N}
\left(
\log\sum_{a=1}^{V}e^{z_{i,a}}-z_{i,y_i}
\right)
$$

其中 $z_{i,a}$ 是第 $i$ 个样本第 $a$ 类的 logit，$y_i$ 是第 $i$ 个样本的正确类别。

### 3.4 两种公式的对照

| | 概率定义 | 工程公式 |
| --- | --- | --- |
| 写法 | $-\log p_y$ | $\log\sum_a e^{z_a}-z_y$ |
| 是否显式算 $p$ | 是 | 否 |
| 主要风险 | $p_y$ 下溢为 0，`log(0)` | $e^{z_a}$ 上溢，需减最大值 |
| 是否等价 | 是 | 是 |
| 推荐用途 | 理解概念 | 实际实现 |

### 3.5 梯度长什么样（进阶，可跳过）

对单个样本，交叉熵对 logits 的梯度是：

$$
\frac{\partial \mathcal{L}}{\partial z_j}=p_j-\mathbb{1}[j=y]
$$

也就是说：

- 正确类的梯度是 $p_y-1$（通常是负数，想把它的 logit 推高）；
- 其他类的梯度是 $p_j$（通常是正数，想把它们的 logit 压低）。

这个梯度形式非常简洁，也是 softmax + 交叉熵组合好用的原因之一。

> 对新手来说，记住“损失对 logits 的梯度 = 预测概率 - one-hot 标签”就够了。

---

## 4. 数值稳定性：Log-Sum-Exp

### 4.1 直接计算会出什么问题

#### 问题 1：`exp` 上溢

float32 的 `exp` 在输入大约超过 88 时就会溢出：

```python
torch.exp(torch.tensor(1000.0))   # inf
```

如果先算 $\sum e^{z_a}$，而某个 logit 是 1000，就会得到 `inf`；再算 `inf / inf`，就会得到 `nan`。

#### 问题 2：概率下溢

softmax 的分母很大时，正确类别的概率可能小到 float32 无法表示：

```python
z = torch.tensor([-1000.0, 0.0, 0.0])
p = torch.softmax(z, dim=-1)
# p[0] = 0.0（真实值约 0，但 float32 下是 0）
```

然后：

```python
torch.log(p[0])   # -inf
```

损失变成 `inf`，梯度也会变成 `nan`。

> 注意：即使 softmax 本身做了“减最大值”，`p_y` 仍可能下溢为 0；问题出在“先算 p，再取 log”。

#### 问题 3：`log(0)`

$$
-\log 0=+\infty
$$

真实数学里 $p_y$ 是一个很小的正数，损失应该是一个很大的有限值；但浮点数把它舍入成 0，就丢了信息。

### 4.2 Log-Sum-Exp 技巧

我们要算：

$$
\log\sum_a e^{z_a}
$$

直接算会溢出。做法是：先减去最大值 $m=\max_a z_a$：

$$
\log\sum_a e^{z_a}
=m+\log\sum_a e^{z_a-m}
$$

为什么安全？

- $z_a-m\le0$，所以 $e^{z_a-m}\le1$，不会上溢；
- 至少有一个 $a$ 使 $z_a=m$，所以那一项是 $e^0=1$；
- 因此 $\sum_a e^{z_a-m}\ge1$，$\log$ 的结果是有限的；
- 其他项的 $e^{z_a-m}$ 可能下溢成 0，但它们本来就非常小，对结果影响可以忽略。

于是稳定的损失：

$$
\boxed{
\mathcal{L}_i
=m_i+\log\sum_{a=1}^{V}e^{z_{i,a}-m_i}-z_{i,y_i}
}
$$

其中 $m_i=\max_a z_{i,a}$。

### 4.3 为什么这能救回“极端错误”的样本

看一个例子：

```text
z = [-1000, 0, 0], y = 0
```

真实数学：

$$
p_0=\frac{e^{-1000}}{e^{-1000}+1+1}
$$

这是一个极小的正数（约 $10^{-435}$ 量级），损失约 $1000.693$（有限，但很大）。float32 直接算：

- softmax 后 $p_0=0$；
- `log(0) = -inf`；
- 损失 `inf`。

Log-Sum-Exp：

- $m=0$；
- $z-m=[-1000,0,0]$；
- $e^{z-m}=[0,1,1]$；
- $\sum=2$；
- $\log 2=0.6931$；
- 损失 $=0+0.6931-(-1000)=1000.6931$。

有限、正确，而且没有算 $p_0$。

### 4.4 标准实现

```python
def cross_entropy(logits, targets):
    # logits: (..., V)，targets: (...)
    logits = logits.float()                         # 低精度先升 float32

    # 1) 每个样本减自己的最大值，防止 exp 上溢
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m                            # (..., V)，每行最大值是 0

    # 2) log-sum-exp：log(sum(exp(shifted)))
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))

    # 3) log 概率 = shifted - log_sum_exp
    log_probs = shifted - log_sum_exp               # (..., V)

    # 4) 取正确类别的 log 概率
    correct_log_prob = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # 5) 负号 + 批量平均
    return -correct_log_prob.mean()
```

也可以用 PyTorch 的 `torch.logsumexp`：

```python
log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
```

它内部同样做了“减最大值”，只是把细节封装了。

### 4.5 关于 `float32`

- 交叉熵里要做 `exp`、`log`，建议用 `float32`；
- 如果 logits 是 `float16`/`bfloat16`，先 `.float()`；
- 作业测试用 float32，`atol=1e-4`；
- 不要用半精度跑 `test_cross_entropy`，误差可能超标。

### 4.6 一个对照实验

```python
import torch

z = torch.tensor([[-1000.0, 0.0, 0.0]])
y = torch.tensor([0])

# 直接 softmax + log
p = torch.softmax(z, dim=-1)
print(-torch.log(p[0, 0]))   # inf

# Log-Sum-Exp
m = z.max(dim=-1, keepdim=True).values
lse = m + torch.log(torch.exp(z - m).sum(dim=-1, keepdim=True))
print((lse - z.gather(-1, y.unsqueeze(-1))).mean())   # 1000.693...
```

这个实验能直观看到：**同一个数学量，换一种算法，结果从 `inf` 变成有限值。**

---

## 5. 小示例：一步一步算

### 5.1 例 1：预测正确且比较自信

```text
logits = [2.0, 0.5, 0.1]
target = 0
```

**第一步：减最大值**

$$
m=2.0,\qquad z-m=[0,-1.5,-1.9]
$$

**第二步：exp**

$$
e^{z-m}=[1,\ 0.2231,\ 0.1496]
$$

**第三步：求和**

$$
S=1+0.2231+0.1496=1.3727
$$

**第四步：log-sum-exp**

$$
\log S=\log1.3727=0.3168
$$

**第五步：减正确类 logit**

$$
\mathcal{L}=2.0+0.3168-2.0=0.3168
$$

解释：模型给正确答案 $p_0\approx0.7285$，损失约 $0.3168$，不算大。

### 5.2 例 2：预测错误且比较自信

```text
logits = [0.1, 0.5, 2.0]
target = 0
```

**第一步：减最大值**

$$
m=2.0,\qquad z-m=[-1.9,-1.5,0]
$$

**第二步：exp**

$$
e^{z-m}=[0.1496,\ 0.2231,\ 1]
$$

**第三步：求和**

$$
S=1.3727
$$

**第四步：log-sum-exp**

$$
\log S=0.3168
$$

**第五步：减正确类 logit**

$$
\mathcal{L}=2.0+0.3168-0.1=2.2168
$$

解释：模型给正确答案只有 $p_0\approx0.1090$，损失约 $2.2168$，明显比例 1 大。

### 5.3 例 3：完全均匀

```text
logits = [0, 0, 0]
target = 1
```

$$
m=0,\quad e^{z-m}=[1,1,1],\quad S=3
$$

$$
\mathcal{L}=0+\log3-0=1.0986
$$

这就是“在 3 个类别里随机猜”的损失：

$$
\log V=\log3=1.0986
$$

### 5.4 例 4：极端错误

```text
logits = [-1000, 0, 0]
target = 0
```

$$
m=0,\quad z-m=[-1000,0,0]
$$

$$
e^{z-m}=[0,1,1],\quad S=2,\quad \log S=0.6931
$$

$$
\mathcal{L}=0+0.6931-(-1000)=1000.6931
$$

- 直接 `softmax` 后 $p_0=0$，`-log(0)=inf`；
- Log-Sum-Exp 给出有限值 $1000.6931$，这才是真实数学结果。

### 5.5 例 5：所有 logits 都很大

```text
logits = [1000, 1000, 1000]
target = 0
```

$$
m=1000,\quad z-m=[0,0,0],\quad S=3
$$

$$
\mathcal{L}=1000+\log3-1000=\log3=1.0986
$$

这说明：**所有 logits 同时加同一个常数，交叉熵不变**。减最大值正是利用了这个性质。

### 5.6 例 6：批量平均

两个样本：

```text
样本 0: logits=[2.0,0.5,0.1], target=0 → loss=0.3168
样本 1: logits=[0.1,0.5,2.0], target=0 → loss=2.2168
```

平均：

$$
\mathcal{L}=\frac{0.3168+2.2168}{2}=1.2668
$$

### 5.7 汇总表

| 例子 | logits | target | $p_y$ | 直接 `-log(p_y)` | 稳定公式 |
| --- | --- | ---: | ---: | ---: | ---: |
| 正确且自信 | [2.0, 0.5, 0.1] | 0 | 0.7285 | 0.3168 | 0.3168 |
| 错误且自信 | [0.1, 0.5, 2.0] | 0 | 0.1090 | 2.2168 | 2.2168 |
| 均匀 | [0, 0, 0] | 1 | 0.3333 | 1.0986 | 1.0986 |
| 极端错误 | [-1000, 0, 0] | 0 | 0（下溢） | **inf** | **1000.6932** |
| 全很大 | [1000, 1000, 1000] | 0 | 0.3333 | 1.0986 | 1.0986 |
| 正确极端 | [1000, 0, 0] | 0 | 1.0000 | 0.0000 | 0.0000 |

> `p_y` 列是数学上的概率；float32 下“极端错误”的 $p_y$ 会舍入成 0，所以直接算法得到 `inf`。

### 5.8 困惑度（Perplexity）

训练时看 loss，评估时常用困惑度：

$$
\operatorname{PPL}=e^{\mathcal{L}}
$$

- loss $=0.3168$ → PPL $\approx1.37$；
- loss $=2.2168$ → PPL $\approx9.18$；
- loss $=\log3=1.0986$ → PPL $=3$。

直觉：困惑度是“模型平均在多少个候选之间犹豫”。PPL 越小越好。

### 5.9 完整可运行示例

把下面的代码保存为 `cross_entropy_demo.py`，运行 `python cross_entropy_demo.py`：

```python
"""交叉熵损失：直接实现 vs 稳定实现。"""
import torch
import torch.nn.functional as F


def softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def cross_entropy_naive(logits, targets):
    """按定义算：先 softmax，再取正确类概率，最后取负对数。"""
    probs = softmax(logits, dim=-1)
    correct = probs[torch.arange(logits.shape[0]), targets]
    return -torch.log(correct).mean()


def cross_entropy(logits, targets):
    """稳定实现：先减最大值，再用 log-sum-exp，避免溢出/下溢。"""
    logits = logits.float()
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    log_probs = shifted - log_sum_exp
    correct_log_prob = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -correct_log_prob.mean()


def main():
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
    targets = torch.tensor([0, 0])

    print("batch naive :", round(cross_entropy_naive(logits, targets).item(), 6))
    print("batch stable:", round(cross_entropy(logits, targets).item(), 6))
    print("torch ref   :", round(F.cross_entropy(logits, targets).item(), 6))

    print("\n单样本例子：")
    cases = [
        ([2.0, 0.5, 0.1], 0),
        ([0.1, 0.5, 2.0], 0),
        ([0.0, 0.0, 0.0], 1),
        ([1000.0, 0.0, 0.0], 0),
        ([1000.0, 1000.0, 1000.0], 0),
    ]
    for z, y in cases:
        z_t = torch.tensor([z])
        y_t = torch.tensor([y])
        loss = cross_entropy(z_t, y_t).item()
        if abs(loss) < 1e-12:
            loss = 0.0
        print(f"  logits={z}, target={y} -> loss={loss:.6f}")

    extreme = torch.tensor([[-1000.0, 0.0, 0.0]])
    target = torch.tensor([0])
    print("\n极端错误预测：")
    print("  naive :", cross_entropy_naive(extreme, target).item())
    print("  stable:", round(cross_entropy(extreme, target).item(), 6))


if __name__ == "__main__":
    main()
```

预期输出：

```text
batch naive : 1.266779

batch stable: 1.266779
torch ref   : 1.266779

单样本例子：
  logits=[2.0, 0.5, 0.1], target=0 -> loss=0.316779
  logits=[0.1, 0.5, 2.0], target=0 -> loss=2.216779
  logits=[0.0, 0.0, 0.0], target=1 -> loss=1.098612
  logits=[1000.0, 0.0, 0.0], target=0 -> loss=0.000000
  logits=[1000.0, 1000.0, 1000.0], target=0 -> loss=1.098612

极端错误预测：
  naive : inf
  stable: 1000.693176
```

> 这里用 `F.cross_entropy` 只是做数值对照；作业要求自己实现，不能直接调用它。

---

## 6. 与 CS336 Assignment 1 的关联

### 6.1 作业要求（讲义 §4.1）

作业要求实现交叉熵损失：

- 输入 logits 和 targets；
- 计算 $-\log\operatorname{softmax}(o_i)[x_{i+1}]$；
- **减去最大值**，保证数值稳定；
- **尽可能抵消 log 和 exp**；
- 支持任意前导 batch 维；
- **返回 batch 上的平均**；
- 实现 `adapters.run_cross_entropy`；
- 运行 `uv run pytest -k test_cross_entropy`；
- 这一题 1 分。

### 6.2 接口与形状

```python
def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    ...
```

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `inputs` | `(N, V)` | 第 $i$ 行是第 $i$ 个样本的 $V$ 个 logits |
| `targets` | `(N,)` | 每个样本的正确类别下标 |
| 返回值 | 标量 | 所有样本的平均交叉熵 |

注意：

- `inputs` 是 **logits**，不是概率；
- `targets` 是整数下标，不是 one-hot；
- 返回 **平均**，不是求和；
- 支持 batch-like 维度在前的形式（测试会先展平）。

### 6.3 测试细节

`test_cross_entropy`：

- 输入 `inputs` 形状 `(2,4,5)`，展平成 `(8,5)`；
- `targets` 形状 `(2,4)`，展平成 `(8,)`；
- 参考值是 `F.cross_entropy(inputs.view(-1,5), targets.view(-1))`；
- 容差 `atol=1e-4`；
- 另外用 `large_inputs = 1000.0 * inputs` 再测一次，专门检查数值稳定性。

这意味着：

- 你的实现不能只支持二维；要能处理展平后的 `(N,V)`；
- 不能直接 `log(softmax(...))`，否则大 logits 会失败；
- 不能返回 sum，否则和参考值差 $N$ 倍；
- 不能用 `F.cross_entropy` 作弊（作业要求自己实现）。

### 6.4 仓库里的 `Cross_Entropy_Loss.py`

当前实现（已按下面几点改过）：

```python
import torch


class CrossEntropyLoss:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.vocab_size = inputs.shape[1]
        self.batch_size = inputs.shape[0]

    def forward(self):
        log_probs = torch.log_softmax(self.inputs, 1)          # 对数域，避开 log(0)
        log_p = log_probs[range(self.batch_size), self.targets]  # 只取正确类别
        return -log_p.mean()                                   # 批平均
```

三段分别对应：

1. `log_softmax` —— 直接给出 $\log\operatorname{softmax}(z)$，内部用 log-sum-exp，大 logits 也不会下溢成 `-inf`；
2. 高级索引 —— 从 `(B, V)` 里挑出每个样本正确类别的那一列，降到 `(B,)`；
3. `mean` —— 返回批平均，和 `F.cross_entropy` 的默认 `reduction="mean"` 对齐。

历史版本的三个问题，现在的状态：

| 问题 | 旧写法 | 现在 |
| --- | --- | --- |
| 导入路径 | `from hw3.softmax import softmax` | 已删掉，改为直接用 `torch.log_softmax` |
| 数值稳定性 | `log(softmax(...))` 会下溢 | 已改成对数域，`1000×logits` 也不会 `inf` |
| 求和 vs 平均 | `-torch.sum(...)` | 已改成 `.mean()` |

还没处理的两点（讲义里也不要求）：

| 问题 | 说明 |
| --- | --- |
| 维度假设 | `inputs.shape[1]` 仍假设二维，喂 `(B, T, V)` 要自己先 `reshape(-1, V)` |
| `ignore_index` | 没有跳过 padding 标签的机制，需要外面 mask 或提前筛行 |

### 6.5 在语言模型训练中的用法

```python
logits = model(ids)                       # (B, T, V)
loss = cross_entropy(
    logits[:, :-1],                       # 前 T-1 个位置的预测
    ids[:, 1:],                           # 后 T-1 个位置的真实 token
)
loss.backward()
optimizer.step()
```

对应关系：

```text
logits[b, t]  → 预测 ids[b, t+1]
```

这也是为什么标签要错开一位。

### 6.6 与困惑度的关系

训练日志里通常同时报告：

```python
loss = cross_entropy(logits[:, :-1], ids[:, 1:])
perplexity = torch.exp(loss)
```

- loss 是平均交叉熵；
- perplexity 是 $e^{\text{loss}}$；
- 两者单调对应，报告哪个都可以，但困惑度更直观。

### 6.7 常见错误对照表

| 现象 | 可能原因 | 修正 |
| --- | --- | --- |
| loss 是参考值的 $N$ 倍 | 用了 `sum` 而不是 `mean` | 改成 `.mean()` |
| loss 是 `inf` / `nan` | 直接 `log(softmax(...))` 下溢 | 用 Log-Sum-Exp |
| 大 logits 时出错 | 没减最大值，`exp` 溢出 | 先减每行最大值 |
| 形状不匹配 | logits 展平了，targets 没展平 | 两边一起展平 |
| 取错类别 | 把 `targets` 当成了行索引 | 用 `gather` 或行/列配对索引 |
| 训练 loss 不下降 | 标签没错位 | `logits[:, :-1]` 对 `ids[:, 1:]` |
| 测试报 `ModuleNotFoundError` | 导入了 `hw3.softmax` | 用 `cs336_basics.softmax` |
| 测试误差略大 | 用了半精度 | 先升 `float32` |
| 返回了概率 | 把 softmax 结果当损失 | 返回标量 loss |
| targets 越界 | 标签不在 `[0,V-1]` | 检查数据/词表 |

### 6.8 实现检查清单

- [ ] `inputs` 最后一维是 $V$；
- [ ] `targets` 是整数，形状和 `inputs` 的前导维一致；
- [ ] 每行减最大值；
- [ ] 用 `log(sum(exp(shifted)))`，不要 `log(softmax)`；
- [ ] 用 `gather` 取正确类别；
- [ ] 返回 `.mean()`；
- [ ] 支持 `(N,V)` 和 `(B,T,V)`；
- [ ] 跑过 `pytest -k test_cross_entropy`；
- [ ] 不用 `F.cross_entropy` 作为实现（只用于对照）。

---

## 7. 常见误区 FAQ

**Q1：交叉熵一定小于 1 吗？**
不一定。它最小是 0，没有上界。给正确答案 0.1 的概率，损失就是 2.303。

**Q2：交叉熵能为负数吗？**
不能。概率在 $(0,1]$，$-\log p\ge0$。

**Q3：为什么不用 MSE？**
MSE 对分类概率的梯度不够“锐利”，而且和“最大似然”不一致；交叉熵是分类/语言模型的标准损失。

**Q4：为什么只取正确类别的概率？**
因为真实分布是 one-hot，求和里只有正确类那一项非零。

**Q5：为什么减最大值不改变结果？**
softmax 对“所有 logits 加同一个常数”不变；减最大值只是选了一个方便计算的常数。

**Q6：Log-Sum-Exp 会不会改变数学结果？**
不会。它只是把 $\log\sum e^{z}$ 换了一种数值稳定的算法。

**Q7：`p_y` 下溢成 0 怎么办？**
不要先算 $p$；用 Log-Sum-Exp 直接算 $\log p_y$。

**Q8：为什么要平均而不是求和？**
让 loss 与 batch 大小无关，学习率更好设；作业也明确要求平均。

**Q9：`targets` 可以是 one-hot 吗？**
数学上可以，但作业接口用的是类别下标；one-hot 需要额外转换。

**Q10：语言模型的 loss 为什么是每个 token 的平均？**
每个位置都是一个“下一个 token 分类”问题；一次前向产生 $T$ 个预测，通常对所有位置取平均。

**Q11：`F.cross_entropy` 能用吗？**
作业要求自己实现；它只能用来对照测试，不要直接调用。

**Q12：交叉熵和困惑度是什么关系？**
$\text{PPL}=e^{\text{loss}}$；loss 越小，困惑度越小。

---

## 8. 小结与自测

### 8.1 一页速查

```text
定义:   L = -log p_y
概率:   p_y = exp(z_y) / sum_a exp(z_a)
工程:   L = log(sum_a exp(z_a)) - z_y
稳定:   m = max_a z_a
        L = m + log(sum_a exp(z_a - m)) - z_y
批量:   L = mean over samples
对齐:   logits[i] 配 targets[i]
LM:     logits[:, :-1] 配 ids[:, 1:]
```

### 8.2 自测题

1. 交叉熵比较的是哪两个分布？为什么真实分布是 one-hot？
2. 写出交叉熵的概率定义和工程公式，并证明它们等价。
3. 为什么 `-log(softmax(z)[y])` 在极端 logits 下会变成 `inf`？
4. Log-Sum-Exp 为什么要减最大值？减了之后为什么结果不变？
5. 给定 `logits=[2,0.5,0.1]`、`target=0`，手算损失。
6. 给定 `logits=[-1000,0,0]`、`target=0`，直接算法和稳定算法分别得到什么？
7. 为什么语言模型的标签要错开一位？
8. `inputs` 是 `(B,T,V)`、`targets` 是 `(B,T)` 时，怎么对齐？
9. 为什么返回平均而不是求和？
10. 交叉熵和困惑度怎么换算？
11. 你给的 `Cross_Entropy_Loss.py` 有哪三个主要问题？
12. CS336 的测试为什么要用 `1000 * inputs`？

> 参考答案在 §1–§6 对应小节；先自己写一遍，再回去核对。

### 8.3 动手清单

- [ ] 手算 §5 的三个例子；
- [ ] 跑通 §4.6 的对照实验，观察 `inf` 和有限值的区别；
- [ ] 用 `(N,V)` 和 `(B,T,V)` 两种形状测试稳定实现；
- [ ] 把 `Cross_Entropy_Loss.py` 改成平均 + Log-Sum-Exp；
- [ ] 跑过 `pytest -k test_cross_entropy`；
- [ ] 在完整 LM 训练里打印 loss 和 perplexity。

---

## 9. 延伸阅读

**课程与讲义**

- CS336 课程主页：<https://cs336.stanford.edu>
- 作业讲义 `cs336_assignment1_basics.pdf` §4.1 —— 交叉熵、困惑度与测试要求。

**论文与背景**

- Goodfellow, Bengio & Courville, *Deep Learning*, 2016 —— 交叉熵与最大似然。
- Bishop, *Pattern Recognition and Machine Learning*, 2006 —— 概率分布与负对数似然。
- PyTorch 文档：`torch.logsumexp`、`torch.gather`、`torch.nn.functional.cross_entropy`（仅作对照）。

**同目录文档**

- `完整语言模型架构的组装.md`：logits 从哪来、训练 loss 怎么接。
- `Transformer_Block.md`：语言模型的 Block 结构。
- `RMSNorm.md`、`SwiGLU.md`、`Attention.md`：模型内部组件。
- `Embedding_Linear.md`：Embedding 与线性层的基础。

> **最后一句**：交叉熵的公式只有一行，但它同时包含了“概率归一化”“正确类对齐”“数值稳定”三件事。把这三件事拆开看，它就不再神秘。
