# Embedding 层与 Linear 层：Transformer 的"查表"与"换维"

> 这份文档讲清楚 Transformer 里最基础、也最容易被误解的两层：**Embedding（嵌入）** 把离散的 token 编号翻译成向量，**Linear（线性层）** 把向量在各个维度之间来回投影。
>
> 面向零基础读者：每个概念都配可运行的 PyTorch 示例（已在本地实跑验证，输出与文中一致），并解释"为什么这么写"。建议边读边把示例敲一遍。

## 目录

- [0. 为什么从这两层开始](#0-为什么从这两层开始)
- [1. Embedding 层：把离散 ID 变成稠密向量](#1-embedding-层把离散-id-变成稠密向量)
- [2. Linear 层：线性变换与维度投影](#2-linear-层线性变换与维度投影)
- [3. 两层在 Transformer 里的位置与协作](#3-两层在-transformer-里的位置与协作)
- [4. 自己实现这两层时要想清楚的事](#4-自己实现这两层时要想清楚的事)
- [5. 手写 Embedding 与 Linear：完整可运行示例](#5-手写-embedding-与-linear完整可运行示例)
- [6. 小结](#6-小结)
- [7. 常见问题 FAQ](#7-常见问题-faq)
- [8. 动手练习](#8-动手练习)

---

## 0. 为什么从这两层开始

Transformer 看起来结构复杂，但如果把"计算"这个动作剥掉，剩下的骨架其实只有两类操作：

| 操作 | 干的事 | 典型代表 |
| --- | --- | --- |
| **查表** | 用离散编号取出预先存好的一行向量 | Embedding |
| **换维** | 拿一个向量做线性组合，得到另一个维度的向量 | Linear |

Attention、FFN、LayerNorm 这些"高级"结构，其实都建立在这两个动作之上。所以先把这两层弄懂，后面的内容会轻松很多。

### 0.1 术语速查

| 术语 | 白话解释 |
| --- | --- |
| 离散 | 取值是有限个、互不相连的编号，比如"第 17 号词" |
| 稠密 | 每个维度都填满浮点数，信息分布均匀，不稀疏 |
| 词表（vocab） | 所有 token 的清单，大小记作 $V$ |
| 隐藏维度（d_model） | 模型内部用多少个数描述一个 token，记作 $D$ |
| 投影 | 把向量从一个维度空间映射到另一个维度空间 |
| 参数 | 需要训练的权重（不是超参数，会通过梯度更新而改变） |
| 超参数 | 训练前就定好的设置，比如层数、维度、学习率 |

### 0.2 环境准备

```python
# 建议用 uv 管理环境（和 CS336 课程仓库一致），在仓库根目录执行：
#   uv run python your_script.py
#
# 如果本机没有 GPU，把示例里的 .cuda() 换成 .cpu() 即可，
# 纯 CPU 也能把本课所有示例跑通。

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)  # 固定随机种子：让每次运行的数字一样，方便对照输出
```

---

## 1. Embedding 层：把离散 ID 变成稠密向量

### 1.1 先看清问题：token ID 到底是什么

经过 BPE 分词后，一段文本变成了一串整数：

```text
"The cat sat"  →  [464, 2068, 333]
```

这三个数字本身**不含任何语义**。它们只是"清单上的第几行"，就像图书馆的书架编号。

如果直接把这些数字丢给模型，会出三个问题：

1. **数字的大小会误导模型**：`333 > 464`，但 "sat" 和 "the" 谁也不比谁重要。模型会误以为编号大的词更"重"。
2. **缺少相似性**：编号 464 和 465 可能分别是 "the" 和 " a"，毫无关联；但编号相差不远的两本书可能讲同一个主题。编号完全表达不了这种关系。
3. **维度爆炸**：用 one-hot（只有一位是 1，其余全是 0）表示，词表多大就要多宽，稀疏且浪费。

**Embedding 的任务就是解决这三点：给每个 token 配一个语义化的向量，并让语义相近的 token 在向量空间里靠得更近。**

### 1.2 Embedding 到底做了什么：一次查表

Embedding 层内部就是**一张表**：

- 行数 = 词表大小 $V$
- 列数 = 隐藏维度 $D$
- 查表方式：拿 token ID 当行号，把那一行取出来

```text
              0      1      2      3     ...    D-1
token 0   → [ 0.31, -1.02,  0.88,  0.05, ... ]   ← 第 0 行
token 1   → [-0.44,  0.15,  1.30, -0.77, ... ]   ← 第 1 行
token 2   → [ 0.77,  0.02, -0.61,  0.33, ... ]   ← 第 2 行
  ...
token 333 → [ ... ]                                ← "sat" 的向量
```

这张表里的数字**不是人填的，是训练学出来的**：在语料里经常一起出现的词，向量会慢慢靠拢。所以它被称为"嵌入"——把词"嵌"进了一个连续的几何空间里。

### 1.3 形状与参数量

| 量 | 形状 | 说明 |
| --- | --- | --- |
| 嵌入表 `weight` | `(V, D)` | $V$ 个 token，每个 $D$ 维向量 |
| 输入 `input_ids` | `(B, T)` | $B$ 条序列，每条 $T$ 个 token |
| 输出 | `(B, T, D)` | 每个 token 位置都变成一个 $D$ 维向量 |

参数量 $= V \times D$。以 $V=10000$、$D=512$ 为例，是 512 万个参数（fp32 约 20 MB）。

### 1.4 示例 1：查表到底查出了什么

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

vocab_size = 5          # 小词表：只有 5 个 token，方便肉眼核对
d_model = 4             # 每个 token 用 4 个数字描述

# 造一张"嵌入表"。这里随机初始化只为演示，真实模型里这张表是训练出来的
embedding = nn.Embedding(vocab_size, d_model)

# 输入：2 句话，每句 3 个 token（数字就是 token ID）
input_ids = torch.tensor([
    [1, 3, 0],     # 第 0 句
    [2, 2, 4],     # 第 1 句
])

vectors = embedding(input_ids)   # 一次查表，把所有 ID 换成向量

print("输出形状:", tuple(vectors.shape))           # (2, 3, 4)：2 句 × 3 个 token × 4 维
print("嵌入表形状:", tuple(embedding.weight.shape))  # (5, 4)：5 行，每行 4 维

# 验证"查表"这件事：第 0 句的第 2 个 token（ID=3）应该等于嵌入表的第 3 行
print("是否等于第 3 行:", torch.allclose(vectors[0, 1], embedding.weight[3]))
```

实际输出：

```text
输出形状: (2, 3, 4)
嵌入表形状: (5, 4)
是否等于第 3 行: True
```

**要点**：Embedding 对每个位置是**独立**的——第 0 句的第 3 个 token 绝不会去看第 1 句。词与词之间此刻毫无交流（那是注意力以后的事）。

### 1.5 示例 2：Embedding 本质上是"one-hot × 权重矩阵"

这个等价关系非常值得知道——它能解释很多事。

```python
import torch
import torch.nn.functional as F

# one-hot：把每个 ID 变成"只有对应那位是 1"的稀疏向量
one_hot = F.one_hot(input_ids, num_classes=vocab_size).to(embedding.weight.dtype)
print("one_hot 形状:", tuple(one_hot.shape))   # (2, 3, 5)：每个位置多了一维词表大小

# 用矩阵乘手动实现 Embedding：sum_j one_hot[j] * W[j]，只有 ID 对应那项不为 0
manual = one_hot @ embedding.weight

print("手写结果与查表一致:", torch.allclose(manual, vectors))
```

实际输出：

```text
one_hot 形状: (2, 3, 5)
手写结果与查表一致: True
```

**为什么要知道这件事？**

- 数学上等价，但 one-hot 版本要造出 `(B, T, V)` 的大矩阵、还做一次 $V$ 维的矩阵乘，纯属浪费。真正的实现是**直接按行号取**（gather/index 操作），把复杂度从 $O(BTV)$ 降到 $O(BT)$。
- 它也解释了 Embedding 和 Linear 的亲缘关系：Embedding ≈ one-hot 版的 Linear。

### 1.6 示例 3：嵌入表是怎么被"训"出来的

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(5, 3)
ids = torch.tensor([[1, 1, 1]])     # 这句话里只出现了 ID=1

embedding(ids).sum().backward()     # 反向传播：算出每个参数的梯度

print("ID=1 这一行的梯度:", embedding.weight.grad[1].tolist())   # 有值：它被用到过
print("ID=2 这一行的梯度:", embedding.weight.grad[2].tolist())   # 全 0：这句里没出现过它
```

实际输出：

```text
ID=1 这一行的梯度: [3.0, 3.0, 3.0]
ID=2 这一行的梯度: [0.0, 0.0, 0.0]
```

**关键结论**：嵌入表的梯度是**稀疏**的——只有这一步真正用到的 token，其对应行才会被更新。这带来两个实际影响：

- 好消息：一个 batch 里常见的词被反复更新，学得快；罕见的词梯度稀少，学得慢。
- 代价：罕见词的嵌入可能长期得不到有效训练（缓解办法通常靠更好的初始化或词表设计，而不是改结构）。

### 1.7 两个常见变体

**(1) `padding_idx`：让填充位不参与训练**

```python
embedding = nn.Embedding(5, 3, padding_idx=0)   # 约定 ID=0 是 padding
```

指定后，第 0 行的梯度**恒为 0**，它永远保持初始化时的样子（通常初始化为 0）。做变长序列批处理时，这能防止"填充位置的噪声"污染训练。

**(2) 权重绑定：输入嵌入和输出投影共用一张表**

很多现代语言模型让输入 Embedding 和顶部的 LM head（输出 Linear）**共享同一份权重**，业内叫 weight tying。好处：

- 词表矩阵只存一份，省显存；
- 语义上更自洽："输入时怎么表示这个词，输出时也倾向于用同一套坐标"。

> CS336 讲义把 Embedding 和 Linear 写成两个独立模块，是否共享属于实现选择，不影响对整体流程的理解。

---

## 2. Linear 层：线性变换与维度投影

### 2.1 公式与形状

Linear（也叫全连接层）做的事就一个公式：

$$
y = x W^\top + b
$$

- $x$：输入向量，维度 $d_{in}$
- $W$：可学习权重，形状 $(d_{out}, d_{in})$
- $b$：可学习偏置，形状 $(d_{out},)$
- $y$：输出向量，维度 $d_{out}$

一句话：**把输入的每个分量按权重重新加权求和，输出成一个新维度的向量。** 权重为正说明"这个输入维度对该输出有正向贡献"，为负说明是反向抑制。

常见的三种用法：

| 用途 | 形状变化 | 例子 |
| --- | --- | --- |
| **升维** | $d_{out} > d_{in}$ | FFN 里的扩张层 $D \to d_{ff}$ |
| **降维** | $d_{out} < d_{in}$ | 把隐藏向量压回词表大小 $D \to V$ |
| **等维变换** | $d_{out} = d_{in}$ | 注意力里的 $W_Q$、$W_K$、$W_V$、$W_O$ |

### 2.2 示例 4：手算一次 Linear

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# 输入 3 维、输出 2 维的全连接层
linear = nn.Linear(in_features=3, out_features=2, bias=True)

x = torch.tensor([[1.0, 2.0, 3.0]])   # 一条 3 维输入
y = linear(x)
print("输出:", y)

# 不调用 linear，手工按 y = xW^T + b 算一遍
W = linear.weight      # 形状 (2, 3)
b = linear.bias        # 形状 (2,)
hand = x @ W.T + b
print("手算:", hand)
print("两者一致:", torch.allclose(hand, y))
```

实际输出：

```text
输出: tensor([[ 1.5261, -1.0865]], grad_fn=<AddmmBackward0>)
手算: tensor([[ 1.5261, -1.0865]], grad_fn=<AddBackward0>)
两者一致: True
```

**注意 `W.T`**：PyTorch 里 `nn.Linear` 的权重存成 `(out_features, in_features)`，而数学上习惯写成 $y = Wx$。这个转置是新手最容易卡住的地方，记住"**PyTorch 用行优先，权重是转置存的**"即可。

### 2.3 示例 5：Linear 只作用于最后一维

这是 Linear 最容易被忽略、也最重要的行为规则。

```python
import torch
import torch.nn as nn

x = torch.randn(2, 5, 8)      # (batch=2, 序列长=5, 特征维=8)
linear = nn.Linear(8, 16)     # 8 维进，16 维出
y = linear(x)
print("输入形状:", tuple(x.shape))
print("输出形状:", tuple(y.shape))   # (2, 5, 16)：只有最后一维被换了
```

实际输出：

```text
输入形状: (2, 5, 8)
输出形状: (2, 5, 16)
```

**规则**：`nn.Linear` 只看**最后一维**。前面所有维（batch、序列位置）都会被自动当成"一批独立样本"并行处理。

推论（很重要）：

- 想让 Linear **逐位置**作用在序列上，什么都不用改——它天然就是逐位置的；
- 想让 Linear **跨位置**作用（比如混合不同时间步的信息），那不能靠 Linear，得用注意力或卷积。

### 2.4 bias 有没有，重要吗

- `bias=True`（默认）：输出额外加一个可学习的常数，相当于给每个输出维度一个"基准起点"。大多数场景下有帮助。
- `bias=False`：不加分量。少 $d_{out}$ 个参数，在**权重绑定**的场景里常用——因为此时输入 Embedding 和输出 Linear 是同一张表，如果再各加一份 bias，就等于"同一张表有两种偏移"，语义上有点别扭。

### 2.5 初始化与数值尺度

Linear 的权重通常用小随机数初始化（如正态分布，标准差约 $1/\sqrt{d_{in}}$ 或 $0.02$ 一类）。为什么不能初始化得太大？

- 权重太大 → 每层输出数值被放大 → 层数一多就**指数爆炸**，训练发散；
- 权重太小 → 输出趋近 0，信号传不下去，梯度也衰减。

尤其是**残差分支上的 Linear**，它们的输出会直接加回主干，初始化时通常要额外小一些，让残差路径在训练初期近似"什么都不改"，训练更稳。

---

## 3. 两层在 Transformer 里的位置与协作

### 3.1 全景图

```text
 token ID  (B, T)
      │
      ▼  ① Embedding（查表）              ← 本课主角 A
 隐藏向量 (B, T, D)
      │
      ▼  ② × N 层 Transformer Block
      │      ├─ RMSNorm
      │      ├─ Linear: W_Q / W_K / W_V   ← 本课主角 B
      │      ├─ 注意力 + RoPE
      │      ├─ Linear: W_O              ← 本课主角 B
      │      ├─ RMSNorm
      │      └─ Linear: W1 / W3 / W2（FFN）← 本课主角 B
      ▼
 最终 RMSNorm
      │
      ▼  ③ LM head = Linear(D → V)        ← 本课主角 B
 logits (B, T, V)
      │
      ▼  ④ softmax
 概率分布 (B, T, V)
```

**结论：整个 Transformer 里只有一个 Embedding，但到处都是 Linear。** 每个 Block 里至少 4 个 Linear（Q/K/V/O），FFN 再加 3 个。

### 3.2 底座：输入 Embedding 的职责

- **把离散变连续**：模型的第一个可学习参数，词表到向量空间的"入口"；
- **提供语义先验**：语义相近的词，向量距离近，这是后续注意力能工作的基础；
- **不负责任何交互**：它只看自己是谁，不看邻居。

### 3.3 顶端：LM head 的职责

- **把向量翻译回词表**：$D$ 维 → $V$ 维，每个维度对应一个候选词，最后 softmax 得到概率；
- **本质是"反向的 Embedding"**：一个把连续空间投影回离散词表的过程。

于是输入和输出形成了一个闭环：

```text
ID --Embedding--> 向量空间 --LM head--> 分数 --softmax--> 概率 --采样--> 新的 ID
```

### 3.4 中间：Block 里的 Linear 干什么

| Linear | 输入 → 输出 | 作用 |
| --- | --- | --- |
| $W_Q$ | $D \to D$ | 把向量变成"我想找什么"（Query） |
| $W_K$ | $D \to D$ | 把向量变成"我是什么"（Key） |
| $W_V$ | $D \to D$ | 把向量变成"我能提供什么内容"（Value） |
| $W_O$ | $D \to D$ | 把各个头的结果重新融合 |
| $W_1, W_3$ | $D \to d_{ff}$ | FFN 的两条支路：一条算门控，一条算内容 |
| $W_2$ | $d_{ff} \to D$ | 把扩张后的结果压回 $D$ 维，好加回残差主干 |

**协作关系一句话**：Embedding 负责"给每个位置一个身份"，Linear 负责"在这个身份基础上换一种问法/答法/说法"。前者是**逐 token 独立**的，后者是**逐位置**的；只有注意力才让不同位置之间发生交流。

### 3.5 示例 6：走一遍完整流程

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

vocab_size, d_model = 10, 8

embedding = nn.Embedding(vocab_size, d_model)   # ① 底座：查表
lm_head = nn.Linear(d_model, vocab_size)        # ③ 顶端：投影回词表

input_ids = torch.randint(0, vocab_size, (2, 4)) # 一批 2 句，每句 4 个 token

hidden = embedding(input_ids)                   # (2, 4) → (2, 4, 8)
logits = lm_head(hidden)                         # (2, 4, 8) → (2, 4, 10)
probs = F.softmax(logits, dim=-1)                # 每个位置一份概率分布

print("input_ids 形状:", tuple(input_ids.shape))
print("hidden 形状:", tuple(hidden.shape))
print("logits 形状:", tuple(logits.shape))
print("每行概率之和:", probs.sum(dim=-1))
```

实际输出：

```text
input_ids 形状: (2, 4)
hidden 形状: (2, 4, 8)
logits 形状: (2, 4, 10)
每行概率之和: tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]])
```

**注意 `lm_head` 为什么能直接吃 `(2, 4, 8)`**：因为 Linear 只看最后一维（见 2.3），它把 8 个"句子位置"当成 8 个独立样本各算一次，分数天然按位置对齐。

### 3.6 示例 7：参数量感受一下

```python
import torch.nn as nn

V, D = 10000, 512
e = nn.Embedding(V, D)
l = nn.Linear(D, V)

print("Embedding 参数数:", e.weight.numel())
print("Linear 参数数:", l.weight.numel() + l.bias.numel())
print("Embedding 显存(fp32):", e.weight.numel() * 4 / 1024 / 1024, "MB")
```

实际输出：

```text
Embedding 参数数: 5120000
Linear 参数数: 5130000
Embedding 显存(fp32): 19.53125 MB
```

可以看出：**词表矩阵是模型里最容易"撑爆"的部分之一**。词表 10 万、维度 4096 时，光这一张表就是 4 亿参数。词表大小、维度、权重绑定这三个决策会直接决定模型的显存账单。

---

## 4. 自己实现这两层时要想清楚的事

> CS336 Assignment 1 要求**不能直接调用** `torch.nn.Embedding` / `torch.nn.Linear`（可用的只有 `torch.nn.Parameter`、容器类和 `torch.optim.Optimizer` 基类），必须自己写。下面是**思路与检查清单**，帮你把事情想明白，而不是直接抄答案。

### 4.1 Embedding 的实现要点

1. **本质是"参数表 + 按 ID 取行"**：参数形状必须是 `(vocab_size, d_model)`；前向就是一次 gather/index，而不是 one-hot 矩阵乘（原因见 1.5）。
2. **要支持 batch 维**：输入 `(B, T)` 时，输出必须是 `(B, T, d_model)`。
3. **参数要能被优化器看到**：用 `torch.nn.Parameter` 包起来，才能出现在 `model.parameters()` 里。
4. **梯度必须是稀疏的**：只有本 batch 用到的行才该有非零梯度；如果你发现没用到的行也有梯度，八成是实现成了 one-hot 矩阵乘。
5. **可选能力**：`padding_idx` 对应的行梯度应恒为 0。

### 4.2 Linear 的实现要点

1. **形状**：权重 `(d_out, d_in)`，偏置 `(d_out,)`；前向等价于 $xW^\top + b$。
2. **只作用在最后一维**：`(..., d_in) → (..., d_out)`，前面的维度原样保留。
3. **别把转置搞混**：PyTorch 内置的 `nn.Linear` 权重是 `(out, in)`，跟数学上习惯的 $Wx$ 差一个转置；和内置层做数值对齐时要保证约定一致。
4. **初始化**：小随机数；残差分支上的输出投影通常还要更小一些。
5. **bias 可选**：权重绑定时通常设为 `False`。

### 4.3 怎么验证自己写对了

课程测试文件里的做法就是最好的自检思路——**拿内置层当标准答案**：

1. **数值对齐**：同一份随机输入，自己写的层和 `nn.Embedding` / `nn.Linear` 的输出逐元素比较（`torch.allclose`）；
2. **形状断言**：`(B, T)` → `(B, T, D)`、`(B, T, D)` → `(B, T, V)`，一处都不能错；
3. **梯度非零检查**：反传一次，确认被用到的行/参数确实拿到了梯度；
4. **padding 检查**：若支持 `padding_idx`，它的梯度应恒为 0；
5. **端到端小样本**：用 3～4 个 token 的极小序列跑通整条链路，确认形状一路对得上。

---

## 5. 手写 Embedding 与 Linear：完整可运行示例

前面几章用的是 PyTorch 内置的 `nn.Embedding` / `nn.Linear`。这一章给出**手写版本**：把两层的内部机制摊开给你看，每一行都写清楚"为什么这么写"。

> 说明：这是**教学示例**（类名叫 `MyEmbedding` / `MyLinear`），目的是理解原理。作业里要按讲义给出的接口（`run_embedding` / `run_linear`）自己实现，参数名和签名以讲义为准（见 5.8 节对照表）。

### 5.1 完整代码：导入与类定义

把下面整段存成 `embedding_linear_demo.py`，可以直接 `uv run python embedding_linear_demo.py` 跑起来。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)  # 固定随机种子，让每次运行的数字一致，方便对照输出


class MyEmbedding(nn.Module):
    """教学版 Embedding：把 token ID 查表换成一整行向量。

    与内置 nn.Embedding 的差别只有两处：
    1. 前向用 self.weight[token_ids] 显式表达"按行号取行"；
    2. 多了输入检查（维度、ID 越界），出错时给人话提示而不是难懂的底层报错。
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None,
                 device=None, dtype=None):
        """
        参数:
            vocab_size: 词表大小，也就是嵌入表的行数。
            d_model:    每个 token 用多少个数描述，也就是嵌入表的列数。
            padding_idx: 填充位置对应的 token ID；指定后该行恒为 0，且不参与梯度更新。
            device:     参数放在哪个设备上（'cuda' / 'cpu'），默认跟随全局默认设备。
            dtype:      参数的数据类型（torch.float32 等），默认跟随全局设置。
        """
        super().__init__()      # 父类初始化必须先做，之后挂参数才不会出问题

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        # nn.Parameter 把张量登记成"可训练参数"：优化器才找得到它，梯度也才会累积
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)   # 随机小值起步，语义靠训练长出来

        if padding_idx is not None:
            # no_grad 块里不做梯度记录：这里只是"把初始值摆正"，不算一次前向计算
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """输入 (B, T) 的 token ID，输出 (B, T, d_model) 的向量。"""
        # 检查一：只接受 (T,) 或 (B, T) 两种形状
        if token_ids.dim() > 2:
            raise ValueError(f"token_ids 最多两维 (batch, seq)，收到 {token_ids.dim()} 维")

        # 检查二：ID 不能越界。内置层遇到越界会直接崩，这里先把话说清楚
        if token_ids.numel() > 0:
            lo = int(token_ids.min().item())
            hi = int(token_ids.max().item())
            if lo < 0 or hi >= self.vocab_size:
                raise IndexError(f"token ID 越界：[{lo}, {hi}]，词表只有 {self.vocab_size} 个")

        # 核心就这一行：把 ID 当行号去取行，输出形状自动变成 token_ids.shape + (d_model,)
        out = self.weight[token_ids]

        if self.padding_idx is not None:
            # 填充位置强制输出 0；用 where 而不是乘 0，避免 0 * inf 这类数值问题
            out = torch.where((token_ids == self.padding_idx).unsqueeze(-1),
                              torch.zeros_like(out), out)
        return out


class MyLinear(nn.Module):
    """教学版 Linear：y = x W^T + b，只作用在输入的最后一维。"""

    def __init__(self, d_in: int, d_out: int, bias: bool = True, device=None, dtype=None):
        """
        参数:
            d_in:  输入的最后一维有多大（Linear 只看这一维）。
            d_out: 输出的最后一维有多大。
            bias:  是否加一个可学习的偏置；权重绑定（与 Embedding 共享参数）时一般设 False。
            device / dtype: 含义同 MyEmbedding。
        """
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.use_bias = bias

        # 权重存成 (d_out, d_in)：PyTorch 沿用行优先布局，前向时用 W^T（见 2.2 节）
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        # 标准差取 1/sqrt(d_in)：让输出数值尺度不随维度膨胀，训练更稳（见 2.5 节）
        nn.init.normal_(self.weight, mean=0.0, std=(1.0 / d_in) ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.empty(d_out, device=device, dtype=dtype))
            nn.init.zeros_(self.bias)     # 偏置从 0 起步，训练时再慢慢调
        else:
            # 不用 bias 时必须注册成 None，否则优化器找不到它、state_dict 也对不上
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (..., d_in)，输出 (..., d_out)；前面的维度原样保留。"""
        # 检查：只有最后一维参与变换，所以它必须刚好等于 d_in
        if x.shape[-1] != self.d_in:
            raise ValueError(f"输入最后一维是 {x.shape[-1]}，但这层要求 {self.d_in}")

        y = x @ self.weight.T          # 最后一维被换掉，batch 维和序列维都不动
        if self.bias is not None:
            y = y + self.bias         # 偏置会广播到所有 batch 与位置
        return y
```

### 5.2 MyEmbedding 的初始化参数速查

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `vocab_size` | int | 是 | 词表大小 = 嵌入表行数 |
| `d_model` | int | 是 | 每个 token 的向量维度 = 嵌入表列数 |
| `padding_idx` | int \| None | 否 | 填充位置 ID；该行输出恒为 0、梯度恒为 0 |
| `device` | str \| torch.device \| None | 否 | 参数所在设备，默认全局默认 |
| `dtype` | torch.dtype \| None | 否 | 参数数据类型，默认全局默认 |

### 5.3 MyLinear 的初始化参数速查

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `d_in` | int | 是 | 输入最后一维 |
| `d_out` | int | 是 | 输出最后一维 |
| `bias` | bool | 否 | 是否加可学习偏置，默认 `True` |
| `device` / `dtype` | — | 否 | 同上 |

### 5.4 示例 8：和内置层对拍

**为什么要对拍**：内置层是"标准答案"。把参数复制过去、同一份输入跑两边，结果一致才说明你的实现没歪——这正是作业测试 `test_embedding` / `test_linear` 的思路。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# --- Embedding 对拍 ---
V, D = 7, 5
mine = MyEmbedding(V, D)
ref = nn.Embedding(V, D)
with torch.no_grad():                 # 复制参数时不要记录梯度
    ref.weight.copy_(mine.weight)     # 两边用同一张表，才谈得上公平比较

ids = torch.tensor([[1, 3, 0], [2, 2, 6]])
print("emb shapes:", tuple(mine(ids).shape), tuple(ref(ids).shape))
print("emb allclose:", torch.allclose(mine(ids), ref(ids)))

# --- Linear 对拍 ---
mlin = MyLinear(3, 2, bias=True)
rlin = nn.Linear(3, 2, bias=True)
with torch.no_grad():
    rlin.weight.copy_(mlin.weight)
    rlin.bias.copy_(mlin.bias)

x = torch.randn(4, 6, 3)              # (batch=4, 序列=6, 特征=3)
print("lin shapes:", tuple(mlin(x).shape), tuple(rlin(x).shape))
print("lin allclose:", torch.allclose(mlin(x), rlin(x), atol=1e-6))
```

实际输出：

```text
emb shapes: (2, 3, 5) (2, 3, 5)
emb allclose: True
lin shapes: (4, 6, 2) (4, 6, 2)
lin allclose: True
```

**注意形状**：`Linear` 把 `(4, 6, 3)` 变成 `(4, 6, 2)`——batch 维和序列维原封不动，只有最后一维被替换。这就是 2.3 节说的"只看最后一维"。

### 5.5 示例 9：padding_idx 的梯度行为

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

emb = MyEmbedding(5, 3, padding_idx=0)     # 约定 ID=0 是 padding
ids = torch.tensor([[1, 1, 1]])             # 这句话里只有 ID=1

emb(ids).sum().backward()                   # 反向传播，看看谁拿到了梯度

print("mine row0:", emb.weight.grad[0].tolist())   # padding 行：应该是全 0
print("mine row1:", emb.weight.grad[1].tolist())   # 用到的行：应该有梯度

# 顺便看看内置层的行为是否一致
emb0 = nn.Embedding(5, 3, padding_idx=0)
emb0(ids).sum().backward()
print("builtin row0:", emb0.weight.grad[0].tolist())
```

实际输出：

```text
mine row0: [0.0, 0.0, 0.0]
mine row1: [3.0, 3.0, 3.0]
builtin row0: [0.0, 0.0, 0.0]
```

**要点**：手写版和内置版行为一致——padding 行梯度恒为 0；用到的行梯度非 0；而**没出现过的行梯度也是 0**（这就是 1.6 节讲的稀疏梯度）。

### 5.6 示例 10：边界与报错信息

```python
import torch

torch.manual_seed(0)
emb = MyEmbedding(5, 3)
mlin = MyLinear(3, 2)

# 越界的 token ID：以前是"index out of range"，现在能直接看懂
try:
    emb(torch.tensor([[99]]))
except IndexError as e:
    print("IndexError:", e)

# 特征维不匹配：以前是 matmul 维数错误，现在告诉你差多少
try:
    mlin(torch.randn(2, 4))
except ValueError as e:
    print("ValueError:", e)
```

实际输出：

```text
IndexError: token ID 越界：[99, 99]，词表只有 5 个
ValueError: 输入最后一维是 4，但这层要求 3
```

**为什么值得写这些检查**：模型训练时一旦形状错了，报错往往出现在很远的地方（比如某个 matmul 内部），很难反推。**在层内部做检查，等于给每一层都装了个"报警器"**。

### 5.7 示例 11：端到端小模型

把 1.6 节的"查表"和 2.1 节的"投影"串起来，就是一个最小可用的语言模型骨架。

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

V, D = 11, 6
emb = MyEmbedding(V, D)            # ① 底座：查表
head = MyLinear(D, V, bias=False)  # ③ 顶端：投影回词表（此时与嵌入共享参数就该关掉 bias）

ids = torch.randint(0, V, (2, 4))  # 一批 2 句，每句 4 个 token
h = emb(ids)                       # (2, 4) → (2, 4, 6)
logits = head(h)                   # (2, 4, 6) → (2, 4, 11)

print("shapes:", tuple(ids.shape), tuple(h.shape), tuple(logits.shape))
probs = F.softmax(logits, dim=-1)
print("probs sum:", probs.sum(-1).tolist())

# 训练一步：算交叉熵、反向传播，看看梯度落在哪
loss = F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1))
loss.backward()
print("loss:", round(loss.item(), 4))
print("rows touched:", int((emb.weight.grad.abs().sum(-1) > 0).sum()))
print("head grad shape:", tuple(head.weight.grad.shape))
```

实际输出：

```text
shapes: (2, 4) (2, 4, 6) (2, 4, 11)
probs sum: [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.9999998211860657, 0.9999998211860657]]
loss: 2.3939
rows touched: 6
head grad shape: (11, 6)
```

**三个值得停下来想一想的观察**：

1. **`probs sum` 全是 1**（除了浮点误差）——softmax 的定义，概率分布成立；
2. **`loss ≈ 2.3939`，而 $\ln(11) \approx 2.398$**——随机初始化时模型在"均匀乱猜"，每个词概率约 $1/11$，交叉熵自然接近 $\ln V$。**这是最好用的 sanity check**：训练刚开始时 loss 应该在 $\ln V$ 附近，如果一开始就远高于它，说明初始化或形状有问题；
3. **`rows touched: 6`**——这一批 8 个 token 覆盖了 11 行词表里的 6 行，梯度只落在这 6 行上，其余保持为 0（稀疏梯度）。

### 5.8 和作业接口的对照

教学示例的参数名是按 PyTorch 内置层的习惯写的，作业里要按**讲义的签名**来：

| 讲义接口 | 关键参数 | 和本例的差别 |
| --- | --- | --- |
| `run_embedding(vocab_size, d_model, weights, token_ids)` | 权重由外部传入（`weights`），不在 `__init__` 里创建 | 本例把权重做成 `self.weight`；讲义版本把权重当输入传进来 |
| `run_linear(d_in, d_out, weights, in_features)` | 同上，权重外部传入 | 本例同理 |
| `run_transformer_lm(...)` / `run_transformer_block(...)` | 还要处理 RoPE、因果掩码、多头拆分与合并 | 本章只覆盖单层，不含注意力 |

**迁移思路不变**：权重是"传进来的张量"还是"自己创建的参数"，只影响参数放在哪儿；前向的计算逻辑（查表 / $xW^\top+b$）完全一致。写完记得按 4.3 节的三条自检方式验证：与内置层对拍、形状断言、梯度检查。

---

## 6. 小结

- **Embedding = 查表**：把离散 token ID 换成稠密的 $D$ 维向量；参数是一张 `(V, D)` 的表；数学上等价于 one-hot 乘权重矩阵，但实现上直接取行更快。
- **Linear = 换维 + 线性组合**：$y = xW^\top + b$；只作用在**最后一维**；既可升维也可降维。
- **位置关系**：Embedding 在最底下把"编号"翻译成"向量"；LM head（一个 Linear）在最顶上把"向量"翻译回"词表分数"；中间的每个 Block 里散布着 4～7 个 Linear。
- **协作关系**：Embedding 赋予身份，Linear 变换视角，注意力负责身份之间的交流，FFN 负责各自的消化。
- **两个都要注意的点**：Embedding 的梯度是稀疏的（罕见词学得慢）；Linear 的权重初始化尺度直接决定训练稳不稳。

---

## 7. 常见问题 FAQ

**Q1：Embedding 和 one-hot 到底什么关系？**
数学上等价（Embedding ≈ one-hot @ 权重矩阵），工程上不同：one-hot 要造 `(B, T, V)` 的大矩阵并做 $V$ 维矩阵乘，而查表只需 $O(BT)$。用 one-hot 实现会慢一个数量级。

**Q2：Embedding 的向量是"训练出来的"还是"人设定的"？**
训练出来的。初始化时是随机数，靠反向传播不断调整，使共现的词彼此靠近。

**Q3：为什么 token 之间的顺序信息不在 Embedding 里？**
因为 Embedding 是"逐 ID 查表"，同一个词在任何位置拿到的向量完全一样。顺序信息要靠**位置编码**（如 RoPE）在注意力里注入。

**Q4：`nn.Linear(8, 16)` 处理 `(2, 5, 8)` 为什么输出是 `(2, 5, 16)`？**
Linear 只看最后一维，前面的维被当作一批独立样本并行处理。它天然是"逐位置"操作的。

**Q5：Linear 的权重为什么是 `(out, in)` 而不是 `(in, out)`？**
PyTorch 遵循行优先（row-major）内存布局，权重按 `(out_features, in_features)` 存，前向时转置使用。记住"PyTorch 里 Linear 权重是转置存的"即可。

**Q6：`nn.Linear` 能跨位置混合信息吗？**
不能，它逐位置独立。要跨位置交互得用注意力（Self-Attention）或卷积。

**Q7：Embedding 表是不是太大了？**
词表 10 万 × 维度 4096 = 4 亿参数，约 1.6 GB（fp32）。这是真实成本。常见缓解手段：降低维度、量化、分层 softmax、以及权重绑定（省一份，但省不了输入侧那份）。

**Q8：`padding_idx` 是干什么的？**
标记"填充位置"，让这些位置的嵌入不参与梯度更新，避免噪声污染训练。批处理变长序列时很有用。

**Q9：Embedding 和 LM head 共享权重有什么好处？**
省一份显存；且"输入端表示"与"输出端打分"用同一套坐标，语义更自洽。代价是两者不能各自有 bias，且可能限制表达能力。

**Q10：这两层需要自己写吗？**
在 CS336 Assignment 1 里**需要**（讲义明确禁止直接使用 `torch.nn.Embedding` / `torch.nn.Linear`）。但学习阶段完全可以先用内置层把概念跑通——本课所有示例就是这么写的，理解之后再自己实现，效率高得多。

**Q11：怎么快速验证我的实现和 PyTorch 一致？**
用同一份随机输入，对比两者的输出张量（`torch.allclose`），并分别检查形状、梯度、padding 行的梯度是否为零。

---

## 8. 动手练习

1. **改形状**：把示例 1 的 `vocab_size` 改成 100、`d_model` 改成 16，先预测输出形状，再跑一遍验证。
2. **验证稀疏梯度**：写一段代码，让一句话里同时出现 ID=1、ID=2，检查这两行都有梯度、其余行梯度为 0。
3. **手算 Linear**：用 `torch.manual_seed(0)` 建一个 `nn.Linear(3, 2)`，把 `weight` 和 `bias` 打印出来，用纸笔按 $xW^\top+b$ 算一遍，和程序输出对比。
4. **观察最后一维规则**：对 `(2, 5, 8)` 的张量做 `nn.Linear(8, 8)`（等维变换），确认输出仍是 `(2, 5, 8)`，理解"序列长度不受影响"。
5. **跨位置对比**：把 `(2, 5, 8)` 先 reshape 成 `(2, 40)` 再做 `nn.Linear(40, 40)`，和"对最后一维做 Linear"比较结果差异——体会 Linear 是逐位置而非跨位置的。
6. **显存估算**：给定词表大小 $V$、维度 $D$，算出嵌入表在 fp32 / fp16 / int8 下各占多少 MB，再对比"整段文本用 one-hot 表示"的体积。

---

*延伸阅读：同目录的 `Transformer 模块.md`（Transformer 整体流程）、`BPE算法原理.md`（token ID 是怎么来的）。*
