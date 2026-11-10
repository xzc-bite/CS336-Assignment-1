# Attention 机制：从零理解

> 这份文档系统讲解 Attention（注意力）机制，面向**零基础新手**。所有示例都用 PyTorch 写过并**实际运行验证**，文中给出的数字是真实输出（随机数每次会波动，但规律一致）。
>
> 文末专门有一节把每个概念对应回 **CS336 Assignment 1** 的接口与测试，方便你把"读懂"变成"写对"。
>
> 如果你想单独吃透最核心的缩放点积公式，配套篇是 `缩放点积注意力.md`；本文负责“总览”，那篇负责“放大镜”。

## 目录

- [0. 从一个具体问题出发](#0-从一个具体问题出发)
- [1. 核心公式：三步走](#1-核心公式三步走)
- [2. 公式每一部分的物理含义](#2-公式每一部分的物理含义)
- [3. 缩放因子 1/√d_k](#3-缩放因子-1d_k)
- [4. Mask：掩码机制](#4-mask掩码机制)
- [5. 注意力权重会怎么变](#5-注意力权重会怎么变)
- [6. 多头注意力](#6-多头注意力)
- [7. 计算与显存成本](#7-计算与显存成本)
- [8. 对应到 CS336 Assignment 1](#8-对应到-cs336-assignment-1)
- [9. 小结](#9-小结)
- [10. 常见问题 FAQ](#10-常见问题-faq)
- [11. 动手练习](#11-动手练习)

---

## 0. 从一个具体问题出发

先看一句话：

```text
"小明把书放在桌上，因为它很重。"
```

"它"指什么？可能是"书"，也可能是"桌子"。要判断，得看上下文，并且**"它"这个词需要主动去查询前面的信息**。

Transformer 的做法是：**每个位置都主动去问其他位置"你跟我有多相关"，然后按相关度取用大家的信息**。这个"主动查询 + 按权重取用"的动作，就是 Attention。

如果只有一层注意力，模型能捕捉的关联还很浅。堆叠多层之后（每层都重新问一遍、重新组合），就能逐层逼近"它 → 书"这种指代关系。这就是"注意力机制让模型理解长距离依赖"的意思。

> **Attention 解决的三个核心难题**（原论文的动机）：
> 1. **顺序无关**：RNN 必须按顺序读，注意力可以并行处理所有位置 → 训练快得多；
> 2. **长距离依赖**：RNN 靠"记忆向量"传递信息，路径长就衰减；注意力直接一步跨越任意距离；
> 3. **可解释性**：注意力权重是"谁看了谁"的显式记录（注意：它不等于完整的因果解释，见 5.6）。

---

## 1. 核心公式：三步走

Attention 的完整公式只有一行：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

别被公式吓到，它说的其实是三件很朴素的事：

```text
① 打分：   我和每个候选有多相关？      →  QKᵀ / √d_k
② 归一化：把分数变成"权重"（加起来=1）→  softmax
③ 取用：   按权重把大家的内容混合起来   →  × V
```

用图书馆的比喻串起来：

> 你拿着检索单（**Q**）走进图书馆；每本书都有标签（**K**）；标签写着"这本书讲狗"、"讲猫"……你把检索单和每个标签比一比，算出匹配度；匹配度归一化成权重；最后按权重把每本书的**内容**（**V**）取回来。

一句话对应：Q 是"我要找什么"，K 是"我是什么"，V 是"我真正提供的内容"。

### 1.1 示例 1：手算一个 3 个 token 的例子

```python
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# 3 个 token，每个 2 维；这里用能手算的简单数字
Q = torch.tensor([[1.0, 0.0],   # 第 0 个 token 的"提问"
                  [0.0, 1.0],   # 第 1 个 token 的"提问"
                  [1.0, 1.0]])  # 第 2 个 token 的"提问"
K = torch.tensor([[1.0, 0.0],   # 第 0 个 token 的"标签"
                  [0.0, 1.0],
                  [1.0, 1.0]])
V = torch.tensor([[1.0,  0.0],  # 第 0 个 token 的"内容"
                  [0.0,  1.0],
                  [-1.0, 0.0]]) # 第 2 个 token 的内容故意给个负数，方便观察

d_k = Q.shape[-1]                  # 2
scores = Q @ K.T / math.sqrt(d_k)  # ① 打分
w = scores.softmax(dim=-1)         # ② 归一化
manual = w @ V                     # ③ 按权重取用
ref = F.scaled_dot_product_attention(Q, K, V)   # PyTorch 内置版做对照

print("scores:\n", scores)
print("weights:\n", w)
print("manual:\n", manual)
print("matches pytorch:", torch.allclose(manual, ref, atol=1e-6))
```

实际输出：

```text
scores:
 tensor([[0.7071, 0.0000, 0.7071],
        [0.0000, 0.7071, 0.7071],
        [0.7071, 0.7071, 1.4142]])
weights:
 tensor([[0.4011, 0.1978, 0.4011],
        [0.1978, 0.4011, 0.4011],
        [0.2483, 0.2483, 0.5035]])
manual:
 tensor([[ 0.0000,  0.1978],
        [-0.2033,  0.4011],
        [-0.2552,  0.2483]])
matches pytorch: True
```

**怎么读这张表**：

- `scores` 第 0 行 `[0.7071, 0, 0.7071]`：token 0 与 token 0、token 2 完全"同向"（点积最大），与 token 1 毫无关系（点积为 0）；
- `weights` 第 0 行 `[0.4011, 0.1978, 0.4011]`：softmax 之后变成权重，token 0 和 token 2 各占 40%，token 1 占 20%（虽然它的原始分是 0，但"分母里其他人也小"所以它并非 0）；
- `manual` 第 0 行 `[0.0000, 0.1978]`：输出 = 0.4011×[1,0] + 0.1978×[0,1] + 0.4011×[-1,0] = [0, 0.1978]。

**注意最后一列 V 是 `[-1, 0]`**，它贡献了负的 x 分量，于是 token 0 的输出 x 恰好被两个 ±1 抵消成 0。数值对上了，说明你真的算懂了。

### 1.2 形状一览

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| Q | `(..., n_queries, d_k)` | 每个"提问者"的向量 |
| K | `(..., n_keys, d_k)` | 每个"被问者"的标签 |
| V | `(..., n_keys, d_v)` | 每个"被问者"提供的内容 |
| $QK^\top$ | `(..., n_queries, n_keys)` | 打分表：谁问谁 |
| softmax 之后 | `(..., n_queries, n_keys)` | 权重表：每行加起来 = 1 |
| 输出 | `(..., n_queries, d_v)` | 每个提问者都拿到一份"混合后的答案" |

`n_queries` 和 `n_keys` **不一定相等**——这就是"交叉注意力"（解码器看编码器输出）能存在的原因。自注意力里两者相等（都是序列长度）。

---

## 2. 公式每一部分的物理含义

### 2.1 Q、K、V 到底是什么

它们都是**同一批 token 向量**经过三个不同线性变换得到的：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

（$X$ 是当前层的输入，形状 `(..., T, d_model)`。）

- **Q（Query，查询）**："我这次想找什么样的信息？"
- **K（Key，键）**："我是个什么东西，可以被这样找到？"
- **V（Value，值）**："如果你决定用我，我真正贡献的内容是它。"

### 2.2 点积作为相似度：几何直觉

点积 $\mathbf{q} \cdot \mathbf{k} = \sum_i q_i k_i$ 有个漂亮的性质：

- 两个向量**方向越接近**，点积越大（越"相关"）；
- 一个分量大、另一个分量大 → 乘积大 → 点积大；
- 符号相反 → 互相抵消 → 点积小甚至为负。

所以点积天然是"带方向的相似度"：同向的两个 token 相互吸引，正交的互不关心。

### 2.3 为什么 Q、K、V 要用三个不同的矩阵

如果 Q、K、V 都等于 X 自己（不做投影），那"相关性"就变成了"两个 token 原始向量有多像"，这受限于嵌入空间本身的结构。引入三个可学习矩阵后：

- 模型可以**为不同角色学不同的坐标系**："作为提问者"时该怎么表达需求，和"作为作答者"时该怎么介绍自己，是两回事；
- 头与头之间、不同层之间都能分化出不同的匹配方式。

**这是注意力最重要的可学习部分**——$W_Q, W_K, W_V, W_O$ 四个矩阵加起来的参数量，比嵌入表还多。

### 2.4 softmax 在这里干什么：它是一个"温度"

softmax 把任意实数分数变成非负且和为 1 的权重，等价于"归一化的打分"。它还有个重要性质：**输入整体放大 $c$ 倍，softmax 输出的比值不变**（因为分子分母同时放大）——这正是 3.2 节数值稳定技巧的原理。

从"温度"角度更好理解：分数放大 → 分布变尖（关注少数几个）；分数缩小 → 分布变平（人人有份）。$1/\sqrt{d_k}$ 就是把温度固定在一个"刚刚好"的值上（见第 3 节）。

### 2.5 示例 2：softmax 的数值稳定性

```python
import torch

v = torch.tensor([1000.0, 1001.0, 1002.0])

# 天真写法：直接 exp
print("naive exp:", torch.exp(v))
print("naive softmax:", torch.exp(v) / torch.exp(v).sum())

# 稳定写法：先减去最大值（softmax 对"整体平移"不敏感）
vs = v - v.max()
print("stable softmax:", torch.exp(vs) / torch.exp(vs).sum())
```

实际输出：

```text
naive exp: tensor([inf, inf, inf])
naive softmax: tensor([nan, nan, nan])
stable softmax: tensor([0.0900, 0.2447, 0.6652])
```

**要点**：`exp(1000)` 直接溢出成 `inf`，再一除就变成 `nan`（`inf/inf`）。而减去最大值后最大项变成 0，`exp(0)=1`，永远不会溢出，结果和朴素写法**数学上完全一致**。

> CS336 讲义 3.4.4 节明确要求 softmax 做这个减最大值的处理，`test_softmax_matches_pytorch` 会检查你的实现。

---

## 3. 缩放因子 1/√d_k

### 3.1 为什么会变大：方差分析

假设 $q$ 和 $k$ 的每个分量都是均值 0、方差 1 的随机数，那么：

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

每一项 $q_i k_i$ 的方差是 1（独立同分布乘积的方差为 $\text{Var}(q)\cdot\text{Var}(k) = 1$），由方差可加性：

$$
\text{Var}(q \cdot k) = d_k \quad\Longrightarrow\quad \text{std}(q \cdot k) = \sqrt{d_k}
$$

**维度越大，点积的数值尺度越大。** $d_k=64$ 时标准差约 8，$d_k=1024$ 时约 32。

### 3.2 不除会怎样：softmax 饱和 + 梯度通道被压扁

分数一大，softmax 就变成"one-hot"：几乎全部概率压在一个位置上。

```python
import math
import torch

torch.manual_seed(0)
n_k = 64

for d in [16, 64, 256, 1024]:
    q = torch.randn(1, 1, d)
    k = torch.randn(1, n_k, d)
    raw = q @ k.transpose(-2, -1)
    for name, sc in [("unscaled", raw), ("scaled", raw / math.sqrt(d))]:
        p = sc.softmax(dim=-1)
        # softmax 雅可比里最大的回传系数 p_i(1-p_i)：p 越接近 1，它越接近 0
        jac = (p * (1 - p)).max().item()
        ent = -(p * (p + 1e-30).log()).sum(-1).mean().item()
        print(f"d_k={d:5d} {name:9s} max_prob={p.max().item():.6f} "
              f"entropy={ent:7.4f} max_grad_coeff={jac:.3e}")
```

实际输出（典型一次）：

```text
d_k=   16 unscaled  max_prob=0.513118 entropy= 1.6034 max_grad_coeff=2.498e-01
d_k=   16 scaled    max_prob=0.085820 entropy= 3.7441 max_grad_coeff=7.845e-02
d_k=   64 unscaled  max_prob=0.542515 entropy= 0.8162 max_grad_coeff=2.482e-01
d_k=   64 scaled    max_prob=0.161177 entropy= 3.2629 max_grad_coeff=1.352e-01
d_k=  256 unscaled  max_prob=0.993724 entropy= 0.0411 max_grad_coeff=6.237e-03
d_k=  256 scaled    max_prob=0.079237 entropy= 3.7332 max_grad_coeff=7.296e-02
d_k= 1024 unscaled  max_prob=0.942736 entropy= 0.2194 max_grad_coeff=5.398e-02
d_k= 1024 scaled    max_prob=0.081841 entropy= 3.7517 max_grad_coeff=7.514e-02
```

**读这张表（关键！）**：

- **不缩放**时，随着 $d_k$ 变大，`max_prob` 飙到 0.99、`entropy` 掉到 0.04——分布已经**饱和成一个尖峰**，模型只能"盯死"一个位置，其余信息全部丢失；
- 关键在第三列 `max_grad_coeff`：softmax 的雅可比对角元是 $p_i(1-p_i)$，**当 $p_i \to 1$ 时它趋于 0**，意味着梯度回传的通道被压扁 → 梯度消失，训练停滞；
- **缩放之后**，`max_prob` 稳定在 0.08 左右、`entropy` 稳定在 3.7 附近、`max_grad_coeff` 也稳定在 $7\times10^{-2}$ 量级——**与 $d_k$ 无关**。

### 3.3 缩放因子在做什么

$$
\frac{QK^\top}{\sqrt{d_k}}
$$

把分数的方差从 $d_k$ 拉回 1：**不管头维度多大，softmax 的输入都处在"健康温度"上**。这一步让注意力的分布既不"人人有份"（学不到东西）也不"死盯一个"（学不到东西）。

### 3.4 常见疑问

**Q：为什么是 $\sqrt{d_k}$ 而不是 $d_k$？**
我们要的是"标准差归一化"，而标准差是 $\sqrt{d_k}$。除以 $d_k$ 会缩得过头（分布过平），除以 1 会缩得不够。

**Q：为什么不直接用余弦相似度（除以各自的模）？**
余弦相似度在某些任务上更稳，但它丢掉了**模长信息**，而且和训练用的矩阵乘法形式不兼容。缩放点积是"可微分 + 能用一次矩阵乘批量算"的折中，也是事实标准。

**Q：$d_k$ 到底是哪个维度？**
是**每个头**的维度 $d_k = d_{model} / n_{heads}$，不是 $d_{model}$。多头拆分之后，每头维度更小，缩放因子也随之变小——这也是多头能让各头关注不同尺度信息的原因之一。

---

## 4. Mask：掩码机制

### 4.1 因果掩码：为什么不能看未来

语言模型训练时，第 $i$ 个位置的预测目标就是第 $i+1$ 个词。如果注意力允许位置 0 直接看到位置 5，那模型只要把"答案"抄过来就能拿满分，训练出来的模型在推理时（没有未来可看）就完全不会用了。

所以要加**因果掩码（causal mask）**：位置 $i$ 只能看到 $j \le i$ 的位置。

```text
        位置0   位置1   位置2   位置3
位置0 [   ✓       ✗       ✗       ✗  ]
位置1 [   ✓       ✓       ✗       ✗  ]
位置2 [   ✓       ✓       ✓       ✗  ]
位置3 [   ✓       ✓       ✓       ✓  ]
```

这就是"下三角"。

### 4.2 掩码要加在 softmax 之前

**正确做法**：在打分矩阵上把不该看的位置设成 $-\infty$，softmax 之后它们的权重自动变成 0。

```python
scores = Q @ K.T / math.sqrt(d_k)
scores = scores.masked_fill(~causal_mask, float("-inf"))   # ← 这里
weights = scores.softmax(dim=-1)
```

**错误做法**：先 softmax，再把权重乘 0。后者有两个问题：

1. 数值上：$0 \times \infty = \text{NaN}$（被屏蔽位置如果分数很大，exp 会溢出）；
2. 语义上：注意力分数被 mask 掉后**不应该影响分母**。乘 0 的做法仍然让被屏蔽位置参与了 softmax 的归一化，等于"虽然不看，但你的分数还是影响了我和别人的相对比例"。

### 4.3 True/False 的约定陷阱（重要！）

**讲义里的约定**：`mask` 是一个布尔张量，`True` 表示"**允许注意**"，`False` 表示"屏蔽"。（讲义原话：*"a value of True at position (i, j) indicates that the query i does attend to the key j"*。）

**PyTorch `F.scaled_dot_product_attention` 的约定**：布尔 `attn_mask` 里 `True` 表示"**参与注意**"——**和讲义一致**，可以原样传进去。（实测验证：`mask=[True, False]` 时只取第 0 个 value。）

**但别把 API 记混**：`nn.MultiheadAttention` 的 `attn_mask` 恰好相反，`True` 表示"**屏蔽**"。换 API 前一定先确认约定——这类错误不报错、只错数，最难查。

**CS336 里怎么做**：`run_scaled_dot_product_attention` 的 `mask` 参数遵循**讲义的约定**（True = 允许）。所以：

- 自己实现时：$\text{scores} = \text{scores.masked\_fill}(\text{mask} == \text{False}, -\infty)$；
- 直接调 `F.scaled_dot_product_attention`：掩码原样传，不用取反；
- 如果用的是 `nn.MultiheadAttention`，则必须先取反：`attn_mask = ~mask`；
- 或者干脆用 `torch.tril` 之类的数值型 mask（给 0 / $-\infty$），语义更直白，不容易搞反。

> 这是作业里最容易踩的坑之一：**形状对、数值全错**，而且不报错，只是测试不过。写完记得单独验证一条：被屏蔽位置的权重是否**精确等于 0**。

### 4.4 整行被屏蔽会怎样

如果一整行（某个 query 的所有 key）都被屏蔽，softmax 的分母就是 0，结果是 `nan`（$-\infty / -\infty$）。

- 因果掩码不会出现这种问题：位置 0 至少能看到自己；
- 但如果将来要支持"某段全是 padding"的场景，就必须小心：常见做法是给被屏蔽行留一个"逃生位置"（比如让它看自己），或者事后把 `nan` 替换成 0。

### 4.5 因果性的可验证性质：前缀不变

这是一个非常好用的自检方法——**改动后面的 token，前面位置的输出必须纹丝不动**。

```python
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
T = 4
q = torch.randn(1, T, 8)
k = torch.randn(1, T, 8)
val = torch.randn(1, T, 8)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))

out_a = F.scaled_dot_product_attention(q, k, val, attn_mask=causal)

# 打印权重矩阵
w = ((q @ k.transpose(-2, -1) / math.sqrt(8))
     .masked_fill(~causal, float("-inf"))).softmax(-1)
print("weights:\n", w)

# 把最后一个 token 的内容改得很大
val2 = val.clone()
val2[:, 3] += 100.0
out_c = F.scaled_dot_product_attention(q, k, val2, attn_mask=causal)

print("prefix unchanged:", torch.allclose(out_a[:, 0], out_c[:, 0], atol=1e-6))
```

实际输出（典型一次）：

```text
weights:
 tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
         [0.8935, 0.1065, 0.0000, 0.0000],
         [0.5828, 0.1109, 0.3062, 0.0000],
         [0.2324, 0.2137, 0.1543, 0.3996]]])
prefix unchanged: True
```

**怎么用这个例子自检**：

1. 权重矩阵严格是下三角——上三角**精确为 0**（不是"很小"，是 0）；
2. 第 0 行必然是 `[1, 0, 0, 0]`（只有自己能看）；
3. `prefix unchanged: True`——把最后一个 token 的 V 改动 100 之后，位置 0 的输出完全不变。这正是因果性的定义。

---

## 5. 注意力权重会怎么变

新手常见困惑："权重矩阵是模型算出来的，那它长什么样？"下面是几条经验规律。

### 5.1 训练初期：接近均匀

刚初始化时，$W_Q, W_K$ 还是随机数，打分没有结构，softmax 后各位置权重**差不多平均**——相当于"每个位置都在平均地看全场"，信息还没有选择性。

### 5.2 训练过程中：出现分工与稀疏

训练推进后，权重开始分化：

- 某些位置对会形成**稳定的高权重**（模型学会了"这里该看那里"）；
- 一些头会变得**非常稀疏**，几乎只盯一两个位置（常被解释为"指代头""复制头""分隔符头"等）；
- 另一些头保持**弥散**，承担"整体语境汇总"的角色。

### 5.3 随输入变化：同一个头，不同句子不同分布

权重不是写死在参数里的常量，而是**由当前输入算出来的**。

```python
import torch

torch.manual_seed(0)
q1 = torch.randn(1, 3, 16)
k1 = torch.randn(1, 3, 16)
q2 = q1 + 3.0          # 只改提问，标签不动
k2 = k1.clone()

w1 = (q1 @ k1.transpose(-2, -1) / 4.0).softmax(-1)
w2 = (q2 @ k2.transpose(-2, -1) / 4.0).softmax(-1)

print("weights input1:\n", w1)
print("weights input2:\n", w2)
```

实际输出（典型一次）：

```text
weights input1:
 tensor([[[0.1407, 0.6998, 0.1594],
         [0.4688, 0.1522, 0.3789],
         [0.5275, 0.2856, 0.1869]]])
weights input2:
 tensor([[[0.0233, 0.0144, 0.9623],
         [0.0328, 0.0013, 0.9659],
         [0.0715, 0.0048, 0.9236]]])
```

同样三个位置，权重分布完全不同：**注意力是"因输入而异"的动态行为，不是固定模式**。

### 5.4 随层数变化：浅层看字面，深层看结构

经验上：

- **浅层**的头更关注**局部窗口**（相邻几个 token）、标点、词形等表层特征；
- **深层**的头更关注**长距离的语义与结构**（比如代词指向的对象、句法依存、跨句的实体）。

### 5.5 注意力自带"位置偏好"

没有位置编码时，注意力**分不清先后顺序**（它对输入的置换是等变的）。因此即使加了因果掩码，模型也只知道自己"能看到左边"，不知道"左边有多远"。

这正是 RoPE 存在的原因：它把"相对距离"编码进 Q/K 的旋转角度里，让注意力能分辨"隔 1 个词"和"隔 100 个词"。详见 `Transformer 模块.md` 与作业的 `run_rope`。

### 5.6 重要提醒：注意力权重 ≠ 因果解释

看到某个头的权重图很漂亮（代词指向了正确的名词），很容易下结论"模型就是这样思考的"。但：

- 注意力权重只是**信息流动的一路证据**，模型可能同时用其他回路（FFN 的逐位置计算、残差主干）传递同样的信息；
- 有研究指出：**把注意力权重"打乱"或换成随机值，模型表现往往不掉**——说明关键信息可能走了别的路。

所以：**注意力图是很好的观察窗口，但不是解释**。

---

## 6. 多头注意力

### 6.1 为什么要多头

单个头只能学一种"关注模式"，而且它的"关注容量"有限（softmax 后一个分布）。多头把 $d_{model}$ 拆成 $H$ 份，每份独立做一次注意力：

- 不同头可以各学一套关注偏好（局部 / 全局 / 指代 / 复制）；
- 每个头维度更小（$d_k = d_{model}/H$），单个头的表达力没那么"拥挤"；
- 最后把各头输出拼回 $d_{model}$，再过一层 $W_O$ 融合。

**类比**：单头 = 一个专家独揽全场；多头 = 专家团各自记录自己关心的部分，最后合议。

### 6.2 形状怎么走

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, d_model, H = 2, 5, 8, 4      # batch=2, 序列长=5, 模型维度=8, 头数=4
d_head = d_model // H               # 每头 2 维

# 关键技巧：一次矩阵乘算出所有头的 Q、K、V
qkv = torch.randn(B, T, 3 * d_model)
q, k, v = qkv.split(d_model, dim=-1)

# 拆头：把 (B, T, d_model) 变成 (B, H, T, d_head)
qh = q.view(B, T, H, d_head).transpose(1, 2)
kh = k.view(B, T, H, d_head).transpose(1, 2)
vh = v.view(B, T, H, d_head).transpose(1, 2)

print("per-head q shape:", tuple(qh.shape))       # (2, 4, 5, 2)
o = F.scaled_dot_product_attention(qh, kh, vh)
print("attn out shape:", tuple(o.shape))          # (2, 4, 5, 2)

# 合头：还原成 (B, T, d_model)
merged = o.transpose(1, 2).reshape(B, T, d_model)
print("merged shape:", tuple(merged.shape))        # (2, 5, 8)
```

实际输出：

```text
per-head q shape: (2, 4, 5, 2)
attn out shape: (2, 4, 5, 2)
merged shape: (2, 5, 8)
```

### 6.3 CS336 参考实现的权重布局（重要）

作业给的参考权重（`state_dict`）里：

| 权重 | 形状 | 行如何排列 |
| --- | --- | --- |
| `layers.0.attn.q_proj.weight` | `(d_model, d_model)` | 前 $d_k$ 行是头 0 的投影，接着是头 1，以此类推 |
| `layers.0.attn.k_proj.weight` | `(d_model, d_model)` | 同上 |
| `layers.0.attn.v_proj.weight` | `(d_model, d_model)` | 同上（V 的输出维度是 $d_v$） |
| `layers.0.attn.output.weight` | `((num_heads * (d_model/num_heads)), d_model)` | 即 `(d_model, d_model)`，负责合头后融合 |

**含义**：你可以用**一次**矩阵乘（`X @ W_Q`）算出所有头的数据，然后把结果的最后一维按 `(H, d_head)` 拆开即可——这就是讲义 3.4.5 节要求"用单个矩阵乘处理所有头的投影"的实现方式，比循环 $H$ 次快得多。

---

## 7. 计算与显存成本

注意力矩阵的规模是 $T \times T$，所以：

- **时间复杂度**：$O(T^2 d)$（T 翻倍 → 计算量 4 倍）
- **显存**：注意力矩阵占 $O(T^2)$ 空间

```python
import time
import torch
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
for T in [512, 1024, 2048]:
    q = torch.randn(1, 4, T, 16, device=dev)
    k = torch.randn(1, 4, T, 16, device=dev)
    v = torch.randn(1, 4, T, 16, device=dev)
    for _ in range(5):                       # 预热：第一次调用要初始化 kernel
        _ = F.scaled_dot_product_attention(q, k, v)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        _ = F.scaled_dot_product_attention(q, k, v)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 50
    print(f"T={T:5d}  time={dt*1000:8.3f} ms   attn_matrix={T*T/1e6:.2f} M entries")
```

实际输出（典型一次，RTX 级显卡）：

```text
T=  512  time=   0.043 ms   attn_matrix=0.26 M entries
T= 1024  time=   0.162 ms   attn_matrix=1.05 M entries
T= 2048  time=   0.630 ms   attn_matrix=4.19 M entries
```

序列长度翻倍，耗时约变成 4 倍——**平方级增长**。

这也是为什么长上下文很贵：上下文从 4K 提到 32K，注意力部分的计算量是 64 倍。业界为此发明了 FlashAttention（分块计算，不显式存下整个 $T\times T$ 矩阵）、稀疏/线性注意力等方案。作业里你用朴素实现即可，但要理解这个代价从何而来。

---

## 8. 对应到 CS336 Assignment 1

### 8.1 需要接的三个接口

> 仓库现状：这三个适配器都已经接好（`run_scaled_dot_product_attention` 转手给
> `Scaled_dot_product_attention.py`，两个多头函数转手给对应的注意力类），
> `pytest -k "attention or rope"` 全绿。下面这张表说明它们各自的关键约束。

| 适配器函数（`tests/adapters.py`） | 关键要求 |
| --- | --- |
| `run_scaled_dot_product_attention(Q, K, V, mask=None)` | 支持**任意前置 batch 维**：3D `(B, T, d_k)` 和 4D `(B, H, T, d_k)` 都要能跑（测试里两个用例各测一种） |
| `run_multihead_self_attention(...)` | 所有头的 Q/K/V 投影要用**一次矩阵乘**完成；不带 RoPE |
| `run_multihead_self_attention_with_rope(...)` | 同上，但 Q/K 施加 RoPE；**RoPE 的旋转维度是 head 维度** `d_model // num_heads` |

### 8.2 实现顺序建议

1. **先写 `run_softmax`**（讲义 3.4.4）：要带"减最大值"的数值稳定处理，`test_softmax_matches_pytorch` 会验证；
2. **再写 `run_scaled_dot_product_attention`**：先用 3D 输入跑通，再补 4D 支持；
3. **然后写 `run_multihead_self_attention`**：投影 → 拆头 → 调 SDPA → 合头 → $W_O$；
4. **最后写 `run_rope`** 与带 RoPE 的多头版本。

### 8.3 自检清单（每写完一步都过一遍）

- [ ] 形状对不对？`(B, T, d_k)` 输入 → `(B, T, d_v)` 输出；4D 版本头维度不能错位；
- [ ] **被屏蔽位置的权重精确等于 0**（不是"接近 0"）；
- [ ] 每一行权重加起来等于 1（未被整行屏蔽时）；
- [ ] 因果性：改动后面的 token，前面位置的输出不变；
- [ ] 缩放因子写了没有？`/ math.sqrt(d_k)`（注意是 **head 维度**，不是 $d_{model}$）；
- [ ] 掩码语义搞反了没有？（讲义与 `F.scaled_dot_product_attention` 都是 True=允许；只有 `nn.MultiheadAttention` 是 True=屏蔽）；
- [ ] softmax 有没有做减最大值的稳定处理？
- [ ] 4D 输入时，`d_k` 取的是最后一维（每头的维度），不是整个 `d_model`；
- [ ] 数值对齐：和 `F.scaled_dot_product_attention` 在相同输入下结果一致（测试用 numpy snapshot，容差 `atol=1e-5`）。

### 8.4 常见错误清单

| 症状 | 多半是 |
| --- | --- |
| 结果和 PyTorch 差一点点，误差约 1e-3 | 漏了 $1/\sqrt{d_k}$ |
| 上三角有微小非零值 | 掩码用了乘 0 而不是 $-\infty$，或掩码语义反了 |
| 出现 `nan` | 某一行被整行屏蔽；或 softmax 没做稳定化 |
| 4D 输入结果错乱 | 拆头时 `transpose` 前后维度搞错；或把 $d_{model}$ 当成了 $d_k$ |
| 权重矩阵形状是 `(T, T)` 但 batch 丢了 | 少写了一个 batch 维的广播/维度处理 |
| 测试只在 4D 失败、3D 通过（或反之） | 只实现了一种输入布局 |

### 8.5 一个能立刻建立信心的小实验

拿 4 个 token 的极小序列，把注意力权重矩阵打印出来，对着第 4.5 节那张下三角表逐格核对；然后把掩码去掉，看输出如何变化；再把 $\sqrt{d_k}$ 换成 1，看 softmax 分布如何变尖。**小实验是理解注意力最快的路。**

---

## 9. 小结

- **Attention = 打分 → 归一化 → 加权求和**：`softmax(QKᵀ/√d_k)V`。
- **Q/K/V 是三个角色**：我想找什么 / 我是什么 / 我能提供什么。点积是带方向的相似度。
- **$1/\sqrt{d_k}$ 是温度校准**：把分数方差从 $d_k$ 拉回 1，防止 softmax 饱和、梯度通道被压扁。
- **Mask 加在 softmax 之前**，用 $-\infty$ 而不是乘 0；**讲义里 True = 允许注意**，`F.scaled_dot_product_attention` 同向，只有 `nn.MultiheadAttention` 反向。
- **权重是动态的**：训练初期近均匀 → 训练后分化出分工；随输入、随层数变化。
- **注意力权重 ≠ 因果解释**。
- **多头**：拆成 $H$ 份并行做注意力，$d_k = d_{model}/H$，最后合头再过 $W_O$。
- **代价是 $O(T^2)$**：序列翻倍，计算量四倍。

---

## 10. 常见问题 FAQ

**Q1：Q、K、V 必须长度相等吗？**
Q 的最后一维必须等于 K 的最后一维（要能做点积）；V 的最后一维 $d_v$ 可以不同（输出就是 $d_v$ 维）。$n_{queries}$ 和 $n_{keys}$ 也可以不同。

**Q2：为什么公式里是 $QK^\top$ 而不是 $K^\top Q$？**
要得到"提问者 × 被问者"的矩阵，行必须是提问者，所以是 $QK^\top$。用列向量记法时写作 $K^\top Q$，形状一样。

**Q3：self-attention 和 cross-attention 的区别？**
Self-attention 的 Q、K、V 来自同一批序列（ encoder/decoder 内部）；cross-attention 的 Q 来自一种序列、K/V 来自另一种（如解码器看编码器输出）。公式完全一样，只是输入来源不同。

**Q4：掩码可以直接用 `float('-inf')` 吗？会不会出问题？**
用 $-\infty$ 是标准做法，softmax 会把它变成 0。但如果整行都是 $-\infty$，分母为 0 → `nan`。另外某些实现用 `-1e9` 这种大负数代替，数值上更安全（不会真的溢出），效果几乎一样。

**Q5：为什么 softmax 而不是其他归一化？**
softmax 可导、平滑、且天然表达"相对份额"；它还提供了 3.2 节讲的可平移不变性。

**Q6：注意力输出还需要归一化吗？**
不需要。Attention 之后（多头合并后）会接 $W_O$ 投影，然后是残差相加 + 下一个 RMSNorm。层归一化在 Block 的 pre-norm 结构里已经就位了。

**Q7：能不能把 $1/\sqrt{d_k}$ 换成可学习参数？**
可以，有人试过（效果相当）。但它会破坏"温度恒定"这个先验，训练更不稳定，主流实现仍用固定的 $1/\sqrt{d_k}$。

**Q8：为什么每个头的 $d_k$ 小一点更好？**
注意力矩阵的规模只和 $T$ 有关，$d_k$ 影响的是表达维度。但 $d_k$ 变小会让 $QK^\top$ 的尺度变小（需要重新缩放），同时单个头的"容量"更小，逼迫不同头分工。

**Q9：注意力能不能双向（不掩码）？**
能。BERT 就是双向的——它做的是"理解"任务，不做"生成"，没有偷看未来的问题。本作业是 decoder-only 语言模型，必须因果掩码。

**Q10：训练时和推理时的注意力有什么不同？**
训练时（teacher forcing）整段序列一次算完，可以并行；推理时逐 token 生成，需要 **KV cache** 缓存已算好的 K/V，把每步代价从 $O(T^2)$ 降到 $O(T)$。

**Q11：注意力权重能用来做"可解释性"吗？**
可以当观察窗口用，但要谨慎（见 5.6）。

**Q12：为什么注意力被称为"加权平均"？**
因为输出确实是所有 V 的凸组合（权重非负且和为 1）。区别在于权重不是人为设定的，而是由内容动态算出来的。

---

## 11. 动手练习

1. **手算验证**：把示例 1 的 Q、K、V 换成自己的数字，在纸上完整算一遍分数、权重、输出，再和程序对照。
2. **验证掩码语义**：写一个 4×4 的因果 mask，用讲义约定（True=允许）构造，传给 `F.scaled_dot_product_attention` 前记得取反；检查权重矩阵是否严格下三角。
3. **前缀不变性**：写一段代码，依次修改每个位置的 V，验证位置 $i$ 的输出只受位置 $\le i$ 影响。
4. **去掉缩放因子**会怎样：用 $d_k=256$ 复现 3.2 节表格中 `max_prob=0.99` 那一行，并计算熵。
5. **换一批输入看权重**：用同一组 Q、K 的权重，对三段不同的输入分别算权重矩阵，观察分布差异（对应 5.3 节）。
6. **多头切分练习**：验证 `view` + `transpose` + `reshape` 的组合确实等价于"逐头循环计算再拼接"（可以写个 for 循环版本对拍）。
7. **成本估算**：给定 $d_{model}=512$、$n_{heads}=8$、上下文 2048，算出注意力矩阵在 fp16 下占多少显存；再估算上下文扩到 8192 时的占用。

---

*延伸阅读：同目录的 `缩放点积注意力.md`（把核心公式单独拆开）、`Transformer 模块.md`（Attention 在整体结构中的位置）、`Embedding_Linear.md`（Q/K/V 投影矩阵的来源）、`BPE算法原理.md`（token 从哪来）。*
