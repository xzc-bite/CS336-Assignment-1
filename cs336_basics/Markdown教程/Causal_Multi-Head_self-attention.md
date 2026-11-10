# 因果多头自注意力（Causal Multi-Head Self-Attention）

> 面向**零基础新手**的系统讲解。Attention 的通用原理见 `Attention.md`，单个头的公式细节见 `缩放点积注意力.md`；本文专门攻克"多头 + 因果 + 位置"这三件套，并把每一步的**形状变化**摊开给你看。
>
> 文中所有示例都用 PyTorch 写过并**实际运行验证**，输出为真实结果。

## 目录

- [0. 一句话看懂这件事](#0-一句话看懂这件事)
- [1. 复习：自注意力到底在做什么](#1-复习自注意力到底在做什么)
- [2. 为什么需要"多头"](#2-为什么需要多头)
- [3. 完整维度流转](#3-完整维度流转)
- [4. 关键技术细节](#4-关键技术细节)
- [5. 对应到 CS336 Assignment 1](#5-对应到-cs336-assignment-1)
- [6. 小结](#6-小结)
- [7. 常见问题 FAQ](#7-常见问题-faq)
- [8. 动手练习](#8-动手练习)

---

## 0. 一句话看懂这件事

> **因果多头自注意力 = 让序列里的每个位置派出 $H$ 个"小分队"，各自去全场挑自己关心的信息（而且只能往左挑），最后把各队的笔记汇总成一份结论。**

三个关键词拆开看：

| 关键词 | 含义 | 通俗说法 |
| --- | --- | --- |
| **自（self）** | Q、K、V 都来自同一个序列 | "自己人查自己人" |
| **多头（multi-head）** | 把 $d_{model}$ 拆成 $H$ 份，每份独立做一次注意力 | "几个专家各看各的" |
| **因果（causal）** | 位置 $i$ 只能看 $j \le i$ | "只能往回看，不许偷看未来" |

先建立整体直觉：**Transformer 里没有"句子"这个整体概念，只有"位置"这一个概念。** 每个位置手里握着一个向量，它要做的事只有一件——"参考一下其他位置，把有用的信息收进我的向量里"。多头自注意力，就是这件事的完整工业级实现。

---

## 1. 复习：自注意力到底在做什么

设输入序列的向量矩阵为 $X \in \mathbb{R}^{B \times T \times d_{model}}$（$B$ = batch，$T$ = 序列长度，$d_{model}$ = 隐藏维度）。

先做三次线性投影，得到三个角色：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

- **Q（查询）**：这个位置"想找什么"
- **K（键）**：这个位置"是什么"
- **V（值）**：这个位置"真正提供的内容"

然后：

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

三步走：**打分**（$QK^\top$，谁和谁相关）→ **归一化**（softmax，变成权重）→ **加权取用**（$\times V$）。

注意"自"体现在哪：$Q$、$K$、$V$ 都由同一个 $X$ 算出来，所以是"序列内部互相参考"。如果 K、V 来自另一个序列（比如解码器看编码器），那叫交叉注意力，公式一模一样。

---

## 2. 为什么需要"多头"

### 2.1 单头的天花板

一个注意力头输出的本质是**一份关注列表**：对每个位置，给其他所有位置打一个权重分。

这份列表受两重限制：

1. **容量限制**：一次 softmax 只能表达"一份"关注模式；
2. **维度限制**：所有判断都发生在同一个 $d_{head}$ 维子空间里，而高维空间能同时容纳的"不同方向"是有限的。

如果一个头既要记住"上一句的主语是谁"，又要关心"下一个词该选哪个"，还要跟踪"标点在哪儿"——一张关注列表根本不够用。

### 2.2 三个直观类比

**类比一：多位专家 / 多台摄像机**

> 同一个会议，单台摄像机只能拍一个机位；多机位（导播切镜头）能同时捕捉全景、近景、字幕。**注意力头就是多机位**——每台机器拍同一场会议，但角度不同。

**类比二：多位读者读同一篇文章**

> 一篇文章，有人划重点是"作者在骂人"，有人划重点是"作者在举例"，有人划重点是"作者在转折"。
> **每个头就是一种读法**；模型最后把大家的划重点汇总，才能完整理解文章。

**类比三：一束光的不同滤镜**

> 同一个向量，进入不同的头，就像同一束白光穿过不同滤镜，得到不同颜色的分解结果。
> 头与头之间**处理的 token 完全相同**，但因为用的投影矩阵不同，问出的问题不同。

### 2.3 实证：两个头真的会关注不同的东西

下面的例子手工构造了两组不同的 K（键），让同一个 Q（查询）产生截然不同的关注分布：

```python
import math
import torch

torch.manual_seed(0)

# 两个头用完全相同的提问
query = torch.tensor([[1.0, 0.0]])

# 头 A 的"标签"：只有第 0 个位置特别显眼（像"指代头"盯着某个特定 token）
k_global = torch.tensor([[5.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

# 头 B 的"标签"：越靠前越亮（像"局部头"偏好邻近位置）
k_local = torch.tensor([[1.0, 0.0], [0.5, 0.0], [0.33, 0.0], [0.25, 0.0]])

for name, kk in [("global-ish head", k_global), ("local-ish head", k_local)]:
    s = (query @ kk.T) / math.sqrt(2)
    print(name, "weights:", [round(v, 4) for v in s.softmax(dim=-1)[0].tolist()])
```

实际输出：

```text
global-ish head weights: [0.9196, 0.0268, 0.0268, 0.0268]
local-ish head weights: [0.3433, 0.241, 0.2137, 0.202]
```

**读法**：输入完全相同、提问完全相同，只是"标签"不同，就得到了完全不同的关注模式——一个死死盯着位置 0（92%），另一个均匀偏向最近的位置。**这就是"多子空间并行关注"的直观含义。**

### 2.4 三个必须澄清的误解

| 误解 | 真相 |
| --- | --- |
| "多头是多跑几遍取平均" | 不是平均，是**先分工、后融合**：每个头的输出拼起来，再过 $W_O$ |
| "不同头看不同的 token" | 不对。**每个头都看全部 token**（或掩码允许的那部分），只是"关注程度"不同 |
| "头越多越好" | 头数 $H$ 增大时每头维度 $d_{head}=d_{model}/H$ 变小，表达更专门；但头太多、维度太小时反而难以学到复杂模式 |

### 2.5 头数与维度的取舍

- 投影矩阵 $W_Q, W_K, W_V, W_O$ 的大小只和 $d_{model}$ 有关，**与 $H$ 无关**；
- 所以"多头"几乎是**免费**的能力扩展：不加参数，只是把同样的参数解释成更多个子空间；
- 代价是每个头维度更小（$d_{model}=512, H=8$ 时每头只有 64 维）。

---

## 3. 完整维度流转

这一节是本文的骨架。用一个具体例子走完全程：**$B=2$（2 条序列）、$T=5$（每条 5 个 token）、$d_{model}=8$、$H=4$（4 个头）**，于是 $d_{head} = 8/4 = 2$。

### 3.1 形状流转表

| 步骤 | 操作 | 形状 | 说明 |
| --- | --- | --- | --- |
| ① | 输入 $X$ | `(B, T, d_model)` = `(2, 5, 8)` | 每条序列 5 个 token，每个 token 8 维 |
| ② | $Q=XW_Q,\ K=XW_K,\ V=XW_V$ | 各 `(B, T, d_model)` = `(2, 5, 8)` | **一次矩阵乘算完所有头**（关键技巧，见 4.5） |
| ③ | 拆头 `view(B,T,H,d_head).transpose(1,2)` | `(B, H, T, d_head)` = `(2, 4, 5, 2)` | 把 8 维切成 4 组 × 2 维 |
| ④ | $Q_hK_h^\top$ | `(B, H, T, T)` = `(2, 4, 5, 5)` | 头与头之间**互不相干**，各自一张 5×5 分数表 |
| ⑤ | $\div \sqrt{d_{head}}$ | 不变 | 把分数温度拉回健康范围 |
| ⑥ | 加因果掩码 → softmax | `(B, H, T, T)` | 上三角被压成 $-\infty$，softmax 后变 0 |
| ⑦ | $\times V_h$ | `(B, H, T, d_head)` = `(2, 4, 5, 2)` | 每个头得到自己那份"混合答案" |
| ⑧ | 合头 `transpose(1,2).reshape(B,T,d_model)` | `(B, T, d_model)` = `(2, 5, 8)` | 4 个头的 2 维拼回 8 维 |
| ⑨ | 输出投影 $\times W_O$ | `(B, T, d_model)` = `(2, 5, 8)` | 融合各头信息，进入残差 |

**记忆这张表的三条主线**：

- 维度轴：$d_{model}$ → 拆成 $H$ 份 $d_{head}$ → 再拼回 $d_{model}$（**进 → 拆 → 合**）；
- 头轴：拆头时插入（$H$），合头时移除；
- 位置轴 $T$：在第 ④ 步**从一维变成二维**（$T \times T$ 的关注表），这正是注意力"两两比较"的本质。

### 3.2 完整可运行示例

```python
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)

B, T, d_model, H = 2, 5, 8, 4
d_head = d_model // H

x = torch.randn(B, T, d_model)                    # (2, 5, 8)
Wq = torch.randn(d_model, d_model) * 0.1           # 随机权重，只为演示形状
Wk = torch.randn(d_model, d_model) * 0.1
Wv = torch.randn(d_model, d_model) * 0.1
Wo = torch.randn(d_model, d_model) * 0.1

# ② 一次矩阵乘得到 Q/K/V
q = x @ Wq                                        # (2, 5, 8)
k = x @ Wk
v = x @ Wv
print("x", tuple(x.shape), "q", tuple(q.shape))

# ③ 拆头：(B, T, d_model) -> (B, H, T, d_head)
qh = q.view(B, T, H, d_head).transpose(1, 2)      # (2, 4, 5, 2)
kh = k.view(B, T, H, d_head).transpose(1, 2)
vh = v.view(B, T, H, d_head).transpose(1, 2)
print("per-head q", tuple(qh.shape))

# ④ 打分
scores = qh @ kh.transpose(-2, -1)                # (2, 4, 5, 5)
print("scores", tuple(scores.shape))

# ⑤ 缩放
scaled = scores / math.sqrt(d_head)

# ⑥ 因果掩码 + softmax
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
masked = scaled.masked_fill(~causal, float("-inf"))
w = masked.softmax(dim=-1)
print("weights", tuple(w.shape))

# ⑦ 加权取用
ctx = w @ vh                                      # (2, 4, 5, 2)
print("ctx", tuple(ctx.shape))

# ⑧ 合头：(B, H, T, d_head) -> (B, T, d_model)
merged = ctx.transpose(1, 2).reshape(B, T, d_model)
print("merged", tuple(merged.shape))

# ⑨ 输出投影
out = merged @ Wo
print("out", tuple(out.shape))

# 和 PyTorch 内置实现对拍
ref = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=causal)
print("matches pytorch:", torch.allclose(ref, ctx, atol=1e-5))
```

实际输出：

```text
x (2, 5, 8) q (2, 5, 8)
per-head q (2, 4, 5, 2)
scores (2, 4, 5, 5)
weights (2, 4, 5, 5)
ctx (2, 4, 5, 2)
merged (2, 5, 8)
out (2, 5, 8)
matches pytorch: True
```

### 3.3 拆头那两行代码到底做了什么

```python
# 拆头 = 两步：view 切分组，transpose 换轴序
qh = q.view(B, T, H, d_head).transpose(1, 2)
#   view(B, T, H, d_head)   (2,5,8) -> (2,5,4,2)：把最后一维 8 拆成 "4 组 × 每组 2"
#   transpose(1, 2)         (2,5,4,2) -> (2,4,5,2)：把"组"这一维挪到 batch 后面
print(tuple(qh.shape))          # (2, 4, 5, 2)
```

两步各司其职：

- `view` 只改"怎么解释内存里的数据"，不搬运数据，因此几乎不花时间；
- `transpose` 交换两个轴的语义。注意它**不复制**数据，所以后面的矩阵乘能高效处理——真正需要连续内存时会自动触发一次拷贝（contiguous）。

合头是它的逆操作：`transpose(1,2)` 换回来，再 `reshape`（必要时拷贝一次）把 4 个头的 2 维拼成 8 维。

> **新手常见卡点**：`view` 要求内存布局兼容，`reshape` 没有这个限制、更"皮实"。写代码时如果 `view` 报 "shape is invalid for input of size ..."，换成 `reshape` 往往就好了。

---

## 4. 关键技术细节

### 4.1 因果掩码：不让模型偷看未来

**为什么要**：语言模型训练时，位置 $i$ 的标准答案是第 $i+1$ 个词。如果注意力允许位置 0 直接看到位置 5，模型只要把答案抄过来就能拿满分——训练出来的模型在推理时（没有未来可看）就彻底不会用了。

**怎么做**：加一个下三角的布尔矩阵，把"未来"的位置在 softmax **之前**设成 $-\infty$，softmax 之后它们的权重自然变成 0。

```text
        位置0   位置1   位置2   位置3
位置0 [   ✓       ✗       ✗       ✗  ]
位置1 [   ✓       ✓       ✗       ✗  ]
位置2 [   ✓       ✓       ✓       ✗  ]
位置3 [   ✓       ✓       ✓       ✓  ]
```

**三个必须注意的细节**：

1. **掩码加在 softmax 之前**（设 $-\infty$），不能先 softmax 再乘 0——后者有 `0 × inf = NaN` 的风险，而且被屏蔽位置仍然参与了分母归一化；
2. **布尔语义要看具体 API**：讲义里 `True` 表示"**允许注意**"；`F.scaled_dot_product_attention` 的布尔 `attn_mask` 也是 `True` = 参与（同向，可原样传），但 `nn.MultiheadAttention` 的 `attn_mask` 是 `True` = 屏蔽（反向）。换 API 前先确认——这类错误不报错、只错数；
3. **整行被屏蔽会出 NaN**（分母为 0）。因果掩码不会出现（位置 0 至少能看到自己），但处理 padding 时要注意。

**一个非常好用的自检方法——前缀不变性**：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
T = 5
q = torch.randn(1, T, 8)
k = torch.randn(1, T, 8)
v = torch.randn(1, T, 8)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))

out_a = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)

# 把最后一个 token 的内容改得很大
v2 = v.clone()
v2[:, 3] += 50.0
out_b = F.scaled_dot_product_attention(q, k, v2, attn_mask=causal)

print("prefix unchanged:", torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-6))
print("last position changed:", not torch.allclose(out_a[:, 3], out_b[:, 3], atol=1e-6))
```

实际输出：

```text
prefix unchanged: True
last position changed: True
```

**怎么用**：改动后面的 token，前面的输出必须**纹丝不动**。如果前面也变了，说明掩码没生效或生效方向反了。

### 4.2 缩放点积：给 softmax 调温度

分数 $q \cdot k$ 的标准差随维度增长（$\text{std}=\sqrt{d_k}$）。不缩放的话，分数一大，softmax 就饱和成 one-hot，softmax 雅可比里的 $p(1-p)$ 趋于 0，**梯度回传的通道被压扁**，训练停滞。

所以要除以 $\sqrt{d_{head}}$：把分数方差拉回 1，让 softmax 的输入永远处在"既能分辨、又不极端"的温度上。

> 这里除的是 **head 维度 $d_{head} = d_{model}/H$**，不是 $d_{model}$——这是多头实现里最容易写错的地方。详细的方差推导和实测数据见 `Attention.md` 第 3 节。

### 4.3 RoPE：是什么，为什么需要

#### 4.3.1 什么是 RoPE

RoPE 全称 Rotary Position Embedding，中文叫**旋转位置编码**，出自《RoFormer》（Su 等，2021）。

一句话概括：**不往向量上"加"位置，而是按位置把 q、k 转一个角度。**

传统的位置编码是"加法"——造一个和位置有关的向量，加到输入上。RoPE 换了个思路，改成"乘法"：

```text
绝对位置嵌入：  x' = x + p[i]        加法，位置信息直接掺进向量里
RoPE：          q' = R(i) · q        乘法，按位置把 q 转一个角度
```

$R(i)$ 是"转过 $i\cdot\theta$ 角度"的旋转矩阵。关键差别在于：**加法是改内容，旋转是改方向**。方向改了，但长度不变；而"方向差"恰好可以用来表达"位置差"。

三种常见方案对比：

| 方案 | 位置信息怎么进模型 | 优点 | 短板 |
| --- | --- | --- | --- |
| 正弦绝对编码 | 加到输入上，数值固定不训练 | 不用学参数，能外推一些 | 相对距离要模型自己算 |
| 可学习绝对嵌入 | 加到输入上，每个位置一个可学向量 | 简单直接 | 超出训练长度直接失效 |
| **RoPE** | 旋转 q、k | 天然表达相对距离、不改模长、不污染词向量 | 超长外推仍需技巧（插值、NTK 等） |

现在主流开源模型（LLaMA、Qwen、Mistral…）基本都用 RoPE，本作业也用它。

#### 4.3.2 为什么需要 RoPE

**先看一个事实：注意力本身对顺序是"瞎"的。**

同样的几个 token 换个顺序喂进去，输出只是跟着换个顺序，数值一点没变：

```python
import torch

torch.manual_seed(0)
T, d = 4, 8
x = torch.randn(T, d)          # 4 个 token，每个 8 维
q = k = v = x                  # 先不管投影，只看注意力本身

def attn(q, k, v):
    w = (q @ k.T / (d ** 0.5)).softmax(-1)
    return w @ v

out = attn(q, k, v)
perm = torch.tensor([2, 0, 3, 1])              # 换一种词序
out_perm = attn(q[perm], k[perm], v[perm])
print("输出只是跟着重排:", torch.allclose(out[perm], out_perm, atol=1e-6))
```

实际输出：

```text
输出只是跟着重排: True
```

"猫追狗"和"狗追猫"在纯注意力眼里是同一袋词。**所以位置信息必须从外部注入。**

**那因果掩码管不管用？** 不管。掩码只回答"能不能看右边"，不回答"左边那个隔多远"。而语言里距离本身就是信息：`dogs` 和 `dog` 差一个 `s`，`the cat` 和 `cat the` 词一样、意思全变。

**那为什么不用"加一个位置向量"就行？** 三个原因：

1. **只认绝对位置**：训练时只见过 $0 \sim T-1$，推理遇到第 $T+1$ 个位置就没见过，表现明显下降；
2. **和内容搅在一起**：位置向量加到词向量上，之后每层线性变换都会把两者混着改，位置信号容易被内容淹没；
3. **相对距离要靠模型自己推**：语言真正需要的是"隔了 3 个词"这种关系，绝对编码得让模型自己学会"把两个位置相减"。

**RoPE 的解法**：不动输入向量，只在算注意力之前把 q、k 转一下。转角由位置决定，于是点积结果天然只依赖相对距离（推导在 4.3.3）。附带两个好处：

- 旋转是正交变换，**不改模长**，原本的相似度尺度不受影响；
- 位置只出现在注意力内部，**不污染词向量本身**。

#### 4.3.3 怎么旋，以及为什么就"相对"了

**做法**：把 $q$、$k$ 当成平面上的箭头，**按各自所在的位置旋转一个角度**（位置越靠后转得越多），再算它们的点积。

**为什么这样就编码了相对位置**？因为两个箭头旋转之后的点积，只取决于**它们转过的角度差**，也就是位置差：

$$
(R(m_i)q)\cdot(R(m_j)k) = q^\top R(m_j - m_i)\,k
$$

于是"距离"被自动写进了分数里。

下面用最小的二维旋转验证这个性质：

```python
import torch

torch.manual_seed(0)
d = 4
theta = 10000.0
freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))

def rope(x, pos):
    """最简 RoPE：把每相邻两维当作复数，按 pos*freq 旋转。"""
    x1, x2 = x[:, 0::2], x[:, 1::2]
    ang = pos.float().unsqueeze(-1) * freqs.unsqueeze(0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    out = torch.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out

q0 = torch.randn(1, d)      # 固定的一支"箭"
k0 = torch.randn(1, d)      # 固定的一支"箭"
T = 6
by_offset = {}
for i in range(T):
    for j in range(T):
        s = float((rope(q0, torch.tensor([i])) * rope(k0, torch.tensor([j]))).sum())
        by_offset.setdefault(j - i, []).append(s)

for off in sorted(by_offset):
    vals = by_offset[off]
    print(f"offset {off:+d}: n={len(vals)} spread={max(vals)-min(vals):.2e}")
```

实际输出：

```text
offset -5: n=1 spread=0.00e+00
offset -4: n=2 spread=0.00e+00
offset -3: n=3 spread=1.49e-07
offset -2: n=4 spread=2.38e-07
offset -1: n=5 spread=4.77e-07
offset +0: n=6 spread=2.38e-07
offset +1: n=5 spread=3.58e-07
offset +2: n=4 spread=4.77e-07
offset +3: n=3 spread=2.38e-07
offset +4: n=2 spread=3.58e-07
offset +5: n=1 spread=0.00e+00
```

**怎么读**：同一个相对距离（offset）下，分数的差异只有 $10^{-7}$ 量级——**就是 float32 的舍入误差**。也就是说：**分数只取决于两者的相对距离，与它们具体在哪无关。** 如果不做旋转（把 `rope` 换成恒等映射），所有距离的分数都完全一样，等于完全没有位置信息。

**两条必须记住的规则**：

1. **RoPE 只作用于 Q 和 K，不作用于 V**。V 是"要被取走的内容"，旋转它没有语义收益，还会打乱数值；
2. **RoPE 的旋转维度是 head 维度** $d_{head} = d_{model}/H$（作业讲义明确要求），不是 $d_{model}$。因为多头是先拆后转的，每个头各自转自己的 $d_{head}$ 维。

**分工总结**（这张表值得背下来）：

| 机制 | 回答的问题 | 作用位置 |
| --- | --- | --- |
| **因果掩码** | "我能不能看那个位置？" | softmax 之前的分数上 |
| **RoPE** | "那个位置离我多远？" | Q、K 的旋转 |
| **缩放因子** | "softmax 的温度合适吗？" | 分数上 |

三者缺一不可：**掩码管"能不能看"，RoPE 管"隔多远"，缩放管"看得清不清"。**

### 4.4 多头 + RoPE：一步步接进去

4.3 说的是"是什么、为什么"，这一节走一遍"多个头一起用"的完整做法。

#### 4.4.1 旋在哪一步

一句话：**投影之后、拆头之后、打分之前。**

```text
x  (B, T, D)
├── @ W_q / W_k / W_v         → q, k, v   (B, T, D)
├── 拆头                       → (B, H, T, d_head)
├── ★ 对 q、k 逐位置旋转（v 不动）
├── 打分 softmax(q kᵀ / √d_head + mask)
├── 加权求和 × v               → (B, H, T, d_head)
└── 合头 + W_o                 → (B, T, D)
```

顺序不能调，两个原因：

- **必须在拆头之后**：旋转是按"维度对"做的，拆完头每个头才有独立的 $d_{head}$ 维，每一对维度对应一个频率；
- **必须在打分之前**：RoPE 的全部意义就是改变 $q\cdot k$ 的结果，放到 softmax 后面等于白旋。

#### 4.4.2 频率表：每个维度对一个转速

$d_{head}$ 维被切成 $d_{head}/2$ 对，每对配一个频率：

$$
\theta_i = \Theta^{-2i/d_{head}}, \quad i = 0, 1, \dots, \frac{d_{head}}{2}-1
$$

位置 $m$ 的某一对维度，转过的角度是 $m\cdot\theta_i$。以 $d_{head}=8,\ \Theta=10000$ 为例：

| i | 频率 $\theta_i$ | 转一圈要几个位置 | 管什么 |
| --- | --- | --- | --- |
| 0 | 1.0 | ≈ 6.3 | 转得最快，只在紧邻的几个位置上有区分度 |
| 1 | 0.1 | ≈ 63 | 中等 |
| 2 | 0.01 | ≈ 628 | 较慢 |
| 3 | 0.00316 | ≈ 1987 | 最慢，负责远距离 |

这就是"低维管近距离、高维管远距离"的由来。另外两点值得记住：

- 所有头维数相同，所以 `cos` / `sin` 表**所有头共用一份**；
- 表里的值完全由位置和频率决定，**不需要训练**，构造时算一次存成 buffer 就够了。

#### 4.4.3 旋转怎么写

把相邻两维当成平面上的一支箭头，转 $m\theta_k$ 角度：

$$
\begin{pmatrix} x'_{2k} \\ x'_{2k+1} \end{pmatrix}
=
\begin{pmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{pmatrix}
\begin{pmatrix} x_{2k} \\ x_{2k+1} \end{pmatrix}
$$

旋转是正交变换，所以 $\|q'\| = \|q\|$——**RoPE 不会改变分数的量级**，只是把方向按位置拧了一下。

#### 4.4.4 配对方式必须和参考实现一致（实测对比）

同样叫 RoPE，维度怎么配对至少有两种写法：

| 写法 | 配成一对的维度 | 常见出处 |
| --- | --- | --- |
| 相邻配对 | (0,1)、(2,3)、(4,5) … | GPT-J、**本作业的参考实现** |
| 前后半配对 | (0, d/2)、(1, d/2+1) … | Llama / GPT-NeoX 的常见写法 |

两种都合法，但**数值完全不同**。拿 `test_rope.npz` 对拍的结果：

```text
相邻两维配对 (0,1)(2,3)        max|diff| = 0.000000   ← 作业用的就是这种
前后半配对  (0,d/2)(1,d/2+1)   max|diff| = 5.376348
```

**遇到"形状全对、数值全错"，先查这一条。**

#### 4.4.5 可运行示例

```python
import torch

torch.manual_seed(0)

def build_rope_cache(max_seq_len, d_head, theta=10000.0):
    # i 取 0,2,4,...：每一对维度对应一个频率 θ_i = theta^(-i/d)
    i = torch.arange(0, d_head, 2).float()
    inv_freq = theta ** (-i / d_head)
    # 位置 m 转过的角度 = m * θ_i，所有位置 × 所有频率一次算完
    ang = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
    return torch.cos(ang), torch.sin(ang)

def apply_rope(x, cos, sin):
    # x: (..., T, d_head)；相邻两维 (0,1)、(2,3)… 配成一对
    T, half = x.shape[-2], x.shape[-1] // 2
    shape = (1,) * (x.dim() - 2) + (T, half)   # 让 cos/sin 能广播到任意前置维
    c, s = cos[:T].view(shape), sin[:T].view(shape)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out

H, T, d_head = 2, 6, 8
cos, sin = build_rope_cache(max_seq_len=16, d_head=d_head)
q, k = torch.randn(H, T, d_head), torch.randn(H, T, d_head)

qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
print("形状不变:", tuple(qr.shape) == tuple(q.shape))
print("模长不变:", torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5))
print("位置0等价于不旋转:", torch.allclose(qr[:, 0], q[:, 0], atol=1e-6))

# 固定一支 q、一支 k，放到不同位置：分数应当只跟相对距离有关
bq, bk = torch.randn(d_head), torch.randn(d_head)

def rotate_at(vec, pos):
    out = torch.empty_like(vec)
    c, s = cos[pos], sin[pos]
    out[0::2] = vec[0::2] * c - vec[1::2] * s
    out[1::2] = vec[0::2] * s + vec[1::2] * c
    return out

for off in (-2, 0, 2):
    vals = [float((rotate_at(bq, i) * rotate_at(bk, i + off)).sum())
            for i in range(T) if 0 <= i + off < T]
    print(f"offset {off:+d}: n={len(vals)} spread={max(vals) - min(vals):.2e}")
```

实际输出：

```text
形状不变: True
模长不变: True
位置0等价于不旋转: True
offset -2: n=4 spread=2.38e-07
offset +0: n=6 spread=4.77e-07
offset +2: n=4 spread=2.38e-07
```

四条输出分别说明什么：

- **形状不变**：RoPE 是"原地换坐标"，张量的形状一个都没变；
- **模长不变**：正交变换保长度，所以分数的量级不会被 RoPE 打乱；
- **位置 0 等价于不旋转**：$R(0)$ 是单位矩阵，位置 0 天然没有偏移；
- **spread 只有 $10^{-7}$**：固定一对向量放到任意位置，只要相对距离相同，分数就相同——这就是"相对位置编码"的含义。

#### 4.4.6 `token_positions` 到底有什么用

训练时位置就是 $0,1,\dots,T-1$，看着这个参数可有可无。推理时就不一样了：

```text
已经生成 100 个 token，现在要预测第 101 个
这个新 token 的位置是 100，不是 0
```

用 KV cache 时每步只送新 token 进模型，如果位置总从 0 数起，模型会以为新词在句子开头，算出来的"距离"全是错的。所以接口留了 `token_positions`：默认取 `arange(T)`，需要时可以外部传入。

#### 4.4.7 对应到两个接口

| 函数 | 要做的事 |
| --- | --- |
| `run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions)` | 对单个张量旋转：输入 `(..., T, d_k)`，输出同形状。这里的 `d_k` 就是 `d_head` |
| `run_multihead_self_attention_with_rope(...)` | 投影 → 拆头 → 旋转 q、k → 打分 → 合头；旋转维度是 `d_model // n_heads` |

`run_rope` 可能收到 `(B, T, D)`、`(B, H, T, d_head)` 等任意前置维，实现里别假设只有 3 维。

#### 4.4.8 六个坑

1. **把 V 也旋了**：V 是"要被取走的内容"，旋转没有收益，还会搅乱数值；
2. **旋转维度用了 `d_model`**：应该是 `head_dim`，否则频率表长度都对不上；
3. **位置用了浮点**：索引和角度的外层乘法都要整数，`torch.arange(..., dtype=torch.long)`；
4. **配对方式写成了 Llama 那种前后半**：形状对、数值全错（见 4.4.4）；
5. **把 `cos` / `sin` 当成可训练参数**：值是固定的，注册成 buffer 即可，别放进优化器；
6. **buffer 只算到训练长度**：推理序列更长时索引越界，需要能按需扩展。

### 4.5 为什么讲义要求"一次矩阵乘"

拆头之前，$Q = XW_Q$ 一次就算完了所有头的数据（$W_Q$ 的行按 $(H, d_{head})$ 排布），之后只是 `view` + `transpose` 重新解释内存。

如果写成循环：

```python
# ❌ 慢：H 次小矩阵乘，每次都要等 kernel 启动
for h in range(H):
    q_h = x @ Wq[h * d_head:(h + 1) * d_head]
```

那么 H 个头就是 H 次独立的 GPU 任务，**kernel 启动开销**可能比计算本身还大。作业讲义 3.4.5 节明确要求"所有头的投影用单个矩阵乘完成"，就是为了这个。

---

## 5. 对应到 CS336 Assignment 1

### 5.1 你要实现的三个接口

| 适配器函数（`tests/adapters.py`） | 形状要求 | 备注 |
| --- | --- | --- |
| `run_scaled_dot_product_attention(Q, K, V, mask=None)` | 支持任意前置 batch 维：`(..., T, d_k)` 和 `(B, H, T, d_k)` 都要能跑 | 讲义约定：`mask` 里 `True` = **允许注意** |
| `run_multihead_self_attention(...)` | 输入 `(B, T, d_model)` → 输出 `(B, T, d_model)` | Q/K/V 投影用**一次矩阵乘**；不带 RoPE |
| `run_multihead_self_attention_with_rope(...)` | 同上 | Q/K 施加 RoPE，**旋转维度是 $d_{model} // n_{heads}$** |

### 5.2 参考实现的权重布局

作业给的参考权重（`state_dict`）：

| 权重 | 形状 | 行如何排列 |
| --- | --- | --- |
| `layers.0.attn.q_proj.weight` | `(d_model, d_model)` | 前 $d_{head}$ 行是头 0，接着是头 1，以此类推 |
| `layers.0.attn.k_proj.weight` | `(d_model, d_model)` | 同上 |
| `layers.0.attn.v_proj.weight` | `(d_model, d_model)` | 同上（输出维度是 $d_v$） |
| `layers.0.attn.output.weight` | `(d_model, d_model)` | 合头之后的融合投影 |

**推论**：既然行是按 $(H, d_{head})$ 排的，那么拆头就是"把最后一维 reshape 成 `(H, d_head)` 再转置"，与 3.2 节演示的完全一致。

### 5.3 RoPE 的两个参数

| 参数 | 含义 |
| --- | --- |
| `theta` | 旋转频率的基数（Llama 系列用 10000）。频率随维度递减：**低维管近距离，高维管远距离** |
| `max_seq_len` | 预计算 sin/cos buffer 的长度 = 支持的最大上下文长度。因为 sin/cos 值是固定的，**不需要训练**，只是提前算好缓存 |

### 5.4 自检清单

- [ ] 4D 输入时，头维度是 $d_{model} // n_{heads}$，不是 $d_{model}$；
- [ ] 缩放因子除的是 $\sqrt{d_{head}}$（head 维度）；
- [ ] 掩码语义是"讲义约定"（`True` = 允许），并确认所用 API 同向（`F.scaled_dot_product_attention` 同向，`nn.MultiheadAttention` 反向）；
- [ ] 掩码加在 softmax **之前**，用 $-\infty$；
- [ ] 被屏蔽位置的权重**精确等于 0**；
- [ ] 每行权重加起来 = 1（未被整行屏蔽时）；
- [ ] 前缀不变性成立：改后面的 token，前面输出不变；
- [ ] RoPE 只作用在 Q、K 上；
- [ ] RoPE 的旋转维度是 head 维度；
- [ ] RoPE 的维度配对方式和参考一致（相邻两维配对，不是前后半配对）；
- [ ] Q/K/V 投影是一次矩阵乘，而不是循环 H 次；
- [ ] 和 `F.scaled_dot_product_attention` 在相同输入下数值一致（测试用 numpy snapshot，容差 `atol=1e-5`）。

### 5.5 常见错误速查

| 症状 | 多半是 |
| --- | --- |
| 形状对但数值差一点点 | 漏了 $1/\sqrt{d_{head}}$ |
| 上三角有微小非零 | 掩码用乘 0 而非 $-\infty$，或语义反了 |
| 出现 `nan` | 某行被整行屏蔽；或 softmax 没做减最大值稳定处理 |
| 4D 结果错乱 | `transpose` 前后维度错位；或把 $d_{model}$ 当成了 $d_{head}$ |
| 权重矩阵在 RoPE 之后不再是"距离相关" | RoPE 作用到了 V 上，或旋转维度用错 |
| RoPE 形状全对、数值全错 | 维度配对写成了前后半（Llama 写法），或频率用了 `theta^(-2i/d_model)` |
| 训练极慢 | 用了循环逐头计算，而不是一次矩阵乘 |

---

## 6. 小结

- **自注意力**：Q、K、V 都来自同一段序列，每个位置按相关度"取用"其他位置的信息。
- **多头**：把 $d_{model}$ 拆成 $H$ 份，每份一个独立的子空间，各自学一套关注方式（有的盯远距离指代，有的只看邻居），最后拼回并用 $W_O$ 融合——**不加参数的能力扩展**。
- **形状流转**：`(B,T,D)` → 拆头 `(B,H,T,D/H)` → 注意力表 `(B,H,T,T)` → 上下文 `(B,H,T,D/H)` → 合头 `(B,T,D)` → $W_O$ → `(B,T,D)`。
- **因果掩码**：softmax 之前把未来设成 $-\infty$；讲义与 `F.scaled_dot_product_attention` 都是 `True` = 允许注意（只有 `nn.MultiheadAttention` 相反）。
- **缩放因子**：$1/\sqrt{d_{head}}$ 把 softmax 温度固定在健康范围，防止饱和与梯度消失。
- **RoPE**：旋转 Q/K（不旋 V），让分数只依赖**相对距离**；与掩码分工——掩码管"能不能看"，RoPE 管"隔多远"。

---

## 7. 常见问题 FAQ

**Q1：为什么叫"自"注意力？**
因为 Q、K、V 都来自同一个序列，序列在"自己看自己"。若 K、V 来自另一个序列（例如解码器看编码器），就是交叉注意力。

**Q2：头数 $H$ 会影响参数量吗？**
不会。$W_Q/W_K/W_V/W_O$ 的规模只取决于 $d_{model}$，与 $H$ 无关。$H$ 变大只是把同样的参数解释成更多个子空间。

**Q3：多个头是同时算的吗？**
在 GPU 上是的——它们被塞进同一个批量矩阵乘法里并行执行。拆头只是重排内存视图，$H$ 个头共享同一次矩阵乘的代价。

**Q4：为什么注意力表是 $T \times T$？**
因为每个位置都要和其他**每一个**位置算一次相关性。这正是注意力 $O(T^2)$ 复杂度的来源，也是长上下文昂贵的根本原因。

**Q5：因果掩码和位置编码能互相替代吗？**
不能。掩码管"能不能看右边"，RoPE 管"左边隔多远"，缺一不可。去掉 RoPE，模型依然看不到未来，但分不清远近。

**Q6：RoPE 会不会破坏注意力权重的可解释性？**
RoPE 只是旋转 Q/K 的坐标，注意力权重矩阵本身的意义（谁关注谁）不变；只是解读"为什么关注"时要考虑相对距离因素。

**Q7：注意力头数怎么选？**
常见配置是 $d_{model}$ 的 1/64 到 1/32（如 512 维配 8 或 16 头）。工程上优先保证 $d_{model}$ 能被 $H$ 整除。

**Q8：多头和"多个 Transformer 层"是一回事吗？**
不是。层是**串行**的：后一层看的是前一层加工过的结果。头是**并行**的：同一层里几个头看的是同一份输入。两者组合起来才形成"多层 × 多头"的结构。

**Q9：为什么注意力要除以 $\sqrt{d_{head}}$ 而不是 $\sqrt{d_{model}}$？**
因为拆头之后每个头独立做注意力，打分用的是头内部的 $d_{head}$ 维向量，点积的方差随 $d_{head}$ 增长，所以缩放因子要跟着头维度走。

**Q10：合并各头之后，为什么还要 $W_O$？**
因为各头的输出是"用不同坐标系记的笔记"，直接拼起来语义混乱。$W_O$ 负责把它们重新混合到同一个坐标系里。

---

## 8. 动手练习

1. **改形状跑一遍**：把 3.2 节示例里的 $B,T,d_{model},H$ 改成 $(3, 7, 16, 4)$，先预测每一步的形状，再跑一遍核对。
2. **验证"每个头都看全部 token"**：打印不同头的权重矩阵，确认它们的形状相同、且非零位置都覆盖整段序列（只是数值不同）。
3. **掩码语义实验**：用讲义约定（`True`=允许）造一个 mask，直接传给 `F.scaled_dot_product_attention`，观察结果与预期相反；再传 `~mask` 验证恢复。
4. **前缀不变性**：写循环，依次篡改每个位置的 V，验证位置 $i$ 的输出只受位置 $\le i$ 影响。
5. **RoPE 相对性**：把 4.3.3 节代码里的 `rope` 改成恒等映射，重跑对比，解释"没有位置信息"意味着什么。
6. **头数对比**：固定 $d_{model}=8$，分别用 $H=1,2,4,8$，观察每头维度与注意力表大小的变化，并计算 4 种情况下注意力矩阵占多少显存。
7. **自检脚本**：把 5.4 的清单逐条写成断言，改一处代码让它失败，确认断言真的能抓到问题。

---

*延伸阅读：同目录的 `Attention.md`（Attention 通用原理与公式推导）、`Transformer 模块.md`（Block 整体结构）、`Embedding_Linear.md`（$W_Q/W_K/W_V/W_O$ 这些投影矩阵的来源）。*
