# SwiGLU：从零理解 Transformer 的前馈网络

> 本文是 CS336 Assignment 1（Building a Transformer LM）的配套概念读物，对应讲义 §3.4.2 的 `positionwise_feedforward`。它不替你做作业，而是把 SwiGLU **是什么、为什么需要、数学上怎么来的、维度怎么定、代码怎么写** 这条链路讲清楚。
>
> 读完你应该能回答五个问题：
>
> 1. 为什么 Transformer 不能只有注意力，还必须有一个前馈网络？
> 2. SiLU 到底是什么函数？为什么它的负区间不是 0，而是有一个小凹陷？
> 3. SwiGLU 里的“门控”到底在门控什么？为什么不是简单地换个激活函数？
> 4. $d_{ff}=\frac{8}{3}d_{model}$ 这个看起来奇怪的比例是怎么来的？
> 5. 在 CS336 的测试里，`run_swiglu` 的三个权重矩阵分别是什么形状、为什么要转置？

## 目录

- [0. 阅读前的准备](#0-阅读前的准备)
- [1. SwiGLU 是什么](#1-swiglu-是什么)
- [2. 作用与适用场景](#2-作用与适用场景)
- [3. 为什么 Transformer 需要 FFN](#3-为什么-transformer-需要-ffn)
- [4. 数学原理与推导](#4-数学原理与推导)
- [5. 参数规模与维度设计](#5-参数规模与维度设计)
- [6. CS336 Assignment 1 中的 SwiGLU](#6-cs336-assignment-1-中的-swiglu)
- [7. 可运行代码](#7-可运行代码)
- [8. 常见误区 FAQ](#8-常见误区-faq)
- [9. 小结与自测](#9-小结与自测)
- [10. 延伸阅读](#10-延伸阅读)

---

## 0. 阅读前的准备

### 0.1 一句话版本

SwiGLU 是 Transformer 中**前馈网络（FFN）**的一种现代写法：

> 先把每个 token 的向量**升维**到更宽的 $d_{ff}$，分成“内容”和“门控”两条支路；门控支路经过 **SiLU** 激活后，与内容支路**逐元素相乘**；最后再用一个线性层**降维**回 $d_{model}$。

写成一行公式：

$$
\operatorname{SwiGLU}(x)=W_2\Big(\operatorname{SiLU}(W_1x)\odot(W_3x)\Big)
$$

其中 $\odot$ 是逐元素相乘。一句话记忆：

> **SiLU 负责“软化”，GLU 负责“门控”，$W_2$ 负责“收回”。**

### 0.2 符号表

| 符号 | 含义 | 一句话解释 |
| --- | --- | --- |
| $B$ | batch size | 一次同时处理多少条序列 |
| $T$ | sequence length | 一条序列有多少个 token |
| $D$ 或 $d_{model}$ | 模型隐藏维度 | 每个 token 用多少个数表示 |
| $d_{ff}$ | FFN 中间层维度 | SwiGLU 内部先升到多宽 |
| $x$ | 单个位置的向量 | $x\in\mathbb{R}^{D}$ |
| $X$ | 一批向量 | $X\in\mathbb{R}^{T\times D}$（省略 batch 维） |
| $W_1,W_3$ | 升维矩阵 | 形状都是 $d_{ff}\times D$ |
| $W_2$ | 降维矩阵 | 形状是 $D\times d_{ff}$ |
| $\sigma$ | sigmoid | $\sigma(z)=\frac{1}{1+e^{-z}}$ |
| $\odot$ | 逐元素乘 | 两个同形状张量对应位置相乘 |

> 本文公式里用 $D$ 和 $d_{model}$ 混用，它们指的是同一件事；代码里统一写 `d_model`。

### 0.3 你需要的前置知识

不需要会推注意力，但下面几件事要熟：

1. **线性层**：$y=xW^\top$，输入最后一维是 `in_features`，输出最后一维是 `out_features`；
2. **矩阵形状**：$(T,D)\times(D,d_{ff})\to(T,d_{ff})$，只有相邻的维度能乘；
3. **逐元素运算**：两个 $(T,d_{ff})$ 的张量可以逐位置相乘，形状不变；
4. **链式法则**：反向传播就是沿着计算图把导数一层层乘回去；
5. **非线性**：没有激活函数，多层线性层叠起来仍然等价于一层线性层。

### 0.4 SwiGLU 在整张图里的位置

```text
        x  (B, T, D)
        │
        ├──► RMSNorm ──► 多头自注意力 ──► (+) ──► y
        │                                  ▲
        │                                  └── 残差
        │
        ├──► RMSNorm ──► SwiGLU FFN ─────► (+) ──► z
        │                                  ▲
        │                                  └── 残差
        │
        └── 重复 N 层
```

三个关键事实：

- **形状不变**：输入 $(B,T,D)$，输出还是 $(B,T,D)$，所以能直接加回残差；
- **逐位置**：位置 $t$ 的 FFN 只看到位置 $t$ 的向量，不会看别的 token；
- **在注意力之后**：先由注意力跨位置交换信息，再由 FFN 在每个位置内部做非线性加工。

> 类比：注意力是“开会时听谁发言”，FFN 是“散会后各自写心得”。会议是全场共享的，心得是个人的。

### 0.5 阅读路线

| 你想知道 | 看哪节 |
| --- | --- |
| 它长什么样、名字怎么来的 | §1 |
| 它能干什么、什么时候用 | §2 |
| 为什么非要有 FFN | §3 |
| SiLU 函数本身（定义、图像、导数） | §4.4 |
| 公式怎么推、梯度怎么流 | §4 |
| $d_{ff}$ 为什么是 $8/3D$ | §5 |
| 作业接口和测试怎么过 | §6 |
| 直接跑代码 | §7 |
| 常见坑 | §8、§9 |

---

## 1. SwiGLU 是什么

### 1.1 名字拆开看：Swish + GLU

`SwiGLU` 是 **Swi**sh（也就是 SiLU）和 **GLU**（Gated Linear Unit）的拼接：

| 部件 | 全称 | 公式 | 作用 |
| --- | --- | --- | --- |
| Swish / SiLU | Sigmoid-weighted Linear Unit | $\operatorname{SiLU}(x)=x\sigma(x)$ | 平滑的非线性激活 |
| GLU | Gated Linear Unit | $\operatorname{GLU}(x)=\sigma(W_1x)\odot(W_2x)$ | 用一路信号去“门控”另一路 |
| SwiGLU | Swish-Gated Linear Unit | $W_2(\operatorname{SiLU}(W_1x)\odot W_3x)$ | 把 GLU 的门换成 SiLU，再降维 |

所以它**不是**一个单纯的激活函数，而是一个**带门控的前馈层**。作业讲义里有时把它归到“activation function”那一节讲，是行文方便；严格说，SiLU 才是激活函数，SwiGLU 是层。

> 名字里的 “Swish” 和 “SiLU” 是同一个函数：Swish 一般写成 $x\sigma(\beta x)$，当 $\beta=1$ 时就是 SiLU。SiLU 的形状、导数和数值稳定性会在 §4.4 单独展开；这里先记住它是 $x\sigma(x)$。

### 1.2 公式与形状

对单个 token 的向量 $x\in\mathbb{R}^{D}$：

$$
\begin{aligned}
a &= W_1x \quad &\text{形状 } d_{ff} \\
b &= W_3x \quad &\text{形状 } d_{ff} \\
g &= \operatorname{SiLU}(a) \quad &\text{形状 } d_{ff} \\
h &= g\odot b \quad &\text{形状 } d_{ff} \\
y &= W_2h \quad &\text{形状 } D
\end{aligned}
$$

矩阵形状：

$$
W_1,W_3\in\mathbb{R}^{d_{ff}\times D},\qquad W_2\in\mathbb{R}^{D\times d_{ff}}
$$

> **实现时最容易错的地方**：数学上写 $W_1x$，但 PyTorch 的 `nn.Linear` 把权重存成 `(out_features, in_features)`，所以代码里是 `x @ W.T`，不是 `W @ x`。这一点在 §6 会反复强调。

### 1.3 三步数据流（带形状）

把 batch 和序列维都省略，只看一个 token：

```text
x: (D)
 │
 ├── W1: (d_ff, D) ──► a: (d_ff) ──► SiLU ──► g: (d_ff)   ← 门控信号
 │
 ├── W3: (d_ff, D) ──► b: (d_ff)                          ← 内容
 │
 └── g ⊙ b ──► h: (d_ff)
                 │
                 └── W2: (D, d_ff) ──► y: (D)
```

如果带 batch 和序列维，把每个 `(D)` 换成 `(B, T, D)`、每个 `(d_ff)` 换成 `(B, T, d_ff)` 即可。**所有线性层都是逐位置作用的**，不会把不同 token 混在一起。

### 1.4 和经典 FFN 的对比

| | 经典 FFN（ReLU/GELU） | SwiGLU FFN |
| --- | --- | --- |
| 公式 | $W_2\,\sigma(W_1x)$ | $W_2(\operatorname{SiLU}(W_1x)\odot W_3x)$ |
| 线性层数 | 2 | 3 |
| 中间宽度 | 通常 $4D$ | 通常 $\frac{8}{3}D$ |
| 参数量 | $8D^2$ | $3D\cdot\frac{8}{3}D=8D^2$ |
| 非线性 | 单路激活 | 激活 + 乘性门控 |
| 直觉 | “升维 → 激活 → 降维” | “升维 → 一路当门、一路当内容 → 相乘 → 降维” |

在**等参数**的前提下，SwiGLU 用“多一个矩阵、窄一点”换来了“门控”这种更灵活的非线性。这是理解它为什么有效、以及为什么 $d_{ff}$ 要取 $8/3D$ 的关键。

### 1.5 一个通道的数值直觉

先只看隐藏层的第 $i$ 个通道。设门控输入 $a$、内容输入 $b$，输出是 $\operatorname{SiLU}(a)\cdot b$。下表是 $\operatorname{SiLU}(a)$ 的实际数值（由 §7 的脚本算出）：

| $a$ | $\operatorname{SiLU}(a)$ | 对内容 $b$ 的影响 |
| ---: | ---: | --- |
| -3.000 | -0.1423 | 负向通过约 14%，且符号翻转 |
| -2.000 | -0.2384 | 小幅负向通过 |
| -1.278 | -0.2785 | 负向最强（SiLU 的最小值附近） |
| -1.000 | -0.2689 | 负向通过 |
| 0.000 | 0.0000 | 完全关断 |
| 1.000 | 0.7311 | 通过约 73% |
| 2.000 | 1.7616 | 放大到 1.76 倍 |
| 3.000 | 2.8577 | 放大到 2.86 倍 |

三个值得注意的点：

1. **门不是 0/1 开关**：它可以是负数，也可以大于 1；
2. **门是输入相关的**：$a=W_1x$ 随 $x$ 变化，所以同一个通道在不同 token 上可以“开”或“关”；
3. **门和内容来自不同的线性变换**：$a$ 和 $b$ 是两个独立的投影，模型可以学出“什么时候该让这个特征通过”。

> 如果 $b>0$ 而门控为负，输出会变成负的——这不是 bug，而是乘性门控允许的“抑制/反转”行为。模型可以选择不用这种模式，但结构上给了它这个自由度。

### 1.6 三个“不是”

| 说法 | 纠正 |
| --- | --- |
| “SwiGLU 是一个激活函数” | 严格说是**带门控的前馈层**；SiLU 才是激活函数 |
| “SwiGLU 是一种注意力” | 它**不跨位置**，每个 token 独立计算，不交换 token 间信息 |
| “SwiGLU 就是把维度变大再变小” | 升维降维只是骨架，**逐元素门控**才是它与经典 FFN 的本质区别 |

---

## 2. 作用与适用场景

### 2.1 它到底做了什么

把 SwiGLU 的作用拆成三层：

1. **逐位置的非线性变换（channel mixing）**
   线性层只做加权求和，激活函数引入非线性；SwiGLU 让每个位置的 $D$ 维向量在更宽的 $d_{ff}$ 维空间里做非线性组合，再压回 $D$ 维。

2. **输入相关的门控（gating / feature selection）**
   $\operatorname{SiLU}(W_1x)$ 是一个由输入自己算出来的“音量旋钮”，逐通道地决定 $W_3x$ 的哪些维度该放大、哪些该压低、哪些该关掉。

3. **提供模型容量**
   在标准 MHA + $d_{ff}=8/3D$ 的配置下，FFN 约占每个 Transformer Block 参数量的 $2/3$。它也是模型存储“知识”的主要位置之一（见 §3.4）。

### 2.2 适用场景

| 场景 | 适合吗 | 说明 |
| --- | --- | --- |
| Decoder-only LLM 的 FFN | **非常适合** | LLaMA、Qwen、Mistral 等主流开源模型都用 SwiGLU；PaLM 也采用了它 |
| Encoder / ViT 的 FFN | **可以用** | 经典实现多用 GELU；SwiGLU 也常见于较新的视觉 Transformer |
| 需要跨 token 建模 | **不能单独用** | SwiGLU 是逐位置的，跨位置必须靠注意力或卷积 |
| MoE 的专家网络 | **常用** | 每个 expert 往往就是一个较小的 SwiGLU |
| 端侧 / 极低延迟推理 | **谨慎** | 3 个矩阵乘法 + 更宽的中间层，参数量和访存量都更高 |
| 非 Transformer 的 MLP | **可以用** | 本质是 gated MLP，不依赖注意力 |

### 2.3 代价：没有免费的午餐

| 代价 | 具体表现 |
| --- | --- |
| 参数量 | 等参数设计下与经典 FFN 相同；若保持 $d_{ff}=4D$，则参数量变为 $12D^2$，是经典 FFN 的 1.5 倍 |
| 计算量 | 3 个矩阵乘法（两个升维、一个降维），而经典 FFN 是 2 个 |
| 激活显存 | 训练时要保存两条 $d_{ff}$ 宽的中间张量；等参数下比经典 FFN 多约 1/3 的中间激活 |
| 推理延迟 | 自回归解码是**访存瓶颈**，FFN 权重越大，每生成一个 token 要读的字节越多 |
| 调参成本 | $d_{ff}$ 需要和硬件对齐、和并行策略对齐，不能随便取 |

### 2.4 FFN 家族对比

| 变体 | 公式 | 线性层数 | 备注 |
| --- | --- | ---: | --- |
| ReLU FFN | $W_2\operatorname{ReLU}(W_1x)$ | 2 | 原版 Transformer，简单、便宜 |
| GELU FFN | $W_2\operatorname{GELU}(W_1x)$ | 2 | GPT-2、BERT 等常用 |
| SiLU FFN | $W_2\operatorname{SiLU}(W_1x)$ | 2 | 没有门控，是作业里做消融的天然基线 |
| GLU | $\sigma(W_1x)\odot(W_2x)$ | 2 | 原版门控单元，通常后面还要接输出投影 |
| GeGLU | $W_2(\operatorname{GELU}(W_1x)\odot W_3x)$ | 3 | 用 GELU 当门，Gemma 等采用 |
| **SwiGLU** | $W_2(\operatorname{SiLU}(W_1x)\odot W_3x)$ | 3 | 本文主角，LLaMA 系列采用 |
| MoE FFN | 多个 SwiGLU 专家 + 路由 | 很多 | 稀疏激活，总参数大但每 token 只算一部分 |

> Shazeer 在 2020 年的论文里系统比较了这些变体，结论是 **GLU 类（尤其是 SwiGLU/GeGLU）在同等参数和算力下通常优于非门控的 ReLU/GELU/SiLU**。这也是 LLaMA 系列选择它的直接原因之一。

### 2.5 为什么主流 LLM 都爱用它

- **经验上稳定**：在语言建模的困惑度上，门控版本通常有小幅但稳定的改善；
- **优化友好**：SiLU 光滑，没有 ReLU 在 0 处的折点；
- **表达力更强**：乘性交互让一层就能建模特征之间的相互作用；
- **工程上成熟**：三个 GEMM 的形状规整，容易做张量并行和融合；
- **生态配套**：LLaMA 系模型的 checkpoint、推理框架、量化方案都围绕它优化。

---

## 3. 为什么 Transformer 需要 FFN

这是理解 SwiGLU 的前提。很多人第一次看 Transformer 结构图时会有疑问：注意力已经让每个 token 看过所有 token 了，为什么还要在每一层后面接一个又大又贵的 FFN？

### 3.1 注意力做了什么，没做什么

先看单个注意力头、单个位置的输出：

$$
o_t=\sum_{s=1}^{T}\alpha_{ts}\,v_s,\qquad \alpha_{ts}\ge 0,\quad \sum_s\alpha_{ts}=1
$$

其中 $v_s=x_sW_V$ 是第 $s$ 个位置的值向量，$\alpha_{ts}$ 是 softmax 出来的注意力权重。

这句话的信息量很大：

- 输出是**值向量的凸组合**（加权平均），权重非负、和为 1；
- 如果固定住注意力权重 $\alpha$，输出对 $V$ 是**线性**的；
- 虽然 $\alpha$ 本身依赖输入（所以整个注意力块不是全局线性的），但注意力**没有对每个通道做非线性变换**；
- 它的主要职责是**跨位置搬运信息**（token mixing），不是“就地加工特征”。

多个头、输出投影 $W_O$ 也只是把多个凸组合再线性混合。注意力可以让“猫”这个位置读到“追”这个位置的信息，但它不会在单个位置上做“把这两个特征组合成一个新特征”的非线性运算。

### 3.2 FFN 补上“逐位置的通道混合”

FFN 的做法正好互补：

- **逐位置**：同一个 MLP 独立地作用在每个 token 上，位置之间不通信；
- **跨通道**：$W_1$ 把 $D$ 维的各个通道混合成 $d_{ff}$ 维的新特征；
- **非线性**：激活函数（ReLU/GELU/SiLU）把线性混合变成非线性；
- **再压回**：$W_2$ 把 $d_{ff}$ 维的新特征组合回 $D$ 维，写回残差流。

用一句话概括：

> **注意力负责“信息交换”（token mixing），FFN 负责“就地思考”（channel mixing）。**

从形状上看，FFN 就是一个**核大小为 1 的卷积**：卷积核在每个位置上共享，不跨位置滑动。

### 3.3 如果没有 FFN 会怎样

可以从三个角度看：

1. **表达力**：没有 FFN，每个 Block 只剩“注意力 + 线性投影”。注意力擅长搬运和加权平均，但不擅长在单个位置上做复杂的非线性特征变换。模型的“计算深度”实际上被削弱了。
2. **参数容量**：在标准配置下 FFN 占 Block 参数的 $2/3$。去掉它，等于把模型容量砍掉一大半；在同样参数量下，只能把注意力层加宽，而注意力参数随 $D^2$ 增长，性价比并不好。
3. **经验结果**：大量消融实验表明，去掉 FFN 或显著减小 $d_{ff}$ 会让语言建模性能明显下降；FFN 宽度是除了层数、$d_{model}$ 之外最重要的缩放旋钮之一。

> 一个常见的误解是“注意力已经能看全局，所以 FFN 可有可无”。事实相反：**注意力决定“能看到什么”，FFN 决定“看到之后能算出什么”**。两者缺一不可。

### 3.4 一个有用的解释：FFN 是键值记忆

Geva 等人（2021）提出过一种有影响力的解释：把 FFN 看成一组“键值对”：

- $W_1$ 的每一行是一个**键** $k_i$，负责检测输入中的某种模式；
- 激活值 $a_i=\phi(k_i^\top x)$ 是**这个模式被激活的强度**；
- $W_2$ 的每一列是一个**值** $v_i$，表示“如果检测到了，就往残差流里写入什么”；
- 最终输出是 $\sum_i a_i v_i$，即按激活强度加权写入的一组方向。

在这个视角下：

- FFN 是一个**大规模的记忆表**，存储的是“什么模式 → 写入什么信息”；
- 注意力负责从上下文里把相关 token 的信息取过来，FFN 负责根据这些信息“查表并写入”；
- SwiGLU 的门控相当于在写入前再加一道**输入相关的音量控制**：不是所有被检测到的模式都值得以同样强度写入。

这不是唯一的解释，也没有覆盖所有现象，但它能很好地解释“为什么 FFN 占了大部分参数，却看起来只是在做逐位置变换”。

### 3.5 容量视角：为什么 FFN 要“先变宽再变窄”

- **变宽**：把 $D$ 维空间中的特征投影到更高维的 $d_{ff}$ 维空间，给模型更大的“工作台”去组合特征；
- **非线性**：在高维空间里做非线性变换，比在低维空间里更容易把纠缠的特征分开；
- **变窄**：把高维结果压回 $D$ 维，以便和残差流相加、进入下一层。

这类似“展开 → 加工 → 折叠”的过程。$d_{ff}$ 越大，工作台越大，但参数、计算和显存也越高——这就是 §5 要讨论的权衡。

### 3.6 一句话总结

> **注意力让 token 之间交换信息，FFN 让每个 token 在自己的向量内部做非线性计算。SwiGLU 是后者的现代实现。**

---

## 4. 数学原理与推导

这一节从“为什么需要非线性”开始，一步步推到 SwiGLU 的公式、逐通道形式和反向传播。

### 4.1 为什么必须有非线性

假设没有激活函数，只有线性层。两层线性层：

$$
W_2(W_1x)=(W_2W_1)x
$$

仍然是一个线性变换。即使加上残差：

$$
x+Wx=(I+W)x
$$

还是线性。堆多少层都一样，最终等价于**一层**线性层。这意味着：

- 模型无法表示 XOR 这类线性不可分函数；
- 无法做“如果特征 A 和特征 B 同时出现，就输出 C”这种条件逻辑；
- 深度带来的表达能力几乎全部浪费。

激活函数的作用就是**打破这种可合并性**：$\sigma(W_1x)$ 不能写成某个矩阵乘 $x$，所以多层堆叠才真正增加了表达力。

> 理论上，一个足够宽的两层 MLP 可以逼近紧集上的任意连续函数（通用逼近定理）。FFN 给每个位置提供的正是这种“通用小网络”的能力。

### 4.2 经典 FFN：升维、激活、降维

原版 Transformer 的 FFN：

$$
\operatorname{FFN}(x)=W_2\,\sigma(W_1x)
$$

其中：

$$
W_1\in\mathbb{R}^{4D\times D},\qquad W_2\in\mathbb{R}^{D\times 4D}
$$

参数量：

$$
\underbrace{D\cdot 4D}_{W_1}+\underbrace{4D\cdot D}_{W_2}=8D^2
$$

为什么是 $4$ 倍？这是一个经验选择：中间层足够宽，模型有空间做非线性组合；又不至于让参数量失控。GPT-2、BERT 等经典模型都采用这个比例。

### 4.3 激活函数的演进：ReLU → GELU → SiLU

| 激活 | 公式 | 在 0 处光滑 | 负区间行为 | 备注 |
| --- | --- | --- | --- | --- |
| ReLU | $\max(0,x)$ | 否（折点） | 恒为 0 | 简单、便宜；负区间梯度为 0，可能“死神经元” |
| GELU | $x\Phi(x)$ | 是 | 有小负值 | $\Phi$ 是标准正态 CDF；GPT-2/BERT 常用 |
| SiLU | $x\sigma(x)$ | 是 | 有小负值，下限约 $-0.2785$ | 自门控、非单调；LLaMA 系列采用 |

三者的共同目标都是：**正区间近似恒等，负区间适当抑制，同时保持梯度可传**。区别主要在“负区间怎么处理”和“函数是否光滑”。

### 4.4 SiLU 函数详解：SwiGLU 的门控核心

> 这一节把 SiLU 单独拿出来讲。它既是 SwiGLU 的激活函数，也是一个值得独立理解的函数。读完你应该知道：它长什么样、导数怎么推、为什么负区间不是 0、代码里怎么稳定地算。

#### 4.4.1 名字与定义

SiLU 全称 **Sigmoid-weighted Linear Unit**（也有文献叫 Sigmoid Linear Unit），别名 **Swish**（$\beta=1$ 的情形）：

$$
\operatorname{SiLU}(x)=x\cdot\sigma(x)=\frac{x}{1+e^{-x}},\qquad
\sigma(x)=\frac{1}{1+e^{-x}}
$$

两个等价视角：

| 视角 | 写法 | 直觉 |
| --- | --- | --- |
| “线性单元 × 门” | $x\cdot\sigma(x)$ | $x$ 是内容，$\sigma(x)$ 是 0 到 1 之间的音量旋钮 |
| “平滑版 ReLU” | $x\sigma(x)$ | 正区间近似 $x$，负区间被压到接近 0，但过渡是光滑的 |

> 名字里的 “Swish” 和 “SiLU” 是同一个函数：Swish 一般写成 $x\sigma(\beta x)$，当 $\beta=1$ 时就是 SiLU。

#### 4.4.2 直觉：为什么是“乘”而不是“取 max”

ReLU 是硬开关：

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

- $x>0$：原样通过；
- $x<0$：直接关掉；
- 在 0 处有一个折点，导数从 0 跳到 1。

SiLU 把“开关”换成平滑的 sigmoid：

$$
\operatorname{SiLU}(x)=x\cdot\sigma(x)
$$

- $x$ 很大：$\sigma(x)\approx1$，输出 $\approx x$，和 ReLU 一样；
- $x$ 很小（很负）：$\sigma(x)\approx0$，输出 $\approx0$；
- $x$ 在 0 附近：平滑过渡，没有折点；
- $x$ 为负但不那么负：$\sigma(x)$ 还是正的，所以输出是一个**小的负值**，而不是直接 0。

一句话：

> **ReLU 是“要么开、要么关”的硬开关；SiLU 是“根据输入自己调节开度”的软开关。**

而且这个“开度”不需要额外的参数——门控信号 $\sigma(x)$ 就来自 $x$ 自己，所以叫 **self-gated（自门控）**。

#### 4.4.3 函数形状与关键数值

先看几个关键点：

| $x$ | $\operatorname{SiLU}(x)$ | 说明 |
| ---: | ---: | --- |
| $-\infty$ | $0$ | 负无穷被压到 0 |
| -3.000 | -0.1423 | 小幅负值 |
| -1.278 | -0.2785 | **全局最小值**附近 |
| -1.000 | -0.2689 | 负值 |
| 0.000 | 0.0000 | 过原点 |
| 1.000 | 0.7311 | 正区间开始接近线性 |
| 3.000 | 2.8577 | 接近 $x$ |
| $+\infty$ | $\approx x$ | 正无穷近似线性 |

形状草图（示意，不是精确比例）：

```text
        SiLU(x)
          ▲
        3 │                    ●
        2 │                 ●
        1 │              ●
        0 │──────────●──────────────► x
          │        ●   ╲
    -0.28 │      ●       ╲  ← 最小值附近
          └──┬──┬──┬──┬──┬──┬──
            -3 -2 -1  0  1  2  3
```

三个容易记住的特征：

1. **过原点**：$\operatorname{SiLU}(0)=0$；
2. **负区间有一个小凹陷**：在 $x\approx-1.278$ 处取最小值 $\approx-0.2785$，所以函数**不是单调递增**的；
3. **正区间近似线性**：$x$ 越大，$\sigma(x)$ 越接近 1，$\operatorname{SiLU}(x)\approx x$。

> 这个“小凹陷”很重要：它意味着 SiLU 允许少量负值通过，而不是像 ReLU 那样把负区间一刀切成 0；相应地，负区间的导数也不是 0，梯度还能继续传。

#### 4.4.4 导数推导

用乘积法则：

$$
\begin{aligned}
\frac{d}{dx}\bigl[x\sigma(x)\bigr]
&=\sigma(x)+x\sigma(x)\bigl(1-\sigma(x)\bigr) \\
&=\sigma(x)\Bigl(1+x\bigl(1-\sigma(x)\bigr)\Bigr)
\end{aligned}
$$

其中用到 $\sigma'(x)=\sigma(x)\bigl(1-\sigma(x)\bigr)$。

关键数值：

| $x$ | $\operatorname{SiLU}'(x)$ | 说明 |
| ---: | ---: | --- |
| -3.000 | -0.0881 | 负但在传 |
| -2.400 | $\approx-0.0998$ | 导数最小值附近 |
| -1.278 | 0.0001 | 函数最小值处，切线接近水平 |
| 0.000 | 0.5000 | 斜率 0.5 |
| 1.000 | 0.9277 | 接近 1 |
| 3.000 | 1.0881 | 略大于 1 |

性质：

- 导数**有下界**，约 $-0.1$，不会像 ReLU 那样在负区间恒为 0；
- 导数在 $x<-1.278$ 时为负（函数在下降），在 $x>-1.278$ 时为正（函数在上升）；
- 导数**处处连续**，因为 SiLU 光滑（$C^\infty$）。

> 注意：导数最小值为负并不代表“训练会变差”。它只是说函数在负区间是下降的；只要梯度不是恒 0，参数就还有更新机会。

#### 4.4.5 与 ReLU、GELU 的对比

| 函数 | 公式 | 0 处光滑 | 负区间 | 主要特点 |
| --- | --- | --- | --- | --- |
| ReLU | $\max(0,x)$ | 否 | 恒为 0 | 便宜；负区间梯度为 0，可能“死神经元” |
| GELU | $x\Phi(x)$ | 是 | 有小负值 | 用正态 CDF 当门；GPT-2/BERT 常用 |
| SiLU | $x\sigma(x)$ | 是 | 有小负值 | 用 sigmoid 当门；LLaMA 系列常用 |

三者的共同目标：**正区间尽量保留输入，负区间适当抑制，同时让梯度可传**。区别在“抑制得多狠”和“函数是否光滑”。

- ReLU 最便宜，但负区间一刀切；
- GELU 和 SiLU 形状很像，都是平滑的“软 ReLU”；GELU 用 $\Phi(x)$，SiLU 用 $\sigma(x)$；
- SiLU 的计算只比 ReLU 多一个 sigmoid，通常比精确 GELU 的 erf 更便宜（和 tanh 近似版 GELU 差别不大）；
- 经验上，SiLU 在深层网络里往往比 ReLU 更好训，但这不是绝对的。

#### 4.4.6 为什么 SwiGLU 用 SiLU 当门

把 SiLU 放进 GLU 的门里：

$$
\operatorname{SwiGLU}(x)=W_2\big(\operatorname{SiLU}(W_1x)\odot(W_3x)\big)
$$

为什么不是 sigmoid？

| 门函数 | 取值范围 | 能否放大 | 饱和 | 备注 |
| --- | --- | --- | --- | --- |
| $\sigma(a)$ | $(0,1)$ | 不能 | 两端都容易饱和 | 原版 GLU |
| $\operatorname{SiLU}(a)$ | $[-0.2785,\infty)$ | 可以 $>1$ | 负端趋 0，正端近似线性 | SwiGLU |

具体好处：

1. **可以放大**：$\operatorname{SiLU}(3)\approx2.86$，门可以开得比 1 还大；sigmoid 门最多到 1；
2. **可以为负**：负门控允许抑制甚至翻转内容，表达更灵活；
3. **更光滑**：没有 sigmoid 两端那么强的饱和，梯度更容易传；
4. **自门控**：门信号来自同一输入，不需要额外参数。

> 但别把它当成定理：Shazeer 的结论是经验性的，SwiGLU 不是“数学上最优”的门控。它的价值在于“在这些任务上通常更好，而且工程上成熟”。

#### 4.4.7 代码：实现、导数与数值稳定

最小实现：

```python
import torch
import torch.nn.functional as F


def silu(x):
    """SiLU(x) = x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def silu_grad(x):
    """SiLU 的导数：s * (1 + x * (1 - s))，其中 s = sigmoid(x)"""
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


x = torch.tensor([-3.0, -1.278, 0.0, 1.0, 3.0], requires_grad=True)
y = silu(x)
y.backward(torch.ones_like(y))

print("SiLU(x)      :", y.detach())
print("autograd 梯度:", x.grad)
print("手写导数     :", silu_grad(x.detach()))
print("F.silu(x)    :", F.silu(x.detach()))
```

预期输出（数值保留）：

```text
SiLU(x)      : tensor([-0.1423, -0.2785,  0.0000,  0.7311,  2.8577])
autograd 梯度: tensor([-8.8104e-02,  1.0121e-04,  5.0000e-01,  9.2767e-01,  1.0881e+00])
手写导数     : tensor([-8.8104e-02,  1.0121e-04,  5.0000e-01,  9.2767e-01,  1.0881e+00])
F.silu(x)    : tensor([-0.1423, -0.2785,  0.0000,  0.7311,  2.8577])
```

> 注意：`y.backward(torch.ones_like(y))` 是因为 `y` 是向量，需要给每个元素一个上游梯度；这里全填 1，等价于对 $y$ 求和再反传。

数值稳定：

```python
z = torch.tensor([1000.0])
print(torch.exp(z) / (1 + torch.exp(z)))  # nan：inf / inf
print(torch.sigmoid(z))                   # 1.0：稳定
```

所以：

- 用 `torch.sigmoid(x)` 或 `F.silu(x)`；
- 不要手写 `exp(x)/(1+exp(x))`；
- 作业里也明确允许直接用 `torch.sigmoid`。

#### 4.4.8 一句话总结

> **SiLU 就是“用输入自己的 sigmoid 当音量旋钮”的平滑激活：正区间近似线性，负区间留一个小负值，处处可导。SwiGLU 把这个旋钮从激活函数里拆出来，变成一条独立的门控支路。**

### 4.5 GLU：把“门控”显式拆出来

Gated Linear Unit（Dauphin 等，2017）定义为：

$$
\operatorname{GLU}(x)=\sigma(W_1x)\odot(W_2x)
$$

其中：

- $\sigma(W_1x)$ 是**门**，每个通道在 $(0,1)$ 之间；
- $W_2x$ 是**内容**；
- $\odot$ 是逐元素相乘。

> 一个有意思的视角：SiLU $x\sigma(x)$ 可以看成 GLU 的“门和内容共享同一输入”的特例（取 $W_1=W_2=I$）；GLU 把门和内容拆成两个独立的线性变换，SwiGLU 再把门换成 SiLU。这条线索能把 §4.4 和本节串起来。

#### 为什么“乘”比“加”更有表达力？

对第 $i$ 个通道：

$$
h_i=\sigma(w_i^\top x)\,(v_i^\top x)
$$

如果不看 sigmoid，把门控函数暂时当成恒等，那么：

$$
h_i=(w_i^\top x)(v_i^\top x)=x^\top w_i v_i^\top x
$$

这是一个由秩 1 矩阵 $w_i v_i^\top$ 定义的二次型（也就是两个线性函数的乘积），里面包含 $x_jx_k$ 这样的交叉项。也就是说，**一次乘法就能建模特征之间的相互作用**。

对比 ReLU MLP：

$$
h=\operatorname{ReLU}(Wx)
$$

它是 $x$ 的**分段线性函数**，本身不包含 $x_jx_k$ 这样的交叉项；要逼近“两个特征相乘”这种函数，需要足够的宽度或深度。门控乘法提供了一种更直接的归纳偏置。

> 注意：这不是说 ReLU MLP 不能表示乘法——通用逼近定理保证只要够宽就能逼近。区别在于**参数效率**和**优化难度**：门控结构把“乘法交互”写进了架构，而不是让模型从数据里硬学。

#### 门控的另外三个好处

1. **输入相关的缩放**：门值随输入变化，模型可以对不同样本做不同的特征选择；
2. **动态稀疏**：门接近 0 的通道相当于被关掉，网络可以“按需使用”容量；
3. **梯度路径**：值分支的反传只乘门控值，不需要再穿过激活函数的导数；门控分支的反传则乘内容值。两条路径相加，比单路饱和激活更容易保留信号。

> 原版 GLU 用 sigmoid，门被限制在 $(0,1)$，容易饱和。SwiGLU 把门换成 SiLU：门可以为负、可以大于 1，且更光滑。这是它比原版 GLU 更强的原因之一。

### 4.6 组装成 SwiGLU

把 GLU 的门从 $\sigma$ 换成 $\operatorname{SiLU}$，再加上输出投影：

$$
\operatorname{SwiGLU}(x)=W_2\Big(\operatorname{SiLU}(W_1x)\odot(W_3x)\Big)
$$

其中：

$$
W_1,W_3\in\mathbb{R}^{d_{ff}\times D},\qquad W_2\in\mathbb{R}^{D\times d_{ff}}
$$

逐通道写出来：

$$
y=\sum_{i=1}^{d_{ff}}(W_2)_{:,i}\;\operatorname{SiLU}(w_{1,i}^\top x)\;(w_{3,i}^\top x)
$$

这里 $(W_2)_{:,i}$ 是 $W_2$ 的第 $i$ 列（$D$ 维向量），$w_{1,i}$、$w_{3,i}$ 分别是 $W_1$、$W_3$ 的第 $i$ 行。

这个式子说明：

- 每个隐藏通道 $i$ 是一个“**门控特征检测器**”；
- 它检测 $w_{1,i}^\top x$（决定门开多大）和 $w_{3,i}^\top x$（决定内容是什么）；
- 输出是所有检测器按门控强度加权后的 $D$ 维向量的和；
- $d_{ff}$ 就是“检测器的数量”。

> **关于支路顺序**：作业的公式是 $\operatorname{SiLU}(W_1x)\odot(W_3x)$；Shazeer 原文写的是 $\operatorname{Swish}(xW)\odot(xV)$。两者都叫 SwiGLU，区别只是“哪条支路做激活”。按作业公式实现即可；交换 $W_1$ 和 $W_3$ 会改变数值，但不会改变结构类型。

#### 为什么没有 bias？

作业和现代 LLM（PaLM、LLaMA 等）通常省略线性层的 bias：

- 在 RMSNorm + 残差的结构里，bias 的边际作用较小，省略后性能通常几乎不受影响；
- 少一组参数，数值更简单；
- 对大模型来说，能省一点显存和带宽。

所以三个线性层都是**纯矩阵乘法**，没有 $+b$。

### 4.7 反向传播：逐项推导

为了严谨，设一个序列（省略 batch）：

$$
X\in\mathbb{R}^{T\times D}
$$

前向：

$$
\begin{aligned}
A&=XW_1^\top &\in\mathbb{R}^{T\times d_{ff}} \\
B&=XW_3^\top &\in\mathbb{R}^{T\times d_{ff}} \\
G&=\operatorname{SiLU}(A) &\in\mathbb{R}^{T\times d_{ff}} \\
H&=G\odot B &\in\mathbb{R}^{T\times d_{ff}} \\
Y&=HW_2^\top &\in\mathbb{R}^{T\times D}
\end{aligned}
$$

设损失对输出的梯度：

$$
\Delta=\frac{\partial L}{\partial Y}\in\mathbb{R}^{T\times D}
$$

则：

$$
\begin{aligned}
\frac{\partial L}{\partial W_2}&=\Delta^\top H
&&\in\mathbb{R}^{D\times d_{ff}} \\[4pt]
\frac{\partial L}{\partial H}&=\Delta W_2
&&\in\mathbb{R}^{T\times d_{ff}} \\[4pt]
\frac{\partial L}{\partial G}&=(\Delta W_2)\odot B
&&\in\mathbb{R}^{T\times d_{ff}} \\[4pt]
\frac{\partial L}{\partial B}&=(\Delta W_2)\odot G
&&\in\mathbb{R}^{T\times d_{ff}} \\[4pt]
\frac{\partial L}{\partial A}&=(\Delta W_2)\odot B\odot\operatorname{SiLU}'(A)
&&\in\mathbb{R}^{T\times d_{ff}} \\[4pt]
\frac{\partial L}{\partial W_1}&=\left(\frac{\partial L}{\partial A}\right)^\top X
&&\in\mathbb{R}^{d_{ff}\times D} \\[4pt]
\frac{\partial L}{\partial W_3}&=\left(\frac{\partial L}{\partial B}\right)^\top X
&&\in\mathbb{R}^{d_{ff}\times D}
\end{aligned}
$$

逐项解释：

1. **$W_2$ 的梯度**：$Y=HW_2^\top$，所以对 $W_2$ 求导得到 $\Delta^\top H$，形状正好是 $(D,d_{ff})$；
2. **$H$ 的梯度**：把误差从 $D$ 维投影回 $d_{ff}$ 维；
3. **$G$ 和 $B$ 的梯度**：乘法 $H=G\odot B$ 的导数就是“乘对方”——$\partial H/\partial G=B$，$\partial H/\partial B=G$；
4. **$A$ 的梯度**：先经过逐元素乘 $B$，再乘 SiLU 的导数；
5. **$W_1$、$W_3$ 的梯度**：把逐位置的梯度按 token 维求和（矩阵形式就是转置相乘）。

> 有 batch 维时，上述矩阵乘变成 batched matmul，或者把 $(B,T)$ 展平成 $(B\cdot T,D)$；PyTorch 的 autograd 会自动处理。你只需要保证前向形状对，反向通常不会错。

### 4.8 梯度视角：门控为什么可能更好训

对比三种情况：

| 结构 | 负区间的梯度 | 潜在问题 |
| --- | --- | --- |
| ReLU | 精确为 0 | 一旦进入负区间，这个神经元就“死”了，梯度永远为 0 |
| SiLU | 很小但非 0（下界约 $-0.1$） | 仍可能饱和，但不会完全断流 |
| SwiGLU | 两条路径相加 | 值分支的梯度只乘门控值；门控分支的梯度乘内容值 |

SwiGLU 的乘积结构让梯度有**两条路**可以走：

- 对内容分支 $B$：$\partial L/\partial B=(\Delta W_2)\odot G$，只乘门控值，不乘激活导数；
- 对门控分支 $A$：$\partial L/\partial A=(\Delta W_2)\odot B\odot\operatorname{SiLU}'(A)$，要乘激活导数，但也乘了内容值 $B$。

只要 $G$ 和 $B$ 不要同时为 0，就至少有一条路径能传梯度。这比“单路激活饱和就全断”要稳健一些。

> 但不要夸大：SwiGLU **不能**完全解决梯度消失。门控仍可能饱和，深层网络仍需要残差、pre-norm、合适的学习率和初始化。SwiGLU 只是众多工具之一。

### 4.9 表达力视角：乘性交互与动态稀疏

把 SwiGLU 的优势总结成三句话：

1. **乘性交互**：$x_jx_k$ 这样的交叉项被显式地写进结构，一层就能建模特征组合；
2. **动态稀疏**：门接近 0 的通道被关掉，模型可以按输入选择“用哪些特征”；
3. **平滑非线性**：SiLU 处处可导，优化地形比 ReLU 更平滑。

这三者共同构成了 SwiGLU 的归纳偏置。它们不是严格的定理，而是“为什么它常常更好”的直觉解释。

### 4.10 一个诚实的边界

Shazeer 在提出 GLU 变体的论文里有一句很有名的话，大意是：

> “我们无法解释这些架构为什么有效；我们把它们的成功，和别的一切一样，归因于神的恩典。”

这句话值得记住，因为它提醒我们：

- SwiGLU 的有效性**主要是经验结果**，不是从第一性原理推出来的；
- 在比较时一定要**控制变量**：同等参数量、同等算力、同等训练 token；
- 经典 FFN（2 个矩阵、$d_{ff}=4D$）换成 SwiGLU（3 个矩阵、$d_{ff}=8/3D$）时，参数量不变；但如果 SwiGLU 仍用 $d_{ff}=4D$，参数量会多 50%，那时的提升可能只是“模型更大了”，不能全归功于门控。

作业后面安排 SwiGLU vs SiLU 的消融，就是为了让你亲手做一次受控比较。

---

## 5. 参数规模与维度设计

### 5.1 一个 SwiGLU 有多少参数

三个矩阵：

$$
W_1: d_{ff}\times D,\qquad
W_3: d_{ff}\times D,\qquad
W_2: D\times d_{ff}
$$

参数量：

$$
\boxed{3\,D\,d_{ff}}
$$

没有 bias，所以这就是全部。对单个 token 的前向计算量（乘加）也正比于 $3Dd_{ff}$。

### 5.2 $d_{ff}=\frac{8}{3}D$ 是怎么来的

经典 FFN 用 $4D$ 的中间层，参数量：

$$
2\cdot D\cdot 4D=8D^2
$$

SwiGLU 有三个矩阵，如果想让参数量与经典 FFN **相等**：

$$
3\cdot D\cdot d_{ff}=8D^2
\quad\Longrightarrow\quad
d_{ff}=\frac{8}{3}D\approx 2.667D
$$

用图表示：

```text
经典 FFN:   D ──W1──► 4D ──W2──► D        2 个矩阵，8D²
SwiGLU:     D ──W1/W3──► 8D/3 ──W2──► D   3 个矩阵，8D²
```

所以 $8/3$ 不是“魔法比例”，而是**等参数比较**自然推出的结果：

- 如果你想让 SwiGLU 和经典 FFN 参数量一样，就用 $8/3D$；
- 如果你更在意效果、算力也充足，可以用更大的 $d_{ff}$（例如 $3.5D$、$4D$），但那时它已经比经典 FFN 更大了，比较时要说明；
- 如果你在写作业，讲义明确建议 $d_{ff}\approx\frac{8}{3}d_{model}$，并取 64 的倍数。

### 5.3 取整与硬件对齐

讲义的要求是：

> $d_{ff}$ 约为 $\frac{8}{3}d_{model}$，并取**最接近的 64 的倍数**，以便更好地利用硬件。

为什么是 64？

- GPU 的矩阵乘法（GEMM）按 tile 计算，维度是 64 的倍数时，tile 利用率更高；
- 显存访问更规整，减少边界处理；
- 张量并行（tensor parallelism）通常要求维度能被并行度整除，64 是一个安全的粒度；
- 这不是数学要求，而是**工程要求**：取 63 或 65 也能算，只是慢。

一个实用的函数：

```python
def suggest_d_ff(d_model: int, multiple: int = 64) -> int:
    """8/3 * d_model，四舍五入到 multiple 的整数倍。"""
    target = 8 * d_model / 3
    return multiple * max(1, round(target / multiple))
```

> 这里用的是四舍五入，所以结果可能略低于 $8/3D$。如果你希望“不低于 $8/3D$”，把 `round` 换成 `math.ceil` 即可；两种做法都符合讲义的“nearby multiple of 64”。

几个例子：

| $D$ | $\frac{8}{3}D$ | 取整到 64 的倍数 | 实际比例 |
| ---: | ---: | ---: | ---: |
| 64 | 170.67 | 192 | 3.000 |
| 256 | 682.67 | 704 | 2.750 |
| 512 | 1365.33 | 1344 | 2.625 |
| 1024 | 2730.67 | 2752 | 2.688 |
| 4096 | 10922.67 | 10944 | 2.672 |

> **小 $D$ 的舍入误差更大**：$D=64$ 时，理论值 170.67 被舍入到 192，比例变成 3.0。这就是为什么作业测试 fixture 里直接给了 `d_ff=128`，而不是让你从 `d_model=64` 推——测试关心的是**数值正确性**，不是比例。实际模型 $D$ 很大时，$8/3$ 才逼近得比较准。

### 5.4 具体算例：FFN 到底占多少参数

标准多头注意力（MHA）每个 Block 有四个 $D\times D$ 矩阵（$W_Q,W_K,W_V,W_O$），参数量 $4D^2$。SwiGLU 用 $3Dd_{ff}$。取 $d_{ff}=\frac{8}{3}D$：

| $D$ | $d_{ff}$ | 注意力参数 $4D^2$ | FFN 参数 $3Dd_{ff}$ | FFN 占比 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 192 | 16,384 | 36,864 | 69.2% |
| 256 | 704 | 262,144 | 540,672 | 67.3% |
| 512 | 1344 | 1,048,576 | 2,064,384 | 66.3% |
| 1024 | 2752 | 4,194,304 | 8,454,144 | 66.8% |
| 4096 | 10944 | 67,108,864 | 134,479,872 | 66.7% |

结论：

- 在等参数设计下，**FFN 约占 Block 的 $2/3$**；
- 如果模型用 GQA（分组查询注意力），注意力参数更少，FFN 占比会更高；
- 所以“FFN 是模型容量的主要载体”不是夸张。

以 LLaMA 2 7B 为例（$D=4096$，$d_{ff}=11008$，32 层）：

- 每层注意力：$4\times4096^2\approx67.1$M；
- 每层 FFN：$3\times4096\times11008\approx135.3$M；
- 每层合计约 $202.4$M，FFN 占 $66.8\%$；
- 32 层约 $6.48$B，再加嵌入层等，接近 6.7B。

### 5.5 真实模型的配置

| 模型 | $D$ | $d_{ff}$ | $d_{ff}/D$ | FFN 类型 |
| --- | ---: | ---: | ---: | --- |
| GPT-2 small | 768 | 3072 | 4.00 | GELU |
| PaLM | 18432 | 73728 | 4.00 | SwiGLU |
| LLaMA 2 7B | 4096 | 11008 | 2.69 | SwiGLU |
| LLaMA 3 8B | 4096 | 14336 | 3.50 | SwiGLU |
| Mistral 7B | 4096 | 14336 | 3.50 | SwiGLU |
| Qwen2.5 7B | 3584 | 18944 | 5.29 | SwiGLU |

这张表说明两件事：

1. **$8/3$ 只是默认值**，不是所有模型都严格遵循；
2. 实际比例取决于**算力预算、数据量、层数、注意力结构（MHA/GQA）、并行策略**等因素。LLaMA 3 用 $3.5D$，Qwen2.5 用 $5.29D$，都取得了很好的效果——但它们不是“等参数比较”，不能直接拿来证明“越大越好”。

> Gemma 7B 甚至用了 $8D$ 的中间层（GeGLU），说明在特定训练预算下，FFN 宽度可以远超 $8/3D$。设计空间很大，关键是用实验说话。

### 5.6 设计权衡：$d_{ff}$ 不是越大越好

| 维度 | $d_{ff}$ 增大时的影响 |
| --- | --- |
| 模型容量 | 变大，能拟合更复杂的函数 |
| 参数量 | 线性增长（$3Dd_{ff}$） |
| 训练算力 | 线性增长；FFN 通常是训练 FLOPs 的大头 |
| 激活显存 | 线性增长；训练时两条 $d_{ff}$ 中间张量都要保存 |
| 推理延迟 | 自回归解码是访存瓶颈，FFN 权重越大，每 token 读取的字节越多，延迟越高 |
| 并行 | 需要能被张量并行度整除；否则要 padding，浪费算力 |
| 过拟合 | 小数据上大 FFN 更容易过拟合；需要正则/早停 |
| 硬件效率 | 太小或不是 64 的倍数，GEMM 利用率低 |

因此，正确的做法不是“背一个比例”，而是：

1. 先用 $8/3D$ 作为默认值；
2. 固定**训练 token 数 / 总算力**，扫几个 $d_{ff}$（例如 $2D, 8/3D, 3.5D, 4D$）；
3. 比较验证集困惑度、训练吞吐、显存占用；
4. 结合部署约束（延迟、显存、并行度）做最终选择。

### 5.7 设计清单

- [ ] 默认取 $d_{ff}=\operatorname{round}(\frac{8}{3}D/64)\times64$；
- [ ] 如果要和经典 FFN 做**等参数**比较，严格用 $8/3D$；
- [ ] 如果要和 SwiGLU 的其他配置比较，固定总算力或总 token 数，而不是只看参数量；
- [ ] 确保 $d_{ff}$ 能被 64（以及张量并行度）整除；
- [ ] 算清楚 FFN 占比，别把预算全花在注意力上；
- [ ] 别忘了嵌入层和 LM head 的参数；
- [ ] 记录你的选择，方便复现和写报告。

---

## 6. CS336 Assignment 1 中的 SwiGLU

### 6.1 作业要求（讲义 §3.4.2）

作业要求你实现 `positionwise_feedforward`，具体是：

- 实现 **SwiGLU** 前馈网络，由 **SiLU** 激活和 **GLU** 门控组成；
- **省略 bias**（跟 PaLM、LLaMA 等现代 LLM 一致）；
- $d_{ff}$ 取约 $\frac{8}{3}d_{model}$，并保证是 **64 的倍数**；
- 实现测试适配器 `adapters.run_swiglu`；
- 运行 `uv run pytest -k test_swiglu` 验证；
- 这一题 2 分。

讲义还特别提示：**可以用 `torch.sigmoid` 来保证数值稳定**。

### 6.2 接口与张量形状

`run_swiglu` 的签名（来自 `tests/adapters.py`）：

```python
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    ...
```

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `d_model` | 标量 | 输入/输出的最后一维 |
| `d_ff` | 标量 | 中间层宽度 |
| `w1_weight` | `(d_ff, d_model)` | $W_1$，门控支路的升维矩阵 |
| `w3_weight` | `(d_ff, d_model)` | $W_3$，内容支路的升维矩阵 |
| `w2_weight` | `(d_model, d_ff)` | $W_2$，降维矩阵 |
| `in_features` | `(..., d_model)` | 任意前导维度，常见是 `(B, T, D)` 或 `(T, D)` |
| 返回值 | `(..., d_model)` | 与输入相同的形状 |

**关键点**：PyTorch 的权重布局是 `(out_features, in_features)`，所以：

- `in_features @ w1_weight.T` → `(..., d_ff)`；
- `in_features @ w3_weight.T` → `(..., d_ff)`；
- `(gate * value) @ w2_weight.T` → `(..., d_model)`。

### 6.3 最短实现（函数版，可直接填进 adapter）

```python
import torch.nn.functional as F


def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    # 门控支路：先升维，再过 SiLU
    gate = F.silu(in_features @ w1_weight.T)      # (..., d_ff)

    # 内容支路：只升维，不激活
    value = in_features @ w3_weight.T             # (..., d_ff)

    # 逐元素门控，再降维回 d_model
    return (gate * value) @ w2_weight.T           # (..., d_model)
```

逐行解释：

1. `in_features @ w1_weight.T`：因为 `w1_weight` 是 `(d_ff, d_model)`，转置后是 `(d_model, d_ff)`，矩阵乘法把最后一维从 `d_model` 变成 `d_ff`；
2. `F.silu`：PyTorch 内置的 SiLU，数值稳定，等价于 `x * torch.sigmoid(x)`；
3. `gate * value`：逐元素相乘，形状不变；
4. `@ w2_weight.T`：`w2_weight` 是 `(d_model, d_ff)`，转置后 `(d_ff, d_model)`，把 `d_ff` 压回 `d_model`。

> 这个实现不依赖你自己写的 `Linear` 类，适合直接过测试；如果你已经在 `cs336_basics` 里实现了 `LinearModule`，也可以用模块版本（见 §7）。

### 6.4 数值稳定：为什么讲义让你用 `torch.sigmoid`

SiLU 的定义里有 $\sigma(x)$。如果你手写：

```python
# 错误示范：x 很大时 exp(x) 溢出，inf / inf = nan
sigmoid = torch.exp(x) / (1.0 + torch.exp(x))
```

在 `x = 1000` 时会得到 `nan`：

```text
exp(x)/(1+exp(x)) : nan
torch.sigmoid(x)  : 1.0
```

正确做法：

```python
# 推荐：PyTorch 内部做了稳定处理
gate = F.silu(x)

# 或者
gate = x * torch.sigmoid(x)
```

`torch.sigmoid` 会根据符号分支计算，避免 $e^{|x|}$ 溢出。作业明确允许使用它，所以**不要自己手写 exp 版本**。

> 另外，`test_swiglu` 的容差是 `atol=1e-5`。不要用 float16 跑测试，否则误差可能超标；用 float32，并直接调用 `F.silu`，通常就能过。

### 6.5 测试 fixture 的 $d_{model}=64$、$d_{ff}=128$

打开 `tests/fixtures/ts_tests/model_config.json`：

```json
{
    "vocab_size": 10000,
    "context_length": 16,
    "d_model": 64,
    "num_layers": 3,
    "num_heads": 4,
    "d_ff": 128,
    "remove_rmsnorm": false,
    "use_post_norm": false,
    "remove_rope": false,
    "rope_theta": 10000.0,
    "ffn_type": null
}
```

这里的 `d_ff=128` 是 $2\times d_{model}$，**不是** $8/3$。原因：

- 这是测试用的微型配置，目的是让快照文件小、跑得快；
- 测试通过 `run_swiglu(d_model, d_ff, ...)` **显式传入** `d_ff`，你的实现不应该自己从 `d_model` 推算；
- 你的 `SwiGLU` 模块可以有一个“默认 $8/3$”的构造函数，但 forward 必须使用传入的 `d_ff`。

测试读取的权重键：

```python
ts_state_dict[0]["layers.0.ffn.w1.weight"]
ts_state_dict[0]["layers.0.ffn.w2.weight"]
ts_state_dict[0]["layers.0.ffn.w3.weight"]
```

没有 `bias` 键——这也在提醒你：**不要加 bias**。

### 6.6 它在 Transformer Block 里的位置

作业采用 **pre-norm** 结构：

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # 第一个子层：注意力（跨位置交换信息）
        x = x + self.attn(self.norm1(x))
        # 第二个子层：SwiGLU FFN（逐位置加工）
        x = x + self.ffn(self.norm2(x))
        return x
```

对应公式：

$$
y=x+\operatorname{MHA}(\operatorname{RMSNorm}(x))
$$
$$
z=y+\operatorname{SwiGLU}(\operatorname{RMSNorm}(y))
$$

三个细节：

1. **先归一化再进子层**（pre-norm），残差主干上的数值更稳；
2. **子层输出加回输入**，不是替换；
3. **顺序是“先注意力、后 FFN”**，每层各一次。

SwiGLU 的输入输出都是 $(B,T,D)$，所以能无缝接在残差上。

### 6.7 常见错误对照表

| 现象 | 可能原因 | 修正 |
| --- | --- | --- |
| 形状报错 `mat1 and mat2 shapes cannot be multiplied` | 写成了 `w1_weight @ x` | 改成 `x @ w1_weight.T` |
| 输出形状是 `(..., d_ff)` | 忘了最后的 `@ w2_weight.T` | 补上降维 |
| 数值和参考差很多 | $W_2$ 转置错，或门控支路搞反 | 按 §6.3 的顺序检查 |
| 输出量级异常，或与参考差很多 | 忘了逐元素乘，或把 gate 当成了 mask | 确认是 `gate * value`，不是 `gate + value` |
| 参数比预期多 | 加了 bias | 删掉 bias |
| `d_ff` 和测试对不上 | 在实现里硬编码了 $8/3D$ | 使用传入的 `d_ff` |
| 出现 `nan` | 手写 sigmoid 溢出 | 用 `F.silu` 或 `torch.sigmoid` |
| autograd 报“modified by in-place operation” | 对需要梯度的张量做了原地修改 | 用 out-of-place 运算 |
| 测试误差略大于 `1e-5` | 用了 float16 / 手动近似 | 用 float32，直接用 `F.silu` |

### 6.8 作业里的消融：SwiGLU vs SiLU

讲义后面会让你比较 SwiGLU 和 SiLU（无门控）。这里有两个不同的问题，别混：

| 比较目的 | 控制变量 | 做法 |
| --- | --- | --- |
| **门控 + 额外参数的总效果** | 固定 $d_{ff}$、层数、训练 token | SwiGLU（3 矩阵） vs SiLU FFN（2 矩阵，保持相同 $d_{ff}$）——此时参数量不同，结论是“门控 + 更多参数”的总效果 |
| **等参数下的结构差异** | 参数量相同 | SwiGLU $d_{ff}=8/3D$ vs SiLU FFN $d_{ff}=4D$——此时参数量相同，结论更接近“结构本身”的差异 |

更严谨的做法是**同时报告参数量、训练 token 数、验证集困惑度**，并说明你控制了什么。只报一个 loss 曲线，很容易把“参数更多”误认为“结构更好”。

---

## 7. 可运行代码

下面是一个**完整、可独立运行**的示例（纯 PyTorch，CPU 即可；已在 PyTorch 2.x 上验证）。它实现了：

- 不带 bias 的 `Linear`；
- `silu`；
- `SwiGLU` 模块；
- 与 PyTorch 官方算子的数值对照；
- 参数量、梯度、SiLU 数值表、不同 $D$ 下的参数占比。

### 7.1 完整脚本

```python
"""SwiGLU 最小实现与自检脚本（纯 PyTorch，CPU 可跑）。

运行：python swiglu_demo.py
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)


class Linear(nn.Module):
    """不带 bias 的线性层：y = x W^T，W 的形状是 (out_features, in_features)。"""

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return x @ self.weight.T


def silu(x):
    """SiLU(x) = x * sigmoid(x)，用 torch.sigmoid 保证数值稳定。"""
    return x * torch.sigmoid(x)


def suggest_d_ff(d_model, multiple=64):
    """8/3 * d_model，四舍五入到 multiple 的整数倍。"""
    target = 8 * d_model / 3
    return multiple * max(1, round(target / multiple))


class SwiGLU(nn.Module):
    """FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )，三个线性层都不带 bias。"""

    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = suggest_d_ff(d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        gate = silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


def swiglu_reference(x, w1, w2, w3):
    """用 PyTorch 官方算子重写一遍，作为数值对照。"""
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


def main():
    B, T, D = 2, 5, 64
    d_ff = suggest_d_ff(D)
    x = torch.randn(B, T, D)
    ffn = SwiGLU(D, d_ff)

    y = ffn(x)
    ref = swiglu_reference(x, ffn.w1.weight, ffn.w2.weight, ffn.w3.weight)

    print(f"输入形状      : {tuple(x.shape)}")
    print(f"d_model={D}, d_ff={d_ff} (比值 {d_ff / D:.3f})")
    print(f"输出形状      : {tuple(y.shape)}")
    print(f"与参考实现误差: {(y - ref).abs().max().item():.3e}")

    n_params = sum(p.numel() for p in ffn.parameters())
    print(f"参数量        : {n_params} (理论值 3*d_model*d_ff = {3 * D * d_ff})")

    loss = y.pow(2).mean()
    loss.backward()
    print("\n各权重的梯度范数：")
    for name, p in ffn.named_parameters():
        print(f"  {name:10s} 形状={str(tuple(p.shape)):12s} ||grad||={p.grad.norm().item():.4f}")

    print("\nSiLU 数值表：")
    xs = torch.tensor([-3.0, -2.0, -1.278, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    print("   x      SiLU(x)   SiLU'(x)")
    for xv in xs:
        t = xv.clone().requires_grad_(True)
        sv = silu(t)
        sv.backward()
        print(f" {xv.item():6.3f}  {sv.item():8.4f}  {t.grad.item():8.4f}")

    print("\n数值稳定性对照（x = 1000）：")
    z = torch.tensor([1000.0])
    naive = torch.exp(z) / (1.0 + torch.exp(z))
    print(f"  exp(x)/(1+exp(x)) : {naive.item()}")
    print(f"  torch.sigmoid(x)  : {torch.sigmoid(z).item()}")

    print("\n参数量对比（attention 4D² vs FFN 3·D·d_ff）：")
    print("  D      d_ff     attn参数     ffn参数    ffn占比")
    for d in [64, 256, 512, 1024, 4096]:
        dff = suggest_d_ff(d)
        attn = 4 * d * d
        ffn_p = 3 * d * dff
        print(f" {d:5d}  {dff:6d}  {attn:10,d}  {ffn_p:10,d}  {ffn_p / (attn + ffn_p):7.1%}")

    with torch.no_grad():
        gate = silu(ffn.w1(x))
    print(f"\n门控值统计（随机初始化，仅演示）：均值={gate.mean():.4f}，"
          f"标准差={gate.std():.4f}，|gate|<0.01 的比例={(gate.abs() < 0.01).float().mean():.2%}")


if __name__ == "__main__":
    main()
```

### 7.2 预期输出（节选）

```text
输入形状      : (2, 5, 64)
d_model=64, d_ff=192 (比值 3.000)
输出形状      : (2, 5, 64)
与参考实现误差: 1.192e-07
参数量        : 36864 (理论值 3*d_model*d_ff = 36864)

各权重的梯度范数：
  w1.weight  形状=(192, 64)    ||grad||=0.2182
  w3.weight  形状=(192, 64)    ||grad||=0.2063
  w2.weight  形状=(64, 192)    ||grad||=0.1159

SiLU 数值表：
   x      SiLU(x)   SiLU'(x)
 -3.000   -0.1423   -0.0881
 -2.000   -0.2384   -0.0908
 -1.278   -0.2785    0.0001
 ...
  3.000    2.8577    1.0881

数值稳定性对照（x = 1000）：
  exp(x)/(1+exp(x)) : nan
  torch.sigmoid(x)  : 1.0

参数量对比（attention 4D² vs FFN 3·D·d_ff）：
  D      d_ff     attn参数     ffn参数    ffn占比
    64     192      16,384      36,864    69.2%
   256     704     262,144     540,672    67.3%
   512    1344   1,048,576   2,064,384    66.3%
  1024    2752   4,194,304   8,454,144    66.8%
  4096   10944  67,108,864  134,479,872    66.7%
```

几个可以核对的地方：

- **误差 $1.19\times10^{-7}$** 是 float32 的正常舍入误差，说明你的实现和官方算子一致；
- **参数量 36,864** 正好等于 $3\times64\times192$；
- **三个权重的梯度形状**与权重形状一致，说明反向传播正常；
- **SiLU 的最小值**出现在 $x\approx-1.278$，值约 $-0.2785$；
- **x=1000 时**，手写 sigmoid 溢出成 `nan`，`torch.sigmoid` 仍然是 1.0。

### 7.3 接入作业

`tests/adapters.py` 的 `run_swiglu` 已经接好（走的是 `cs336_basics/SwiGLU.py` 的 `SwiGLU` 类），
§6.3 那段函数是它的等价写法。跑测试：

```bash
cd /home/kevin/assignment1-basics
uv run pytest -k test_swiglu
# 如果不用 uv，也可以：
.venv/bin/python -m pytest -k test_swiglu
```

仓库里已经有模块版本 `cs336_basics/SwiGLU.py`（类名 `SwiGLU`，内部用三个 `nn.Linear`），
adapter 直接接上就行：

```python
import torch
# 文件名和类名都是大驼峰 SwiGLU；Linux 区分大小写，写成 swiglu 会 ModuleNotFoundError
from cs336_basics.SwiGLU import SwiGLU

def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    ffn = SwiGLU(d_model, d_ff)
    # nn.Linear 存权重的属性名是 weight，形状 (out, in)，和参考实现一致
    with torch.no_grad():
        ffn.w1.weight.copy_(w1_weight)
        ffn.w2.weight.copy_(w2_weight)
        ffn.w3.weight.copy_(w3_weight)
    return ffn(in_features)
```

> 注意：用模块版本时，`Linear` 的权重名要和 `copy_` 对应上；`nn.Linear` 的权重名是 `weight`，形状同样是 `(out, in)`。

### 7.4 三个小实验

**实验 1：去掉门控，看看差多少**

把 `forward` 改成：

```python
def forward(self, x):
    return self.w2(silu(self.w1(x)))   # 没有 W3，也没有逐元素乘
```

参数量从 $3Dd_{ff}$ 降到 $2Dd_{ff}$。在同一个小语言模型上训练，比较验证集困惑度。注意：这时参数量变了，结论是“门控 + 额外参数”的总效果；要做等参数比较，把 $d_{ff}$ 调大。

**实验 2：扫 $d_{ff}$**

固定层数、$d_{model}$、训练 token 数，分别取 $d_{ff}=2D, 8/3D, 3.5D, 4D$，记录：

- 参数量；
- 训练吞吐（token/s）；
- 验证集困惑度；
- 显存峰值。

你会看到“容量 ↑、吞吐 ↓”的典型权衡。

**实验 3：观察门控分布**

```python
with torch.no_grad():
    gate = torch.sigmoid(ffn.w1(x)) * ffn.w1(x)
print("接近 0 的比例:", (gate.abs() < 0.01).float().mean())
```

- 随机初始化时，门控值大致围绕 0 分布；
- 训练后，很多通道会变成“常开”或“常关”，这就是动态稀疏的直观证据；
- 可以画直方图，对比不同层、不同 token 的门控分布。

---

## 8. 常见误区 FAQ

**Q1：SwiGLU 是激活函数吗？**
不是。SiLU 是激活函数；SwiGLU 是“带门控的前馈层”。作业里把它放在 activation 相关章节讲，是编排习惯。

**Q2：为什么门控用 SiLU，不用 sigmoid？**
sigmoid 的门被限制在 $(0,1)$，容易饱和；SiLU 允许负值和大于 1 的放大，更光滑。Shazeer 的实验表明 SwiGLU 通常优于原版 GLU。

**Q3：为什么不用 bias？**
现代 LLM 通常省略。RMSNorm + 残差的结构下，bias 的作用有限；省参数、省带宽。作业明确要求无 bias。

**Q4：$d_{ff}$ 必须是 $8/3D$ 吗？**
不是。$8/3$ 是“与经典 $4D$ FFN 等参数”的推导结果，是一个默认值。实际模型有 $2.69D$、$3.5D$、$4D$、$5.29D$ 等各种选择。

**Q5：SwiGLU 能替代注意力吗？**
不能。它逐位置计算，不跨 token 交换信息。没有注意力，模型无法利用上下文。

**Q6：参数量一样时，SwiGLU 一定更好吗？**
不一定。它是经验上常常更好，但不是定理。数据、训练策略、初始化都会影响结论；要做受控实验。

**Q7：SiLU 和 Swish 是同一个东西吗？**
是。Swish 一般写作 $x\sigma(\beta x)$，$\beta=1$ 时就是 SiLU。

**Q8：门控值可以是负数吗？**
可以。SiLU 在 $x<0$ 时为负，所以门控不仅能把通道关掉，还能把内容符号翻转。这是它与 sigmoid 门的重要区别。

**Q9：为什么输出还要一个 $W_2$？**
因为中间是 $d_{ff}$ 维，残差流是 $D$ 维；必须压回 $D$ 才能相加。$W_2$ 同时负责把高维特征组合成最终输出。

**Q10：SwiGLU 和 GeGLU 哪个好？**
两者都是门控 FFN，差别在门控激活（SiLU vs GELU）。经验上接近，具体取决于模型和数据；LLaMA 选 SwiGLU，Gemma 选 GeGLU。

**Q11：为什么说 FFN 是“键值记忆”？**
$W_1$ 的行像“键”，检测输入模式；$W_2$ 的列像“值”，决定写入什么。激活强度是写入权重。这是一种有实验支持的解释，不是严格定义。

**Q12：SwiGLU 能缓解梯度消失吗？**
它提供了两条梯度路径，负区间也不会像 ReLU 那样完全断流，所以**有助于**优化；但不能完全解决深层网络的梯度问题，仍需要残差、pre-norm 等。

---

## 9. 小结与自测

### 9.1 一页速查

```text
公式:   SwiGLU(x) = W2( SiLU(W1 x) ⊙ (W3 x) )
SiLU:   SiLU(x) = x * sigmoid(x) = x / (1 + e^-x)
导数:   SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
形状:   (..., D) --W1/W3--> (..., d_ff) --W2--> (..., D)
参数:   3 * D * d_ff        （无 bias）
默认:   d_ff = round(8D/3 / 64) * 64
位置:   pre-norm Block 的第二个子层，逐位置，不跨 token
作用:   非线性 + 输入相关门控 + 模型容量
占比:   等参数下 FFN ≈ 2/3 的 Block 参数
作业:   无 bias；用 F.silu / torch.sigmoid；实现 run_swiglu；
        uv run pytest -k test_swiglu
```

### 9.2 自测题

1. 为什么说“没有 FFN 的 Transformer 只剩下 token mixing”？注意力到底缺了什么？
2. 写出 SwiGLU 的公式，并标出每个矩阵的形状。
3. SiLU 的导数是什么？它在负区间的最小值大约是多少？为什么这比 ReLU 更不容易“死”？
4. GLU 的门控为什么用“乘”而不是“加”？从二次型的角度解释。
5. $d_{ff}=8/3D$ 是怎么推出来的？如果 $d_{ff}=4D$，SwiGLU 的参数量是多少？
6. 为什么 $d_{ff}$ 要取 64 的倍数？这是数学要求还是工程要求？
7. 在 `run_swiglu` 里，为什么是 `x @ w1_weight.T` 而不是 `w1_weight @ x`？
8. 如果手写 `exp(x)/(1+exp(x))` 实现 sigmoid，`x=1000` 时会发生什么？应该怎么改？
9. 作业测试的 `d_ff=128`、`d_model=64`，为什么不是 $8/3$？你的实现应该从哪里拿 `d_ff`？
10. SwiGLU 的梯度有几条路径？分别乘了什么？为什么说它“有助于”但不能“完全解决”梯度消失？
11. 做 SwiGLU vs SiLU 消融时，如何区分“门控的贡献”和“参数更多的贡献”？
12. 为什么说 FFN 可以看成一个“键值记忆”？$W_1$、$W_2$ 分别对应什么？

> 参考答案在 §1–§6 对应小节；先自己写一遍，再回去核对。

### 9.3 动手清单

- [ ] 手算一个 2 维输入的 SwiGLU（取 $d_{ff}=2$），核对每一步形状；
- [ ] 跑通 §7.1 的脚本，确认误差、参数量、SiLU 数值表；
- [ ] 跑过 `pytest -k test_swiglu`（仓库的 `run_swiglu` 已接好，可以直接跑）；
- [ ] 做一次“去掉门控”的对比实验；
- [ ] 扫 3 个 $d_{ff}$，记录参数量、吞吐、验证集 loss；
- [ ] 画一张门控值分布图，观察训练前后的变化。

---

## 10. 延伸阅读

**论文**

- Shazeer, *GLU Variants Improve Transformer*, 2020. arXiv:2002.05202 —— SwiGLU 的提出与系统比较。
- Dauphin et al., *Language Modeling with Gated Convolutional Networks*, 2017 —— 原版 GLU。
- Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*, 2016 —— GELU。
- Elfwing et al., *Sigmoid-Weighted Linear Units for Neural Network Function Approximation*, 2017 —— SiLU/Swish。
- Vaswani et al., *Attention Is All You Need*, 2017 —— Transformer 与经典 FFN。
- Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories*, 2021 —— FFN 的键值记忆解释。
- Grattafiori et al., *The Llama 3 Herd of Models*, 2024 —— LLaMA 3 的 SwiGLU 配置。
- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, 2022 —— PaLM 的 SwiGLU + 4x 配置。

**课程与文档**

- CS336 课程主页：<https://cs336.stanford.edu>
- 作业讲义 `cs336_assignment1_basics.pdf` §3.4.2
- PyTorch 文档：`torch.nn.functional.silu`、`torch.sigmoid`

**同目录文档**

- `Transformer 模块.md`：从文本到下一个词预测的完整链路，§4.8 是 SwiGLU 的简要版。
- `Attention.md`：注意力的 Q/K/V、掩码与多头机制。
- `Causal_Multi-Head_self-attention.md`：因果多头注意力的实现细节。
- `Embedding_Linear.md`：线性层与嵌入层的基础。
- `BPE算法原理.md`：输入阶段的分词。

> **最后一句**：SwiGLU 的公式只有一行，但背后是“注意力负责交换、FFN 负责计算”的分工，以及“门控 + 等参数设计”的工程权衡。把这两件事想清楚，比记住 $8/3$ 更有用。
