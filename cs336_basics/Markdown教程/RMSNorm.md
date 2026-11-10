# RMSNorm：给大模型“调音量”的归一化层

> 本文是 CS336 Assignment 1（Building a Transformer LM）的配套概念读物，对应讲义 §3.4.1 的 `rmsnorm`。它面向零基础读者，把 RMSNorm **是什么、解决什么问题、公式怎么推、代码怎么写、作业怎么过** 这条链路讲清楚。
>
> 读完你应该能回答五个问题：
>
> 1. RMSNorm 和 LayerNorm 到底差在哪？为什么说它“只缩放、不挪中心”？
> 2. 为什么大模型要在每个子层前做归一化？不做会怎样？
> 3. 公式里的 RMS 是什么？为什么 $\epsilon$ 要放在根号里面？
> 4. 为什么输入要先升到 `float32`，算完再降回原 dtype？
> 5. 在 CS336 的 `run_rmsnorm` 里，`weights` 是什么形状、应该怎么广播？

## 目录

- [0. 阅读前的准备](#0-阅读前的准备)
- [1. 什么是 RMSNorm](#1-什么是-rmsnorm)
- [2. 它有什么用](#2-它有什么用)
- [3. 核心定义与公式](#3-核心定义与公式)
- [4. 深度价值：为什么大模型离不开它](#4-深度价值为什么大模型离不开它)
- [5. 数学逻辑：一步步推导](#5-数学逻辑一步步推导)
- [6. 可运行代码](#6-可运行代码)
- [7. 联系 CS336 Assignment 1](#7-联系-cs336-assignment-1)
- [8. 常见误区 FAQ](#8-常见误区-faq)
- [9. 小结与自测](#9-小结与自测)
- [10. 延伸阅读](#10-延伸阅读)

---

## 0. 阅读前的准备

### 0.1 一句话版本

> **RMSNorm 做的事：对每个 token 的向量，算一个“整体能量” RMS，把整个向量除以这个能量，让它的均方根回到 1 附近；再乘一个可学习的逐通道增益 $g$。**

公式只有一行：

$$
\operatorname{RMSNorm}(x)=\frac{x}{\operatorname{RMS}(x)}\odot g
$$

其中：

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2+\epsilon}
$$

一句话记忆：

> **LayerNorm 是“先把数据挪到 0，再缩放”；RMSNorm 是“不挪 0，只缩放”。**

### 0.2 符号表

| 符号 | 含义 | 一句话解释 |
| --- | --- | --- |
| $B$ | batch size | 一次处理多少条序列 |
| $T$ | sequence length | 一条序列有多少个 token |
| $D$ 或 $d_{model}$ | 隐藏维度 | 每个 token 的向量有多少个数 |
| $x$ | 一个 token 的向量 | $x\in\mathbb{R}^{D}$ |
| $x_i$ | 向量的第 $i$ 个分量 | 也常叫“第 $i$ 个通道” |
| $\mu$ | 均值 | $\frac{1}{D}\sum_i x_i$ |
| $\sigma^2$ | 方差 | $\frac{1}{D}\sum_i (x_i-\mu)^2$ |
| $\operatorname{RMS}(x)$ | 均方根 | Root Mean Square，$\sqrt{\text{mean}(x^2)}$ |
| $g$ | 可学习增益 | 形状 $(D,)$，每个通道一个数 |
| $\epsilon$ | 数值稳定常数 | 通常取 $10^{-5}$，防止除以 0 |
| $\odot$ | 逐元素乘 | 两个同形状张量对应位置相乘 |

### 0.3 你需要的前置知识

不需要会推 Transformer，但下面几件事要熟：

1. **向量和形状**：一个 token 是一个 $D$ 维向量；一批 token 是 $(B,T,D)$；
2. **逐元素运算**：两个 $(D,)$ 向量可以逐位置相加、相乘；
3. **广播**：$(B,T,D)$ 可以和 $(D,)$ 逐通道相乘，PyTorch 会自动把 $(D,)$ 复制到每个位置；
4. **均值/平方/开根号**：知道“先平方、再平均、再开根”就是均方根；
5. **归一化**：把数值调整到某个标准范围，让训练更稳定。

### 0.4 RMSNorm 在整张图里的位置

在 pre-norm Transformer（先归一化、再进子层）里，每个 Block 有两个子层，每个子层前都有一次 RMSNorm：

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
        └── 重复 N 层，最后再做一次 RMSNorm ──► LM head
```

三个关键事实：

- **逐位置**：位置 $t$ 的 RMSNorm 只看位置 $t$ 的向量，不会看别的 token；
- **形状不变**：输入 $(B,T,D)$，输出还是 $(B,T,D)$，所以能直接接残差；
- **不混信息**：它不改变向量的“方向”，只改变整体长度（在没有 gain 时）。

> 注意最后那个“final RMSNorm”：pre-norm 结构里，残差主干从输入到输出是“干净”的，所以最后一层 Block 之后、送进 LM head 之前，还要再归一化一次。

### 0.5 阅读路线

| 你想知道 | 看哪节 |
| --- | --- |
| 它到底在干什么 | §1 |
| 它解决什么问题、什么时候用 | §2 |
| 公式和形状 | §3 |
| 为什么大模型需要它 | §4 |
| 公式怎么一步步推出来 | §5 |
| 直接跑代码 | §6 |
| 作业接口和测试怎么过 | §7 |
| 常见坑 | §8、§9 |

---

## 1. 什么是 RMSNorm

### 1.1 生活化类比：调音台

想象你在录一首歌，录音里有 $D$ 个声道（鼓、贝斯、吉他、人声……）。每个声道的音量差别很大：

- 鼓可能很响，数值是 100；
- 人声可能很轻，数值是 0.1；
- 有些声道整体偏正，有些整体偏负。

直接把这段录音送进下一级设备，可能会出现两个问题：

1. **整体太响**：数值太大，后面一放大就“爆音”（溢出）；
2. **整体太轻**：数值太小，后面一缩小就“听不见”（梯度消失）。

调音台的做法是：

> 先测一下这段录音的**整体能量**（RMS），然后把整段录音的音量统一调到标准水平；如果某个声道还需要单独调整，再用**每通道的增益旋钮** $g$ 微调。

RMSNorm 就是这样一个“自动增益”：

- **RMS**：这段录音的整体能量（不是平均值，而是“平方的平均再开根”）；
- **除以 RMS**：把整体能量拉到 1 附近；
- **乘 $g$**：给每个通道一个可学习的音量旋钮，模型可以自己决定“哪个通道该放大、哪个该压低”。

注意：调音台**没有**把录音的“直流偏移”（也就是整体偏正或偏负的部分）挪到 0——它只调音量，不调中心。这正是 RMSNorm 和 LayerNorm 的最大区别。

### 1.2 一句话定义

> **RMSNorm 对每个 token 的 $D$ 维向量，计算均方根 RMS，把向量按这个值整体缩放，再逐通道乘一个可学习增益。**

它不减去均值，也不加 bias。它只做两件事：**缩放** + **逐通道增益**。

### 1.3 和 LayerNorm 的第一印象

LayerNorm 的公式：

$$
\operatorname{LayerNorm}(x)_i
=g_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+b_i
$$

RMSNorm 的公式：

$$
\operatorname{RMSNorm}(x)_i
=g_i\frac{x_i}{\sqrt{\frac{1}{D}\sum_j x_j^2+\epsilon}}
$$

把两者放在一起看：

| 操作 | LayerNorm | RMSNorm |
| --- | --- | --- |
| 减均值 | 是：$x_i-\mu$ | **否** |
| 除标准差 | 是：$\sqrt{\sigma^2+\epsilon}$ | 否 |
| 除均方根 | 否 | 是：$\sqrt{\text{mean}(x^2)+\epsilon}$ |
| 乘增益 $g$ | 是 | 是 |
| 加 bias $b$ | 是 | **否** |

所以：

> **RMSNorm 不是“LayerNorm 去掉 bias”，而是“连减均值也一起去掉”的简化版。**

### 1.4 三个关键点

1. **没有均值中心化**：输出均值不一定是 0；
2. **用 RMS 而不是标准差**：分母是“平方的平均”开根，包含均值的影响；
3. **只有增益，没有 bias**：$g$ 形状是 $(D,)$，初始化为全 1。

### 1.5 三个“不是”

| 说法 | 纠正 |
| --- | --- |
| “RMSNorm 就是 LayerNorm 去掉 bias” | 不止。它连**减均值**也去掉了，用的是均方根而不是标准差 |
| “RMSNorm 是 BatchNorm 的一种” | 不是。它**逐 token** 计算，不使用 batch 统计量，也没有 running mean/var |
| “RMSNorm 会混合不同 token 的信息” | 不会。它逐位置作用，形状不变，不跨 token |

---

## 2. 它有什么用

### 2.1 它解决的问题

深度网络里，每一层的输出都会经过一堆矩阵乘法和激活函数。如果没有归一化，数值尺度很容易“漂”：

- 有的层输出越来越大 → 前向溢出、梯度爆炸；
- 有的层输出越来越小 → 信号消失、梯度消失；
- 不同层、不同 token 的尺度差异很大 → 优化器很难选一个合适的学习率；
- 混合精度训练时，平方、除法等操作容易溢出。

RMSNorm 的作用就是**在每个子层前，把输入的尺度重新锚定到 1 附近**，让后面的矩阵乘法和激活函数工作在稳定的数值范围里。

> 注意：这不是说“归一化解决了所有训练问题”。它只是把尺度这个变量控制住，让优化更容易；残差、初始化、学习率、数据质量仍然重要。

### 2.2 适用场景

| 场景 | 适合吗 | 说明 |
| --- | --- | --- |
| Transformer 的 pre-norm Block | **非常适合** | LLaMA 系模型的标准配置 |
| 大语言模型（LLM） | **非常适合** | 训练稳定、计算便宜、易融合 |
| 视觉 Transformer / 语音模型 | **可以用** | 很多新模型也用 RMSNorm 替代 LayerNorm |
| 需要逐 token 独立归一化 | **适合** | 不依赖 batch，推理时行为一致 |
| 均值偏移很重要的特征 | **谨慎** | RMSNorm 不消除均值，LayerNorm 可能更合适 |
| 很小的模型 / 简单任务 | **不一定** | LayerNorm 已经够用，收益可能不明显 |

### 2.3 代价与注意点

| 代价 | 说明 |
| --- | --- |
| 表达能力略减 | 没有减均值、没有 bias，理论上比 LayerNorm 少一点自由度 |
| 对均值敏感 | 如果输入整体偏移很大，RMS 会被均值主导，输出可能“挤在一起” |
| 不是万能 | 有些任务/模型上 LayerNorm 仍然更好，需要实验 |
| 实现细节多 | $\epsilon$ 的位置、dtype 提升、广播方向都容易写错 |

### 2.4 与 LayerNorm、BatchNorm 的对比

| | LayerNorm | RMSNorm | BatchNorm |
| --- | --- | --- | --- |
| 归一化范围 | 每个 token 的所有通道 | 每个 token 的所有通道 | 每个通道跨 batch |
| 减均值 | 是 | 否 | 是（用 batch 均值） |
| 除什么 | 标准差 | 均方根 | batch 标准差 |
| 是否依赖 batch | 否 | 否 | 是 |
| 推理时行为 | 与训练一致 | 与训练一致 | 用 running stats |
| 可学习参数 | $g,b$，共 $2D$ | $g$，共 $D$ | $g,b$ |
| 典型用途 | 原版 Transformer、CV | LLaMA 系 LLM | CNN |

> 对新手最重要的一句话：**RMSNorm 和 LayerNorm 都是“逐 token”的，BatchNorm 是“跨 batch”的。** 不要把三者混为一谈。

---

## 3. 核心定义与公式

### 3.1 定义

给定一个 token 的向量 $x\in\mathbb{R}^{D}$，RMSNorm 定义为：

$$
\boxed{
\operatorname{RMSNorm}(x)=\frac{x}{\operatorname{RMS}(x)}\odot g
}
$$

其中：

$$
\boxed{
\operatorname{RMS}(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2+\epsilon}
}
$$

- $g\in\mathbb{R}^{D}$ 是可学习的**增益**（gain），每个通道一个数；
- $\epsilon$ 是防止除零的小常数，作业里取 $1\times10^{-5}$；
- $\odot$ 是逐元素乘。

逐分量写出来：

$$
\operatorname{RMSNorm}(x)_i
=g_i\cdot\frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}}
$$

### 3.2 公式拆解：五步

| 步骤 | 操作 | 形状 | 说明 |
| ---: | --- | --- | --- |
| 1 | $s_i=x_i^2$ | $(D,)$ | 每个通道平方 |
| 2 | $\bar{s}=\frac{1}{D}\sum_i s_i$ | 标量 | 平方的平均（mean square） |
| 3 | $r=\sqrt{\bar{s}+\epsilon}$ | 标量 | 加 $\epsilon$ 后开根，得到 RMS |
| 4 | $\hat{x}_i=x_i/r$ | $(D,)$ | 整个向量除以 RMS |
| 5 | $y_i=g_i\hat{x}_i$ | $(D,)$ | 逐通道乘增益 |

注意第 2 步是“平方的平均”，不是“平均的平方”；第 3 步的 $\epsilon$ 在**根号里面**。

### 3.3 形状与广播

对一个 batch：

```text
x:        (B, T, D)
x.pow(2): (B, T, D)
.mean(-1, keepdim=True): (B, T, 1)
+ eps, sqrt:             (B, T, 1)   ← 每个 token 一个 RMS
x / rms:                 (B, T, D)   ← (B,T,1) 自动广播到 D
* g:                     (B, T, D)   ← (D,) 自动广播到 (B,T,D)
```

三个关键点：

- **`dim=-1`**：只在最后一维（通道维）上求平均，绝不跨 token；
- **`keepdim=True`**：保留最后一维为 1，才能正确广播；
- **$g$ 的形状是 $(D,)$**：不是 $(D,D)$，也不是标量。

### 3.4 手算一个例子

取 $x=[1,2,3,4]$，$D=4$，$\epsilon=10^{-5}$，$g=[1,1,1,1]$。

**第一步：平方**

$$
x^2=[1,4,9,16]
$$

**第二步：求平均**

$$
\bar{s}=\frac{1+4+9+16}{4}=\frac{30}{4}=7.5
$$

**第三步：加 $\epsilon$ 开根**

$$
r=\sqrt{7.5+10^{-5}}\approx2.7386
$$

**第四步：除以 RMS**

$$
\hat{x}=\frac{[1,2,3,4]}{2.7386}
\approx[0.3651,\ 0.7303,\ 1.0954,\ 1.4606]
$$

**第五步：乘增益**

$g$ 全是 1，所以输出不变。

结果：

| 量 | 值 |
| --- | --- |
| $\operatorname{RMS}(x)$ | 2.7386 |
| $\operatorname{RMSNorm}(x)$ | [0.3651, 0.7303, 1.0954, 1.4606] |
| 输出均值 | 0.9129（**不是 0**） |
| 输出 RMS | 1.0000 |

对比 LayerNorm（同样的 $x$，无仿射）：

| 量 | LayerNorm |
| --- | --- |
| 均值 $\mu$ | 2.5 |
| 标准差 $\sigma$ | 1.1180 |
| 输出 | [-1.3416, -0.4472, 0.4472, 1.3416] |
| 输出均值 | 0 |
| 输出标准差 | 1 |

这张表把区别说得很清楚：

> **LayerNorm 的输出被“挪”到均值为 0；RMSNorm 的输出没有被挪，均值仍然是 0.9129。**

### 3.5 输出性质

在没有增益（或 $g$ 全 1）时：

1. **输出 RMS ≈ 1**：
   $$
   \frac{1}{D}\sum_i\hat{x}_i^2
   =\frac{\frac{1}{D}\sum_i x_i^2}{\frac{1}{D}\sum_i x_i^2+\epsilon}
   \approx1
   $$
2. **输出均值不一定为 0**：因为没有减均值；
3. **输出方向与输入相同**：除以一个正数不改变方向；
4. **逐 token 独立**：不同 token 之间互不影响。

有增益 $g$ 时，输出的每个通道会被 $g_i$ 单独缩放，方向和 RMS 都可能改变。

### 3.6 与 LayerNorm 的公式对比

| | LayerNorm | RMSNorm |
| --- | --- | --- |
| 中心 | $\mu=\frac{1}{D}\sum x_i$ | 无 |
| 尺度 | $\sigma=\sqrt{\frac{1}{D}\sum(x_i-\mu)^2}$ | $\operatorname{RMS}=\sqrt{\frac{1}{D}\sum x_i^2}$ |
| 归一化 | $(x-\mu)/\sqrt{\sigma^2+\epsilon}$ | $x/\sqrt{\text{mean}(x^2)+\epsilon}$ |
| 仿射 | $g\odot\hat{x}+b$ | $g\odot\hat{x}$ |
| 输出均值 | 0 | 不一定 |
| 输出尺度 | 方差 ≈ 1 | 均方 ≈ 1 |

一个有用的恒等式：

$$
\operatorname{RMS}(x)^2
=\frac{1}{D}\sum_i x_i^2
=\sigma^2+\mu^2
$$

也就是说，**RMS 同时包含“波动”和“整体偏移”**；LayerNorm 的标准差只包含“波动”。当 $\mu\approx0$ 时，RMS ≈ 标准差，RMSNorm 就非常接近“不减均值的 LayerNorm”。

### 3.7 两种关键行为：缩放与平移

| 输入变换 | LayerNorm（不计 bias） | RMSNorm |
| --- | --- | --- |
| 正比例缩放：$\alpha x$（$\alpha>0$） | 输出不变 | 输出不变（忽略 $\epsilon$） |
| 加常数：$x+c$ | 输出不变 | **输出会变** |
| 乘负数：$-\alpha x$（$\alpha>0$） | 输出符号翻转 | 输出符号翻转 |
| 通道重排 | 跟着重排 | 跟着重排 |

> 这里只比较“归一化部分”。RMSNorm 本身没有 bias；LayerNorm 如果带 bias，bias 不随输入的缩放/平移改变，结论会多出一个常数项。

- **缩放不变**：因为分子分母同时乘 $\alpha$，约掉了；
- **不平移不变**：因为分母的 RMS 会随 $c$ 改变，而分子没有减去 $c$。

> 这两个性质是理解 RMSNorm 的钥匙：**它对“整体缩放”不敏感，但对“整体偏移”敏感**。它只负责把音量拉到标准，不负责把中心挪到 0——这正是它和 LayerNorm 的分工差异。

---

## 4. 深度价值：为什么大模型离不开它

> 先说清楚：不是“没有 RMSNorm 就训不了”，LayerNorm 也能训。准确的说法是：**在现代大模型的 pre-norm 架构里，RMSNorm 已经成了默认选择之一**，因为它同时满足了“稳”和“快”。

### 4.1 训练稳定性：把尺度锚在 1 附近

深层 Transformer 的残差流是一条“高速公路”：

$$
x_{l+1}=x_l+F_l(\operatorname{Norm}(x_l))
$$

- 如果没有 Norm，$F_l$ 的输出尺度可能随层数指数级放大或缩小；
- 有了 Norm，每个子层的输入都被拉回稳定范围，$F_l$ 不会因为输入太大/太小而饱和或消失；
- 残差加法要求两边尺度相近，Norm 让这件事更容易满足。

RMSNorm 的输出 RMS ≈ 1（在没有 gain 时），意味着：

- 线性层的输入尺度稳定，权重初始化更容易设；
- 激活函数不容易进入饱和区；
- 梯度不会因为尺度连乘而爆炸/消失；
- 学习率可以设得更激进一些。

### 4.2 计算效率：少一次减均值，少一次方差

LayerNorm 要做：

1. 求均值（一次 reduction，也就是把 $D$ 个数合成一个数）；
2. 求方差（再围绕均值算一次）；
3. 每个元素减均值；
4. 除以标准差；
5. 乘 $g$、加 $b$。

RMSNorm 要做：

1. 求平方的均值（一次 reduction）；
2. 加 $\epsilon$、开根；
3. 每个元素除以 RMS；
4. 乘 $g$。

少了“减均值”和“bias”，也少了一次围绕均值的方差计算。对 GPU 来说，归一化往往是**访存瓶颈**（读一遍、写一遍），少一次读写、少一个 GPU 算子（kernel），就可能带来实际加速。

> 不要背具体的加速百分比：不同模型、不同框架、不同融合方式差异很大。你只需要记住：**RMSNorm 的操作更少，更容易和前后算子融合。**

### 4.3 参数效率：只要 $D$ 个增益

| | LayerNorm | RMSNorm |
| --- | --- | --- |
| 增益 $g$ | $D$ | $D$ |
| bias $b$ | $D$ | 0 |
| 合计 | $2D$ | $D$ |

在大模型里，这点参数不算多，但：

- 少一组参数，少一次加法；
- 权重加载、保存、通信都更简单；
- 对多卡并行训练更友好。

### 4.4 混合精度：平方容易溢出，所以要升 float32

半精度（float16）的最大值约为 65504。如果激活值是 300，平方就是 90000，直接溢出成 `inf`；再开根、相除就可能得到 `nan`。

所以 RMSNorm 的实现通常：

```python
in_dtype = x.dtype
x = x.to(torch.float32)   # 平方/求和在 float32 里做
# ... 归一化 ...
return result.to(in_dtype)  # 再降回原来的 dtype
```

这也是 CS336 讲义特别强调的一点。

### 4.5 为什么 LLaMA 系模型选它

- **稳**：pre-norm + RMSNorm 的组合在大规模训练里表现稳定；
- **快**：操作少，容易融合；
- **简单**：没有 bias，没有均值，代码短，硬件友好；
- **生态**：LLaMA、Mistral、Qwen、Gemma 等模型都采用 RMSNorm，推理框架和量化工具也跟着优化。

> 但请保留一个“经验主义”的清醒：RMSNorm 的成功有很大一部分是**工程经验**，不是“数学上证明了 LayerNorm 一定差”。在具体任务上，LayerNorm 仍然可能是更好的选择。

### 4.6 一个重要的边界

RMSNorm **不消除均值**。如果某个特征的均值携带重要信息，或者不同通道的整体偏移差异很大，RMSNorm 可能会把有用的“直流分量”一起缩放掉。这时 LayerNorm 的减均值可能更合适。

> 选择归一化层，和选择激活函数一样：先理解它的归纳偏置，再用实验说话。

---

## 5. 数学逻辑：一步步推导

这一节把公式拆开，每一步都解释“为什么”。

### 5.1 从 LayerNorm 出发

LayerNorm 对向量 $x\in\mathbb{R}^{D}$ 做两件事：

1. **中心化**：减去均值，把数据中心挪到 0；
2. **缩放**：除以标准差，把波动缩到 1。

$$
\mu=\frac{1}{D}\sum_i x_i,\qquad
\sigma^2=\frac{1}{D}\sum_i(x_i-\mu)^2
$$

$$
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
y_i=g_i\hat{x}_i+b_i
$$

RMSNorm 的问题是：**这两件事都必要吗？**

### 5.2 去掉均值中心化

RMSNorm 的假设是：

> 对深层网络来说，**重新缩放**比**重新中心化**更关键；均值中心化可以省掉。

于是把 $x_i-\mu$ 换回 $x_i$：

$$
\hat{x}_i=\frac{x_i}{\text{某个尺度}}
$$

剩下的问题就是：分母用什么尺度？

### 5.3 从“标准差”换成“均方根”

标准差是“围绕均值的波动”：

$$
\sigma=\sqrt{\frac{1}{D}\sum_i(x_i-\mu)^2}
$$

如果不减均值，最自然的替代就是“围绕 0 的波动”，也就是**均方根**：

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{D}\sum_i x_i^2}
$$

两者的关系：

$$
\operatorname{RMS}(x)^2=\sigma^2+\mu^2
$$

所以 RMS 同时包含“波动”和“整体偏移”。当 $\mu\approx0$ 时，RMS ≈ 标准差；当 $\mu$ 很大时，RMS 主要反映均值。

### 5.4 加上 $\epsilon$：防止除以 0

如果 $x$ 全是 0，RMS 就是 0，除以 0 会得到 `inf` 或 `nan`。所以分母加一个很小的 $\epsilon$：

$$
r=\sqrt{\frac{1}{D}\sum_i x_i^2+\epsilon}
$$

- $\epsilon$ 放在**根号里面**：$r=\sqrt{\bar{s}+\epsilon}$；
- 如果放在外面：$r=\sqrt{\bar{s}}+\epsilon$，也能防除零，但数值不同；
- 讲义采用“里面”的写法，作业按这个实现。

当 $x=0$ 时：

$$
r=\sqrt{0+\epsilon}=\sqrt{\epsilon},\qquad
\hat{x}_i=\frac{0}{\sqrt{\epsilon}}=0
$$

输出是 0，不会出现 `nan`。

### 5.5 归一化：除以 RMS

$$
\hat{x}_i=\frac{x_i}{r}
=\frac{x_i}{\sqrt{\frac{1}{D}\sum_j x_j^2+\epsilon}}
$$

这一步把整个向量按同一个正数缩放，所以：

- 向量方向不变；
- 向量的整体长度被拉到 1 附近；
- 每个通道之间的相对比例不变。

### 5.6 输出 RMS 为什么约等于 1

计算归一化后向量的均方：

$$
\frac{1}{D}\sum_i\hat{x}_i^2
=\frac{1}{D}\sum_i\frac{x_i^2}{\frac{1}{D}\sum_j x_j^2+\epsilon}
=\frac{\frac{1}{D}\sum_i x_i^2}{\frac{1}{D}\sum_i x_i^2+\epsilon}
$$

令 $S=\frac{1}{D}\sum_i x_i^2$，则：

$$
\frac{1}{D}\sum_i\hat{x}_i^2=\frac{S}{S+\epsilon}\approx1
$$

所以：

$$
\operatorname{RMS}(\hat{x})=\sqrt{\frac{S}{S+\epsilon}}\approx1
$$

$\epsilon$ 的作用是“防止 $S=0$ 时除零”，同时让结果略小于 1；$\epsilon$ 很小的时候，差异可以忽略。

### 5.7 缩放不变性：为什么输入乘 10 输出不变

设 $\alpha>0$，计算 $\alpha x$ 的 RMS：

$$
\operatorname{RMS}(\alpha x)
=\sqrt{\frac{1}{D}\sum_i(\alpha x_i)^2+\epsilon}
=\sqrt{\alpha^2\cdot\frac{1}{D}\sum_i x_i^2+\epsilon}
\approx\alpha\operatorname{RMS}(x)
$$

忽略 $\epsilon$ 时：

$$
\frac{\alpha x}{\operatorname{RMS}(\alpha x)}
\approx\frac{\alpha x}{\alpha\operatorname{RMS}(x)}
=\frac{x}{\operatorname{RMS}(x)}
$$

所以输出不变。严格来说，$\epsilon$ 不随 $\alpha$ 缩放，所以只是**近似**不变；$\epsilon$ 很小时误差很小。

如果 $\alpha<0$：

$$
\operatorname{RMS}(\alpha x)\approx|\alpha|\operatorname{RMS}(x)
$$

分母是正的，分子带负号，所以输出**符号翻转**。

### 5.8 平移不变性不成立：为什么加 100 输出会变

设 $x'=x+c$，则：

$$
\operatorname{RMS}(x')=\sqrt{\frac{1}{D}\sum_i(x_i+c)^2+\epsilon}
$$

这个值一般不等于 $\operatorname{RMS}(x)$，所以：

$$
\frac{x+c}{\operatorname{RMS}(x+c)}\neq\frac{x}{\operatorname{RMS}(x)}
$$

举例：$x=[1,2,3,4]$，$x+100=[101,102,103,104]$。

- 原来输出：$[0.3651,0.7303,1.0954,1.4606]$；
- 加 100 后输出：$[0.9853,0.9951,1.0048,1.0146]$。

差异很大。这说明 RMSNorm **不消除整体偏移**；而 LayerNorm 对 $x+c$ 的输出完全不变（忽略 $\epsilon$）。

### 5.9 梯度长什么样（进阶，可跳过）

如果只想理解“怎么用”，这一小节可以跳过；想理解“为什么它能训”，可以看下去。

设：

$$
r=\sqrt{\frac{1}{D}\sum_k x_k^2+\epsilon},\qquad
y_i=g_i\frac{x_i}{r}
$$

对 $x_j$ 求偏导：

$$
\frac{\partial y_i}{\partial x_j}
=\frac{g_i}{r}\left(\delta_{ij}-\frac{x_i x_j}{D r^2}\right)
$$

其中 $\delta_{ij}$ 是“$i=j$ 时为 1，否则为 0”的指示符号。

设上游梯度 $u_i=\partial L/\partial y_i$，则：

$$
\frac{\partial L}{\partial x_j}
=\frac{g_j u_j}{r}
-\frac{x_j}{D r^3}\sum_i g_i u_i x_i
$$

写成向量形式：

$$
\frac{\partial L}{\partial x}
=\frac{g\odot u}{r}
-\frac{x}{D r^3}\left(x^\top(g\odot u)\right)
$$

怎么理解：

- 第一项：把上游梯度按 $g/r$ 缩放；
- 第二项：减掉梯度中沿 $x$ 方向的径向分量，它来自分母对 $x$ 的依赖；
- 这跟 LayerNorm 的梯度结构很像，但**没有减均值那一项**；所以计算更简单。

> 对新手来说，记住结论就够了：**RMSNorm 的梯度会把“沿着向量方向”的分量投影掉一部分，从而让尺度稳定；它不需要计算均值，所以比 LayerNorm 少一些操作。**

### 5.10 为什么可以去掉均值中心化

这不是一条定理，而是一个**有实验支持的设计假设**：

1. 在深层网络里，真正让训练变难的是**尺度漂移**；中心漂移相对次要；
2. 很多层的输入已经近似零均值（前一层有 bias/激活/残差），再减均值的收益不大；
3. 去掉均值可以减少计算、保留“整体偏移”这一信息；
4. 实验上，RMSNorm 在语言模型等任务上表现不输 LayerNorm，甚至更好。

> 所以：**RMSNorm 是一个“简化 + 归纳偏置”的设计，不是“LayerNorm 的严格升级版”。**

---

## 6. 可运行代码

下面是一个完整、可独立运行的示例（纯 PyTorch，CPU 即可；已在 PyTorch 2.x 上验证）。它实现了：

- `RMSNorm` 模块；
- 与手写公式、`F.rms_norm` 的数值对照；
- 手算例子、输出 RMS、输出均值；
- 缩放/平移行为、全 0 输入、半精度输入。

### 6.1 完整脚本

```python
"""RMSNorm 最小实现与自检脚本（纯 PyTorch，CPU 可跑）。

运行：python rmsnorm_demo.py
"""
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。"""

    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 可学习的逐通道增益，初始化为全 1：
        # 一开始等价于“只归一化、不额外缩放”，让模型自己学要不要放大/缩小某个通道
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        # 平方在 float16 下容易溢出，先升到 float32 做归一化
        x = x.to(torch.float32)

        # 1) 每个 token 的所有通道平方
        # 2) 在最后一维（通道维）求平均，keepdim 保留 (..., 1) 便于广播
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # 3) 加 eps 后开根号，得到每个 token 的 RMS
        rms = torch.sqrt(mean_square + self.eps)

        # 4) 整个向量除以 RMS
        x_hat = x / rms

        # 5) 逐通道乘可学习增益
        out = x_hat * self.weight

        # 6) 还原输入 dtype
        return out.to(in_dtype)


def rmsnorm_reference(x, weight, eps=1e-5):
    """按公式手写一遍，作为数值对照。"""
    x32 = x.to(torch.float32)
    rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x32 / rms * weight).to(x.dtype)


def rms(t):
    """计算最后一维的均方根，便于打印。"""
    return t.float().pow(2).mean(dim=-1).sqrt()


def main():
    # ---------- 1. 手算例子 ----------
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    norm = RMSNorm(4)
    y = norm(x)

    print("x               :", x.tolist())
    print("RMS(x)          :", torch.sqrt(x.pow(2).mean()).item())
    print("RMSNorm(x)      :", [round(v, 4) for v in y.tolist()])
    print("输出均值        :", round(y.mean().item(), 4))
    print("输出 RMS        :", round(rms(y).item(), 4))

    ln = nn.LayerNorm(4, eps=1e-5, elementwise_affine=False)
    print("LayerNorm(x)    :", [round(v, 4) for v in ln(x).tolist()])
    print("LayerNorm 均值  :", round(ln(x).mean().item(), 4))

    # ---------- 2. batch 形状与数值对照 ----------
    B, T, D = 2, 4, 8
    xb = torch.randn(B, T, D) * 5 + 3  # 刻意让均值不为 0
    norm = RMSNorm(D)
    yb = norm(xb)
    ref = rmsnorm_reference(xb, norm.weight)

    print("\n输入形状        :", tuple(xb.shape))
    print("输出形状        :", tuple(yb.shape))
    print("与手写参考误差  :", f"{(yb - ref).abs().max().item():.3e}")

    if hasattr(F, "rms_norm"):
        ref2 = F.rms_norm(xb, (D,), norm.weight, 1e-5)
        print("与 F.rms_norm 误差:", f"{(yb - ref2).abs().max().item():.3e}")

    print("输入 RMS        :", [round(v, 3) for v in rms(xb).flatten().tolist()[:4]], "...")
    print("输出 RMS        :", [round(v, 3) for v in rms(yb).flatten().tolist()[:4]], "...")

    # ---------- 3. 梯度 ----------
    loss = yb.pow(2).mean()
    loss.backward()
    print("weight 梯度范数 :", f"{norm.weight.grad.norm().item():.4f}")

    # ---------- 4. 缩放 / 平移 ----------
    y_scale = norm(xb * 10)
    y_shift = norm(xb + 100)
    print("输入乘 10 的差异:", f"{(yb - y_scale).abs().max().item():.3e}")
    print("输入加 100 的差异:", f"{(yb - y_shift).abs().max().item():.3f}")

    # ---------- 5. 全 0 与半精度 ----------
    z = torch.zeros(2, 3, D)
    print("全 0 输入输出   :", f"{norm(z).abs().max().item():.3e}")

    xh = xb.half()
    yh = norm(xh)
    print("半精度输出 dtype:", yh.dtype)
    print("半精度 vs fp32  :", f"{(yh.float() - yb).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
```

### 6.2 预期输出（节选）

```text
x               : [1.0, 2.0, 3.0, 4.0]
RMS(x)          : 2.7386128902435303
RMSNorm(x)      : [0.3651, 0.7303, 1.0954, 1.4606]
输出均值        : 0.9129
输出 RMS        : 1.0
LayerNorm(x)    : [-1.3416, -0.4472, 0.4472, 1.3416]
LayerNorm 均值  : 0.0

输入形状        : (2, 4, 8)
输出形状        : (2, 4, 8)
与手写参考误差  : 0.000e+00
与 F.rms_norm 误差: 2.384e-07
输入 RMS        : [4.635, 5.586, 6.496, 6.569] ...
输出 RMS        : [1.0, 1.0, 1.0, 1.0] ...
weight 梯度范数 : 0.7556
输入乘 10 的差异: 5.960e-07
输入加 100 的差异: 2.552
全 0 输入输出   : 0.000e+00
半精度输出 dtype: torch.float16
半精度 vs fp32  : 1.162e-03
```

几个可以核对的地方：

- **手算例子**：输出均值 0.9129，说明 RMSNorm 没有把数据中心挪到 0；
- **输出 RMS = 1.0**：归一化后均方根回到 1；
- **缩放差异约 $6\times10^{-7}$**：乘 10 后输出几乎不变（$\epsilon$ 带来的微小误差）；
- **平移差异 2.552**：加 100 后输出明显变化，说明 RMSNorm 不平移不变；
- **全 0 输入输出 0**：$\epsilon$ 防止了除零；
- **半精度输入**：内部升 float32，输出仍是 float16。

### 6.3 代码逐行解释

| 代码 | 作用 | 易错点 |
| --- | --- | --- |
| `in_dtype = x.dtype` | 记住输入 dtype | 忘了最后转回去，测试会 dtype 不匹配 |
| `x = x.to(torch.float32)` | 平方/求和在 float32 里做 | 不升精度，fp16 下平方可能溢出 |
| `x.pow(2).mean(dim=-1, keepdim=True)` | 求每个 token 的平方均值 | `dim` 写错会跨 token；`keepdim=False` 无法广播 |
| `torch.sqrt(mean_square + self.eps)` | 加 $\epsilon$ 后开根 | $\epsilon$ 放到根号外，公式就不一样了 |
| `x / rms` | 按 RMS 缩放 | 除反了（`rms / x`）会完全错 |
| `x_hat * self.weight` | 逐通道增益 | `weight` 形状必须是 $(D,)$ |
| `out.to(in_dtype)` | 还原 dtype | 不还原，混合精度训练会出问题 |

### 6.4 更短的写法（工程上常见）

用 `torch.rsqrt` 可以少一次除法：

```python
def rmsnorm_short(x, weight, eps=1e-5):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * rms * weight).to(in_dtype)
```

`rsqrt(t) = 1 / sqrt(t)`，数值上等价，通常更快。两种写法都可以；作业里用哪种都行，只要结果对。

### 6.5 接入作业的 adapter

> 仓库现状：`tests/adapters.py` 的 `run_rmsnorm` 已经接好（走的是 `cs336_basics/RMSNorm.py` 的 `RMSNorm` 类），
> 下面这段是"只用函数、不建模块"的等价写法，留作对照。自己实现时思路是一样的。

把下面这个函数填进 `tests/adapters.py` 的 `run_rmsnorm`：

```python
import torch


def run_rmsnorm(d_model, eps, weights, in_features):
    in_dtype = in_features.dtype
    x = in_features.to(torch.float32)

    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x / rms * weights

    return out.to(in_dtype)
```

- `weights` 形状是 `(d_model,)`，直接和 `(..., d_model)` 广播；
- `in_features` 可以有任意前导维度（`(B,T,D)`、`(T,D)`、甚至 `(D,)`）；
- 不需要 `d_model` 参数参与计算，它只是接口的一部分。

### 6.6 四个小实验

> 下面几段接在 `main()` 里运行即可（`norm`、`xb`、`y1` 都是前面定义过的变量）。实验 3 会修改增益，做完记得重新初始化或改回 1。

**实验 1：验证缩放不变性**

```python
y1 = norm(xb)
y2 = norm(xb * 10)
print((y1 - y2).abs().max())   # 应该很小，约 1e-7
```

**实验 2：验证平移会改变输出**

```python
y3 = norm(xb + 100)
print((y1 - y3).abs().max())   # 应该明显大于 0
```

**实验 3：把 gain 改成别的值**

```python
with torch.no_grad():
    norm.weight.fill_(2.0)
y4 = norm(xb)
print(y4 / y1)   # 每个通道应该约等于 2
```

**实验 4：去掉 $\epsilon$**

```python
z = torch.zeros(1, 1, 8)
# 把 eps 设为 0，再对全 0 输入做 RMSNorm
bad = RMSNorm(8, eps=0.0)
print(bad(z))   # 会出现 nan
```

这个实验说明：$\epsilon$ 不是“可选的装饰”，而是数值稳定的必需品。

---

## 7. 联系 CS336 Assignment 1

### 7.1 作业要求（讲义 §3.4.1）

作业要求你实现 `RMSNorm`，具体是：

- 实现一个 `torch.nn.Module` 子类；
- 接口建议为 `__init__(self, d_model, eps=1e-5, device=None, dtype=None)`；
- `forward(x)` 接收 `(batch_size, sequence_length, d_model)`，返回同形状；
- 使用 RMSNorm 公式（不是 LayerNorm）；
- 输入先升到 `torch.float32`，算完再降回原 dtype；
- 实现测试适配器 `adapters.run_rmsnorm`；
- 运行 `uv run pytest -k test_rmsnorm` 验证；
- 这一题 1 分。

### 7.2 接口与形状

`run_rmsnorm` 的签名（来自 `tests/adapters.py`）：

```python
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    ...
```

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `d_model` | 标量 | 隐藏维度 |
| `eps` | 标量 | 数值稳定常数，测试里是 `1e-5` |
| `weights` | `(d_model,)` | 可学习增益 $g$，不是矩阵 |
| `in_features` | `(..., d_model)` | 任意前导维度，常见 `(B,T,D)` |
| 返回值 | `(..., d_model)` | 与输入同形状 |

**关键点**：`weights` 是**一维**的增益，不是 $(D,D)$ 的矩阵。它通过广播作用在最后一维上。

### 7.3 实现要点

1. **继承 `nn.Module`，调用 `super().__init__()`**；
2. **增益是 `nn.Parameter`**，形状 `(d_model,)`，初始化为全 1；
3. **不要用 `nn.LayerNorm` 或 `nn.RMSNorm`**：作业要求自己实现；
4. **不要加 bias**：RMSNorm 只有增益；
5. **归一化维度是最后一维**：`dim=-1`；
6. **`keepdim=True`**：否则无法广播；
7. **$\epsilon$ 放在根号里面**：和讲义公式一致；
8. **先升 float32，最后降回原 dtype**；
9. **不要对 batch/sequence 维做平均**：每个 token 独立。

一个符合接口的模块：

```python
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x / rms * self.weight
        return out.to(in_dtype)
```

### 7.4 测试细节

`test_rmsnorm` 会：

- 从 `state_dict["layers.1.ln1.weight"]` 取增益；
- 用 `d_model = weights.shape[0]` 推断维度；
- 用 `eps=1e-5`；
- 输入是 `in_embeddings`，形状 `(batch_size, n_queries, d_model)`；
- 容差 `atol=1e-4`。

这意味着：

- 你的实现必须接受任意前导维度；
- 不能假设输入一定是三维；
- 增益必须正确广播；
- 数值误差要足够小（用 float32 计算，通常没问题）。

权重键的命名也值得记住：

| 键 | 位置 |
| --- | --- |
| `layers.{i}.ln1.weight` | Block 里第一个 RMSNorm（注意力前） |
| `layers.{i}.ln2.weight` | Block 里第二个 RMSNorm（FFN 前） |
| `ln_final.weight` | 最后一层 Block 之后的最终 RMSNorm |

没有 `bias` 键——再次提醒：**RMSNorm 没有 bias**。

### 7.5 在 Block 里的位置

pre-norm Block：

```python
from torch import nn


class TransformerBlock(nn.Module):
    def forward(self, x):
        # 第一个子层：RMSNorm → 注意力 → 残差
        x = x + self.attn(self.ln1(x))
        # 第二个子层：RMSNorm → SwiGLU FFN → 残差
        x = x + self.ffn(self.ln2(x))
        return x
```

对应公式：

$$
y=x+\operatorname{MHA}(\operatorname{RMSNorm}(x))
$$

$$
z=y+\operatorname{SwiGLU}(\operatorname{RMSNorm}(y))
$$

最后：

$$
\text{logits}=\operatorname{LMHead}(\operatorname{RMSNorm}(z))
$$

三个细节：

1. **先归一化，再进子层**（pre-norm）；
2. **残差加的是子层输出**，不是归一化结果；
3. **最后还要一次 final norm**，因为残差主干上没有归一化。

### 7.6 常见错误对照表

| 现象 | 可能原因 | 修正 |
| --- | --- | --- |
| 输出均值是 0 | 你写了 LayerNorm，减了均值 | RMSNorm 不减均值 |
| 数值和参考差很多 | 用了标准差而不是 RMS | 分母是 $\sqrt{\text{mean}(x^2)+\epsilon}$ |
| 形状对不上 | `dim` 写成了 `0` 或 `1` | 用 `dim=-1`，只对最后一维 |
| 广播报错 | `keepdim=False` | 改成 `keepdim=True` |
| 增益不生效 | `weight` 形状写成了 `(D,D)` 或标量 | 形状必须是 `(D,)` |
| 参数比预期多 | 加了 bias | 删掉 bias |
| `nan` / `inf` | 没升 float32，或 $\epsilon$ 太小/为 0 | 升 float32，$\epsilon=10^{-5}$ |
| dtype 不匹配 | 忘了最后 `.to(in_dtype)` | 还原输入 dtype |
| 用了 `nn.RMSNorm` | 作业要求自己实现 | 手写公式 |
| 测试误差略大 | 在半精度下算平方 | 先升 float32 |

### 7.7 实现检查清单

- [ ] 继承 `nn.Module`，调用 `super().__init__()`；
- [ ] `self.weight` 是 `nn.Parameter`，形状 `(d_model,)`，初始化为 1；
- [ ] `forward` 里先保存 `in_dtype`，再升 `float32`；
- [ ] 用 `x.pow(2).mean(dim=-1, keepdim=True)`；
- [ ] 分母是 `sqrt(mean_square + eps)`，$\epsilon$ 在根号内；
- [ ] 乘 `self.weight` 后再降回 `in_dtype`；
- [ ] 不加 bias，不用 `nn.LayerNorm`/`nn.RMSNorm`；
- [ ] 跑过 `pytest -k test_rmsnorm`。

---

## 8. 常见误区 FAQ

**Q1：RMSNorm 就是 LayerNorm 去掉 bias 吗？**
不是。它还去掉了**减均值**，并且用均方根代替标准差。这是两个独立的简化。

**Q2：为什么 RMSNorm 的输出均值不是 0？**
因为它没有减均值。它只保证输出的 RMS ≈ 1，不保证中心在 0。

**Q3：$\epsilon$ 为什么放在根号里面？**
讲义的定义是 $r=\sqrt{\text{mean}(x^2)+\epsilon}$。放在里面可以保证 $x=0$ 时分母是 $\sqrt{\epsilon}$，输出为 0；放在外面也能防除零，但数值不同，不要混用。

**Q4：为什么增益初始化为 1？**
初始时让 RMSNorm 等价于“只归一化、不额外缩放”，不破坏原始信号的相对大小；训练中模型再自己学每个通道该放大还是缩小。

**Q5：为什么没有 bias？**
现代 LLM 通常省略线性层和归一化层的 bias；RMSNorm 只保留增益，参数更少、计算更简单。作业也要求无 bias。

**Q6：为什么要升 `float32`？**
平方会让数值变大，float16 很容易溢出（最大值约 65504）。升到 float32 再平方、求和、开根，最后降回原 dtype，兼顾稳定和速度。

**Q7：RMSNorm 对输入乘 10 真的不变吗？**
忽略 $\epsilon$ 时严格不变；$\epsilon$ 很小时近似不变。对负数缩放，输出会符号翻转。

**Q8：RMSNorm 对输入加 100 不变吗？**
**不是，输出会变。** RMS 会随整体偏移改变，所以归一化结果也跟着变。这是它和 LayerNorm 的关键区别。

**Q9：RMSNorm 比 LayerNorm 一定好吗？**
不一定。它在很多大模型上更稳、更快，但这是经验结果；某些任务上 LayerNorm 可能更合适。

**Q10：可以用 `torch.nn.RMSNorm` 吗？**
作业要求自己实现，不能用内置的 `nn.RMSNorm` 或 `nn.LayerNorm`。自己写一遍也更容易理解公式。

**Q11：RMSNorm 会混合不同 token 吗？**
不会。它逐 token 计算，每个位置的归一化只依赖自己的 $D$ 个通道。

**Q12：为什么叫“Root Mean Square”？**
先平方（Square）、再平均（Mean）、再开根（Root），合起来就是 Root Mean Square。

---

## 9. 小结与自测

### 9.1 一页速查

```text
公式:   RMSNorm(x) = x / RMS(x) * g
RMS:    RMS(x) = sqrt( mean(x^2) + eps )
形状:   (..., D) -> (..., D)
参数:   g 形状 (D,)，初始化为 1；无 bias
关键:   不减均值；用均方根；eps 在根号内；先升 float32
位置:   pre-norm Block 的每个子层前；最后还有 final norm
作用:   稳定尺度、改善梯度、计算更省、混合精度更稳
作业:   实现 nn.Module + run_rmsnorm；pytest -k test_rmsnorm
```

### 9.2 自测题

1. 写出 RMSNorm 的公式，并标出 $g$ 和 $\epsilon$ 的位置。
2. RMS 和标准差有什么关系？什么时候两者近似相等？
3. 为什么 RMSNorm 的输出均值不一定是 0？
4. 输入乘 10，RMSNorm 的输出会变吗？加 100 呢？分别解释。
5. 为什么 $\epsilon$ 要放在根号里面？如果 $x$ 全是 0，输出是什么？
6. 为什么平方之前要升 `float32`？不升会发生什么？
7. 在 `run_rmsnorm` 里，`weights` 的形状是什么？它和输入怎么广播？
8. 为什么 RMSNorm 只有增益、没有 bias？
9. 画出 pre-norm Block 的结构，标出 RMSNorm 的位置。
10. 如果误用了 LayerNorm，输出会有什么明显不同？
11. 为什么说 RMSNorm “只缩放、不挪中心”？
12. 为什么 RMSNorm 的计算比 LayerNorm 少？少在哪里？

> 参考答案在 §3–§7 对应小节；先自己写一遍，再回去核对。

### 9.3 动手清单

- [ ] 手算 $x=[1,2,3,4]$ 的 RMSNorm，核对 §3.4 的表；
- [ ] 跑通 §6.1 的脚本，确认输出 RMS ≈ 1、均值不为 0；
- [ ] 跑过 `pytest -k test_rmsnorm`（仓库的 `run_rmsnorm` 已接好，可以直接跑）；
- [ ] 做“乘 10 / 加 100”两个实验，观察缩放不变、平移会变；
- [ ] 把 `eps` 改成 0，对全 0 输入跑一次，观察 `nan`；
- [ ] 把输入改成 `float16`，确认内部升 float32、输出 dtype 不变。

---

## 10. 延伸阅读

**论文**

- Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019. arXiv:1910.07467 —— RMSNorm 原论文。
- Ba, Kiros & Hinton, *Layer Normalization*, 2016. arXiv:1607.06450 —— LayerNorm 原论文。
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, 2020 —— pre-norm 与训练稳定性。
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023 —— LLaMA 采用 RMSNorm。
- Grattafiori et al., *The Llama 3 Herd of Models*, 2024 —— LLaMA 3 的 RMSNorm 配置。

**课程与文档**

- CS336 课程主页：<https://cs336.stanford.edu>
- 作业讲义 `cs336_assignment1_basics.pdf` §3.4.1
- PyTorch 文档：`torch.rsqrt`、`torch.nn.functional.rms_norm`

**同目录文档**

- `Transformer 模块.md`：从文本到下一个词预测的完整链路，§4.2 是 RMSNorm 的简要版。
- `SwiGLU.md`：pre-norm Block 里另一个子层的前馈网络。
- `Attention.md`：注意力的 Q/K/V、掩码与多头机制。
- `Embedding_Linear.md`：线性层与嵌入层的基础。

> **最后一句**：RMSNorm 的公式很短，但它体现了大模型工程里一个很重要的取舍：**去掉不必要的中心化，只保留最关键的尺度控制**。理解这个取舍，比记住公式更有价值。
