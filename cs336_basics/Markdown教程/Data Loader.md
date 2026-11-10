# Data Loader：把一整条 token 流切成训练用的 batch

> 面向零基础读者。本文以 `cs336_basics/Data_Loader.py` 为主线，结合 CS336 Assignment 1 的 §5.1 与 `tests/test_data.py`，讲清楚三件事：
>
> 1. 语言模型的训练任务到底在做什么；
> 2. 数据加载器吐出来的 `x` 和 `y` 分别是什么；
> 3. 数据很大时，怎样用 `np.memmap` 之类的办法避免把内存撑爆。
>
> 读完你应该能自己写出一个正确的 `get_batch`，并看懂现有 `Data_Loader.py` 里哪些地方需要修。

---

## 目录

| 章节 | 内容 |
| --- | --- |
| §1 | 训练任务的本质：预测下一个 token |
| §2 | `x` 和 `y` 分别代表什么 |
| §3 | 窗口、起点与长度的边界 |
| §4 | batch、context、vocab 这些术语 |
| §5 | 内存优化：为什么需要 `np.memmap` |
| §6 | 逐行拆解 `Data_Loader.py`（含问题与修正版） |
| §7 | 把 Data Loader 接进训练循环 |
| §8 | 联系 CS336 Assignment 1 |
| §9 | 常见问题 FAQ |
| §10 | 自测清单与一页速查 |

---

## 1. 训练任务的本质：预测下一个 token

### 1.1 语言模型只做一件事

给定一段已经看过的文本，预测**下一个 token 是什么**。

比如已经看到：

```text
今天天气真
```

模型要输出一个概率分布，表示下一个 token 可能是“好”“不错”“冷”……的概率。训练时，如果真实的下一个 token 是“好”，我们就让“好”的概率变大一点。

这就是**自回归语言模型**（autoregressive language model）的训练目标。它不需要人工标注标签：**文本里下一个 token 本身就是标签**。

### 1.2 一整篇文档，就是一条很长的 token 流

原始数据可能是一篇篇文章、网页或代码文件。常见的处理方式是：

1. 用 tokenizer 把每篇文档编码成 token ID；
2. 在文档之间插入一个分隔符，例如 `<|endoftext|>`；
3. 把所有文档**拼成一条很长的序列**：

```text
[文档1的 token ...] <|endoftext|> [文档2的 token ...] <|endoftext|> ...
```

拼成一条长流以后，整份语料就变成了一个一维整数数组：

```text
data = [x0, x1, x2, x3, x4, x5, ..., xN-1]
```

其中 `N` 是 token 总数，每个元素是一个整数 token ID。

> 为什么要把文档拼起来？因为语言模型训练时并不关心“这句话属于哪篇文章”，它只关心“在当前位置，下一个 token 是什么”。拼成一条流以后，任意一段连续的子序列都可以拿来做训练样本，采样非常方便。

### 1.3 一个长度为 T 的窗口，能提供 T 个预测任务

假设我们从长流里切出一个长度为 `T` 的窗口：

```text
窗口：  [x2, x3, x4, x5]
```

这个窗口可以拆成 4 个“看前缀、猜下一个”的任务：

| 模型看到的前缀 | 要预测的下一个 token |
| --- | --- |
| `x2` | `x3` |
| `x2, x3` | `x4` |
| `x2, x3, x4` | `x5` |
| `x2, x3, x4, x5` | 窗口外的 `x6` |

前 3 个任务完全在窗口内部；第 4 个任务需要窗口外的 `x6`。

所以在数据加载器里，我们通常这样构造：

```text
x = [x2, x3, x4, x5]
y = [x3, x4, x5, x6]
```

- `x` 是**输入**；
- `y` 是**目标**；
- `y` 的每个位置，正好是 `x` 对应位置的**下一个 token**。

这样一次前向传播就能同时训练 `T` 个预测任务，而不是只训练最后一个。

### 1.4 为什么可以一次算 T 个位置

Transformer 的注意力层带有 **causal mask**（因果掩码），它保证第 `t` 个位置只能看到 `0..t` 的 token，不能偷看未来。

因此：

```text
输入 x:      [x2, x3, x4, x5]
模型输出:    每个位置一个概率分布
位置 0 的输出：只能基于 x2，预测 x3
位置 1 的输出：只能基于 x2,x3，预测 x4
位置 2 的输出：只能基于 x2,x3,x4，预测 x5
位置 3 的输出：只能基于 x2,x3,x4,x5，预测 x6
```

这正好和 `y = [x3, x4, x5, x6]` 一一对应。

> 如果没有 causal mask，位置 0 就能看到 `x3`，那预测 `x3` 就变成“抄答案”，训练就失去意义了。

### 1.5 训练损失长什么样

模型输出 `logits`，形状是：

```text
(B, T, V)
```

其中：

- `B`：batch size，一次处理几条序列；
- `T`：context length，每条序列多长；
- `V`：vocab size，词表大小。

把前两维压平：

```text
logits:  (B, T, V)  ->  (B*T, V)
y:       (B, T)     ->  (B*T,)
```

然后做交叉熵：

```python
loss = cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
```

直观理解：对 `B*T` 个位置，每个位置都做一次“V 分类”，正确答案就是 `y` 里对应的 token ID。

> 交叉熵的细节见 `Cross_Entropy_Loss.md`。这里只需要记住：**`y` 已经是“下一个 token”，所以损失函数里不需要再 shift 一次。**

---

## 2. `x` 和 `y` 分别代表什么

### 2.1 一句话定义

| 名字 | 形状 | 含义 | 类型 |
| --- | --- | --- | --- |
| `x` | `(B, T)` | 输入 token ID | `torch.long`（整数） |
| `y` | `(B, T)` | 目标 token ID，`y[b,t] = x[b,t+1]` | `torch.long`（整数） |

注意：`y` 不是 one-hot，也不是概率，而是**整数 token ID**。交叉熵函数会在内部把 logits 转成概率，并和这个整数标签比较。

### 2.2 一个具体窗口

假设长流是：

```text
data = [10, 11, 12, 13, 14, 15, 16]
```

我们取起点 `s = 2`，窗口长度 `T = 4`：

```text
下标:   0   1   2   3   4   5   6
data:  10  11  12  13  14  15  16
                ↑
              起点 s=2

x = data[2:6] = [12, 13, 14, 15]
y = data[3:7] = [13, 14, 15, 16]
```

逐位置看：

| 位置 `t` | `x[:, t]` | `y[:, t]` | 含义 |
| ---: | ---: | ---: | --- |
| 0 | 12 | 13 | 看到 12，预测 13 |
| 1 | 13 | 14 | 看到 12,13，预测 14 |
| 2 | 14 | 15 | 看到 12,13,14，预测 15 |
| 3 | 15 | 16 | 看到 12,13,14,15，预测 16 |

> `y` 不是“窗口的最后一个 token”，而是“每个位置的下一个 token”。这一点是初学者最容易误解的地方。

### 2.3 一个 batch 的例子

假设 `B = 2`，`T = 4`，两个起点分别是 10 和 20：

```text
起点 10：
x[0] = [10, 11, 12, 13]
y[0] = [11, 12, 13, 14]

起点 20：
x[1] = [20, 21, 22, 23]
y[1] = [21, 22, 23, 24]
```

堆成一个 batch：

```text
x = [[10, 11, 12, 13],
     [20, 21, 22, 23]]

y = [[11, 12, 13, 14],
     [21, 22, 23, 24]]
```

形状都是 `(2, 4)`。两行之间**不需要相邻**，它们只是从长流里随机抽出来的两个窗口。

### 2.4 `y` 是“生成”的，不是单独存一份

你不需要为 `y` 单独准备一个文件。它只是 `x` 在长流上向右看一格：

```text
x:  data[s   : s+T]
y:  data[s+1 : s+T+1]
```

所以数据加载器每次只需要：

1. 随机选 `B` 个起点；
2. 按起点切出 `x`；
3. 把起点加 1，切出 `y`。

这样既省存储，也保证 `x` 和 `y` 永远对齐。

### 2.5 和 Hugging Face 的 `labels` 有什么区别

有些库（例如 Hugging Face Transformers）在模型内部会自动把 `labels` 左移一位：

```python
# 某些模型的内部逻辑（示意）
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

而 CS336 的数据加载器是**在数据侧**先 shift 好：

```python
x = data[s : s+T]
y = data[s+1 : s+T+1]
```

两种做法都对，但**不要重复 shift**。

- 如果你用本作业的 `get_batch`，`y` 已经是下一个 token，损失函数直接用它；
- 如果你用某个 HF 模型并传 `labels=x`，要确认该模型内部是否会自己 shift；
- 最稳妥的检查方法：打印一个 batch，确认 `y[0,0] == x[0,1]`。

### 2.6 常见误解

| 误解 | 正确理解 |
| --- | --- |
| `y` 是 `x` 的最后一个 token | `y` 的每个位置都是对应位置的**下一个** token |
| `y` 的形状应该比 `x` 少一维 | 两者形状相同，都是 `(B,T)`；只是内容错开一位 |
| `y` 需要 one-hot 编码 | 不需要，交叉熵接收整数类别 ID |
| `x` 和 `y` 要分别存成两个大文件 | 不需要，`y` 可以由同一份数据 shift 得到 |
| 模型内部还会再 shift 一次 | 取决于模型/库；本作业手动算损失，不会再 shift |

---

## 3. 窗口、起点与长度的边界

### 3.1 合法起点到底有哪些

设：

- 长流长度 `N`；
- 窗口长度 `T`；
- 起点 `s`（从 0 开始数）。

`x` 占用的下标是：

```text
s, s+1, ..., s+T-1
```

`y` 占用的下标是：

```text
s+1, s+2, ..., s+T
```

要让 `y` 的最后一个下标不越界，必须：

```text
s + T <= N - 1
```

移项：

```text
s <= N - T - 1
```

所以合法起点是：

```text
0, 1, 2, ..., N-T-1
```

一共有 `N - T` 个。

### 3.2 一个具体数字

`N = 100`，`T = 7`：

```text
合法起点：0 .. 92
一共：100 - 7 = 93 个
```

验证一下最后一个起点 `s = 92`：

```text
x = data[92:99]   -> 下标 92..98，合法
y = data[93:100]  -> 下标 93..99，合法
```

如果再往后取 `s = 93`：

```text
y = data[94:101]  -> 下标 100 越界
```

所以 `93` 不是合法起点。

### 3.3 采样公式

用 NumPy 随机选 `B` 个起点：

```python
num_starts = len(data) - context_length
starts = np.random.randint(0, num_starts, size=batch_size)
```

`np.random.randint(low, high)` 的 `high` 是**不包含**的，所以：

```python
np.random.randint(0, 93, size=...)  # 结果范围 0..92
```

正好覆盖全部 93 个合法起点。

然后用广播一次性构造下标矩阵：

```python
idx = starts[:, None] + np.arange(context_length)
```

`starts` 形状是 `(B,)`，`starts[:, None]` 变成 `(B,1)`；`np.arange(T)` 形状是 `(T,)`；广播后得到 `(B,T)`：

```text
starts = [10, 20]
idx = [[10, 11, 12, 13],
       [20, 21, 22, 23]]
```

最后：

```python
x = data[idx]       # (B,T)
y = data[idx + 1]   # (B,T)
```

### 3.4 最常见的 off-by-one

有三种写法，只有第一种是对的：

| 写法 | 实际范围 | 结果 |
| --- | --- | --- |
| `np.random.randint(0, N - T, size=B)` | `0 .. N-T-1` | 正确 |
| `np.random.randint(0, N - T - 1, size=B)` | `0 .. N-T-2` | 少了最后一个起点 |
| `np.random.randint(0, N - T + 1, size=B)` | `0 .. N-T` | 最后一个起点会越界 |

现有 `Data_Loader.py` 用的是第二种：

```python
idxs = np.random.randint(0, self.data_len - self.context_length - 1, size=(self.batch_size,))
```

当 `N=100, T=7` 时，它只采到 `0..91`，而测试要求能采到 `92`。这就是 `test_get_batch` 会失败的原因之一。

> 记忆口诀：**高位写 `N-T`，不要多减 1。** 因为 `randint` 的高位本来就不包含。

### 3.5 边界保护

如果 `N <= T`，连一个合法窗口都没有。此时 `num_starts <= 0`，`np.random.randint(0, 0)` 会报错。最好在构造函数里提前检查：

```python
if len(data) <= context_length:
    raise ValueError("数据太短：至少需要 context_length + 1 个 token")
```

如果 `N = T + 1`，只有 `s = 0` 一个起点，`np.random.randint(0, 1, size=B)` 会返回全 0。这是合法的，只是每个 batch 的窗口都一样。

### 3.6 随机采样 vs 顺序采样

| 场景 | 是否随机 | 原因 |
| --- | --- | --- |
| 训练 | 通常随机 | 打散样本顺序，减少相关性，提升泛化 |
| 验证/测试 | 通常顺序 | 保证每次评估用同一批数据，结果可比较 |

训练时用**有放回**随机采样：同一个起点在一个 batch 里出现两次，甚至连续几个 batch 重复，都是正常的。不要因为“看到重复”就以为代码有 bug。

如果你想要“一个 epoch 内不重复”，需要额外维护一个打乱后的起点列表或游标；本作业的简单实现没有做这件事。

---

## 4. batch、context、vocab 这些术语

### 4.1 四个核心符号

| 符号 | 名称 | 含义 | 例子 |
| --- | --- | --- | --- |
| `N` | 数据总量 | 长流里有多少个 token | 1,000,000 |
| `B` | batch size | 一次喂给模型几条序列 | 32 |
| `T` | context length | 每条序列多长 | 1024 |
| `V` | vocab size | 词表里有多少种 token | 50,000 |

### 4.2 形状一路怎么变

```text
data:      (N,)
x:         (B, T)
y:         (B, T)
logits:    (B, T, V)
flat logits:(B*T, V)
flat y:    (B*T,)
loss:      scalar
```

### 4.3 显存主要花在哪里

`x` 和 `y` 本身通常很小。假设 `B=32, T=1024`，用 `int64`：

```text
x: 32 * 1024 * 8 B = 262,144 B ≈ 256 KiB
y: 32 * 1024 * 8 B = 262,144 B ≈ 256 KiB
```

但 `logits` 是 `(B,T,V)`，假设 `V=50,000`，用 `float32`：

```text
32 * 1024 * 50,000 * 4 B
= 6,553,600,000 B
≈ 6.1 GiB
```

所以：

> **数据加载器通常不是显存瓶颈；真正的大头是 logits、激活值和模型参数。** 但如果你把整个语料读成 Python list，CPU 内存会先爆掉。

### 4.4 context length 不一定是整篇文档

`T` 是模型一次能看的最大长度，不是文档长度。一篇很长的文档会被切成很多个长度为 `T` 的窗口；一篇很短的文档也可能和后面的文档拼在同一个窗口里。

这就是为什么要在文档之间插入 `<|endoftext|>`：让模型知道“这里换文档了”。

---

## 5. 内存优化：为什么需要 `np.memmap`

### 5.1 先算一笔账

假设语料有 10 亿个 token。

| 存储方式 | 每个 token 大约占 | 10 亿 token 大约占 |
| --- | ---: | ---: |
| Python `list[int]` | 28 B 整数对象 + 8 B 指针 ≈ 36 B | ≈ 36 GB |
| `np.int64` | 8 B | ≈ 8 GB |
| `np.int32` | 4 B | ≈ 4 GB |
| `np.uint16` | 2 B | ≈ 2 GB |

Python 的 `int` 是对象，不是裸的 4 字节或 8 字节；列表里存的还是指针。所以“把 token 读成 Python list”会非常占内存。

但即使换成 `np.uint16`，2 GB 也可能超过某些机器的可用内存；而且训练时还要留内存给模型、优化器、数据副本。这时就需要 **memory-mapped file（内存映射文件）**。

### 5.2 mmap 的直觉

`mmap` 是操作系统提供的一个系统调用：把一个磁盘文件“映射”到进程的虚拟内存地址空间。

你可以把它想象成：

> 不是把整座图书馆搬回家，而是拿到一张“借书证”。需要哪一页，操作系统就帮你把那一页从磁盘加载到内存；暂时不用的页，可以留在磁盘上，必要时被换出。

所以你可以“假装”整个数据集都在内存里：

```python
data = np.memmap("train.bin", dtype=np.uint16, mode="r")
print(len(data))       # 看起来就是一个普通数组
print(data[100:110])   # 只加载这一小段
```

实际上，只有你访问到的那些页会被加载。第一次访问会触发磁盘 I/O，之后如果还在 page cache 里，就会很快。

> 注意：`mmap` 不会让磁盘变快，也不会减少总 I/O；它主要解决“内存装不下”和“启动时不想全量加载”的问题。

### 5.3 NumPy 的两种用法

#### 用法一：`.npy` 文件

如果数据是用 `np.save` 保存的：

```python
np.save("train.npy", tokens)
```

读取时加 `mmap_mode="r"`：

```python
data = np.load("train.npy", mmap_mode="r")
print(type(data))    # <class 'numpy.memmap'>
print(data.shape)    # (N,)
print(data.dtype)    # uint16
```

#### 用法二：裸二进制文件

如果数据是用 `tofile()`、`np.memmap` 或其他方式写成的裸二进制：

```python
data = np.memmap("train.bin", dtype=np.uint16, mode="r")
```

如果文件里没有形状信息，`np.memmap` 会根据文件大小自动推断一维长度：

```text
len(data) = 文件字节数 / dtype.itemsize
```

也可以显式指定形状：

```python
data = np.memmap("train.bin", dtype=np.uint16, mode="r", shape=(N,))
```

#### 两种方式不要混用

`.npy` 文件开头有 NumPy 自己的头部（magic string、shape、dtype 等）。如果你用 `np.memmap` 直接读 `.npy`，会把头部也当成 token 数据，得到错误的值和长度。

| 文件格式 | 正确读法 |
| --- | --- |
| `.npy`（`np.save` 保存） | `np.load(path, mmap_mode="r")` |
| 裸 `.bin`（`tofile` 保存） | `np.memmap(path, dtype=..., mode="r")` |

### 5.4 dtype 必须匹配

`np.memmap` 不会猜你的 dtype，你必须告诉它：

```python
data = np.memmap("train.bin", dtype=np.uint16, mode="r")
```

如果实际写文件时用的是 `uint16`，你读的时候却写 `int32`，会得到：

- 长度变成原来的一半；
- 每个“token”由两个 uint16 拼成，数值完全错误；
- 可能超出词表范围。

所以保存和读取要用同一个 dtype。常见选择：

| 词表大小 | 推荐 dtype | 说明 |
| --- | --- | --- |
| `V <= 65535` | `uint16` | 省一半空间 |
| `V <= 2^32-1` | `uint32` | 更大词表 |
| 不确定/图省事 | `int32` / `int64` | 更占空间，但不容易错 |

> CS336 的 GPT-2 词表是 50257，`uint16` 够用。但你的 tokenizer 如果更大，要换 `uint32`。

### 5.5 读取后先做一次“体检”

不要一上来就训练。先确认 memmap 读出来的东西是对的：

```python
data = np.load("train.npy", mmap_mode="r")

print("类型:", type(data))
print("形状:", data.shape)
print("dtype:", data.dtype)
print("前 10 个:", data[:10])
print("最小值:", int(data.min()))
print("最大值:", int(data.max()))
print("长度:", len(data))
```

重点检查：

- `dtype` 是否和保存时一致；
- `max` 是否小于 `vocab_size`；
- `min` 是否大于等于 0；
- 长度是否符合预期；
- 前几个 token 解码出来是否像正常文本。

> `data.min()` / `data.max()` 会扫一遍整个文件，第一次可能比较慢；不要在训练的每个 step 里调用。

### 5.6 采样时只复制一个小 batch

即使用 memmap，也不要把整个数据集变成 Tensor：

```python
# 错误：会把整个文件读进内存，并复制成 Tensor
x_all = torch.tensor(data)

# 错误：会生成一个巨大的 Python list
all_tokens = list(data)

# 正确：只取 B*T 个 token
idx = starts[:, None] + np.arange(context_length)
x = torch.from_numpy(data[idx].astype(np.int64))
```

`data[idx]` 是 NumPy 的**高级索引**，会返回一个新的 `(B,T)` 数组，只包含被选中的元素。这个数组很小，复制它是完全可以接受的。

然后：

- `torch.from_numpy(...)`：把 NumPy 数组共享成 Tensor（不复制）；
- `.astype(np.int64)`：转成 embedding 需要的整数类型（这一步可能复制）；
- `.to(device)`：搬到 GPU。

### 5.7 一个完整的 memmap 例子

```python
import numpy as np
import torch


def get_batch(dataset, batch_size, context_length, device="cpu"):
    num_starts = len(dataset) - context_length
    if num_starts <= 0:
        raise ValueError("dataset 太短")

    starts = np.random.randint(0, num_starts, size=batch_size)
    idx = starts[:, None] + np.arange(context_length)

    x = torch.from_numpy(dataset[idx].astype(np.int64))
    y = torch.from_numpy(dataset[idx + 1].astype(np.int64))
    return x.to(device), y.to(device)


# 假设 train.npy 是用 np.save 保存的 uint16 token ID
data = np.load("train.npy", mmap_mode="r")
x, y = get_batch(data, batch_size=4, context_length=8)
print(x.shape, y.shape, x.dtype)
```

### 5.8 内存不只是 data

训练一个大模型时，内存/显存大致包括：

| 项目 | 说明 |
| --- | --- |
| 模型参数 | 每个参数 2/4/8 字节，取决于精度 |
| 优化器状态 | AdamW 通常要为每个参数存一阶、二阶矩，约 2 份额外参数 |
| 梯度 | 和参数同量级 |
| 激活值 | 前向保存下来给反向用，和 `B*T*D`、层数有关 |
| logits | `B*T*V`，词表大时很可观 |
| 数据 batch | `B*T` 个整数，通常最小 |

所以：

> 数据加载器的主要职责是“别把数据全读进内存”，而不是“省出训练的所有内存”。真正的大头通常在模型和激活值。

---

## 6. 逐行拆解 `Data_Loader.py`

> 说明：这一节先保留**修正前旧版**的代码，用来对照两个典型 bug；当前 `Data_Loader.py` 已经改成 §6.5 的修正版。阅读时请先看旧版哪里错，再看修正版怎么改。

下面是修正前旧版的完整内容：

```python
import random
import numpy as np
from typing import List
import torch

class DataLoader:
    def __init__(self, data:List[int], batch_size:int, context_length:int, shuffle=True):
        self.data = data
        self.data_len = len(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.context_length = context_length
    # def __iter__(self):
    #     if self.shuffle:
    #         self.data = random.shuffle(self.data)
    #     for i in range(0, len(self.data), self.batch_size):
    #         yield self.data[i:i+self.batch_size]

    def get_train_batch_data(self):
        """
        随机获取一个batch的数据，y 正好是 x 向左移动一个位置的结果。模型在看到 x 的第 i 个位置的输入时，需要努力预测出 y 在第 i 个位置的词元，这正是我们想要的“预测下一个词”的效果。
        """
        idxs = np.random.randint(0,self.data_len-self.context_length-1,size=(self.batch_size,)) # 随机选择一个batch_size大小的索引,最后的括号其实写不写都行，写上了就更明确输出是一个数组
        x = np.stack([self.data[i:i+self.context_length] for i in idxs])
        y = np.stack([self.data[i+1:i+self.context_length+1] for i in idxs])
        return torch.tensor(x),torch.tensor(y)

    def get_valid_batch_data_iter(self):
        """
        验证集的区别之处在于，不需要随机选择，而是直接从数据集中按顺序获取所有数据，并用迭代器返回
        """
        start_num = (self.data_len - self.context_length - 1) // self.batch_size # 表示有多少个batch
        for i in range(start_num):
            bias = i * self.batch_size     # 表示每一个batch开始的位置
            x = np.stack([self.data[bias:bias+self.context_length] for i in range(self.batch_size)])
            y = np.stack([self.data[bias+1:bias+self.context_length+1] for i in range(self.batch_size)])
            yield torch.tensor(x), torch.tensor(y)

    def __len__(self):
        """
        返回数据集的batch数量
        """
        return self.data_len // self.batch_size
```

### 6.1 构造函数

```python
def __init__(self, data:List[int], batch_size:int, context_length:int, shuffle=True):
    self.data = data
    self.data_len = len(data)
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.context_length = context_length
```

做得好的地方：

- **没有复制 `data`**：只是保存引用。传入 `np.memmap` 时，不会把整个文件读进内存。
- `len(data)` 对 NumPy 数组和 memmap 都有效。
- 把 `batch_size`、`context_length` 存下来，后续方法都能用。

可以改进的地方：

| 问题 | 说明 | 建议 |
| --- | --- | --- |
| 类型标注 `List[int]` 不准确 | `data` 更常见的是 `np.ndarray` 或 `np.memmap` | 改成 `np.ndarray` 或 `Sequence[int]` |
| `shuffle` 没有被使用 | 训练方法始终随机；`shuffle=False` 没有效果 | 要么删掉，要么实现顺序采样 |
| `import random` 没用到 | 只 import 了但没调用 | 删掉 |
| 没有检查数据长度 | `N <= T` 时采样会报错 | 构造函数里提前 `raise` |

### 6.2 训练采样方法

```python
idxs = np.random.randint(0, self.data_len - self.context_length - 1, size=(self.batch_size,))
x = np.stack([self.data[i:i+self.context_length] for i in idxs])
y = np.stack([self.data[i+1:i+self.context_length+1] for i in idxs])
return torch.tensor(x), torch.tensor(y)
```

逐句看：

1. `idxs` 形状是 `(B,)`，每个元素是一个起点。
2. 列表推导对每个起点切出 `x` 的窗口，`np.stack` 把 `B` 个 `(T,)` 堆成 `(B,T)`。
3. `y` 的切片整体加 1，得到下一个 token。
4. `torch.tensor` 把 NumPy 数组转成 Tensor。

问题一：**起点范围少了一个**。

```python
self.data_len - self.context_length - 1
```

当 `N=100, T=7` 时，高位是 `92`，`randint(0, 92)` 返回 `0..91`，漏掉了合法的 `92`。应该写成：

```python
self.data_len - self.context_length
```

问题二：**dtype 可能不对**。

如果 `self.data` 是 `np.uint16` 的 memmap，`np.stack` 得到 `uint16` 数组，`torch.tensor(x)` 可能得到 `torch.uint16`，而 `nn.Embedding` 需要 `torch.long` 或 `torch.int`。更稳妥的写法是显式转换：

```python
x = torch.from_numpy(np.asarray(x, dtype=np.int64))
```

问题三：**没有 device 参数**。

旧版方法返回 CPU Tensor。如果模型在 GPU 上，训练循环里还要手动 `.to(device)`。这不算错，但 CS336 的 `run_get_batch` 要求直接返回指定 device 上的 Tensor。

问题四：**`np.stack` + `torch.tensor` 有两次复制**。

对小白来说这不是大问题；但用高级索引可以更简洁：

```python
idx = idxs[:, None] + np.arange(self.context_length)
x = self.data[idx]
y = self.data[idx + 1]
```

### 6.3 验证集迭代方法

```python
start_num = (self.data_len - self.context_length - 1) // self.batch_size
for i in range(start_num):
    bias = i * self.batch_size
    x = np.stack([self.data[bias:bias+self.context_length] for i in range(self.batch_size)])
    y = np.stack([self.data[bias+1:bias+self.context_length+1] for i in range(self.batch_size)])
    yield torch.tensor(x), torch.tensor(y)
```

这段有两个明显问题。

#### 问题一：内层 `i` 把外层 `i` 覆盖了

列表推导写的是：

```python
for i in range(self.batch_size)
```

但切片里用的却是 `bias`，不是 `i`：

```python
self.data[bias:bias+self.context_length]
```

所以每一行都用同一个 `bias`，`batch_size` 行完全一样。实际运行会看到：

```text
x[0] = [0, 1, 2, 3, 4, 5, 6]
x[1] = [0, 1, 2, 3, 4, 5, 6]
```

这不是“一个 batch 里有多条不同序列”，而是同一条序列复制了 `B` 遍。

#### 问题二：`start_num` 和起点步长不对

`start_num = (N - T - 1) // B` 只是“能放多少个 batch”，但 `bias = i * B` 每次只前进 `B` 个 token，而不是 `B * T` 个 token。这样验证集覆盖的窗口会大量重叠，而且还会漏掉尾部。

更重要的是：验证集通常希望**每个窗口只评估一次**，所以起点应该按 `T` 步进（不重叠），或者至少让一个 batch 内的 `B` 行有不同起点。

一个简单的修正思路：

```python
# 相邻起点：每个 batch 的 B 行起点是 bias, bias+1, ..., bias+B-1
for batch_start in range(0, num_starts, batch_size):
    starts = np.arange(batch_start, min(batch_start + batch_size, num_starts))
    idx = starts[:, None] + np.arange(context_length)
    ...
```

如果你希望验证窗口不重叠，可以把 `starts` 改成：

```python
starts = np.arange(batch_start, min(batch_start + batch_size * context_length, num_starts), context_length)
```

### 6.4 `__len__`

```python
return self.data_len // self.batch_size
```

这个返回值叫“batch 数量”，但它没有考虑 `context_length`：

- `N=100, B=32, T=7` 时返回 `3`；
- 但合法起点有 `93` 个，按每批 32 个算，验证批次数约为 `ceil(93/32)=3`，碰巧一样；
- 换个参数就不对了，例如 `N=100, B=7, T=20`：返回 `14`，但合法起点只有 `80` 个，按每批 7 个算约 `12` 批。

更合理的定义取决于你想表达什么：

```python
# 如果 __len__ 表示“验证集有多少个 batch”
def __len__(self):
    num_starts = max(0, self.data_len - self.context_length)
    return (num_starts + self.batch_size - 1) // self.batch_size
```

> 如果训练是“随机采样、无限循环”，其实没有“epoch 长度”的概念；`__len__` 只对顺序验证集有意义。

### 6.5 当前文件采用的修正版

当前 `Data_Loader.py` 采用的核心实现如下（省略了文件里的类 docstring、`batch_size` 检查和少量类型标注）。它保留了原文件的接口风格，但修掉了上面提到的关键问题：

```python
import numpy as np
import torch


class DataLoader:
    def __init__(self, data, batch_size, context_length, shuffle=True):
        # 允许传入 list / np.ndarray / np.memmap。
        # memmap 本身就是 ndarray 的子类，不会被复制。
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        self.data = data
        self.data_len = len(data)
        self.batch_size = batch_size
        self.context_length = context_length
        self.shuffle = shuffle
        self.num_starts = self.data_len - self.context_length
        self._cursor = 0

        if self.num_starts <= 0:
            raise ValueError("数据太短：至少需要 context_length + 1 个 token")

    def _make_batch(self, starts, device):
        """starts: (B,)；返回 x, y，形状都是 (B, T)。"""
        idx = starts[:, None] + np.arange(self.context_length)
        x = torch.from_numpy(np.asarray(self.data[idx], dtype=np.int64))
        y = torch.from_numpy(np.asarray(self.data[idx + 1], dtype=np.int64))
        return x.to(device), y.to(device)

    def get_train_batch_data(self, device="cpu"):
        """随机（或顺序）取一个 batch。"""
        if self.shuffle:
            starts = np.random.randint(0, self.num_starts, size=self.batch_size)
        else:
            # 顺序模式：从上次的位置继续，走到末尾后回到开头。
            starts = (self._cursor + np.arange(self.batch_size)) % self.num_starts
            self._cursor = (self._cursor + self.batch_size) % self.num_starts
        return self._make_batch(starts, device)

    def get_valid_batch_data_iter(self, device="cpu"):
        """按顺序遍历验证集：起点按 context_length 步进，窗口互不重叠。"""
        window_starts = np.arange(0, self.num_starts, self.context_length)
        for i in range(0, len(window_starts), self.batch_size):
            starts = window_starts[i:i + self.batch_size]
            yield self._make_batch(starts, device)

    def __len__(self):
        """验证集 batch 数（窗口互不重叠，最后一个 batch 可能不满）。"""
        windows = len(range(0, self.num_starts, self.context_length))
        return (windows + self.batch_size - 1) // self.batch_size
```

这版的关键变化：

1. `num_starts = N - T`，不再多减 1；
2. 用 `idx = starts[:, None] + arange(T)` 一次构造整个 batch；
3. 显式 `astype(np.int64)`，避免 uint16 直接进 embedding；
4. 支持 `device` 参数；
5. 验证集起点按 `T` 步进，窗口互不重叠，每个 token 只被评估一次；
6. `__len__` 按“窗口数 / B”计算，和迭代器实际产出的批次数一致；
7. `shuffle=False` 时用游标顺序前进，而不是每批都从 0 开始。

> 不重叠的代价是批次数变少（约 `N/(T·B)`），一轮覆盖的 token 也少于“相邻起点”方案。好处是 `val_loss` 不会被重复样本稀释——同一条序列被算 256 遍平均出来的数值会明显更平滑，看起来稳定，其实没有信息量。

### 6.6 手动验证当前文件

```python
import numpy as np
from cs336_basics.Data_Loader import DataLoader

np.random.seed(0)
data = np.arange(100)
loader = DataLoader(data, batch_size=32, context_length=7)

starts = []
for _ in range(1000):
    x, y = loader.get_train_batch_data()
    starts.append(int(x[0, 0]))

print("最小起点:", min(starts))
print("最大起点:", max(starts))   # 应该是 92
print("y 是否等于 x+1:", bool((y == x + 1).all()))

for x, y in loader.get_valid_batch_data_iter():
    print("验证 batch 形状:", tuple(x.shape))
    print("前两行起点:", x[0, 0].item(), x[1, 0].item())  # 应该不同
    break
```

预期：

```text
最小起点: 0
最大起点: 92
y 是否等于 x+1: True
验证 batch 形状: (14, 7)
前两行起点: 0 7
```

两点说明：

- `data_len=100, T=7` 时合法起点有 93 个，按 `T` 步进能切出 `ceil(93/7)=14` 个不重叠窗口，正好一批装满，所以形状是 `(14, 7)` 而不是 `(32, 7)`；
- 相邻窗口起点相差 `7`，所以前两行是 `0` 和 `7`。

> 修正前旧版在同样设置下最大起点是 `91`，而且验证 batch 的前两行起点都是 `0`。你可以把旧版代码贴到临时文件里跑一遍，再和当前文件对比，理解“off-by-one”和“变量遮蔽”这两个 bug。

---

## 7. 把 Data Loader 接进训练循环

### 7.1 最小训练循环

```python
import numpy as np
import torch

from cs336_basics.Data_Loader import DataLoader
from cs336_basics.Cross_Entropy_Loss import CrossEntropyLoss
from cs336_basics.AdamW import AdamW

# 1. 用 memmap 打开 token 数据，不要全量读入内存
#    文件是训练脚本产出的裸 uint16 二进制（见 prepare_data.py），不是 .npy
data = np.memmap("data/bin/train.bin", dtype=np.uint16, mode="r")

# 2. 构造 Data Loader
loader = DataLoader(
    data=data,
    batch_size=32,
    context_length=256,
    shuffle=True,
)

# 3. 训练
model = TransformerLM(...).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3)

for step in range(num_steps):
    x, y = loader.get_train_batch_data(device=device)   # (B,T), (B,T)

    logits = model(x)                                   # (B,T,V)
    # CrossEntropyLoss 是类不是函数，数据在构造时传入，调 forward() 拿张量；
    # 内部已经按位置取平均，不用再除 B*T
    loss = CrossEntropyLoss(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
    ).forward()

    optimizer.zero_grad()
    loss.backward()
    # 仓库里用的是自己的 GradientClip；这里写 clip_grad_norm_ 是因为语义一致，
    # 两者都是对全部梯度求全局 L2 范数再等比缩放
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### 7.2 形状检查

在第一次循环时打印：

```python
print("x:", x.shape, x.dtype, x.device)
print("y:", y.shape, y.dtype, y.device)
print("logits:", logits.shape)
print("loss:", loss.item())
```

应该看到（`V=10000` 是本作业的词表大小，不是 GPT-2 的 50257）：

```text
x: torch.Size([32, 256]) torch.int64 cuda:0
y: torch.Size([32, 256]) torch.int64 cuda:0
logits: torch.Size([32, 256, 10000])
loss: 9.2...
```

> 随机初始化时 loss 应该接近 $\ln V$：$\ln 10000 \approx 9.21$，
> 而 $\ln 50257 \approx 10.82$。开工前先看一眼 loss 落没落在 $\ln V$ 附近，
> 是排查"初始化/形状/标签错位"最快的一步。

如果 `x.dtype` 是 `torch.uint16` 或 `torch.float32`，embedding 会报错或结果不对。

### 7.3 损失里不要再 shift

```python
# 正确：y 已经是下一个 token
loss = CrossEntropyLoss(logits.reshape(-1, V), y.reshape(-1)).forward()

# 错误：又 shift 了一次，标签和预测错位
loss = CrossEntropyLoss(
    logits[:, :-1].reshape(-1, V), y[:, 1:].reshape(-1)
).forward()
```

第二种写法只有在 `y` 还是“原始序列”、没有提前 shift 时才需要。本作业的 `get_batch` 已经帮你 shift 好了。

### 7.4 验证循环

```python
model.eval()
total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for x, y in loader.get_valid_batch_data_iter(device=device):
        logits = model(x)
        loss = cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()

print("验证集平均损失:", total_loss / total_tokens)
```

> 最后一个 batch 可能不满 `B`，所以用 `sum` 累加再除以 token 总数，比直接平均每个 batch 的 loss 更准确。

### 7.5 关于 `shuffle`

当前 `Data_Loader.py` 的 `shuffle` 行为：

- `shuffle=True`（默认）：`get_train_batch_data` 每次随机选 `B` 个起点；
- `shuffle=False`：用一个游标按顺序前进，走到末尾后回到开头；
- 验证集使用 `get_valid_batch_data_iter`，始终按顺序遍历，和 `shuffle` 无关。

> 顺序模式只是“依次取窗口”，并不是严格的“一个 epoch 内不重复”。如果你需要按 epoch 打乱且不重复，可以额外维护一个打乱后的起点列表。

---

## 8. 联系 CS336 Assignment 1

### 8.1 作业要求

CS336 讲义 §5.1 的 `data_loading` 题目（2 分）要求：

> 写一个函数，接收一个 NumPy 整数数组 `x`（token ID）、`batch_size`、`context_length` 和一个 PyTorch device 字符串，返回一对 Tensor：
>
> - 输入序列；
> - 对应的下一个 token 目标；
>
> 两者形状都是 `(batch_size, context_length)`，并且都在指定 device 上。

测试入口是 `tests/adapters.py` 里的 `run_get_batch`，测试文件是 `tests/test_data.py`。

### 8.2 适配器签名

```python
def run_get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

返回值要求：

```text
x: torch.LongTensor, shape (batch_size, context_length), device=device
y: torch.LongTensor, shape (batch_size, context_length), device=device
```

### 8.3 最短通过实现

```python
import numpy as np
import torch


def run_get_batch(dataset, batch_size, context_length, device):
    num_possible_starts = len(dataset) - context_length

    # 高位不包含，所以 0 .. num_possible_starts-1 全部合法
    starts = np.random.randint(0, num_possible_starts, size=batch_size)

    # (B,1) + (T,) -> (B,T)
    idx = starts[:, None] + np.arange(context_length)

    x = torch.from_numpy(dataset[idx].astype(np.int64))
    y = torch.from_numpy(dataset[idx + 1].astype(np.int64))

    return x.to(device), y.to(device)
```

这段代码和 §3.3 的公式完全对应。

### 8.4 测试到底在查什么

`tests/test_data.py` 的核心逻辑：

```python
dataset = np.arange(0, 100)
context_length = 7
batch_size = 32
device = "cpu"

for _ in range(1000):
    x, y = run_get_batch(dataset, batch_size, context_length, device)

    assert x.shape == (32, 7)
    assert y.shape == (32, 7)
    np.testing.assert_allclose((x + 1).numpy(), y.numpy())

    starting_indices.update(x[:, 0].tolist())
```

它检查四件事：

1. **形状**：`x`、`y` 都是 `(32, 7)`；
2. **偏移**：`y == x + 1`；
3. **起点范围**：必须能取到 `0` 和 `92`，不能越界；
4. **分布**：1000 次 × 32 个样本，每个起点的出现次数应落在均值 ±5σ 内。

此外还会传入 `device="cuda:99"`，确认你确实把 Tensor 放到了指定 device 上；非法 device 应该报错。

### 8.5 修正前旧版为什么过不了测试

| 测试要求 | 修正前旧版行为 | 结果 |
| --- | --- | --- |
| 最大起点 `92` | `randint(0, 92)` 最大 `91` | 失败 |
| `y == x + 1` | 切片正确 | 通过 |
| 形状 `(32,7)` | `np.stack` 正确 | 通过 |
| device 参数 | 方法没有 device 参数 | 适配器没调用它，所以测试不直接查这个 |

当前 `Data_Loader.py` 已把起点高位改成 `N-T`，并补上了 `device` 参数，所以训练采样能正确覆盖 `0..92`。

注意：`tests/test_data.py` 调用的是 `tests/adapters.py` 里的 `run_get_batch`，不是直接调用 `cs336_basics/Data_Loader.py` 的类。所以：

- 适配器里已经有一份正确实现；
- `Data_Loader.py` 是你自己的练习代码，测试不会直接跑它；
- 但你应该把两者的逻辑对齐，避免“测试过了，自己的类却是错的”。

### 8.6 数据准备流程

一个完整流程大致是：

```text
原始文本
  -> 训练 tokenizer
  -> 用 tokenizer 编码每篇文档
  -> 文档之间插入 <|endoftext|>
  -> 拼成一条长 token 流
  -> 保存为 .npy 或裸 .bin
  -> 训练时用 np.load(..., mmap_mode="r") 或 np.memmap 打开
  -> get_batch 采样
```

保存示例：

```python
tokens = np.concatenate(all_token_arrays).astype(np.uint16)
np.save("train.npy", tokens)
```

读取示例：

```python
train_data = np.load("train.npy", mmap_mode="r")
```

### 8.7 常见扣分点

| 扣分点 | 后果 | 修正 |
| --- | --- | --- |
| 起点高位写成 `N-T-1` | 采不到最后一个合法起点 | 改成 `N-T` |
| `y` 没有 shift | 模型学“复制输入” | `y = data[idx+1]` |
| `y` shift 了两次 | 标签错位 | 只 shift 一次 |
| dtype 是 float | embedding/交叉熵报错 | `.astype(np.int64)` |
| 忘记 `.to(device)` | 测试设备检查失败 | 返回前 `.to(device)` |
| 把整个数据集转成 Tensor | 内存爆炸 | 只对 `(B,T)` 小 batch 转换 |
| memmap dtype 写错 | token 值乱码 | 保存/读取 dtype 一致 |
| 验证集所有行同一起点 | 评估重复、覆盖不全 | 每行用不同起点 |

---

## 9. 常见问题 FAQ

### Q1：为什么 `y` 不是 `x` 的最后一个 token？

因为一个长度为 `T` 的窗口要提供 `T` 个训练信号。`y` 的每个位置都是对应位置的“下一个 token”，不只是最后一个。

### Q2：batch 里的序列可以重叠吗？

可以。训练时随机采样，两个窗口完全可能重叠，甚至起点相同。这不是 bug，反而能增加样本利用率。

### Q3：为什么同一个起点会在一个 batch 里出现两次？

因为训练采样是**有放回**的。`np.random.randint` 每次独立选，重复是正常的。如果你要求一个 batch 内不重复，需要额外的无放回采样逻辑，但本作业不需要。

### Q4：`context_length` 和 `batch_size` 怎么选？

- `context_length` 受模型位置编码/注意力范围限制，也受显存限制；
- `batch_size` 受显存限制；
- 两者乘积 `B*T` 决定一次前向的 token 数；
- 先用小 `B`、短 `T` 跑通，再逐步加。

### Q5：验证集为什么要顺序采样？

为了让每次评估用同一批数据，loss 可比较。如果验证集也随机，不同 step 的验证 loss 会抖动，难以判断模型是否真的变好。

### Q6：能不能直接把整个 `.txt` 读成 Python list？

小数据可以；大数据不建议。Python list 的每个 int 都是对象，内存占用可能是原始 token 的十几倍。大语料应该先转成 `uint16`/`uint32` 的 NumPy 数组，再用 memmap。

### Q7：用了 memmap，为什么训练还是慢？

`mmap` 只解决“内存装不下”，不解决磁盘随机读慢。可能的原因：

- 数据在机械硬盘上，随机窗口导致大量寻道；
- 每个 batch 都触发缺页，page cache 命中率低；
- 数据预处理（tokenize）本身很慢；
- GPU 利用率低，瓶颈其实在模型或日志。

可以先顺序读几个 batch 测吞吐，再判断是不是 I/O 问题。

### Q8：`uint16` 够用吗？

如果 `vocab_size <= 65535`，够用。GPT-2 的 50257 就可以。若词表更大，用 `uint32`。不确定时先用 `int32`，确认没问题再压缩。

### Q9：为什么 loss 要 flatten？

交叉熵通常接收 `(N, C)` 的 logits 和 `(N,)` 的标签。`(B,T,V)` 和 `(B,T)` 要先压平成 `(B*T,V)` 和 `(B*T,)`。这只是改变形状，不改变元素顺序。

### Q10：`x` / `y` 需要 one-hot 吗？

不需要。交叉熵内部会做 log-softmax，标签只需要整数类别 ID。one-hot 反而浪费内存。

### Q11：怎样复现同一个 batch？

在采样前设置随机种子：

```python
np.random.seed(42)
x1, y1 = loader.get_train_batch_data()
np.random.seed(42)
x2, y2 = loader.get_train_batch_data()
assert torch.equal(x1, x2)
```

注意：如果你同时用了 Python 的 `random` 和 NumPy 的 `np.random`，它们是两套独立的随机数生成器，要分别设种子。

### Q12：为什么类名叫 `DataLoader`，和 PyTorch 的 `DataLoader` 重名？

重名不会报错，但容易混淆。导入时可以起别名：

```python
from cs336_basics.Data_Loader import DataLoader as LMDataLoader
```

PyTorch 的 `torch.utils.data.DataLoader` 是一个通用的批处理迭代器；这里的 `DataLoader` 是专门为语言模型下一个 token 预测写的小类，两者不是一回事。

### Q13：`x` 和 `y` 需要放到同一个 device 吗？

需要。模型在哪个 device，`x` 就要在哪个 device；`y` 也要和 logits 在同一 device，否则交叉熵会报 device mismatch。

### Q14：为什么最后一个验证 batch 可能不满？

如果合法起点数不是 `batch_size` 的整数倍，最后一个 batch 会小一些。训练时随机采样永远能凑满 `B`，验证时顺序遍历就可能遇到“尾批”。处理方式：

- 保留尾批，loss 用 token 数加权；
- 或者丢掉尾批，保证每个 batch 形状固定。

---

## 10. 自测清单与一页速查

### 10.1 自测清单

- [ ] 我能解释为什么 `y` 和 `x` 形状相同，而不是少一列；
- [ ] 我能写出合法起点的范围 `0 .. N-T-1`；
- [ ] 我知道 `np.random.randint(0, N-T)` 的高位不包含；
- [ ] 我能手算 `N=100, T=7` 时的最大起点是 `92`；
- [ ] 我能解释 `y[b,t] = x[b,t+1]`；
- [ ] 我知道 `x`/`y` 的 dtype 应该是 `torch.long`；
- [ ] 我能说出 `np.memmap` 和 `np.load(..., mmap_mode="r")` 的区别；
- [ ] 我知道 `.npy` 不能直接用 `np.memmap` 读；
- [ ] 我能解释为什么不能 `torch.tensor(整个数据集)`；
- [ ] 我能指出修正前旧版 `Data_Loader.py` 的两个 bug；
- [ ] 我能把 `get_batch` 接进训练循环，并且不在损失里重复 shift；
- [ ] 我知道验证集为什么要顺序采样。

### 10.2 一页速查

```text
数据:       data = [x0, x1, ..., xN-1]
窗口长度:   T
合法起点:   s = 0 .. N-T-1
采样:       starts = np.random.randint(0, N-T, size=B)
下标:       idx = starts[:, None] + np.arange(T)
输入:       x = data[idx]          # (B,T)
目标:       y = data[idx + 1]      # (B,T)
dtype:      int64 / torch.long
device:     x.to(device), y.to(device)

训练损失:
  logits = model(x)                # (B,T,V)
  loss = cross_entropy(logits.reshape(-1,V), y.reshape(-1))

内存:
  小数据: np.ndarray
  大数据: np.load(..., mmap_mode="r") 或 np.memmap(...)
  只复制 B*T 个小 batch，不要复制整个数据集
```

### 10.3 一句话总结

> **Data Loader 做的事，就是从一条很长的 token 流里随机切出 `B` 个长度为 `T` 的窗口：窗口本身是 `x`，窗口整体右移一格是 `y`；训练就是让模型在每个位置根据 `x` 的前缀预测 `y`。数据太大时，用 memmap 按页读取，只把当前 batch 复制进内存。**

---

## 附：相关文档

- `BPE算法原理.md`：token 是怎么来的；
- `Cross_Entropy_Loss.md`：`logits` 和 `y` 怎么算损失；
- `完整语言模型架构的组装.md`：Data Loader 在完整训练流程中的位置；
- `AdamW.md`：拿到 loss 之后，参数怎么更新。
