# Python 和 PyTorch 基础：读懂 CS336 教程的预备课

> 这份文档是整套 `Markdown教程/` 的**地基**。它不追求把 Python 讲成一本语言手册，而是把教程里**真正出现过**的 Python 语法、标准库、NumPy 和 PyTorch API 挑出来，逐个说明：
>
> - 它是什么；
> - 它解决什么问题；
> - 在教程的哪个场景会用到；
> - 用一个生活化比喻帮你记住。
>
> 建议读法：**先通读一遍建立地图，再回到具体教程里边看边查**。不要试图一次背完。

---

## 目录

| 章节 | 内容 |
| --- | --- |
| §0 | 使用说明、运行环境与知识地图 |
| §1 | Python 语法：变量、数字、字符串 |
| §2 | Python 容器：list、tuple、dict、set 与切片 |
| §3 | 控制流：if、for、while、推导式 |
| §4 | 函数：参数、返回值、lambda、生成器 |
| §5 | 类与对象：`class`、`self`、继承、`nn.Module` 的前置知识 |
| §6 | 异常、文件、路径、JSON、命令行、正则 |
| §7 | NumPy：数组、dtype、memmap |
| §8 | PyTorch 张量：形状、索引、广播、autograd |
| §9 | PyTorch 模型：Module、Parameter、层、损失、优化器 |
| §10 | 把知识串起来：训练与推理 |
| §11 | 常见报错速查 |
| §12 | 学习路线与练习 |
| §13 | 一页速查表 |

---

## 0. 使用说明、运行环境与知识地图

### 0.1 这份文档怎么用

你可以把这份文档想成一张**城市地图**：

- 教程是具体的街道和景点；
- 本文是“地铁线路图”，告诉你 Python 和 PyTorch 的哪些站会经过；
- 第一遍不需要记住每个细节，知道“遇到问题该查哪一节”就够了。

例如：

- 看到 `@` 不懂 → 查 §8.7“矩阵乘法”；
- 看到 `yield` 不懂 → 查 §4.9“生成器”；
- 看到 `shape`、`view`、`transpose` 不懂 → 查 §8.4～§8.6；
- 看到 `state_dict` 不懂 → 查 §9.5；
- 看到 `with open(...)` 不懂 → 查 §6.2。

### 0.2 运行环境：`uv`、模块与包

项目要求：

```text
Python >= 3.12, < 3.14
PyTorch ~= 2.11
NumPy >= 2.4
```

推荐用 `uv` 管理环境：

```bash
cd /home/kevin/assignment1-basics

# 运行一个脚本
uv run python your_script.py

# 以模块方式运行项目里的入口
uv run python -m cs336_basics.main_train --help

# 跑测试
uv run pytest
```

> **比喻**：`uv` 像一间已经配好厨具的厨房。你不需要每次做饭前先买锅碗瓢盆，直接进去用就行。

#### 包（package）和模块（module）

```text
cs336_basics/
├── __init__.py          # 告诉 Python：这是一个包
├── AdamW.py             # 模块
├── RMSNorm.py           # 模块
├── Markdown教程/        # 教程文档
└── main_train.py        # 入口脚本
```

导入方式：

```python
from cs336_basics.AdamW import AdamW
from cs336_basics.RMSNorm import RMSNorm

import torch
import numpy as np
```

- `import torch`：把整个工具箱搬进来；
- `from torch import nn`：只拿工具箱里的一层抽屉；
- `from cs336_basics.AdamW import AdamW`：从项目自己的模块里拿一个类。

`__init__.py` 里可能用 `importlib.metadata.version(...)` 读取包版本，并用 `try/except PackageNotFoundError` 兜底；`pass` 表示“这里什么都不做”。这些都是工程细节，初学时知道有这回事即可。

> **比喻**：`import` 是“去仓库取工具”；`from ... import ...` 是“只取某个抽屉里的某一件”。

#### `if __name__ == "__main__":`

```python
def main():
    print("只有直接运行这个文件时才执行")


if __name__ == "__main__":
    main()
```

- 直接运行 `python main_train.py`：`__name__` 是 `"__main__"`，执行 `main()`；
- 被别的文件 `import`：`__name__` 是模块名，不执行 `main()`，避免“导入一下就开始训练”。

> **比喻**：`__main__` 是“前门”。只有从前门进来才开灯；别人从窗户递东西进来，不要顺手把整栋楼点亮。

### 0.3 教程知识地图

这张表把 18 篇教程和它们主要用到的 Python/PyTorch 知识对应起来：

| 教程 | 主要 Python / PyTorch 知识 |
| --- | --- |
| `BPE算法原理.md` | `bytes`/`str`、`dict`、`tuple`、`Counter`、`defaultdict`、正则、排序、文件读写 |
| `ASCII_Unicode_UTF-8.md` | `ord`/`chr`、`str.encode`/`bytes.decode`、UTF-8 |
| `Embedding_Linear.md` | `nn.Module`、`nn.Parameter`、`nn.Linear`、`nn.Embedding`、索引、`state_dict` |
| `Attention.md` | 张量形状、矩阵乘法、`softmax`、转置、广播 |
| `缩放点积注意力.md` | `@`/`matmul`、`transpose`、`masked_fill`、`tril`、`allclose` |
| `Causal_Multi-Head_self-attention.md` | `view`/`transpose`/`contiguous`、`ModuleList`、广播、因果 mask |
| `RMSNorm.md` | `mean`、`pow`、`rsqrt`、dtype 转换、`nn.Parameter` |
| `SwiGLU.md` | `nn.Linear`、`sigmoid`、逐元素乘法、`@staticmethod` |
| `Transformer_Block.md` | `nn.Module`、残差、`ModuleList`/`Sequential`、`register_buffer` |
| `完整语言模型架构的组装.md` | 形状流转、`Embedding`、`Linear`、`softmax`、`cross_entropy` |
| `Cross_Entropy_Loss.md` | `log_softmax`、高级索引、`mean`、`F.cross_entropy` |
| `AdamW.md` | `Optimizer` 子类、`param_groups`、原地运算、`state_dict` |
| `Gradient Clipping.md` | `cat`、`flatten`、`norm`、`mul_`、生成器 |
| `学习率调度：Cosine Schedule.md` | `math`、`__call__`、浮点运算 |
| `Data Loader.md` | NumPy `memmap`、随机整数、高级索引、`torch.from_numpy`、`__len__` |
| `CheckPoint.md` | `state_dict`、`torch.save/load`、`BytesIO`、`os.replace` |
| `端到端训练和模型推理.md` | `argparse`、训练循环、`no_grad`、`sort`/`cumsum`/`multinomial`/`gather` |
| `Python和PyTorch基础.md` | 本文：把上面这些基础一次补齐 |

### 0.4 三条学习路线

| 你的情况 | 建议顺序 |
| --- | --- |
| 完全没写过 Python | §1 → §2 → §3 → §4 → §5 → §6，再读教程 |
| 会 Python，但不熟 PyTorch | §7 → §8 → §9，再读模型和训练教程 |
| 只是想把教程跑通 | §0.2 → §8.1～§8.4 → §9.1～§9.5 → §10，遇到再查 |

### 0.5 形状记号先约定

教程里经常出现：

```text
(B, T, D)
B = batch size
T = sequence length / context length
D = d_model / hidden size
V = vocab size
H = num_heads
```

还有：

```text
(..., T, D)
```

这里的 `...` 表示“前面可以有任意多个维度”，例如：

- `(T, D)`：没有 batch；
- `(B, T, D)`：有 batch；
- `(B, H, T, D)`：有 batch 和 head。

> **注意**：Python 里也有真正的 `...`（Ellipsis），但教程类型标注字符串里的 `...` 只是“任意前置维度”的记号，不是代码要执行的省略号。

---

## 1. Python 语法：变量、数字、字符串

### 1.1 变量：带标签的盒子

```python
n = 3
lr = 6e-4
name = "tiny"
ok = True
nothing = None
```

**概念**：变量是“名字 → 对象”的绑定。Python 是动态类型，同一个名字可以先指向整数，再指向字符串。

**类比**：变量像储物柜上的标签。标签写“学习率”，柜子里可以放 `0.0006`；换一份实验，可以把柜子里的东西换成 `0.001`。

**教程场景**：

- `batch_size = 32`
- `d_model = 512`
- `lr = 6e-4`
- `device = "cuda" if torch.cuda.is_available() else "cpu"`

### 1.2 数字与运算

| 写法 | 含义 | 例子 |
| --- | --- | --- |
| `+ - *` | 加减乘 | `2 + 3` |
| `/` | 真除法，结果一定是浮点数 | `3 / 2 == 1.5` |
| `//` | 整除，向下取整 | `7 // 2 == 3` |
| `%` | 取余 | `7 % 2 == 1` |
| `**` | 幂 | `2 ** 10 == 1024` |
| `1e-5` | 科学计数法 | `0.00001` |

教程里的例子：

```python
vocab_size = 10000
num_merges = vocab_size - 256 - len(special_tokens)
std = (2.0 / (in_features + out_features)) ** 0.5
eps = 1e-5
```

> **比喻**：`/` 是“精确切蛋糕”，`//` 是“只数完整的蛋糕块”，`%` 是“切完剩下的边角料”。

### 1.3 布尔值与真值判断

```python
is_cuda = torch.cuda.is_available()
use_rope = not no_rope


def encode(text):
    if not text:
        return []
```

Python 里很多值可以直接当条件：

| 值 | 真假 |
| --- | --- |
| `0`、`0.0` | 假 |
| `""`、`[]`、`{}`、`set()` | 假 |
| `None`、`False` | 假 |
| 其他非空值 | 真 |

> **比喻**：`if` 像门卫，不只看“是不是 `True`”，还看“东西是不是空的”。空箱子、空名单，一律不放行。

### 1.4 `is None` 与 `==`

```python
for p in model.parameters():
    if p.grad is None:
        continue

if mask is not None:
    ...
```

- `is None`：判断“是不是同一个特殊对象 None”；
- `==`：判断“值是否相等”。

> **比喻**：`is None` 是“看身份证号是不是空”，`==` 是“看内容是不是一样”。

### 1.5 字符串：人类可读的文本

```python
name = "tiny"
message = f"模型参数量: {n_params / 1e6:.2f} M"
```

字符串方法在教程里很常见：

| 方法 | 作用 | 教程场景 |
| --- | --- | --- |
| `split()` | 按分隔符切分 | `line.split(" ")` |
| `join()` | 用分隔符拼接 | `"|".join(tokens)` |
| `strip()` / `rstrip()` | 去首尾空白 | `line.rstrip("\n")` |
| `startswith()` | 是否以某前缀开头 | 跳过表头 |
| `find()` | 查找子串位置 | 字节块查找 |
| `encode()` | 字符串 → 字节 | `text.encode("utf-8")` |
| `decode()` | 字节 → 字符串 | `bytes.decode("utf-8")` |

#### f-string

```python
loss = 1.23456
print(f"loss={loss:.4f}, lr={6e-4:.2e}")
```

输出：

```text
loss=1.2346, lr=6.00e-04
```

- `:.4f`：保留 4 位小数；
- `:.2f`：保留 2 位小数；
- `:.2e`：科学计数法；
- `:7.2f`：总宽度 7，右对齐。

> **比喻**：f-string 是“填空题模板”，大括号里可以放变量，也可以规定“这个空要写几位小数”。

#### 原始字符串与三引号

```python
pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+"""
doc = """这是一段
可以换行的说明文字"""
```

- `r"..."`：原始字符串，反斜杠不当转义符；正则里常用；
- `"""..."""`：三引号，可以跨行，常用于 docstring 或长正则。

### 1.6 `bytes`：计算机真正认识的字节

这是本教程非常重要的一对概念：

| 类型 | 例子 | 是什么 |
| --- | --- | --- |
| `str` | `"你好"` | 人类可读的 Unicode 文本 |
| `bytes` | `b"\xe4\xbd\xa0\xe5\xa5\xbd"` | 原始字节序列 |

转换：

```python
text = "你好"
raw = text.encode("utf-8")     # str -> bytes
back = raw.decode("utf-8")     # bytes -> str
```

`bytes` 的三个关键点：

1. `b"abc"[0]` 得到的是整数 `97`，不是字符 `"a"`；
2. `bytes([97, 98, 99])` 得到 `b"abc"`；
3. `b"".join([b"ab", b"cd"])` 得到 `b"abcd"`；
4. `b"abc".find(b"b")` 返回 `1`，找不到返回 `-1`（预分词切块时用来找特殊 token）。

教程场景：

```python
# BPE：把每个字节变成独立的 token
byte_parts = [bytes([b]) for b in word.encode("utf-8")]


def decode(full_bytes):
    # 解码：不完整的字节序列用替换字符兜底
    return full_bytes.decode("utf-8", errors="replace")
```

> **比喻**：`str` 是“菜谱上的文字”，`bytes` 是“冰箱里一袋袋原料”。UTF-8 是“把文字翻译成原料编号”的规则。

### 1.7 注释、docstring 与缩进

```python
# 单行注释

def forward(self, x):
    """这是 docstring：解释这个函数做什么。"""
    return x
```

- `#` 后面是注释，不执行；
- 函数/类/模块开头的三引号字符串是 docstring，`help()` 可以看到；
- Python 用缩进表示代码块，通常 4 个空格。

> **比喻**：缩进是“文件夹层级”。同一层的语句放在同一个抽屉里，缩进错了，抽屉就装错东西。

### 1.8 `ord` 与 `chr`

```python
ord("A")   # 65
chr(65)    # "A"
```

教程里用于构造“字节 → 可见字符”的映射表：

```python
bs = list(range(ord("!"), ord("~") + 1))
cs = [chr(c) for c in cs]
```

> **比喻**：`ord` 是查“字符的门牌号”，`chr` 是按门牌号找回字符。

---

## 2. Python 容器：list、tuple、dict、set 与切片

### 2.1 `list`：可以变的队伍

```python
ids = [3, 1, 3, 2]
ids.append(5)        # 末尾加一个
ids.extend([7, 8])   # 把另一个列表摊开加进来
ids[0] = 9           # 改第一个
```

**类比**：`list` 像一列火车，车厢可以增删、可以改座位，顺序很重要。

教程场景：

- `words_list` 存所有词；
- `tokens` 收集编码后的 ID；
- `val_batches` 存多个验证 batch；
- `grads` 存所有梯度。

常用操作：

| 操作 | 作用 |
| --- | --- |
| `len(a)` | 长度 |
| `a[i]` | 第 i 个 |
| `a[-1]` | 最后一个 |
| `a[1:3]` | 切片 |
| `a.append(x)` | 末尾追加一个 |
| `a.extend(xs)` | 末尾追加一批 |
| `a.pop()` | 弹出末尾 |
| `a.sort()` | 原地排序 |
| `sorted(a)` | 返回新列表 |

> **注意**：`b = a` 不是复制，两个名字指向同一列火车；要复制用 `b = a.copy()` 或 `list(a)`。

### 2.2 `tuple`：封好的快递盒

```python
pair = (b"t", b"h")
vocab_item = (0, b"!")
```

**类比**：`tuple` 像封好的快递盒，里面的东西不能换；因此可以当字典的 key。

教程场景：

- BPE 的 `(byte_a, byte_b)` 作为 `stats` 的 key；
- 函数返回多个值：
  ```python
  def f():
      return 1, 2


  a, b = f()   # 解包
  ```

> `tuple` 不可变，`list` 可变。`Counter` 的 key 必须是可哈希的，所以 BPE 用 `tuple` 而不是 `list`。

### 2.3 `dict`：电话簿

```python
vocab = {0: b"!", 1: b'"', 2: b"#"}

print(vocab[0])          # 按 key 查 value
print(vocab.get(99, 0))  # 查不到时给默认值
```

**类比**：`dict` 是电话簿：名字（key）→ 电话（value）。查名字很快，但不保证顺序（Python 3.7+ 按插入顺序保留，但语义上仍把它当映射）。

教程场景：

- `vocab: {id: bytes}`；
- `byte_to_id: {bytes: id}`；
- `state_dict: {参数名: 张量}`；
- `stats: {(byte_a, byte_b): 频率}`；
- `indices: {pair: {单词下标}}`。

常用操作：

```python
d = {"a": 1}
d["b"] = 2
for key, value in d.items():
    print(key, value)
print(d.keys())
print(d.values())
```

#### 字典推导式

```python
byte_to_id = {v: k for k, v in vocab.items()}
```

> **比喻**：字典推导式是“流水线翻转机”：把“ID → 字节”的表格整体翻成“字节 → ID”。

### 2.4 `set`：不重复的名单

```python
seen = set()
seen.add("a")
seen.add("a")
print(seen)   # {"a"}
```

**类比**：`set` 像签到名单，同一个人签两次也只算一次。

教程场景：

- `indices[pair]` 存“包含这个 pair 的单词下标”，用 `set` 自动去重；
- `sorted(set(boundaries))` 去重并排序；
- 判断 `pair in self.merges`。

> **注意**：`set` 无序，不能按下标取；需要顺序时用 `sorted(...)`。

### 2.5 索引与切片

```python
a = [10, 11, 12, 13, 14, 15]

a[0]      # 10
a[-1]     # 15
a[1:4]    # [11, 12, 13]
a[::2]    # [10, 12, 14]
a[1::2]   # [11, 13, 15]
a[::-1]   # 倒序
```

规则：

```text
a[start:stop:step]
```

- 包含 `start`，不包含 `stop`；
- 省略 `start` 从 0 开始；
- 省略 `stop` 到末尾；
- 负数从末尾数；
- `step=2` 隔一个取一个。

教程场景：

```python
x[..., 0::2]   # 取偶数维
x[..., 1::2]   # 取奇数维
x[:, -1, :]    # 取序列最后一个位置
data[i:i+T]    # 取窗口
```

> **比喻**：切片是“用尺子在传送带上量一段”，不是把整条传送带搬走。对 list/str 会生成新对象；对 NumPy/Tensor 可能是“视图”，后面会讲。

### 2.6 解包：把盒子拆开

```python
a, b = 1, 2
first, *rest = [1, 2, 3, 4]
```

教程场景：

```python
q, k, v = qkv.split(d_model, dim=-1)
sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
p1, p2 = pair
```

函数调用时用 `*` 展开：

```python
blocks = nn.Sequential(*[TransformerBlock(...) for _ in range(n_layers)])
```

> **比喻**：`*` 是“把一盒鸡蛋一个个拿出来摆到台面上”，而不是把整盒塞进去。

### 2.7 推导式：一条流水线

```python
squares = [x * x for x in range(5)]
even = [x for x in range(10) if x % 2 == 0]
lengths = {name: len(name) for name in ["a", "bb"]}
unique = {x % 3 for x in range(10)}
```

**类比**：推导式是“装配线”：从左边拿原料，经过 `for` 和 `if`，右边出成品。

教程场景：

```python
vocab = {i: bytes([i]) for i in range(256)}
indices = defaultdict(set)
byte_parts = [bytes([b]) for b in word.encode("utf-8")]

# 生成器表达式：圆括号，按需产出，不一次性建列表
word_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
total = sum(p.numel() for p in model.parameters())
```

> 列表推导式用 `[]`，生成器表达式用 `()`。前者一次建好列表，后者边用边算，适合大对象或流式数据。

### 2.8 可变对象、别名与复制

```python
a = [1, 2]
b = a          # 别名：同一列火车
b.append(3)
print(a)       # [1, 2, 3]

c = a.copy()   # 浅复制：新车，但车厢里的东西仍可能共享
```

**类比**：

- `b = a` 是“给同一把钥匙配了个新钥匙扣”；
- `a.copy()` 是“复制一列新车，但车厢里的货物如果是箱子，仍然可能是同一批箱子”；
- `copy.deepcopy(a)` 是“连车厢里的箱子也重新做一份”。

教程场景：

- `state_dict()` 返回的张量通常和参数共享内存，`torch.save` 会立刻写出，所以安全；如果要保存历史快照，需要 `copy.deepcopy`；
- 检查点里的 `state_dict` 不是深拷贝。

### 2.9 `Counter`、`defaultdict`、`OrderedDict`

#### `Counter`：自动计数的字典

```python
from collections import Counter

counts = Counter(["a", "b", "a"])
print(counts["a"])          # 2
print(counts.most_common(1))
```

**类比**：`Counter` 是“自动点票器”，每放进一个元素，票数加一。

教程场景：BPE 统计词频 `raw_counts[word] += 1`。

#### `defaultdict`：自动补默认值的字典

```python
from collections import defaultdict

stats = defaultdict(int)      # 不存在的 key 默认 0
stats["a"] += 1

indices = defaultdict(set)    # 不存在的 key 默认空 set
indices["pair"].add(3)
```

**类比**：`defaultdict` 是“会自动补货的货架”：你要一个不存在的商品，它先给你一个空盒子，而不是报错。

#### `OrderedDict`

`OrderedDict` 是“有顺序的字典”。普通 `dict` 在 Python 3.7+ 也保留插入顺序，但 `OrderedDict` 提供更明确的有序语义和一些额外方法。

教程场景：加载 checkpoint 时处理 DDP 的 `module.` 前缀。

### 2.10 容器选择表

| 需求 | 选择 | 类比 |
| --- | --- | --- |
| 有序、可修改 | `list` | 火车 |
| 有序、不可修改、可当 key | `tuple` | 封好的快递盒 |
| 按名字查值 | `dict` | 电话簿 |
| 去重、快速判断存在 | `set` | 签到名单 |
| 计数 | `Counter` | 点票器 |
| 自动补默认值 | `defaultdict` | 自动补货货架 |

---

## 3. 控制流：if、for、while、推导式

### 3.1 `if / elif / else`

```python
if temperature <= 0:
    raise ValueError("temperature 必须 > 0")
elif temperature < 1:
    print("更保守")
else:
    print("更随机")
```

**类比**：`if` 是岔路口，`elif` 是“如果不是，再检查下一个路牌”，`else` 是“都不符合就走默认路”。

教程场景：

- 学习率三段式：warmup / cosine / flat；
- `norm_mode == "pre"` 选择残差结构；
- `mask is not None` 决定是否加 mask。

### 3.2 `for` 与 `range`

```python
for i in range(5):
    print(i)          # 0..4

for i in range(2, 10, 2):
    print(i)          # 2,4,6,8
```

`range(start, stop, step)` 同样包含 `start`、不包含 `stop`。

**类比**：`for` 是传送带，`range` 是传送带上的编号牌。

教程场景：

```python
for layer in range(self.num_layers):
    ...
for group in optimizer.param_groups:
    ...
for x, y in loader.get_valid_batch_data_iter():
    ...
```

### 3.3 `while`

```python
i = 0
while i < len(byte_parts) - 1:
    ...
    i += 1
```

**类比**：`while` 是“只要门还开着就继续排队”，适合不知道要循环多少次、但知道停止条件的场景。

教程场景：BPE 合并时反复扫描相邻 pair，直到没有可合并的 pair。

### 3.4 `break` 与 `continue`

```python
for x in xs:
    if x is None:
        continue      # 跳过本次，进入下一次
    if x == target:
        break         # 直接结束整个循环
```

**类比**：

- `continue` 是“这盘菜不要了，直接下一盘”；
- `break` 是“整桌撤了，不吃了”。

教程场景：

- `if p.grad is None: continue`：跳过没有梯度的参数；
- `if best_pair is None: break`：没有可合并的 pair，退出 while；
- `if eos_token_id ...: break`：生成到结束符，停止。

### 3.5 `enumerate` 与 `zip`

```python
for i, word in enumerate(words_list):
    print(i, word)

for a, b in zip(xs, ys):
    print(a, b)
```

**类比**：

- `enumerate` 是“给每个元素发一个座位号”；
- `zip` 是“拉链”，把两条列表一对一咬合。

教程场景：

```python
for idx, word in enumerate(words_list):
    ...
for start, end in zip(boundaries[:-1], boundaries[1:]):
    ...
```

### 3.6 三元表达式

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
loss = "ok" if x > 0 else "bad"
```

**类比**：三元表达式是“岔路口压缩成一行”：条件为真走左边，否则走右边。

教程场景：

```python
norm = RMSNorm(...) if use_norm else nn.Identity()
rope_theta = 10000.0 if use_rope else 0.0
```

### 3.7 `assert`：代码里的烟雾报警器

```python
assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
assert isinstance(split_special_token, bytes), "必须是 bytes"
```

**作用**：在开发阶段检查“绝不应该发生”的情况；失败会抛 `AssertionError`。

> **注意**：`python -O` 会关闭 assert，所以真正面向用户的输入检查要用 `raise ValueError`，不要只靠 assert。

### 3.8 `del`：删除引用或元素

```python
del stats[pair]       # 删除字典里的键
del word[i + 1]       # 删除列表里的元素
```

教程场景：BPE 合并后删除旧 pair，避免频率为 0 的 pair 干扰 `max`。

> **比喻**：`del` 是“把货架上的某个商品下架”，不是把整个货架拆了。

---

## 4. 函数：参数、返回值、lambda、生成器

### 4.1 `def` 与 `return`

```python
def add(a, b):
    return a + b

result = add(1, 2)
```

**类比**：函数是一台机器：参数是投料口，`return` 是出料口。

教程场景：

```python
def softmax(x, dim):
    ...
    return out
```

### 4.2 默认参数与关键字参数

```python
def train(model, lr=1e-3, device="cpu"):
    ...

train(model)                    # 使用默认值
train(model, lr=3e-4)           # 只改 lr
train(model, device="cuda")     # 用关键字，顺序不重要
```

**注意**：不要用可变对象当默认值：

```python
def bad(x, cache=[]):   # 这个 [] 只创建一次，会在调用之间共享
    ...
```

正确写法：

```python
def good(x, cache=None):
    if cache is None:
        cache = []
```

> **比喻**：默认参数是“机器上的预设旋钮”。如果预设旋钮里装的是一个会变的篮子，所有机器会共用同一个篮子，容易出问题。

### 4.3 `*args`：可变数量的参数

```python
class Tee:
    def __init__(self, *streams):
        self.streams = streams
```

调用：

```python
tee = Tee(sys.stdout, log_file)
```

`streams` 得到 `(sys.stdout, log_file)` 这个 tuple。

教程场景：`main_train.py` 的 `_Tee` 把输出同时写到终端和日志文件。

### 4.4 多返回值与解包

```python
def min_max(xs):
    return min(xs), max(xs)

lo, hi = min_max([3, 1, 4])
```

Python 实际上返回一个 tuple，然后解包给两个变量。

教程场景：

```python
vocab, merges = train_bpe(...)
sorted_probs, sorted_idx = torch.sort(...)
```

### 4.5 `lambda` 与 `key`

```python
pairs = [("a", 2), ("b", 1)]
pairs.sort(key=lambda x: x[1])
```

**类比**：`lambda` 是“一次性小纸条函数”，写完就扔，常用于告诉排序/最大值“按什么规则比”。

教程场景：

```python
best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
```

含义：先比频率 `x[1]`，频率相同再比 pair 本身 `x[0]`。

### 4.6 作用域、闭包与 `nonlocal`

```python
def outer():
    count = 0

    def inner():
        nonlocal count
        count += 1
        return count

    return inner
```

- 函数内部定义的变量默认是局部的；
- 内层函数想改外层函数的变量，用 `nonlocal`；
- 想改模块级变量，用 `global`（本项目基本没用到）。

教程场景：`prepare_data.py` 的 `flush` 函数修改外层的 `written` 和 `buffer`：

```python
def process():
    written = 0
    buffer = []

    def flush(fh):
        nonlocal written, buffer
        ...
```

> **比喻**：`nonlocal` 是“允许你改家里客厅的摆设”，而不是只在自己房间里摆。

### 4.7 `@staticmethod` 与 `@classmethod`

```python
class SwiGLU:
    @staticmethod
    def silu(x):
        return x * torch.sigmoid(x)


class BPETokenizer:
    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        ...
        return cls(vocab={}, merges=[], special_tokens=special_tokens)
```

| 装饰器 | 第一个参数 | 用途 |
| --- | --- | --- |
| `@staticmethod` | 没有 `self/cls` | 工具函数，不依赖实例 |
| `@classmethod` | `cls` | 从文件等来源构造实例 |

> **比喻**：`staticmethod` 是“贴在墙上的公共工具”，谁都能用；`classmethod` 是“工厂入口”，负责按配方造一个新对象。

### 4.8 装饰器是什么

```python
@torch.no_grad()
def evaluate():
    ...
```

`@` 是“给函数套一层外壳”的语法糖。`torch.no_grad()` 这个装饰器会让函数内部不记录梯度。

教程场景：

- `@staticmethod`、`@classmethod`；
- 文档里可能用 `@torch.no_grad()` 包住评估/生成。

> **比喻**：装饰器像“给手机套壳”：手机还是那部手机，但多了一层功能（防摔、防水）。

### 4.9 生成器：`yield` 与 `yield from`

```python
def count_up(n):
    for i in range(n):
        yield i

for x in count_up(3):
    print(x)      # 0,1,2
```

**类比**：生成器是“自动贩卖机”：你按一次，它出一个；不会一次性把所有商品都倒出来。

教程场景：

- `DataLoader.get_valid_batch_data_iter()` 逐批产出验证数据；
- `Tokenizer.encode_iterable()` 逐行编码大文件；
- `yield from` 把一个可迭代对象的元素逐个转交。

```python
def encode_iterable(self, text):
    yield from self.encode(text)
```

> **比喻**：`yield from` 是“把另一台贩卖机的出货口接到我这台”，顾客拿到的还是一件一件的商品。

### 4.10 常用内置函数

| 函数 | 作用 | 教程场景 |
| --- | --- | --- |
| `len(x)` | 长度 | `len(data)`、`len(vocab)` |
| `range(...)` | 数字序列 | 循环 |
| `enumerate(x)` | 下标 + 元素 | BPE 遍历词表 |
| `zip(a, b)` | 成对 | 切块边界 |
| `sorted(x, key=...)` | 排序 | 特殊 token 按长度排序 |
| `max(x, key=...)` | 最大值 | BPE 选最高频 pair |
| `sum(x)` | 求和 | 参数量统计 |
| `int/float/str/bytes` | 类型转换 | 解析/编码 |
| `isinstance(x, T)` | 类型判断 | `isinstance(out, (str, os.PathLike))` |
| `getattr/setattr` | 动态读写属性 | `setattr(self, f"{name}_proj_weight", w)` |
| `next(iter, default)` | 取下一个 | 从 tokenizer 词表找 EOS |
| `print` | 输出 | 日志 |

### 4.11 函数常见坑

| 坑 | 后果 | 正确做法 |
| --- | --- | --- |
| 可变默认参数 `def f(x=[])` | 多次调用共享同一个列表 | 用 `None` 再在函数内创建 |
| 忘记 `return` | 得到 `None` | 明确返回 |
| 变量名遮蔽外层变量 | 逻辑错乱 | 换名字，如内层用 `j` |
| 生成器被消费两次 | 第二次为空 | 需要重复用时先 `list(...)` |
| 函数修改了传入的 list | 调用方被“偷改” | 明确是否原地修改，或先复制 |

---

## 5. 类与对象：`class`、`self`、继承、`nn.Module` 的前置知识

### 5.1 类与实例

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        return f"{self.name} 汪！"

d = Dog("旺财")
print(d.bark())
```

- `class` 是蓝图；
- `d = Dog("旺财")` 是按蓝图造出来的一只具体的狗（实例）；
- `self` 是“当前这只狗”；
- `__init__` 是构造器，创建实例时自动调用。

> **比喻**：类是小区的户型图，实例是真正建好的一套房子；`self` 是“这套房子的钥匙”，进哪套房就用哪把钥匙。

### 5.2 `__init__` 与属性

```python
class DataLoader:
    def __init__(self, data, batch_size, context_length):
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
```

- `self.data` 是实例属性；
- 每个实例各有一份；
- 在 `__init__` 里初始化，后面方法通过 `self.xxx` 访问。

### 5.3 方法、`forward` 与 `__call__`

```python
class MyModule:
    def forward(self, x):
        return x * 2

    def __call__(self, x):
        return self.forward(x)

m = MyModule()
print(m(3))   # 6
```

PyTorch 的 `nn.Module` 就是这样：

- 你实现 `forward()`；
- 调用 `model(x)` 时，`__call__` 会自动转到 `forward()`，并处理 hooks。

> **比喻**：`forward` 是“机器内部的生产流程”，`__call__` 是“机器外面的投料口”。用户按投料口，机器自动走流程。

### 5.4 继承与 `super()`

```python
class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        ...
```

- `class AdamW(optim.Optimizer)` 表示继承；
- `super().__init__(...)` 调用父类构造器，让父类帮忙注册参数组、初始化 `state`；
- 子类重写 `step()`，只负责“怎么更新”。

> **比喻**：继承是“子承父业”。`super()` 是“先请父亲把家底铺好”，然后儿子再决定具体怎么干。

### 5.5 常见 dunder 方法

| 方法 | 什么时候调用 | 教程场景 |
| --- | --- | --- |
| `__init__` | 创建对象时 | 所有类 |
| `__call__` | 对象像函数一样调用时 | `CosineSchedule`、`GradientClip`、`nn.Module` |
| `__len__` | `len(obj)` | `DataLoader` |
| `__name__` | 模块名 | `if __name__ == "__main__"` |
| `__main__` | 直接运行时的模块名 | 入口保护 |

> **比喻**：dunder 方法是“标准插座”。Python 看到 `len(obj)`，就去找 `__len__` 这个插座；看到 `obj(x)`，就去找 `__call__`。

### 5.6 组合：把对象装进对象

```python
class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = EmbeddingModule(...)
        self.layers = nn.ModuleList([...])
        self.lm_head = LinearModule(...)
```

**类比**：组合是“乐高积木”：大模型由小模块拼成，每个小模块自己负责一小步。

> **重要**：`nn.ModuleList` 不是普通 list。普通 list 里的模块不会被注册，`model.parameters()` 看不到；`nn.ModuleList` 会注册。

### 5.7 `isinstance`、`getattr`、`setattr`

```python
isinstance(out, (str, os.PathLike))   # 判断类型
getattr(model, "context_length", None) # 取属性，没有就返回默认值
setattr(self, f"{name}_proj_weight", w) # 动态设置属性
```

教程场景：

- `setattr` 批量创建 `q_proj_weight`、`k_proj_weight` 等参数；
- `getattr` 从模型读取 `context_length`；
- `isinstance` 判断保存目标是路径还是文件对象。

> **比喻**：`getattr` 是“按名字找抽屉”，`setattr` 是“按名字贴一个新抽屉”。

### 5.8 类与对象的常见坑

| 坑 | 后果 | 正确做法 |
| --- | --- | --- |
| 忘记 `super().__init__()` | 子模块/参数没注册 | 在 `nn.Module` 子类开头调用 |
| 用普通 list 装模块 | 参数不在 `model.parameters()` 里 | 用 `nn.ModuleList` |
| 把普通 Tensor 当参数 | 不会被优化器更新 | 用 `nn.Parameter` |
| 在 `__init__` 外创建参数 | 可能未注册 | 在 `__init__` 中创建 |
| 类属性当实例状态 | 所有实例共享 | 在 `__init__` 里 `self.x = ...` |

---

## 6. 异常、文件、路径、JSON、命令行、正则

### 6.1 `try / except / finally`

```python
try:
    import wandb
except ImportError:
    wandb = None

try:
    ...
finally:
    model.train(was_training)
```

- `try`：可能出错的代码；
- `except`：捕获指定异常；
- `finally`：无论是否出错都执行，常用于清理/恢复状态。

**类比**：`try` 是“走钢丝”，`except` 是下面的安全网，`finally` 是“不管有没有掉下去，最后都要收绳子”。

教程场景：

- `wandb` 没安装时降级为 `None`；
- `decode_token` 用 `finally` 恢复模型的 `train/eval` 模式；
- `except (OSError, ValueError)` 读取损坏的 iter 标记文件。

### 6.2 `raise` 与异常类型

```python
if temperature <= 0:
    raise ValueError("temperature 必须 > 0")

if not os.path.exists(path):
    raise FileNotFoundError(path)
```

常见异常：

| 异常 | 典型原因 |
| --- | --- |
| `ValueError` | 参数值不合法 |
| `TypeError` | 类型不对 |
| `KeyError` | 字典没有这个 key |
| `IndexError` | 下标越界 |
| `RuntimeError` | PyTorch 运行错误（形状/设备等） |
| `FileNotFoundError` | 文件不存在 |
| `ImportError` | 模块没安装 |

> **比喻**：`raise` 是“主动拉响火警”，比让程序带着错误数据继续跑更安全。

### 6.3 文件读写与 `with`

```python
with open("vocab.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("train.bin", "wb") as f:
    arr.tofile(f)
```

- `"r"` 读文本，`"rb"` 读二进制；
- `"w"` 覆盖写，`"wb"` 写二进制，`"a"` 追加；
- `encoding="utf-8"` 指定文本编码；
- `with` 自动关闭文件。

**类比**：`with` 是“自动门”：进去时开门，出来时自动关门，哪怕你中途摔倒了也会关。

文件指针：

```python
import os

f.seek(0)                  # 回到开头
pos = f.tell()             # 当前位置
f.seek(0, os.SEEK_END)     # 跳到文件末尾
size = f.tell()            # 文件大小
```

> `torch.save` 写完 `BytesIO` 后，指针在末尾；`torch.load` 前要 `seek(0)`。

### 6.4 路径：`os.path` 与 `pathlib`

```python
import os
from pathlib import Path

path = os.path.join("out", "ckpt.pt")
os.makedirs(os.path.dirname(path), exist_ok=True)
print(os.path.exists(path))

p = Path("out") / "ckpt.pt"
p.parent.mkdir(parents=True, exist_ok=True)
```

| API | 作用 |
| --- | --- |
| `os.path.join` | 拼路径 |
| `os.path.exists` | 是否存在 |
| `os.makedirs(..., exist_ok=True)` | 建目录，已存在不报错 |
| `os.path.dirname` | 取目录部分 |
| `os.path.abspath` | 转绝对路径 |
| `os.path.getsize` | 文件大小 |
| `os.fspath` | 把 PathLike 转成字符串 |
| `os.replace` | 原子替换（checkpoint 保存） |
| `Path.parent` | 父目录 |
| `Path.mkdir(parents=True)` | 递归建目录 |
| `Path.with_suffix(".tmp")` | 替换后缀 |

> **比喻**：`os.path` 是“手工拼地址”，`pathlib` 是“用地址对象自动拼”。

### 6.5 JSON

```python
import json

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump({"0": "!", "1": '"'}, f, indent=4)

with open("vocab.json", encoding="utf-8") as f:
    obj = json.load(f)
```

- JSON 的 key 一定是字符串；
- `json.load` 读出来是 dict/list/str/int/float/bool/None；
- `indent=4` 只是让文件好看。

教程场景：BPE 词表保存为 `vocab.json`，读取时把字符串 key 转回 `int`。

> **比喻**：JSON 是“跨语言的打包盒”，任何语言都能打开；但盒子上的标签只能是字符串。

### 6.6 `argparse`：命令行参数

```python
import argparse

p = argparse.ArgumentParser()
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--no_wandb", action="store_true")
p.add_argument("--norm_mode", choices=["pre", "post"], default="pre")
p.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])
args = p.parse_args()

print(args.batch_size, args.no_wandb)
```

| 参数 | 作用 |
| --- | --- |
| `type=int` | 自动转整数 |
| `default=...` | 默认值 |
| `required=True` | 必填 |
| `action="store_true"` | 出现就是 True |
| `choices=[...]` | 限制可选值 |
| `nargs="*"` | 接收 0 个或多个值 |
| `vars(args)` | 把 Namespace 转成 dict |

教程场景：`main_train.py`、`prepare_data.py`、`sample.py` 的入口。

### 6.7 `sys.stdout`、`io.BytesIO` 与 `_Tee`

```python
import io
import sys

buffer = io.BytesIO()
torch.save({"x": 1}, buffer)
buffer.seek(0)
obj = torch.load(buffer)
```

`_Tee` 把 `print` 同时写到终端和日志：

```python
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)
        return len(data)

    def flush(self):
        for st in self.streams:
            st.flush()
```

`print` 不关心对象是不是“真正的文件”，只要有 `write` 方法即可，这叫**鸭子类型**：

> **比喻**：鸭子类型是“只要会嘎嘎叫，就当你是一只鸭子”。`print` 只要求对象能 `write`，不要求它继承某个文件类。

### 6.8 `copy.deepcopy`

```python
import copy

snapshot = copy.deepcopy(model.state_dict())
```

- 浅拷贝：外层新对象，内层共享；
- 深拷贝：递归复制，内层也独立。

教程场景：保存历史模型状态、避免后续修改影响快照。

### 6.9 正则表达式：`regex` 模块

项目用第三方 `regex`，不是标准库 `re`，因为它支持 `\p{L}`（任意 Unicode 字母）：

```python
import regex as re

pat = re.compile(r"\p{L}+")
print(pat.findall("hello 你好 123"))
```

常用 API：

| API | 作用 |
| --- | --- |
| `re.compile(pattern)` | 编译正则，重复使用 |
| `findall(text)` | 找出所有匹配的字符串 |
| `finditer(text)` | 返回匹配对象，可拿 `start/end/group` |
| `split(pattern, text)` | 按模式切分 |
| `search(pattern, text)` | 找第一个匹配 |
| `re.escape(token)` | 把 token 里的正则特殊字符转义 |

教程场景：

- GPT-2 预分词正则；
- 特殊 token 的正则匹配与切分；
- `re.split(f"({special_regex})", text)` 保留分隔符。

> **比喻**：正则是一张“文字捕鱼网”，网眼形状由模式决定；`findall` 是把网到的鱼全倒出来，`finditer` 是一条一条拿，还能知道鱼在哪个位置。

### 6.10 `math`、`time`、随机数

```python
import math
import time

print(math.sqrt(16))   # 4.0
print(math.cos(math.pi))  # -1.0

start = time.time()
...
elapsed = time.time() - start
```

- `math.sqrt`：标量平方根；
- `math.cos/pi`：余弦调度；
- `time.time()`：墙上时钟秒数；
- `torch.cuda.synchronize()`：等 GPU 真正算完再计时。

随机数有三套常用入口：

```python
import random
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

教程里：

- `torch.manual_seed` 管模型初始化和 torch 采样；
- `np.random.seed` 管 DataLoader 的随机起点；
- 两者独立，要分别设。

> **比喻**：三套随机数像三台不同的摇奖机，各有各的种子；只设一台，另一台仍会变。

### 6.11 类型标注、`jaxtyping` 与 `einops`

类型标注是“给变量贴标签”，帮助读代码的人和编辑器，不是运行时强制检查：

```python
def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    ...

def get_batch(dataset, batch_size: int, context_length: int, device: str):
    ...

ids: list[int] = [1, 2, 3]
table: dict[str, torch.Tensor] = model.state_dict()
```

常见写法：

| 写法 | 含义 |
| --- | --- |
| `x: int` | 参数是整数 |
| `-> torch.Tensor` | 返回值是 Tensor |
| `list[int]` | 整数列表（Python 3.9+） |
| `dict[str, Tensor]` | 字符串到张量的字典 |
| `tuple[int, int]` | 二元组 |
| `int \| None` | 整数或 None（Python 3.10+） |
| `Iterable[str]` | 任何能 for 的字符串序列 |
| `BinaryIO` | 二进制文件对象 |

> 你可能会在旧代码里看到 `from typing import List` + `List[int]`。这是 Python 3.9 之前的写法；现代写法直接用 `list[int]`。测试文件开头的 `from __future__ import annotations` 只是把类型注解延迟求值，不影响运行逻辑。

`jaxtyping` 在测试/适配器里给出更细的形状提示：

```python
from jaxtyping import Float, Bool
from torch import Tensor

def run_sdpa(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    ...
```

这里的 `" ... queries d_k"` 是**字符串标签**，不是运行时的形状检查；它只是让人和类型检查器一眼看出维度含义。

`einops.rearrange` 用一行字符串重排维度：

```python
from einops import rearrange

# (batch*head, seq, d) -> (batch, head, seq, d)
q = rearrange(q, "(batch head) seq d -> batch head seq d", head=2)
```

- 左边是输入维度名，右边是输出维度名；
- `(batch head)` 表示这两个维度被合并在一起；
- `head=2` 告诉 einops 按 2 个头拆开。

> **比喻**：`rearrange` 是“带标签的搬家”：你不是说“把第 1 维挪到第 0 维”，而是说“把 batch 和 head 分开”，可读性更高。

### 6.12 测试：`pytest`

```python
def test_add():
    assert 1 + 2 == 3

def test_raises():
    import pytest
    with pytest.raises(ValueError):
        int("abc")
```

常见概念：

| 概念 | 作用 |
| --- | --- |
| `test_` 开头的函数 | pytest 自动发现 |
| `assert` | 断言 |
| fixture | 测试前准备数据（如 `q`、`k`、`v`） |
| `tmp_path` | 临时目录 |
| `pytest.raises` | 断言会抛异常 |
| `torch.allclose` | 浮点近似比较 |
| `numpy.testing.assert_allclose` | NumPy 近似比较 |

教程场景：`tests/adapters.py` 把作业实现接到测试上；测试用 `jaxtyping` 标注形状，用 `einops.rearrange` 改形状。

> **比喻**：测试是“出厂质检”：每个零件都要过一遍卡尺，不能只看最后组装起来像不像。

---

## 7. NumPy：数组、dtype、memmap

### 7.1 为什么先学 NumPy

PyTorch 负责模型计算，NumPy 负责数据准备：

- 读取 token 文件；
- 随机切窗口；
- 保存/加载 `.npy`；
- 内存映射大文件。

> **比喻**：NumPy 是“仓库管理员”，PyTorch 是“加工车间”。仓库先把原料按形状摆好，车间再加工。

### 7.2 `ndarray`、`shape`、`dtype`

```python
import numpy as np

a = np.array([1, 2, 3])
print(a.shape)   # (3,)
print(a.dtype)   # int64
print(a.size)    # 3
```

- `shape`：每个维度多大；
- `dtype`：每个元素怎么存；
- `size`：元素总数；
- `itemsize`：每个元素字节数。

> **比喻**：`shape` 是“鸡蛋托盘的格子布局”，`dtype` 是“每个格子放什么尺寸的蛋”。

### 7.3 创建数组

```python
np.array([1, 2, 3])
np.arange(0, 10)          # 0..9
np.zeros((2, 3))
np.ones((2, 3))
np.full((2, 2), 7)
np.random.randint(0, 10, size=(2, 3))
```

教程场景：

```python
data = np.arange(0, 100)
starts = np.random.randint(0, num_starts, size=batch_size)
```

### 7.4 索引、切片、布尔索引

```python
a = np.arange(10)

a[0]        # 0
a[-1]       # 9
a[2:5]      # [2,3,4]
a[::2]      # [0,2,4,6,8]
a[a % 2 == 0]  # 布尔索引
```

二维：

```python
a = np.arange(12).reshape(3, 4)
a[0, 1]       # 第 0 行第 1 列
a[:, 1]       # 所有行第 1 列
a[1:, :2]     # 子矩阵
```

### 7.5 高级索引

```python
data = np.arange(100)
starts = np.array([2, 10, 20])
idx = starts[:, None] + np.arange(5)  # (3,5)
x = data[idx]                          # (3,5)
```

- `starts[:, None]` 把 `(3,)` 变成 `(3,1)`；
- 加 `(5,)` 广播成 `(3,5)`；
- `data[idx]` 按矩阵每个位置取值，返回 `(3,5)`。

> **比喻**：高级索引是“拿一张座位表，把所有指定座位上的东西一次取出来”，不是切连续的一段。

### 7.6 `stack`、`concatenate`

```python
a = np.array([1, 2])
b = np.array([3, 4])

np.stack([a, b])          # [[1,2],[3,4]]，形状 (2,2)
np.concatenate([a, b])    # [1,2,3,4]，形状 (4,)
```

- `stack`：新加一个维度；
- `concatenate`：沿已有维度接长。

> **比喻**：`stack` 是“把两张纸叠起来”，`concatenate` 是“把两张纸首尾粘成一张长纸”。

### 7.7 `dtype` 与 `astype`

```python
a = np.array([1, 2, 3], dtype=np.int64)
b = a.astype(np.uint16)       # 转换 dtype，通常会产生新数组
c = np.asarray(a)             # 已经是 ndarray 时通常不复制
d = np.asarray([1, 2, 3])     # 从 list 转成 ndarray
```

常见 dtype：

| dtype | 字节 | 范围/用途 |
| --- | ---: | --- |
| `np.uint16` | 2 | 0~65535，token ID 常用 |
| `np.int32` | 4 | 整数 |
| `np.int64` | 8 | 默认整数，PyTorch 索引常用 |
| `np.float32` | 4 | 浮点 |
| `np.bool_` | 1 | 布尔 |

> **注意**：`np.uint16` 最大 65535；如果你有 10 万词表，要用 `uint32`。

### 7.8 `np.memmap`：把大文件当数组用

```python
data = np.memmap("train.bin", dtype=np.uint16, mode="r")
print(len(data))
print(data[:10])
```

- 文件是裸二进制，没有头部；
- `dtype` 必须和写文件时一致；
- 只在你访问时按页加载，不会一次性读进内存；
- `mode="r"` 只读。

如果文件是 `.npy`：

```python
data = np.load("train.npy", mmap_mode="r")
```

> **注意**：`.npy` 有头部，不能直接用 `np.memmap` 按 token 读，否则会把头部当数据。

> **比喻**：`memmap` 是“图书馆借书证”：不把整座图书馆搬回家，需要哪一页就借哪一页。

### 7.9 保存与加载

```python
np.save("tokens.npy", arr)
arr = np.load("tokens.npy")

arr.tofile("tokens.bin")                 # 裸二进制
arr = np.fromfile("tokens.bin", dtype=np.uint16)
```

- `np.save` 带 `.npy` 头；
- `tofile` 写裸数据；
- 读裸数据要指定 dtype。

### 7.10 NumPy 与 PyTorch 互转

```python
arr = np.arange(6, dtype=np.int64)

t1 = torch.from_numpy(arr)   # 共享内存（CPU）
t2 = torch.tensor(arr)       # 复制
back = t1.numpy()            # 共享内存（CPU）
```

- `from_numpy` 不复制，改一个会影响另一个；
- `torch.tensor` 复制，彼此独立；
- `t.numpy()` 只适用于 CPU、不需要梯度的张量；GPU 张量要先 `.detach().cpu()`。

> **比喻**：`from_numpy` 是“给同一块白板装两个边框”，`torch.tensor` 是“把白板拍照复印一份”。

### 7.11 NumPy 常见坑

| 坑 | 后果 | 修正 |
| --- | --- | --- |
| dtype 不匹配 | token 变乱码/长度不对 | 保存和读取用同一 dtype |
| `.npy` 用 `np.memmap` | 把头部当数据 | 用 `np.load(..., mmap_mode="r")` |
| 切片是视图 | 改切片影响原数组 | 需要独立副本时 `.copy()` |
| 高级索引返回副本 | 改结果不影响原数组 | 明确是否需要写回 |
| 广播形状错 | 结果形状意外 | 打印 `shape` 检查 |
| `np.random.randint` 高位不含 | 少一个起点 | 记住 `high` 不包含 |

---

## 8. PyTorch 张量：形状、索引、广播、autograd

### 8.1 Tensor 是什么

Tensor 是“多维数组 + 设备 + 梯度”：

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)    # torch.Size([2, 2])
print(x.dtype)    # torch.float32
print(x.device)   # cpu
```

**类比**：Tensor 像一张多维 Excel 表：

- 行、列、页……每个方向都是一个维度；
- `dtype` 是单元格类型；
- `device` 是这张表放在哪张桌子上（CPU/GPU）。

`torch.Tensor` 是**类型**，`torch.tensor(...)` 是**创建函数**。日常写 `torch.tensor(...)` 就好；`torch.Tensor(...)` 也能创建，但语义和默认 dtype 更容易混淆，初学者先避开。

### 8.2 创建 Tensor

```python
torch.tensor([1, 2, 3])          # 从 Python 列表，复制数据
torch.as_tensor([1, 2, 3])       # 尽量复用已有数据
torch.zeros(2, 3)
torch.ones(2, 3)
torch.empty(2, 3)                # 不初始化，内容随机
torch.randn(2, 3)                # 标准正态
torch.rand(2, 3)                 # [0,1) 均匀
torch.randint(0, 10, (2, 3))     # 整数随机
torch.arange(0, 10)              # 0..9
torch.zeros_like(x)              # 形状同 x
torch.empty_like(x)
torch.from_numpy(arr)            # 共享 NumPy 内存
```

> **复制还是共享**：`torch.tensor(...)` 通常复制；`torch.as_tensor(...)` 在类型兼容时尽量复用；`torch.from_numpy(arr)` 与 NumPy 数组共享内存。改一个可能影响另一个，调试时要留意。

教程场景：

- `nn.Parameter(torch.empty(...))`；
- `torch.zeros_like(p.data)` 初始化 AdamW 的 `m/v`；
- `torch.arange(seq_len)` 生成位置编号；
- `torch.tril(torch.ones(T, T, dtype=torch.bool))` 生成因果 mask。

### 8.3 dtype：整数、浮点、布尔

| dtype | 用途 | 例子 |
| --- | --- | --- |
| `torch.float32` | 默认浮点，模型权重 | `torch.randn` |
| `torch.float16` / `bfloat16` | 混合精度 | 大模型训练 |
| `torch.int64` / `torch.long` | token ID、索引 | `nn.Embedding` 通常要求 |
| `torch.int32` | 整数 | 较少 |
| `torch.bool` | mask | `torch.tril(..., dtype=torch.bool)` |

```python
x = torch.tensor([1, 2, 3])          # int64
y = torch.tensor([1.0, 2.0])         # float32
mask = torch.tensor([True, False])   # bool
```

> **重要**：`nn.Embedding` 的索引最稳妥用 `long`/`int64`。数据文件里常用 `np.uint16` 省空间，但转成 Tensor 后如果是 `torch.uint16`，要先 `.long()` 才能进 Embedding。

### 8.4 设备：CPU、GPU、MPS

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
model = model.to(device)
```

- 模型和输入必须在同一个 device；
- `.to()` 返回张量，通常要重新赋值：`x = x.to(device)`；
- `next(model.parameters()).device` 可以查模型所在设备；
- Apple Silicon 可以用 `"mps"`（如果 `torch.backends.mps.is_available()` 为真）；
- `torch.cuda.synchronize()` 等 GPU 真正完成，常用于计时。

> **比喻**：CPU 是书桌，GPU 是高速印刷机。模型和输入必须放在同一台机器上，不能把纸放在书桌却让印刷机去印。

### 8.5 形状、维度与 `dim`

```python
x = torch.randn(2, 3, 4)
print(x.shape)    # (2,3,4)
print(x.dim())    # 3
print(x.numel())  # 24
```

- `dim=0` 第 0 维（batch）；
- `dim=1` 第 1 维（序列）；
- `dim=-1` 最后一维（特征/词表）；
- `keepdim=True` 保留维度，方便广播。

```python
x.mean(dim=-1, keepdim=True)  # (2,3,1)
x.mean(dim=-1)                # (2,3)
```

### 8.6 索引与切片

```python
x = torch.arange(24).reshape(2, 3, 4)

x[0]          # 第 0 个 batch
x[:, -1, :]   # 每个 batch 的最后一个位置
x[..., 0::2]  # 最后一维取偶数下标
x[:, None]    # 在第二维插入一个长度为 1 的维度
```

- `...` 表示“前面所有维度”；
- `None` 在索引里表示“新增一个维度”；
- 布尔 mask：

```python
mask = torch.tensor([True, False, True])
x = torch.arange(3)
print(x[mask])   # tensor([0, 2])
```

教程场景：

- `logits[:, -1, :]` 取最后一个位置；
- `mask.unsqueeze(0).unsqueeze(0)` 变成 `(1,1,T,T)`；
- `q.view(B,T,H,D/H)` 拆头。

### 8.7 算术、矩阵乘法与广播

```python
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])

a + b       # 逐元素
a * b       # 逐元素
a @ b       # 点积，结果是标量
```

矩阵乘法：

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B          # (2,4)
D = torch.matmul(A, B)
```

**关键区分**：

| 运算符 | 含义 | 例子 |
| --- | --- | --- |
| `*` | 逐元素乘 | `gate * content` |
| `@` | 矩阵乘/点积 | `Q @ K.transpose(-2,-1)` |
| `+` | 逐元素加 | 残差 `x + f(x)` |
| `torch.outer(a,b)` | 外积，得到矩阵 | RoPE 的“位置 × 频率”表 |
| `torch.sin/cos` | 逐元素三角函数 | RoPE 旋转 |
| `torch.exp/log` | 逐元素指数/对数 | softmax、交叉熵 |
| `torch.sigmoid` | 压缩到 (0,1) | SiLU 的门控 |

> **比喻**：`*` 是“同一排货架对应格子相乘”，`@` 是“整行乘整列再求和”，`outer` 是“每个位置和每个频率两两配一张表”。

#### 广播

```python
x = torch.randn(2, 3, 4)
bias = torch.randn(4)
y = x + bias          # (2,3,4) + (4,) -> (2,3,4)
```

广播规则：

1. 从右往左对齐维度；
2. 维度相等，或其中一个为 1，或其中一个缺失；
3. 为 1 的维度会被“拉伸”。

```python
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))  # (T,T)
mask = mask.unsqueeze(0).unsqueeze(0)                  # (1,1,T,T)
scores = scores.masked_fill(~mask, float("-inf"))      # 广播到 (B,H,T,T)
```

> **比喻**：广播是“把一张小镂空模板盖到一张大桌子上”，小模板会自动铺开，不需要真的复印 B×H 份。

### 8.8 常见形状操作

| 操作 | 作用 | 类比 |
| --- | --- | --- |
| `view(...)` | 重新解释形状，要求内存连续 | 重新贴座位号，不搬椅子 |
| `reshape(...)` | 重新塑形，必要时复制 | 重新摆桌椅，可能换房间 |
| `transpose(d1,d2)` | 交换两个维度 | 把行和列对调 |
| `permute(...)` | 一次重排多个维度 | 把多个书架整体换顺序 |
| `unsqueeze(dim)` | 插入长度为 1 的维度 | 加一层隔板 |
| `squeeze(dim)` | 删除长度为 1 的维度 | 拆掉空隔板 |
| `flatten(start,end)` | 把连续维度压平 | 把几层抽屉并成一排 |
| `contiguous()` | 复制成连续内存 | 把书按顺序重新摆好 |
| `cat([...], dim)` | 沿已有维度拼接 | 火车车厢接长 |
| `stack([...], dim)` | 新维度堆叠 | 把纸叠起来 |
| `split(size, dim)` | 按长度切分 | 一次 QKV 投影拆成三份 |
| `.T` | 二维转置（高维语义不同） | 把一张纸翻面 |

> 二维时 `K.T` 和 `K.transpose(-2, -1)` 等价；三维/四维时 `.T` 会反转所有维度，语义容易混淆，教程统一用 `transpose(-2, -1)` 交换最后两维。

> 形状里的 `-1` 表示“这一维让 PyTorch 自己算”。例如 `x.reshape(-1, V)` 会把前面所有维度压平，最后一维固定为 `V`。

教程里的经典组合：

```python
q = q.view(B, T, H, D // H)   # (B,T,D) -> (B,T,H,Dh)
q = q.transpose(1, 2)         # (B,H,T,Dh)
...
out = out.transpose(1, 2)     # (B,T,H,Dh)
out = out.contiguous().view(B, T, D)
```

> **为什么 `contiguous()`**：`transpose` 只改“怎么看”，不搬数据；此时直接 `view` 可能因为内存不连续而报错。`contiguous()` 先复制成连续块，再 `view` 就安全。

### 8.9 归约与统计

```python
x = torch.randn(2, 3, 4)

x.sum()                 # 所有元素和
x.sum(dim=-1)           # (2,3)
x.sum(dim=-1, keepdim=True)  # (2,3,1)
x.mean(dim=-1)
x.max(dim=-1)           # 返回值 + 索引
x.argmax(dim=-1)        # 最大值的下标
x.pow(2)                # 逐元素平方
torch.rsqrt(x)          # 1 / sqrt(x)，RMSNorm 用
```

教程场景：

- RMSNorm：`x.pow(2).mean(-1, keepdim=True)`；
- 梯度裁剪：`torch.norm(all_grads, 2)`；
- 参数量：`sum(p.numel() for p in model.parameters())`。

### 8.10 `softmax` 与 `log_softmax`

```python
logits = torch.tensor([[1.0, 2.0, 3.0]])
probs = torch.softmax(logits, dim=-1)
log_probs = torch.log_softmax(logits, dim=-1)
```

- `softmax`：变成概率，非负、和为 1；
- `log_softmax`：在对数域归一化，数值更稳定；
- 交叉熵内部用 `log_softmax`，所以不要把 softmax 后的概率再传给它。

> **比喻**：`logits` 是“原始投票数”，`softmax` 是“换算成百分比”，`log_softmax` 是“用对数记账，防止大数溢出”。

### 8.11 mask、`masked_fill` 与 `tril`

```python
T = 4
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
print(causal)

scores = torch.randn(1, T, T)
scores = scores.masked_fill(~causal, float("-inf"))
weights = scores.softmax(dim=-1)
```

- `tril`：下三角（含对角线）；
- `masked_fill(condition, value)`：条件为 True 的位置填 value；
- `~causal` 把“允许”变成“屏蔽”；
- softmax 之前填 `-inf`，被屏蔽位置权重精确为 0。

> **比喻**：mask 是“考场座位表”，`True` 是允许看，`False` 是隔板；隔板必须在“算比例”之前放，不能算完比例再遮。

### 8.12 `sort`、`cumsum`、`multinomial`、`gather`

Top-p 采样的核心：

```python
probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])

sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
cumulative = torch.cumsum(sorted_probs, dim=-1)
keep = (cumulative - sorted_probs) < 0.9
filtered = sorted_probs.masked_fill(~keep, 0.0)
normalized = filtered / filtered.sum(dim=-1, keepdim=True)
rank = torch.multinomial(normalized, 1)
token = torch.gather(idx, dim=-1, index=rank)
```

- `sort` 返回“排序后的值”和“原下标”；
- `cumsum` 累积和；
- `multinomial` 按概率抽一个“排名”；
- `gather` 用排名查回原始 token ID。

> **比喻**：`sort` 是把候选人按票数排队，`cumsum` 是看“前几名加起来够不够 90%”，`multinomial` 是在入围者里抽签，`gather` 是把“第几名”翻译回“候选人编号”。

### 8.13 `where`、`clamp`、`nan_to_num`

```python
torch.where(condition, a, b)  # 条件为真取 a，否则取 b
x.clamp(min=0, max=1)         # 截断到区间
torch.nan_to_num(x, nan=0.0)  # NaN 替换成 0
torch.allclose(a, b, atol=1e-6)  # 浮点近似比较
torch.equal(a, b)             # 逐元素完全相等
```

教程场景：

- top-p 的 fallback：`torch.where(denom > 0, ..., fallback)`；
- softmax 整行被屏蔽：`nan_to_num`；
- 概率下限：`clamp_min(1e-12)`。

### 8.14 autograd：自动求导

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x ** 2).sum()
loss.backward()
print(x.grad)   # tensor([2., 4., 6.])
```

- `requires_grad=True`：告诉 PyTorch“记录这个张量上的运算”；
- `loss.backward()`：沿计算图反向求导；
- `.grad`：梯度会**累加**，所以每轮要 `optimizer.zero_grad()`；
- `torch.no_grad()`：关闭记录，用于推理/评估；
- `detach()`：把张量从计算图里摘出来。

> **比喻**：autograd 是“全程记账”。前向每做一步，就在小票上记一笔；`backward` 是沿着小票倒着算“这一步对最终结果贡献多少”。

#### 叶子张量与原地操作

```python
x = torch.tensor([1.0], requires_grad=True)
# x += 1   # 可能报错：leaf Variable was used in an in-place operation
with torch.no_grad():
    x += 1  # 明确不记录梯度
```

优化器里常用 `p.data.mul_()` 绕过计算图。

### 8.15 `no_grad`、`detach`、`item`

```python
with torch.no_grad():
    y = model(x)

loss_value = loss.item()          # 0 维张量 -> Python float
values = tensor.tolist()          # 张量 -> 嵌套 Python 列表
arr = tensor.detach().cpu().numpy()
```

- `no_grad`：不建图，省显存；
- `detach`：返回共享内存但不需要梯度的张量；
- `item`：取标量值，会触发 GPU 同步，别放在训练主循环里频繁调用。

> **比喻**：`no_grad` 是“关掉监控录像”，`detach` 是“把录像带从系统里拿出来”，`item` 是“把画面上的一个数字抄到纸上”。

### 8.16 原地操作

带下划线的方法通常是原地操作：

```python
x.add_(1)
x.mul_(2)
x.div_(3)
x.clamp_min_(0)
x.fill_(0)
```

- 优点：省内存；
- 风险：可能破坏 autograd 记录，尤其是叶子变量；
- 优化器更新参数时常用 `p.data.add_/mul_`。

相关但不同的两个方法：

```python
y = x.clone()     # 复制数据，仍然连着计算图
z = x.detach()    # 共享数据，但从计算图里摘出来
```

> **比喻**：`clone` 是“复印一份，但还在同一个账本体系里”；`detach` 是“把这一页从账本上撕下来”，之后怎么改都不影响原来的梯度记录。

> **比喻**：原地操作是“直接在原文件上改”，不是“复制一份再改”。快，但改坏了原文件就没了。

---

## 9. PyTorch 模型：Module、Parameter、层、损失、优化器

### 9.1 `nn.Module`：所有模型的积木底座

```python
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
y = model(torch.randn(2, 4))
```

- 继承 `nn.Module`；
- `__init__` 里先 `super().__init__()`；
- 子模块赋值给 `self.xxx` 会自动注册；
- 实现 `forward`；
- 用 `model(x)` 调用，不要直接 `model.forward(x)`（除非你知道自己在做什么）。

**类比**：`nn.Module` 是“乐高底板”，子模块是插在上面的积木；底板负责统一管理所有积木。

### 9.2 `nn.Parameter`：可训练的参数

```python
class LinearModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features))
```

- `nn.Parameter` 是特殊 Tensor；
- 默认 `requires_grad=True`；
- 会出现在 `model.parameters()` 里；
- 优化器只会更新注册过的参数。

普通 Tensor 属性：

```python
self.W = torch.empty(...)   # 不会注册，不会被优化器更新
```

> **比喻**：`nn.Parameter` 是“带调节旋钮的零件”，优化器会转它；普通 Tensor 是“贴在墙上的装饰”，优化器看不见。

#### 参数初始化

参数不能全是 0，也不能随便取很大的数。常见做法：

```python
std = (2.0 / (in_features + out_features)) ** 0.5
nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)
```

- `torch.empty` 只分配内存，内容是垃圾值；
- `nn.init.trunc_normal_` 用截断正态分布填充；
- `std` 控制初始幅度；
- `a`/`b` 是绝对值边界，不是“几倍标准差”。

> **比喻**：初始化像“给新员工定初始工作量”：太大一上来就压垮，太小又学不到东西；Xavier/截断正态是在找一个适中的起点。

### 9.3 buffer：不训练但要跟着模型走的状态

```python
self.register_buffer("cos_cache", values, persistent=False)
```

- buffer 不是参数，不参与优化；
- 会跟着 `model.to(device)` 搬设备；
- `persistent=False` 表示不写进 `state_dict`；
- RoPE 的 `cos/sin` 位置表就是 buffer。

> **比喻**：buffer 是“模型随身带的尺子”，不训练，但换房间要一起带走。

### 9.4 常用层

#### `nn.Linear`

```python
linear = nn.Linear(in_features=4, out_features=3)
y = linear(x)   # (...,4) -> (...,3)
```

- 权重形状 `(out_features, in_features)`；
- 前向做 `x @ W.T + b`；
- `bias=False` 时不加偏置；
- 参数：`weight`、`bias`。

#### `nn.Embedding`

```python
emb = nn.Embedding(num_embeddings=10000, embedding_dim=512)
y = emb(token_ids)   # (...,T) -> (...,T,512)
```

- 本质是查表；
- 输入必须是整数；
- 输出在最后加一维 `embedding_dim`；
- 只有被取到的行会得到非零梯度；没出现的行梯度为 0（`sparse=True` 时才会用稀疏梯度表示）；
- `padding_idx=0` 表示 ID 0 是填充位，这一行输出固定为 0、梯度也为 0。

#### `nn.ModuleList` 与 `nn.Sequential`

```python
self.layers = nn.ModuleList([Block(...) for _ in range(n)])
for layer in self.layers:
    x = layer(x)

self.block = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
)
```

- `ModuleList`：只负责“注册并保存一组模块”，不自动执行；
- `Sequential`：按顺序自动执行；
- 普通 Python list 不会注册模块。

#### `nn.Identity`

```python
norm = nn.Identity()   # 什么都不做
```

用于“关掉某个模块”时占位，避免 forward 里写一堆 if。

### 9.5 `state_dict`：模型参数的行李清单

```python
state = model.state_dict()
print(state.keys())
```

- `state_dict` 是 `dict[str, Tensor]`；
- key 是参数名，如 `"layers.0.attn.q_proj.weight"`；
- 只包含参数和 persistent buffer，不包含模型结构/代码；
- `load_state_dict` 按名字把值复制进已有模型；
- 默认 `strict=True`，key/形状不匹配会报错；
- `strict=False` 会返回 `missing_keys` / `unexpected_keys`，适合只加载一部分。

```python
model2 = MyModel()
model2.load_state_dict(state)

result = model2.load_state_dict(state, strict=False)
print(result.missing_keys, result.unexpected_keys)
```

> **比喻**：`state_dict` 是“行李清单”：告诉你每个箱子里装了什么，但不告诉你房子怎么盖。加载前必须先把房子（模型结构）盖好。

### 9.6 `train()` 与 `eval()`

```python
model.train()   # 训练模式：dropout 开启，BatchNorm 更新
model.eval()    # 评估模式：dropout 关闭，BatchNorm 使用固定统计
```

- 只影响有训练/推理行为差异的层；
- 不会自动关闭梯度，推理还要 `torch.no_grad()`；
- `model.training` 可以查看当前模式。

> **比喻**：`train()` 是“练习模式”，允许试错；`eval()` 是“考试模式”，行为要稳定。

### 9.7 `torch.nn.functional`：函数式 API

```python
import torch.nn.functional as F

F.silu(x)
F.softmax(x, dim=-1)
F.log_softmax(x, dim=-1)
F.cross_entropy(logits, targets)
F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
F.dropout(x, p=0.1, training=model.training)
F.linear(x, weight, bias)
```

- `F.xxx` 是无状态函数；
- `nn.Linear` 等是有状态模块；
- 二者经常混用：模块负责保存权重，函数负责计算。

### 9.8 损失：交叉熵

```python
logits = model(x)                     # (B,T,V)
loss = F.cross_entropy(
    logits.reshape(-1, V),
    y.reshape(-1),
)
```

- `logits` 是未归一化分数；
- `y` 是整数类别 ID；
- `F.cross_entropy` 默认 `reduction="mean"`，返回批平均；
- 本仓库的 `CrossEntropyLoss` 也返回批平均；
- 不要先 softmax 再传进去。

教程场景：`main_train.py` 直接调用 `CrossEntropyLoss(...).forward()`，不再除 `B*T`。

> **比喻**：交叉熵是“惊讶程度”：模型给正确答案的概率越低，越惊讶，loss 越大。

### 9.9 优化器：`Optimizer` 与 `param_groups`

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.1,
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

- `model.parameters()` 是可迭代的参数集合（本质是生成器，遍历一次就空了）；
- 如果后面还要重复使用，先 `params = list(model.parameters())`；
- `param_groups`：一组参数 + 一组超参；
- `optimizer.state`：每个参数的动量等状态；
- `zero_grad()`：清空梯度；
- `step()`：更新参数；
- `state_dict()/load_state_dict()`：保存/恢复优化器。

> **比喻**：优化器是“下山向导”，`param_groups` 是不同小队（可以有不同的步速），`state` 是每个队员的惯性和历史。

### 9.10 自定义优化器

```python
from torch import optim


class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                ...
```

要点：

- 继承 `torch.optim.Optimizer`；
- `super().__init__(params, defaults)` 必须调用；
- 重写 `step()`；
- 参数更新要原地修改；
- 用 `p.grad` 读梯度，用 `p.data` 绕开 autograd；
- `state` 用参数对象做 key。

### 9.11 保存与加载

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "iteration": iteration,
}, "ckpt.pt")

ckpt = torch.load("ckpt.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
```

- `map_location` 处理 GPU/CPU 设备差异；
- `weights_only=False` 只用于可信文件；
- `load_state_dict` 复制到已有参数，保持 Parameter 对象不变；
- 优化器状态要和模型参数顺序一致。

> **比喻**：模型权重是“现在的位置”，优化器状态是“速度和惯性”，iteration 是“跑到第几圈”；三者一起存，才能接着跑。

### 9.12 随机种子与复现

```python
torch.manual_seed(0)
np.random.seed(0)

g = torch.Generator().manual_seed(0)
x = torch.randn(3, generator=g)
```

- `torch.manual_seed`：全局 torch RNG；
- `np.random.seed`：NumPy RNG；
- `torch.Generator`：独立的随机数生成器，适合局部复现；
- `multinomial(..., generator=g)` 可以固定采样。

### 9.13 性能相关的基础

| 做法 | 原因 |
| --- | --- |
| 尽量用批量张量运算 | GPU 擅长并行 |
| 避免 Python 循环逐 token 算 | 启动开销大 |
| 推理用 `torch.no_grad()` | 不建图，省显存 |
| `model.eval()` | 关闭 dropout |
| `torch.cuda.synchronize()` | 计时前等 GPU 完成 |
| 少用 `.item()` | 会触发同步 |
| `contiguous()` 后再 `view` | 避免非连续内存报错 |

> **比喻**：GPU 是“大卡车”，一次拉一车货最划算；一件一件送，油费都比货贵。

---

## 10. 把知识串起来：训练与推理

### 10.1 训练一个 step

```python
model.train()

x, y = loader.get_train_batch_data()   # (B,T), (B,T)
x, y = x.to(device), y.to(device)

logits = model(x)                      # (B,T,V)
V = logits.shape[-1]
loss = F.cross_entropy(
    logits.reshape(-1, V),
    y.reshape(-1),
)

optimizer.zero_grad()
loss.backward()
clip()
optimizer.step()
```

用到的基础：

- Tensor 形状 `(B,T)`、`(B,T,V)`；
- `reshape` 拉平；
- `cross_entropy`；
- autograd `backward`；
- 梯度裁剪；
- 优化器 `step`。

> 这里的 `loader` 是项目自定义的 `DataLoader`，不是 `torch.utils.data.DataLoader`。它的 `get_train_batch_data(device)` 可以直接把 batch 放到目标设备；`get_valid_batch_data_iter(device)` 是一个生成器，逐批产出验证数据。`x`、`y` 都是 `int64`，形状 `(B,T)`。

### 10.2 推理生成一个 token

```python
model.eval()

with torch.no_grad():
    logits = model(ids)                # (1,T,V)
    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
    next_id = torch.multinomial(probs, 1)
    ids = torch.cat([ids, next_id], dim=-1)
```

用到的基础：

- 索引 `[:, -1, :]`；
- 温度缩放；
- `softmax`；
- `multinomial`；
- `cat`；
- `no_grad`。

### 10.3 形状总表

| 阶段 | 形状 | 含义 |
| --- | --- | --- |
| 输入 ID | `(B,T)` | batch × 序列长度 |
| Embedding | `(B,T,D)` | 每个 token 变成 D 维 |
| Attention scores | `(B,H,T,T)` | 每个头一张注意力表 |
| Attention 输出 | `(B,H,T,D/H)` | 每个头一份结果 |
| 合并后 | `(B,T,D)` | 回到模型维度 |
| FFN 中间 | `(B,T,d_ff)` | 升维再压回 |
| logits | `(B,T,V)` | 每个位置对词表的打分 |
| loss | `()` | 标量 |

### 10.4 教程知识点串联表

| 教程中的写法 | 背后的基础 |
| --- | --- |
| `x @ W.T` | 矩阵乘法、转置 |
| `q.view(B,T,H,D/H).transpose(1,2)` | 形状重排、拆头 |
| `scores.masked_fill(~mask, -inf).softmax(-1)` | 索引、mask、softmax |
| `x.pow(2).mean(-1, keepdim=True)` | 逐元素幂、归约、广播 |
| `nn.Parameter(torch.empty(...))` | 参数注册 |
| `self.layers = nn.ModuleList([...])` | 模块容器 |
| `model.state_dict()` | 参数字典 |
| `optimizer.zero_grad(); loss.backward(); optimizer.step()` | 训练三连 |
| `torch.save/load` | 序列化 |
| `np.memmap` + `torch.from_numpy` | 大数据加载 |
| `torch.sort/cumsum/multinomial/gather` | Top-p 采样 |

---

## 11. 常见报错速查

| 报错/症状 | 常见原因 | 修正 |
| --- | --- | --- |
| `mat1 and mat2 shapes cannot be multiplied` | 矩阵维度不匹配 | 打印 `shape`，检查最后一维/转置 |
| `Expected all tensors to be on the same device` | 模型/输入不在同一设备 | `.to(device)` |
| `Expected tensor for argument #1 'indices' to have ... Long` | Embedding 输入不是 int64 | `.long()` |
| `one of the variables needed for gradient computation has been modified by an inplace operation` | 原地操作破坏计算图 | 改用非原地，或 `no_grad`/`clone` |
| `a leaf Variable that requires grad is being used in an in-place operation` | 对叶子变量原地改 | 用 `with torch.no_grad()` 或 `p.data` |
| `view size is not compatible with input tensor's size and stride` | 张量不连续 | 先 `.contiguous()`，或改用 `reshape` |
| `size mismatch for ...` | checkpoint 形状和模型不一致 | 检查模型配置/参数名 |
| `KeyError: ...` | 字典没有这个 key | 检查 tokenizer/state_dict key |
| `IndexError: index out of range` | 位置/词表越界 | 检查 `context_length`、token ID |
| `RuntimeError: probability tensor contains ... nan` | 温度/概率非法，或自定义 top-p 把整行清零 | 检查 `temperature > 0`、归一化，并保证至少保留一个候选 |
| `ModuleNotFoundError` | 包没安装/路径不对 | `uv sync`，检查 import |
| `IndentationError` | 缩进不一致 | 统一 4 空格 |
| `NameError` | 变量名拼错/未定义 | 检查作用域 |
| `TypeError: 'Tensor' object is not callable` | 把 Tensor 当函数/模型 | 检查变量名是否覆盖了 model |

---

## 12. 学习路线与练习

### 12.1 三遍阅读法

**第一遍：建立地图**

- 读 §0～§2；
- 知道变量、容器、切片、字典是什么；
- 不要求记住所有 API。

**第二遍：配合教程**

- 读 `Embedding_Linear.md` 时查 §9.4；
- 读 `Attention.md` 时查 §8.7～§8.10；
- 读 `Data Loader.md` 时查 §7；
- 读 `CheckPoint.md` 时查 §9.5、§9.11。

**第三遍：做小实验**

- 改形状、打印 `shape`；
- 改 dtype，看报错；
- 改 mask，看权重；
- 改 batch size，看显存/速度。

### 12.2 练习清单

1. 用 `list` 存 10 个 token ID，切片取前 3 个和后 2 个。
2. 用 `dict` 统计一段文本里每个字符出现的次数。
3. 用 `Counter` 完成上一题，比较写法。
4. 写一个函数，接受 `(B,T,D)` 的张量，返回 `(B,T)` 的均值。
5. 用 `torch.randn(2,3,4)` 练习 `view`、`transpose`、`unsqueeze`、`flatten`，打印每一步形状。
6. 用 `torch.tril` 构造 4×4 因果 mask，打印它。
7. 用 `requires_grad=True` 计算 `y=(x**2).sum()`，打印 `x.grad`。
8. 写一个 `nn.Module`，包含一个 `nn.Linear`，打印 `state_dict` 的 key。
9. 用 `torch.save/load` 保存这个模型的 `state_dict`，加载到另一个同结构模型。
10. 用 `np.memmap` 读一个自己写的 `uint16` 文件，取前 10 个 token。

### 12.3 自测问题

- `list` 和 `tuple` 的区别是什么？
- `dict` 的 key 为什么不能是 `list`？
- `a[1:4]` 包含下标 4 吗？
- `*args` 和 `**kwargs` 的区别？
- `yield` 和 `return` 的区别？
- `__init__` 和 `__call__` 的区别？
- `view` 和 `reshape` 的区别？
- `transpose` 后为什么有时要 `contiguous()`？
- `softmax(dim=-1)` 的 `-1` 是哪一维？
- `loss.backward()` 前为什么要 `zero_grad()`？
- `model.eval()` 和 `torch.no_grad()` 分别做什么？
- `state_dict` 里有什么、没有什么？
- `np.memmap` 为什么能省内存？
- `torch.from_numpy` 和 `torch.tensor` 的区别？

---

## 13. 一页速查表

### 13.1 Python 语法

```text
变量:        x = 3
类型:        int / float / bool / str / bytes / None
除法:        / 真除   // 整除   % 取余   ** 幂
字符串:      f"{x:.2f}"   encode/decode
列表:        a[i]  a[1:3]  append  extend
元组:        (a, b)  不可变，可当 dict key
字典:        d[key]  d.get(k, default)  d.items()
集合:        set()  去重
切片:        a[start:stop:step]  不含 stop
推导式:      [f(x) for x in xs if cond]
条件:        if/elif/else   三元 a if cond else b
循环:        for x in xs   while cond   break/continue
函数:        def f(x, y=1): return ...
可变参数:    def f(*args)
lambda:      lambda x: x[1]
类:          class A: def __init__(self): ...
继承:        class B(A): super().__init__()
异常:        try/except/finally  raise  assert
文件:        with open(path, encoding="utf-8") as f:
生成器:      yield / yield from
```

### 13.2 PyTorch 张量

```text
创建:        torch.tensor / zeros / ones / empty / randn / arange
dtype:       float32 / int64(long) / bool
设备:        x.to(device)   next(model.parameters()).device
形状:        x.shape  x.dim()  x.numel()
变形:        view / reshape / transpose / permute / unsqueeze / flatten
连续:        x.contiguous()
拼接:        torch.cat([a,b], dim)   torch.stack([a,b], dim)
索引:        x[:, -1, :]   x[..., 0::2]   x[mask]
矩阵乘:      a @ b   torch.matmul(a,b)
归约:        sum / mean / max / argmax   dim, keepdim
softmax:     torch.softmax(x, dim=-1)
mask:        masked_fill(~mask, -inf)   torch.tril
采样:        sort / cumsum / multinomial / gather
求导:        requires_grad=True   loss.backward()   x.grad
清梯度:      optimizer.zero_grad()
关闭梯度:    with torch.no_grad():
```

### 13.3 模型与训练

```text
模块:        class M(nn.Module): super().__init__(); forward()
参数:        nn.Parameter(...)
容器:        nn.ModuleList / nn.Sequential / nn.Identity
层:          nn.Linear / nn.Embedding
状态:        model.state_dict() / load_state_dict()
模式:        model.train() / model.eval()
损失:        F.cross_entropy(logits.reshape(-1,V), y.reshape(-1))
优化器:      optimizer.zero_grad(); loss.backward(); optimizer.step()
保存:        torch.save({...}, path)
加载:        torch.load(path, map_location="cpu", weights_only=False)
```

### 13.4 一句话总结

> **Python 负责“把数据和流程组织好”，NumPy 负责“把原始数据摆成数组”，PyTorch 负责“把数组变成可训练的张量”。三者合起来，才是教程里从 token 到 loss、从 logits 到生成的一整条流水线。**

---

## 附：和教程的对应关系

- `BPE算法原理.md`：重点看 §1.6、§2、§4.9、§6.9；
- `Embedding_Linear.md`：重点看 §5、§8.6、§9.4、§9.5；
- `Attention.md` / `缩放点积注意力.md`：重点看 §8.6～§8.11；
- `Causal_Multi-Head_self-attention.md`：重点看 §8.6、§8.8、§9.4；
- `RMSNorm.md` / `SwiGLU.md`：重点看 §8.9、§9.4；
- `Transformer_Block.md` / `完整语言模型架构的组装.md`：重点看 §5、§9；
- `Cross_Entropy_Loss.md`：重点看 §8.10、§9.8；
- `AdamW.md` / `Gradient Clipping.md` / `学习率调度：Cosine Schedule.md`：重点看 §9.9、§9.10、§8.14；
- `Data Loader.md`：重点看 §7；
- `CheckPoint.md`：重点看 §9.5、§9.11；
- `端到端训练和模型推理.md`：重点看 §10。
