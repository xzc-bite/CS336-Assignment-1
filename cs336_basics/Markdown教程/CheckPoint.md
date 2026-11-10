# CheckPoint：把训练进度“存档”和“读档”

> 面向新手。本文以 `cs336_basics/CheckPoint.py` 为主线，结合 CS336 Assignment 1 的 §5.2 与 `tests/test_serialization.py`，讲清楚：
>
> 1. 为什么训练到一半要保存 checkpoint；
> 2. checkpoint 里到底装了什么；
> 3. `state_dict` 是什么，为什么不能只保存模型权重；
> 4. 保存和加载的完整流程；
> 5. 怎样在 CS336 的作业里通过 `test_checkpointing`。
>
> 你可以把 checkpoint 想成游戏的“存档”：关掉游戏、电脑重启、任务超时，下次都能从上次的进度继续，而不是从第一关重新打。

---

## 目录

| 章节 | 内容 |
| --- | --- |
| §1 | 为什么需要 CheckPoint |
| §2 | 一个 CheckPoint 里有什么 |
| §3 | `state_dict`：模型和优化器的“行李清单” |
| §4 | 保存逻辑：把状态装进文件 |
| §5 | 加载逻辑：先搭空模型，再读档 |
| §6 | 完整可运行案例 |
| §7 | 恢复训练到底需要哪些信息 |
| §8 | 联系 CS336 Assignment 1 |
| §9 | 常见坑与 FAQ |
| §10 | 自测清单与一页速查 |

---

## 1. 为什么需要 CheckPoint

### 1.1 训练不是一瞬间完成的

训练一个语言模型，可能要跑几小时、几天，甚至几周。这期间会发生很多意外：

- 云服务器被抢占，任务突然终止；
- 训练脚本超时，被调度器杀掉；
- 机器断电、磁盘故障；
- 你不小心按了 `Ctrl+C`；
- 你想比较第 1000 步、第 5000 步、第 10000 步的模型效果；
- 你想在训练中途先拿模型生成几句话看看。

如果没有 checkpoint，上述任何一种情况发生，你都只能**从第 0 步重新开始**。

### 1.2 CheckPoint 就是训练的“游戏存档”

打游戏时，你会在 boss 战前存一个档。即使打输了，也能从存档点重来，而不是从游戏开头重来。

训练里的 checkpoint 也一样：

```text
训练到第 1000 步
  -> 保存 checkpoint
训练到第 1500 步
  -> 机器挂了
重启后
  -> 加载第 1000 步的 checkpoint
  -> 从第 1000 步继续跑
```

> checkpoint 不是“最终模型文件”的同义词。它更像是“训练现场的快照”：除了模型权重，还包括优化器状态和训练进度。

### 1.3 为什么不能只保存模型权重

假设你只保存了模型权重，恢复时会发生什么？

| 恢复内容 | 只保存权重时 | 后果 |
| --- | --- | --- |
| 模型参数 | 有 | 模型能加载 |
| AdamW 的一阶、二阶动量 | 没有 | 优化器“失忆”，前几步更新会抖动 |
| 优化器的 `step` 计数 | 没有 | 偏差修正从头开始，更新幅度异常 |
| 训练到第几步 | 没有 | 学习率 schedule 从 0 开始，进度错乱 |
| 数据加载位置/RNG | 没有 | 数据顺序变了，难以精确复现 |

对新手来说，最直观的类比是：

> **模型权重 = 你现在的位置；优化器状态 = 你的速度和惯性；iteration = 你跑到第几圈。**
>
> 只保存位置，不保存速度和圈数，恢复后虽然“人还在原地”，但运动状态已经不对了。

### 1.4 CheckPoint 的其他用途

除了“防止训练白跑”，checkpoint 还常用于：

- **模型选择**：保存验证集效果最好的模型；
- **中间采样**：观察模型在不同训练阶段的能力变化；
- **消融实验**：从同一个 checkpoint 出发，尝试不同学习率或数据；
- **部署**：训练结束后，把权重交给推理服务；
- **调试**：复现某个 step 的 loss 或梯度异常。

---

## 2. 一个 CheckPoint 里有什么

### 2.1 最小三件套

CS336 作业要求保存三样东西：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "iteration": iteration,
}
```

| 内容 | 是什么 | 为什么需要 |
| --- | --- | --- |
| `model_state_dict` | 模型所有参数和 buffer 的字典 | 恢复模型权重 |
| `optimizer_state_dict` | 优化器状态和参数组 | 恢复 AdamW 的动量、步数、学习率等 |
| `iteration` | 已经完成的训练步数 | 恢复学习率 schedule、日志和训练进度 |

> 这三样合起来，才足以“继续训练”。如果只是拿模型做推理，通常只需要第一样。

### 2.2 三件套分别长什么样

#### 模型 state_dict

```python
for name, tensor in model.state_dict().items():
    print(name, tuple(tensor.shape))
```

一个简单的 `nn.Linear(4, 3)` 可能输出：

```text
weight (3, 4)
bias (3,)
```

一个 Transformer 可能输出（本作业 vocab=10000、d_model=512；GPT-2 原版的 50257 不要照搬）：

```text
token_embeddings.weight (10000, 512)
layers.0.attn.q_proj.weight (512, 512)
layers.0.attn.k_proj.weight (512, 512)
...
ln_final.weight (512,)
```

它只包含**张量值**，不包含“这个模型有几层、每层用什么结构”。所以加载前必须先用代码把模型结构搭出来。

#### 优化器 state_dict

AdamW 的 state_dict 大致长这样：

```python
{
    "state": {
        0: {
            "step": 10,
            "m": tensor(...),   # 一阶动量
            "v": tensor(...),   # 二阶矩
        },
        1: {...},
        ...
    },
    "param_groups": [
        {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "params": [0, 1, 2, ...],
        }
    ],
}
```

几个关键点：

- `state` 的 key 是参数的整数索引，不是参数名；
- 每个参数通常有 `step`、`m`、`v`；
- `param_groups` 保存学习率、betas、eps、weight decay 等超参数；
- 参数没有梯度时，可能没有对应的 state。

> 不同的优化器 state_dict 结构不同。这里讲的是 AdamW；SGD 可能只有 `momentum_buffer`，没有 `m/v`。

#### iteration

`iteration` 是一个整数，表示“已经完成了多少步”。

它的用途：

- 计算当前学习率：`lr = schedule(iteration)`；
- 记录日志：第几个 step 的 loss；
- 命名 checkpoint：`checkpoint_5000.pt`；
- 控制训练结束：`for it in range(start_iter, max_iters)`。

> `iteration` 不是 epoch。语言模型预训练通常按 step 计数，不按“把整个数据集遍历几遍”计数。

### 2.3 作业最小版 vs 生产环境完整版

CS336 只要求三件套，但真实训练通常还会保存更多东西：

| 内容 | 作业需要 | 生产环境 | 作用 |
| --- | --- | --- | --- |
| 模型权重 | 是 | 是 | 恢复模型 |
| 优化器状态 | 是 | 是 | 恢复动量、步数 |
| iteration | 是 | 是 | 恢复 schedule |
| 模型配置 | 间接需要 | 是 | 重建模型结构 |
| tokenizer/vocab | 否 | 是 | 推理/继续 tokenize |
| 学习率 scheduler 状态 | 否（纯函数） | 视情况 | 恢复内部状态 |
| RNG 状态 | 否 | 精确复现时需要 | 数据顺序、dropout |
| DataLoader 位置 | 否 | 精确续训时需要 | 从同一数据位置继续 |
| GradScaler 状态 | 否 | 混合精度训练需要 | 恢复 loss scaling |
| 最佳验证指标 | 否 | 是 | 模型选择 |
| EMA 权重 | 否 | 视情况 | 恢复指数滑动平均 |

> 对 CS336 来说，先把三件套做对；等你要做“完全可复现”的实验时，再逐步加上 RNG、数据位置等。

### 2.4 恢复推理 vs 恢复训练

这两个场景需要的东西不一样：

| 场景 | 需要的信息 |
| --- | --- |
| **只想拿模型推理** | 模型结构/配置 + 模型权重 + tokenizer |
| **想继续训练** | 模型权重 + 优化器状态 + iteration + 模型配置 + 数据/RNG（可选） |

所以：

- 发布模型时，通常只发 `model_state_dict`；
- 训练中断恢复时，必须加载完整 checkpoint；
- 不要拿“推理用的权重文件”去恢复训练，否则优化器会从零开始。

---

## 3. `state_dict`：模型和优化器的“行李清单”

### 3.1 一句话定义

`state_dict` 是一个 Python 字典（严格说是 `OrderedDict`），把**名字**映射到**张量**：

```python
{
    "layer1.weight": tensor(...),
    "layer1.bias": tensor(...),
    ...
}
```

它描述的是“这个对象当前有哪些状态值”，不是“这个对象是什么类”。

### 3.2 模型 state_dict 不包含什么

| 不包含 | 例子 | 为什么要另外保存/重建 |
| --- | --- | --- |
| 模型结构 | 有几层、每层维度 | `state_dict` 只有张量，无法知道怎么搭网络 |
| forward 代码 | `forward()` 里做了什么 | 代码在 `.py` 文件里，不在权重里 |
| 超参数 | `d_model`、`n_heads` | 需要配置文件另行保存 |
| 优化器状态 | AdamW 的 `m/v` | 在 `optimizer.state_dict()` 里 |
| 训练进度 | 第几步 | 需要单独存 `iteration` |

所以加载模型的标准姿势是：

```python
# 1. 先用代码搭一个“空模型”
model = TransformerLM(config)

# 2. 再把 state_dict 填进去
model.load_state_dict(checkpoint["model_state_dict"])
```

不能反过来：

```python
# 错误：state_dict 不是模型对象，不能直接 forward
model = checkpoint["model_state_dict"]
model(x)  # 报错
```

### 3.3 `state_dict` 里有什么：参数和 buffer

`nn.Module` 里有两类会被保存的东西：

| 类型 | 是否参与训练 | 例子 |
| --- | --- | --- |
| `Parameter` | 是，需要梯度 | Linear 的 weight/bias |
| `buffer` | 否，但属于模型状态 | BatchNorm 的 running_mean/var |

`state_dict()` 会把它们都收进去。对 Transformer 来说，常见的 buffer 可能包括位置编码缓存等；本作业的 RMSNorm 没有 running stats，所以主要是参数。

### 3.4 `load_state_dict` 做了什么

`load_state_dict` 不是“把模型替换掉”，而是：

> 按照名字，把字典里的张量值**复制进**当前模型已经存在的参数里。

所以：

- 模型必须先存在；
- 参数名要能对上；
- 形状要能对上；
- 默认 `strict=True`，缺 key 或多 key 都会报错。

```python
result = model.load_state_dict(state_dict)
print(result)
# _IncompatibleKeys(missing_keys=[], unexpected_keys=[])
```

如果只想加载一部分（例如只加载 backbone，不加载新加的 head）：

```python
result = model.load_state_dict(state_dict, strict=False)
print("缺失:", result.missing_keys)
print("多余:", result.unexpected_keys)
```

> `strict=False` 不是“万能修复”。它只是允许 key 不匹配；如果形状不匹配，仍然会报错。使用时要确认缺失/多余是你预期的。

### 3.5 `state_dict` 是“浅拷贝”

这一点很容易被忽略：

```python
sd = model.state_dict()

with torch.no_grad():
    model.weight.zero_()

print(sd["weight"].abs().sum())   # 0，因为 sd 里的张量和模型共享内存
```

`state_dict()` 返回的张量通常和模型参数共享存储，不是深拷贝。它适合“立刻拿去 `torch.save`”，但不适合“先存着，等会儿再比较”。

如果你需要一个不会变的快照：

```python
import copy
snapshot = copy.deepcopy(model.state_dict())
```

> 对 checkpoint 来说，`torch.save` 会立刻把当前值写出去，所以一般不需要手动 deepcopy；但如果你要在内存里保留历史版本，就要注意。

### 3.6 优化器 state_dict 为什么重要

AdamW 的更新不是只看当前梯度，还看历史：

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g
v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
```

其中 `m` 和 `v` 是**训练状态**，不是模型参数。

如果恢复时只加载模型权重，优化器的 `m/v` 会从 0 开始：

- 前几步的更新方向会变；
- 偏差修正的 `step` 从 0 开始；
- loss 可能出现一个短暂的“抖动”；
- 训练轨迹和原来不一致。

对于小模型、短训练，这个差异可能不明显；对于大模型、长训练，它可能影响收敛。

> 一句话：**模型权重是“结果”，优化器状态是“过程”。要接着跑，过程也得恢复。**

### 3.7 iteration 为什么重要

假设学习率 schedule 是“前 1000 步 warmup，然后余弦衰减”。

训练到第 5000 步时机器挂了。如果恢复时 `iteration` 变成 0：

- 学习率会重新 warmup；
- 模型已经训练过的参数会被小学习率“冻”一段时间；
- 训练曲线会出现一个明显的平台或抖动。

所以 `iteration` 不是“可有可无的日志”，它是 schedule 的输入。

---

## 4. 保存逻辑：把状态装进文件

### 4.1 当前代码

`cs336_basics/CheckPoint.py`：

```python
import torch

def save_checkpoint(model, optimizer, iteration, out):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration
```

先看保存部分：

```python
def save_checkpoint(model, optimizer, iteration, out):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)
```

### 4.2 逐行解释

#### 第一行：`model.state_dict()`

```python
model.state_dict()
```

拿到模型当前所有参数和 buffer 的字典。

注意：

- 它不复制模型结构；
- 它不包含优化器；
- 它返回的张量通常和模型共享内存；
- `torch.save` 会在这个调用发生时读取当前值。

#### 第二行：`optimizer.state_dict()`

```python
optimizer.state_dict()
```

拿到优化器的：

- `state`：每个参数的 `step`、`m`、`v` 等；
- `param_groups`：学习率、betas、eps、weight decay、参数索引。

#### 第三行：`iteration`

```python
'iteration': iteration
```

保存训练进度。它应该是一个普通的 Python 整数。

#### 第四行：`torch.save(..., out)`

```python
torch.save({...}, out)
```

把整个字典序列化到 `out`。

`out` 可以是：

- 字符串路径：`"checkpoint.pt"`；
- `pathlib.Path`：`Path("checkpoints") / "ckpt_1000.pt"`；
- 文件对象：`open("ckpt.pt", "wb")`；
- 内存中的二进制流：`io.BytesIO()`。

> `torch.save` 使用 Python 的 pickle 机制，再加 PyTorch 自己的张量序列化。它适合保存“自己训练出来的、可信的”文件；不要加载来源不明的 checkpoint。

### 4.3 保存的流程

```text
model ──state_dict()──► 权重字典 ─┐
                                  ├──► checkpoint 字典 ──torch.save──► 文件
optimizer ─state_dict()─► 状态字典 ─┤
                                  │
iteration ────────────────────────┘
```

对应代码：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "iteration": iteration,
}
torch.save(checkpoint, out)
```

### 4.4 保存到路径 vs 保存到内存

#### 保存到文件

```python
save_checkpoint(model, optimizer, iteration=1000, out="checkpoint_1000.pt")
```

#### 保存到 `BytesIO`

```python
import io

buffer = io.BytesIO()
save_checkpoint(model, optimizer, iteration=1000, out=buffer)

# 之后要读取，必须先把指针移回开头
buffer.seek(0)
```

> 文件对象有“当前位置”。`torch.save` 写完后，指针在末尾；`torch.load` 从当前位置读，所以要先 `seek(0)`。

### 4.5 文件里到底存了什么

一个 checkpoint 文件通常包含：

```text
checkpoint.pt
├── model_state_dict
│   ├── token_embeddings.weight
│   ├── layers.0.attn.q_proj.weight
│   └── ...
├── optimizer_state_dict
│   ├── state
│   │   ├── 0: {step, m, v}
│   │   └── ...
│   └── param_groups
└── iteration
```

它不是一个“可以直接运行的程序”，而是“一堆张量 + 一些 Python 对象”的序列化结果。

### 4.6 保存频率与磁盘占用

保存太频繁：

- 训练被 I/O 拖慢；
- 磁盘很快写满；
- 每个 checkpoint 都很大。

保存太少：

- 机器挂了会丢很多进度；
- 无法细粒度比较中间模型。

常见策略：

```text
每 N 步保存一次 latest
每 M 次验证保存一次 best
保留最近 K 个 checkpoint
训练结束时保存 final
```

例如：

```text
checkpoints/
├── checkpoint_latest.pt
├── checkpoint_best.pt
├── checkpoint_1000.pt
├── checkpoint_2000.pt
└── checkpoint_3000.pt
```

### 4.7 一个 checkpoint 有多大

粗略估算：

```text
模型参数量 P
模型权重（fp32）      ≈ 4P 字节
AdamW 的 m 和 v       ≈ 8P 字节
合计                  ≈ 12P 字节
```

对于 1B 参数：

```text
1,000,000,000 * 12 B ≈ 12 GB
```

这还没算 buffer、临时对象和文件格式开销。

所以：

- 不要每一步都保存；
- 不要无限保留所有 checkpoint；
- 大模型可以用 `bf16` 存权重，但优化器状态通常仍是 fp32；
- 保存到持久化存储，不要只放在临时磁盘。

### 4.8 更稳妥的保存方式：原子写入

`torch.save` 直接写目标文件时，如果写到一半断电，文件可能损坏。生产环境常用“临时文件 + 原子重命名”：

```python
from pathlib import Path
import os
import torch


def save_checkpoint_atomic(model, optimizer, iteration, out):
    out = Path(out)
    tmp = out.with_suffix(out.suffix + ".tmp")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, tmp)
    os.replace(tmp, out)   # 原子替换
```

这样即使保存过程中崩溃，原来的 `out` 仍然完好。

> CS336 作业不要求原子写入；但如果你真的在云上跑长训练，这个习惯很有用。

---

## 5. 加载逻辑：先搭空模型，再读档

### 5.1 当前代码

```python
def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration
```

### 5.2 加载的顺序非常重要

正确顺序：

```text
1. 搭好模型结构（和保存时一致）
2. 搭好优化器（和保存时一致）
3. torch.load 读取 checkpoint
4. model.load_state_dict(...)
5. optimizer.load_state_dict(...)
6. 取出 iteration
7. 继续训练
```

错误顺序：

```python
# 错误：模型还没搭，无法 load_state_dict
checkpoint = torch.load(src)
model.load_state_dict(checkpoint["model_state_dict"])  # model 未定义

# 错误：先 load，再换模型结构
model = TransformerLM(config_a)
model.load_state_dict(ckpt["model_state_dict"])
model = TransformerLM(config_b)   # 刚加载的权重被丢掉
```

### 5.3 逐行解释

#### `torch.load(src)`

把文件里的字典读回内存。

它做的是反序列化：

```python
checkpoint = {
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "iteration": 1000,
}
```

#### `model.load_state_dict(...)`

按名字把权重复制进模型。

默认 `strict=True`：

- checkpoint 里有、模型里没有的 key → `unexpected_keys`；
- 模型里有、checkpoint 里没有的 key → `missing_keys`；
- 两者都会报错。

#### `optimizer.load_state_dict(...)`

恢复：

- `state`：`step`、`m`、`v`；
- `param_groups`：学习率、betas、eps、weight decay。

注意：`param_groups` 里包含 `lr`，所以加载后**优化器的学习率会被 checkpoint 覆盖**。如果你打算用新的学习率，要在加载后重新设置，或者让 scheduler 每一步重新写入。

#### `return iteration`

把训练进度返回给调用者，训练循环用它决定从哪一步继续。

### 5.4 设备问题：`map_location`

如果 checkpoint 是在 GPU 上保存的，而你在 CPU 机器上加载，直接 `torch.load(src)` 可能报错：

```text
RuntimeError: Attempting to deserialize object on a CUDA device
but torch.cuda.is_available() is False
```

解决：

```python
checkpoint = torch.load(src, map_location="cpu")
```

如果要在 GPU 上恢复：

```python
checkpoint = torch.load(src, map_location="cuda:0")
```

或者：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(src, map_location=device)
```

> 最稳的做法：先在 CPU 上加载，再 `model.to(device)`；或者明确把 `map_location` 设成目标设备。

### 5.5 安全参数：`weights_only`

较新的 PyTorch 版本对 `torch.load` 的 `weights_only` 默认值更严格。它限制反序列化时可以创建哪些 Python 对象，目的是防止恶意 checkpoint 执行任意代码。

- 如果 checkpoint 只包含张量、整数、字典等安全类型，`weights_only=True` 通常可以；
- 如果包含自定义对象、NumPy RNG 等，可能需要 `weights_only=False`；
- `weights_only=False` 只应该用于**你自己生成的、可信的** checkpoint。

CS336 的适配器用的是：

```python
checkpoint = torch.load(src, map_location="cpu", weights_only=False)
```

你可以在自己的代码里采用同样的写法，但要明白它的安全含义。

### 5.6 更稳妥的加载版本

```python
import torch


def load_checkpoint(src, model, optimizer, map_location="cpu"):
    checkpoint = torch.load(
        src,
        map_location=map_location,
        weights_only=False,   # 仅用于可信的、自己保存的 checkpoint
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]
```

如果你不确定 checkpoint 里有什么，先用 `weights_only=True` 试；失败再评估是否信任来源。

### 5.7 加载后不要忘记切换模式

```python
model.train()   # 继续训练
# 或
model.eval()    # 推理/验证
```

`load_state_dict` 只恢复参数，不会自动改变 `train/eval` 状态，也不会恢复 dropout 的随机状态。

---

## 6. 完整可运行案例

下面这个例子会：

1. 训练 5 步；
2. 保存 checkpoint；
3. 原模型继续训练 5 步；
4. 新建模型和优化器，加载 checkpoint 后也继续 5 步；
5. 比较两条路线的 loss、模型权重和 AdamW 状态。

### 6.1 代码

```python
"""CheckPoint 小示例：保存、恢复、继续训练。"""
import io

import torch
from torch import nn

from cs336_basics.AdamW import AdamW
from cs336_basics.CheckPoint import load_checkpoint, save_checkpoint


def make_model():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )


def make_optimizer(model):
    return AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )


def get_batch(step):
    g = torch.Generator().manual_seed(1000 + step)
    x = torch.randn(8, 4, generator=g)
    y = torch.randn(8, 3, generator=g)
    return x, y


def train(model, optimizer, start, end):
    last_loss = None
    for step in range(start, end):
        x, y = get_batch(step)
        optimizer.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    return last_loss


def main():
    torch.manual_seed(0)
    model = make_model()
    optimizer = make_optimizer(model)

    train(model, optimizer, 0, 5)

    buffer = io.BytesIO()
    save_checkpoint(model, optimizer, iteration=5, out=buffer)
    print("checkpoint bytes:", len(buffer.getvalue()))

    # 路线 A：原模型继续训练 5 步
    loss_a = train(model, optimizer, 5, 10)

    # 路线 B：新建空模型/优化器，加载 checkpoint 后继续 5 步
    restored_model = make_model()
    restored_optimizer = make_optimizer(restored_model)
    buffer.seek(0)
    iteration = load_checkpoint(buffer, restored_model, restored_optimizer)
    loss_b = train(restored_model, restored_optimizer, iteration, 10)

    print("iteration:", iteration)
    print("loss A:", loss_a)
    print("loss B:", loss_b)
    print("loss diff:", abs(loss_a - loss_b))

    for (name_a, p_a), (name_b, p_b) in zip(
        model.state_dict().items(), restored_model.state_dict().items()
    ):
        assert name_a == name_b
        assert torch.equal(p_a, p_b), name_a
    print("all model weights equal: True")

    state_a = optimizer.state_dict()["state"]
    state_b = restored_optimizer.state_dict()["state"]
    print("optimizer step:", state_a[0]["step"], state_b[0]["step"])
    print("m equal:", torch.equal(state_a[0]["m"], state_b[0]["m"]))
    print("v equal:", torch.equal(state_a[0]["v"], state_b[0]["v"]))


if __name__ == "__main__":
    main()
```

### 6.2 实际输出

```text
checkpoint bytes: 5119
iteration: 5
loss A: 1.0647128820419312
loss B: 1.0647128820419312
loss diff: 0.0
all model weights equal: True
optimizer step: 10 10
m equal: True
v equal: True
```

### 6.3 这个例子说明了什么

- 路线 A 和路线 B 的 loss 完全相同，说明恢复后的训练轨迹一致；
- 所有模型权重逐位相等；
- AdamW 的 `step` 都是 10（5 步保存 + 5 步继续）；
- `m` 和 `v` 也完全相等，说明优化器状态真的被恢复了。

如果只恢复模型权重、不恢复优化器，`m/v` 会从 0 开始，loss 就会和路线 A 不同。

### 6.4 用路径保存

```python
from pathlib import Path

out = Path("checkpoints") / "ckpt_5.pt"
out.parent.mkdir(parents=True, exist_ok=True)
save_checkpoint(model, optimizer, iteration=5, out=out)
```

加载：

```python
iteration = load_checkpoint(out, model, optimizer)
```

### 6.5 用文件对象保存

```python
with open("checkpoint.pt", "wb") as f:
    save_checkpoint(model, optimizer, iteration=5, out=f)

with open("checkpoint.pt", "rb") as f:
    iteration = load_checkpoint(f, model, optimizer)
```

> 文件对象在读取前要确保指针在开头；如果是刚写完的同一个对象，记得 `seek(0)`。

### 6.6 查看 state_dict 的 key

```python
model = make_model()
for name, tensor in model.state_dict().items():
    print(name, tuple(tensor.shape))
```

对于上面的 `nn.Sequential`，大致是：

```text
0.weight (8, 4)
0.bias (8,)
2.weight (3, 8)
2.bias (3,)
```

注意层名是 `0`、`2`，因为 `ReLU` 没有参数，不会出现在 state_dict 里。

---

## 7. 恢复训练到底需要哪些信息

### 7.1 一张表看清

| 信息 | 作用 | 不保存会怎样 |
| --- | --- | --- |
| 模型结构/配置 | 知道怎么搭模型 | 不知道 `state_dict` 该往哪装 |
| 模型权重 | 恢复参数 | 模型还是随机初始化 |
| 优化器类型/超参数 | 构造同样的优化器 | 优化器行为不同 |
| 优化器 state | 恢复动量、步数 | 训练轨迹改变 |
| iteration | 恢复 schedule/日志 | 学习率从头开始 |
| tokenizer/vocab | 保证 token ID 含义一致 | 同一 ID 解码成不同文字 |
| RNG 状态 | 数据顺序、dropout 可复现 | 每次恢复的 batch 不同 |
| 数据位置 | 从同一位置继续 | 可能重复/跳过数据 |
| GradScaler | 混合精度恢复 | loss scaling 异常 |
| 代码版本 | 结构/逻辑一致 | 可能加载失败或行为不同 |

### 7.2 CS336 的最小要求

CS336 的 checkpoint 题目只要求：

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "iteration": ...,
}
```

为什么这就够了？

- 模型结构由你的代码和配置决定，测试里会重新 `_TestNet(...)`；
- 优化器由测试重新构造，超参数相同；
- 学习率 schedule 在 CS336 里是“iteration 的纯函数”，不需要额外状态；
- 数据加载是随机采样，测试不要求精确恢复数据顺序。

### 7.3 一个恢复训练的模板

```python
import torch

from cs336_basics.CheckPoint import load_checkpoint
from cs336_basics.Lr_Cosine_Schedule import CosineSchedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 先搭模型和优化器
model = TransformerLM(config).to(device)
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

# 2. 如果需要恢复，加载 checkpoint
start_iter = 0
if resume_path is not None:
    start_iter = load_checkpoint(
        resume_path,
        model,
        optimizer,
        map_location=device,
    )

# 3. 构造 schedule
scheduler = CosineSchedule(
    max_learning_rate=1e-3,
    min_learning_rate=1e-4,
    warmup_iters=100,
    cosine_cycle_iters=10000,
)

# 4. 从 start_iter 继续
for it in range(start_iter, max_iters):
    for group in optimizer.param_groups:
        group["lr"] = scheduler(it)

    x, y = get_batch(...)
    logits = model(x)
    loss = cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if (it + 1) % save_every == 0:
        save_checkpoint(model, optimizer, iteration=it + 1, out=ckpt_path)
```

### 7.4 iteration 的语义要统一

推荐约定：

> `iteration` = **已经完成的优化步数**。

训练循环：

```python
for it in range(start_iter, max_iters):
    ...
    optimizer.step()
    # 第 it 步完成
```

保存时：

```python
save_checkpoint(model, optimizer, iteration=it + 1, out=path)
```

恢复时：

```python
start_iter = load_checkpoint(path, model, optimizer)
for it in range(start_iter, max_iters):
    ...
```

这样：

- 保存 `iteration=5` 表示第 0..4 步已完成；
- 下次从第 5 步开始；
- schedule 用 `scheduler(5)`，和原来一致。

> 如果你在 `optimizer.step()` 之前保存，语义就变成“下一步的编号”，要相应调整循环。关键是**保存和恢复用同一套约定**。

### 7.5 为什么只加载模型权重不够

假设你只做：

```python
model.load_state_dict(ckpt["model_state_dict"])
```

会发生：

- 模型参数正确；
- 优化器的 `m/v` 全为 0；
- `step` 从 0 开始；
- 学习率可能被新构造的 optimizer 设为初始值；
- 数据顺序重新开始。

短训练可能看不出问题；长训练会出现明显的 loss 抖动或收敛差异。

### 7.6 为什么只加载优化器也不行

优化器状态是“相对于参数”的：

- `state` 的 key 是参数索引；
- 如果模型参数是随机初始化的，恢复动量没有意义；
- 必须先恢复模型权重，再恢复优化器状态。

---

## 8. 联系 CS336 Assignment 1

### 8.1 作业要求

CS336 讲义 §5.2 的 `checkpointing` 题目（1 分）要求实现：

```python
def save_checkpoint(model, optimizer, iteration, out):
    ...

def load_checkpoint(src, model, optimizer):
    ...
```

要求：

- `save_checkpoint` 把模型、优化器、iteration 保存到 `out`；
- `out` 可以是路径，也可以是文件对象；
- `load_checkpoint` 从 `src` 恢复模型和优化器；
- 返回保存时的 iteration。

### 8.2 适配器

测试不直接调用 `CheckPoint.py`，而是调用 `tests/adapters.py`：

```python
def run_save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
```

注意 key 名字：

| 文件 | 模型 key | 优化器 key |
| --- | --- | --- |
| `CheckPoint.py` | `model_state_dict` | `optimizer_state_dict` |
| `tests/adapters.py` | `model` | `optimizer` |

两种写法本身都对，但**保存和加载必须使用同一套 key**。如果你用 `CheckPoint.py` 保存，却用适配器的 key 去加载，会得到 `KeyError`。

> 作业测试调用的是适配器，所以适配器内部的 key 是一致的。你自己写训练脚本时，建议统一成一种命名，避免混用。

### 8.3 测试在做什么

`tests/test_serialization.py` 的流程：

```text
1. 固定随机种子 torch.manual_seed(42)
2. 构造 _TestNet(100 -> 200 -> 100 -> 10)
3. 构造 AdamW(lr=1e-3, wd=0.01)
4. 训练 10 步，it 从 0 加到 10
5. 保存 checkpoint 到临时路径
6. 新建一个结构相同的 _TestNet 和 AdamW
7. 加载 checkpoint，得到 loaded_iterations
8. 断言 loaded_iterations == 10
9. 比较原模型和新模型的 state_dict
10. 比较原优化器和新优化器的 state_dict
```

关键代码：

```python
serialization_path = tmp_path / "checkpoint.pt"

run_save_checkpoint(
    model,
    optimizer,
    iteration=it,
    out=serialization_path,
)

new_model = _TestNet(d_input=100, d_output=10)
new_optimizer = get_adamw_cls()(
    new_model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
)

loaded_iterations = run_load_checkpoint(
    src=serialization_path,
    model=new_model,
    optimizer=new_optimizer,
)
```

测试会逐个比较模型参数：

```python
for key in original_model_state:
    numpy.testing.assert_allclose(
        original_model_state[key].detach().numpy(),
        new_model_state[key].detach().numpy(),
    )
```

也会比较优化器状态（包括 `m/v` 和 `param_groups`）。

### 8.4 运行测试

```bash
cd /home/kevin/assignment1-basics
uv run pytest -k test_checkpointing
```

如果不用 `uv`：

```bash
.venv/bin/python -m pytest -k test_checkpointing
```

预期：

```text
tests/test_serialization.py::test_checkpointing PASSED
```

### 8.5 当前 `CheckPoint.py` 的注意点

当前实现：

```python
checkpoint = torch.load(src)
```

在 CPU、同版本 PyTorch、纯张量/int 的 checkpoint 上可以正常工作。但更稳妥的写法是：

```python
checkpoint = torch.load(
    src,
    map_location="cpu",
    weights_only=False,
)
```

原因：

- `map_location` 解决“GPU 保存、CPU 加载”的设备不匹配；
- `weights_only` 明确反序列化策略；
- CS336 适配器也是这么写的。

> 如果你只是跑作业测试，当前实现通常能过；但做真实训练脚本时，建议补上这两个参数。

### 8.6 常见扣分点

| 扣分点 | 后果 | 修正 |
| --- | --- | --- |
| 只保存模型权重 | 优化器失忆 | 加 `optimizer.state_dict()` |
| 忘记保存 iteration | schedule 从头开始 | 加入 checkpoint 字典 |
| `torch.save(model)` 而不是 state_dict | 依赖类定义，不好迁移 | 保存 `model.state_dict()` |
| 加载前没搭模型 | `load_state_dict` 报错 | 先构造同结构模型 |
| 模型结构不一致 | missing/unexpected keys | 配置和保存时保持一致 |
| 优化器参数顺序不一致 | state 对不上 | 用相同的构造方式 |
| key 名字混用 | `KeyError` | 保存/加载统一 key |
| 忘记 `map_location` | GPU checkpoint 在 CPU 上加载失败 | 指定 `map_location` |
| 加载后没设学习率 | 用了 checkpoint 里的旧 lr | 用 scheduler 重新写 `group["lr"]` |
| 文件对象没 `seek(0)` | 读到空数据/报错 | 读取前回到开头 |

---

## 9. 常见坑与 FAQ

### Q1：`torch.save(model, path)` 和 `torch.save(model.state_dict(), path)` 有什么区别？

| 方式 | 保存内容 | 优点 | 缺点 |
| --- | --- | --- | --- |
| `torch.save(model, path)` | 整个模型对象 | 加载简单 | 依赖类定义、代码路径；可移植性差；安全风险更高 |
| `torch.save(model.state_dict(), path)` | 只保存张量字典 | 轻量、可移植、推荐 | 加载前要自己搭模型 |

CS336 要求的是后者。

### Q2：为什么加载后 loss 和原来不一样？

可能原因：

1. 只加载了模型，没有加载优化器；
2. `iteration` 没有恢复，学习率 schedule 从头开始；
3. 数据顺序/RNG 没有恢复；
4. 模型结构或参数名有细微差异；
5. 加载后没有调用 `model.train()`；
6. 优化器的 `lr` 被 checkpoint 覆盖，或者被新 scheduler 改错。

### Q3：为什么 `load_state_dict` 报 missing keys？

说明模型里有参数，但 checkpoint 里没有。常见原因：

- 保存时模型结构和加载时不一样；
- 保存的是 DDP 模型（key 带 `module.` 前缀），加载的是单卡模型；
- 保存时用了 `strict=False` 漏掉了一部分；
- 你修改了模型代码，新增了一层。

解决：

```python
result = model.load_state_dict(state_dict, strict=False)
print(result.missing_keys)
print(result.unexpected_keys)
```

确认差异是你预期的，再决定是否忽略。

### Q4：DDP 保存的 checkpoint 怎么加载到单卡模型？

DDP 会给参数名加 `module.` 前缀。可以去掉：

```python
from collections import OrderedDict

new_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith("module."):
        key = key[len("module."):]
    new_state_dict[key] = value

model.load_state_dict(new_state_dict)
```

反过来，单卡 checkpoint 加载到 DDP 模型，可以给 key 加前缀，或者用 `model.module.load_state_dict(...)`。

### Q5：为什么优化器的学习率加载后变了？

`optimizer.state_dict()["param_groups"]` 里包含 `lr`。`load_state_dict` 会把它恢复成保存时的值。

如果你希望用新的学习率：

```python
optimizer.load_state_dict(ckpt["optimizer"])
for group in optimizer.param_groups:
    group["lr"] = new_lr
```

但在有 scheduler 的训练里，通常每一步都会重新写 `lr`，所以这不是大问题。

### Q6：`iteration` 到底从 0 还是 1 开始？

两种约定都可以，关键是统一：

- 约定 A：`iteration` 是“已完成的步数”，保存 5 表示第 0..4 步已完成，下次从 5 开始；
- 约定 B：`iteration` 是“下一步的编号”，保存 5 表示下次执行第 5 步。

CS336 测试里：

```python
it = 0
for _ in range(10):
    ...
    it += 1
# 保存 iteration=10
```

这是约定 A。

### Q7：checkpoint 文件太大怎么办？

- 降低保存频率；
- 只保留最近 K 个 + 最好的一个；
- 用 `bf16` 保存模型权重（优化器状态通常仍用 fp32）；
- 如果只做推理，保存一个只含模型权重的 `model.pt`，不含优化器；
- 不要把 checkpoint 提交到 Git。

### Q8：保存 checkpoint 时模型在 GPU 上，加载到 CPU 会怎样？

没有 `map_location` 时，`torch.load` 会尝试把张量放到保存时的设备；CPU 机器上没有 CUDA 就会报错。

解决：

```python
ckpt = torch.load(path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.to("cpu")
```

### Q9：`weights_only=True` 和 `False` 怎么选？

- `True`：更安全，只允许安全类型；适合加载不可信来源；
- `False`：更宽松，可以加载自定义对象；只用于你自己信任的 checkpoint。

如果你不确定，先用 `True`；失败再检查 checkpoint 里是否有特殊对象，而不是直接关掉安全限制。

### Q10：`state_dict` 可以直接修改吗？

可以，但要明白它是浅拷贝：

```python
sd = model.state_dict()
sd["weight"].zero_()   # 会改到模型本身
```

如果你想要一个独立快照：

```python
import copy
sd = copy.deepcopy(model.state_dict())
```

### Q11：保存 checkpoint 需要 `model.eval()` 吗？

不需要。`state_dict` 保存的是参数值，和 `train/eval` 模式无关。

但如果你在保存前做了 BatchNorm 的 forward，running stats 可能已经更新；这是模型状态的一部分。对 Transformer 的 RMSNorm 来说没有 running stats，影响较小。

### Q12：为什么测试用 `weights_only=False`？

因为测试的 checkpoint 里包含优化器 state 字典等 Python 对象。`weights_only=True` 在新版本 PyTorch 中对可接受类型有限制；适配器选择显式关闭限制，并配合 `map_location="cpu"`。

再次强调：只对可信文件这么做。

### Q13：checkpoint 可以跨 PyTorch 版本加载吗？

通常可以，但不保证。保存时的类结构、参数名、优化器实现变化都可能导致不兼容。

建议：

- 记录 PyTorch 版本；
- 记录模型代码版本/commit；
- 长期保存时同时保存配置和 tokenizer；
- 重要的 checkpoint 做一次“加载 + 前向”验证。

### Q14：训练脚本应该什么时候保存？

常见做法：

```text
每 N 步：保存 latest
每次验证：如果 val_loss 更好，保存 best
训练结束：保存 final
异常退出：在 finally 里尝试保存一次
```

N 的选择取决于：

- 单步时间；
- checkpoint 大小；
- 你能接受丢失多少进度；
- 存储成本。

---

## 10. 自测清单与一页速查

### 10.1 自测清单

- [ ] 我能解释为什么只保存模型权重不够；
- [ ] 我知道 checkpoint 的三件套是什么；
- [ ] 我能说出 `model.state_dict()` 里有什么、没有什么；
- [ ] 我知道 `load_state_dict` 是“复制值”，不是“替换模型对象”；
- [ ] 我能解释优化器的 `m/v` 为什么属于训练状态；
- [ ] 我知道 `iteration` 对学习率 schedule 的作用；
- [ ] 我能写出保存和加载的完整顺序；
- [ ] 我知道 `map_location` 解决什么问题；
- [ ] 我知道 `weights_only` 的安全含义；
- [ ] 我能指出 `CheckPoint.py` 和 `adapters.py` 的 key 名差异；
- [ ] 我能跑通 `test_checkpointing`；
- [ ] 我能解释为什么恢复后 loss 可能和原来不同。

### 10.2 一页速查

```text
保存:
  checkpoint = {
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "iteration": iteration,
  }
  torch.save(checkpoint, out)

加载:
  ckpt = torch.load(src, map_location="cpu", weights_only=False)
  model.load_state_dict(ckpt["model_state_dict"])
  optimizer.load_state_dict(ckpt["optimizer_state_dict"])
  iteration = ckpt["iteration"]

顺序:
  搭模型 -> 搭优化器 -> torch.load -> load_state_dict -> 取 iteration -> 继续训练

三件套:
  模型权重   = 现在在哪
  优化器状态 = 速度/惯性
  iteration  = 跑到第几圈

设备:
  GPU -> CPU: map_location="cpu"
  CPU -> GPU: map_location=device 或先 CPU 加载再 model.to(device)

安全:
  weights_only=True  更安全，适合不可信来源
  weights_only=False 更宽松，只用于可信 checkpoint
```

### 10.3 一句话总结

> **Checkpoint 是训练现场的快照：模型权重告诉你“参数是什么”，优化器状态告诉你“之前怎么走的”，iteration 告诉你“走到第几步”。三者一起保存、一起加载，才能真正接着训练。**

---

## 附：相关文档

- `Data Loader.md`：训练数据怎样变成 `x/y`；
- `AdamW.md`：优化器状态里到底存了什么；
- `学习率调度：Cosine Schedule.md`：`iteration` 怎样影响学习率；
- `Transformer 模块.md`：模型结构怎样决定 `state_dict` 的 key。
