# CS336 Assignment 1：从零实现语言模型

> Stanford CS336《Language Modeling from Scratch》Assignment 1 的完整实现与配套学习笔记。
> 从字节级 BPE 分词器到 Transformer 语言模型，再到训练循环与采样解码，全部基于 PyTorch 手工构建，不依赖任何现成的语言模型库。

**作者**：晓来云绕天

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230)](https://docs.astral.sh/uv/)
[![Tests](https://img.shields.io/badge/tests-47%20passed-brightgreen)](#测试与-adapter-层)

---

## 目录

- [项目简介](#项目简介)
- [设计要点](#设计要点)
- [实现范围](#实现范围)
- [目录结构](#目录结构)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [训练结果](#训练结果)
- [测试与 Adapter 层](#测试与-adapter-层)
- [推荐学习路线](#推荐学习路线)
- [常见问题](#常见问题)
- [后续计划](#后续计划)
- [说明](#说明)

---

## 项目简介

本项目是 CS336 Assignment 1 的个人实现版本，目标是回答一个问题：**一个能读写文本、能训练、能生成的语言模型，最少需要哪些零件？**

答案是四层：

1. **分词器**：把文本切成字节，再用 BPE 逐步合并成子词，得到词表与合并规则；
2. **模型**：Embedding、RMSNorm、SwiGLU、RoPE、因果多头注意力与残差堆叠成 decoder-only Transformer；
3. **训练**：AdamW、余弦学习率、梯度裁剪、memmap 数据加载、检查点与日志；
4. **推理**：温度缩放与 top-p 核采样，从 prompt 续写到遇到 `<|endoftext|>`。

与"调库跑通"不同，这里的每一层都是手写的：分词器没有用 `tiktoken`，注意力没有用 `F.scaled_dot_product_attention`，优化器没有用 `torch.optim.AdamW`。这样做的意义在于，所有细节——字节如何映射成 ID、位置编码如何旋转、梯度如何被裁剪、概率如何被截断——都必须在代码里显式表达，没有黑箱可躲。

代码按模块拆分，每个核心组件都可以单独测试、单独阅读；`cs336_basics/Markdown教程/` 下配有 19 篇逐模块的推导笔记，构成一条从张量基础到端到端训练的完整路径。

---

## 设计要点

实现中有几处不太显眼但影响正确性的选择，值得单独说明：

**BPE 的平局规则**
每轮合并选"出现频率最高、同频时字节对字典序最大"的候选。代码把比较键写成 `(freq, pair)`，依赖 Python 元组的逐元素比较天然实现这条规则；如果写成 `max(stats, key=stats.get)`，同频时的结果就是不确定的。

**倒排索引维护**
合并一个 pair 会同时改变它左右邻居的计数。实现用 `pair → 包含该 pair 的词下标集合` 做倒排索引，只更新受影响的词，而不是每轮全量重扫；索引中的陈旧条目由内层循环的显式比对兜底。

**RoPE 的配对方式**
相邻两维一组（`[0::2]` 与 `[1::2]`）进行旋转，而不是前后半段配对。两种写法都能让内积只依赖相对位置，但与讲义给定的权重布局不兼容，换错会表现为"训练能跑、效果变差"。

**RMSNorm 的精度提升**
计算均方根前把输入提升到 fp32，算完再转回原精度。bf16 下直接求平方和会损失有效位，几步之后归一化因子就开始漂移。

**损失函数的缩放归属**
`CrossEntropyLoss` 返回的是**求和**而非平均，训练循环里再除以 `B × T`。这个 1/(B·T) 放在哪里都行，但必须只放一次——重复除会让梯度小几个数量级，表现为"loss 正常、模型不学"。

**top-p 的最小覆盖前缀**
截断条件是"前缀累积概率（不含当前 token）仍小于 p"，等价于讲义定义的"使累积概率达到 p 的最小集合"。实现上避免了"整行被清零"的退化情况，也顺手绕开了 0/0。

---

## 实现范围

| 层次 | 内容 | 主要文件 |
| --- | --- | --- |
| 分词 | BPE 训练、编解码、流式编码、特殊 token 隔离 | `BPE_Tokenizer_train.py`、`Tokenizer.py`、`prepare_data.py` |
| 模型 | Linear / Embedding、RMSNorm、SwiGLU、RoPE、注意力、Transformer Block / LM | `Linear_Embedding_model.py`、`RMSNorm.py`、`SwiGLU.py`、`run_rope.py`、`Causal_MultiHead_Self_Attention*.py`、`Transformer_Block.py`、`Transformer_LM.py` |
| 训练 | AdamW、余弦调度、梯度裁剪、数据加载、检查点、训练入口 | `AdamW.py`、`Lr_Cosine_Schedule.py`、`Gradient_Clipping.py`、`Data_Loader.py`、`CheckPoint.py`、`main_train.py` |
| 推理 | 温度缩放、top-p 采样、自回归生成 | `Inference.py`、`sample.py` |
| 辅助 | softmax、交叉熵、权重初始化、RoPE 演示 | `softmax.py`、`Cross_Entropy_Loss.py`、`run_rope.py` |

模型支持四组消融开关：`--no_rms_norm`、`--norm_mode post`、`--no_rope`、`--ffn_type silu`。

---

## 目录结构

```text
.
├── cs336_basics/
│   ├── main_train.py                   # 训练入口：参数解析、训练循环、验证、checkpoint
│   ├── prepare_data.py                 # 文本 → uint16 token 二进制（memmap 友好）
│   ├── sample.py                       # 加载 checkpoint 并用 prompt 生成文本
│   ├── BPE_Tokenizer_train.py          # BPE 词表训练 + 词表落盘
│   ├── Tokenizer.py                    # 分词器：encode / decode / encode_iterable
│   ├── AdamW.py                        # 优化器
│   ├── Lr_Cosine_Schedule.py           # warmup + 余弦退火
│   ├── Gradient_Clipping.py            # 全局 L2 范数裁剪
│   ├── Data_Loader.py                  # memmap 随机切片
│   ├── CheckPoint.py                   # 模型 / 优化器 / 步数存取
│   ├── Cross_Entropy_Loss.py
│   ├── Inference.py                    # temperature + top-p + 生成循环
│   ├── Transformer_Block.py            # 单个 Block
│   ├── Transformer_LM.py               # 完整语言模型
│   ├── Causal_MultiHead_Self_Attention.py
│   ├── Causal_MultiHead_Self_Attention_with_rope.py
│   ├── Linear_Embedding_model.py
│   ├── RMSNorm.py
│   ├── SwiGLU.py
│   ├── run_rope.py                     # RoPE 实现与位置编码演示
│   ├── softmax.py
│   ├── Scaled_dot_product_attention.py
│   ├── pretokenization_example.py      # 分块预分词示例
│   └── Markdown教程/                    # 19 篇配套笔记与推导
├── tests/
│   ├── adapters.py                     # 测试适配层：把课程接口翻译成本仓库的实现
│   ├── test_tokenizer.py               # 分词器：roundtrip、特殊 token、与 tiktoken 对齐
│   ├── test_train_bpe.py               # BPE 训练：结果、特殊 token、速度
│   ├── test_model.py                   # 模型组件与完整 LM
│   ├── test_nn_utils.py                # softmax、交叉熵、梯度裁剪
│   ├── test_optimizer.py               # AdamW、余弦调度
│   ├── test_serialization.py           # checkpoint 存取
│   ├── test_data.py                    # 数据加载
│   ├── fixtures/                       # 测试语料与参考词表
│   └── _snapshots/                     # 参考输出快照
├── data/                               # 语料、tokenized bin、checkpoint（不入库）
├── pyproject.toml                      # 依赖与工具配置
├── uv.lock
├── make_submission.sh
└── cs336_assignment1_basics.pdf        # 作业说明
```

---

## 环境准备

项目要求 **Python 3.12 ~ 3.13**，依赖由 [uv](https://docs.astral.sh/uv/) 管理。首次安装后会按 `uv.lock` 精确还原环境，无需手动处理虚拟环境。

### Linux / macOS

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 在项目根目录同步依赖（自动创建 .venv 并安装 PyTorch 等）
cd assignment1-basics
uv sync
```

若已有 Python 但版本不符，uv 会自动下载并管理符合要求的解释器，无需手动安装。

### Windows

在 **PowerShell** 中执行：

```powershell
# 1. 安装 uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 同步依赖
cd assignment1-basics
uv sync
```

安装完成后重开一个终端，确保 `uv` 已进入 `PATH`。

> Windows 下所有命令均可直接在 PowerShell 或 Windows Terminal 中运行；若使用 WSL2 / Git Bash，按 Linux 步骤操作即可。

验证环境：

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 数据准备

本项目使用 TinyStoriesV2-GPT4 作为主力语料。在项目根目录创建 `data/` 并下载：

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

Windows PowerShell 等价写法（`wget` 不是系统自带命令）：

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" -OutFile "data/TinyStoriesV2-GPT4-train.txt"
Invoke-WebRequest -Uri "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" -OutFile "data/TinyStoriesV2-GPT4-valid.txt"
```

下载完成后目录应如下：

```text
data/
├── TinyStoriesV2-GPT4-train.txt   # 约 2.2 GB
└── TinyStoriesV2-GPT4-valid.txt   # 约 22 MB
```

---

## 快速开始

完整流程共四步：训练分词器 → 编码语料 → 训练模型 → 生成文本。

### 1. 训练 BPE 词表

```bash
uv run cs336_basics/BPE_Tokenizer_train.py
```

输出写入 `data/TinyStoriesV2-GPT4-train/`：

```text
data/TinyStoriesV2-GPT4-train/
├── vocab.json     # 10000 个 token，ID → byte-level 字符串
└── merges.txt     # 9743 条合并规则
```

脚本只在开头打印两行提示，中途无输出属正常现象；判断是否跑完，看 `merges.txt` 是否 9743 行、`vocab.json` 是否 10000 条。

### 2. 把语料编码成 token 二进制

```bash
# 验证集
uv run python -m cs336_basics.prepare_data \
    --text data/TinyStoriesV2-GPT4-valid.txt \
    --out  data/bin/valid.bin \
    --vocab  data/TinyStoriesV2-GPT4-train/vocab.json \
    --merges data/TinyStoriesV2-GPT4-train/merges.txt

# 训练集（先用 --max_mb 处理一小段，跑通后再放开）
uv run python -m cs336_basics.prepare_data \
    --text data/TinyStoriesV2-GPT4-train.txt \
    --out  data/bin/train.bin \
    --vocab  data/TinyStoriesV2-GPT4-train/vocab.json \
    --merges data/TinyStoriesV2-GPT4-train/merges.txt \
    --max_mb 200
```

输出为裸 `uint16` 小端数组，训练时用 `np.memmap` 按页读取，无需一次性载入内存。

### 3. 训练语言模型

```bash
uv run python -m cs336_basics.main_train \
    --train_data_path data/bin/train.bin \
    --valid_data_path data/bin/valid.bin \
    --out_dir data/ckpt_tinystories \
    --batch_size 32 \
    --context_length 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 8 \
    --d_ff 2048 \
    --lr 6e-4 \
    --max_iters 4000 \
    --warmup_iters 400 \
    --no_wandb
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--train_data_path` / `--valid_data_path` | tokenized `.bin` 路径（必填） |
| `--out_dir` | checkpoint 与日志输出目录 |
| `--batch_size` / `--context_length` | 每步样本数与序列长度 |
| `--d_model` / `--num_layers` / `--num_heads` / `--d_ff` | 模型规模 |
| `--lr` / `--min_lr` / `--warmup_iters` / `--max_iters` | 优化配置 |
| `--eval_every` / `--save_every` | 验证与存盘间隔 |
| `--no_wandb` | 纯本地训练，不联网记录 |

启动时若 `out_dir/ckpt.pt` 已存在，会从该检查点续训；删除或改名可从头开始。

### 4. 生成文本

```bash
uv run python -m cs336_basics.sample \
    --ckpt data/ckpt_tinystories/ckpt_final.pt \
    --prompt "Once upon a time" \
    --max_tokens 150 \
    --temperature 0.8 \
    --top_p 0.95
```

---

## 训练结果

TinyStories 子集（约 6000 万 token）上的基线配置：

| 配置项 | 取值 |
| --- | --- |
| 层数 / 隐藏维 / 头数 | 4 / 512 / 8 |
| 上下文长度 | 256 |
| 批大小 | 32 |
| 学习率 | 6e-4，warmup 400 步后余弦衰减 |
| 训练步数 | 4000 |

| 指标 | 数值 |
| --- | --- |
| 最终训练 loss | 1.6968 |
| 最终验证 loss | **1.6603** |
| 验证困惑度 | 5.26 |

训练曲线与各消融实验（RMSNorm、pre/post-norm、RoPE、SwiGLU）仍在补全中。

---

## 测试与 Adapter 层

### 为什么需要 adapter

课程测试不直接导入你的模块。测试文件只认一套固定的函数签名，真正的"翻译"发生在 `tests/adapters.py`：它接收测试传来的权重与张量，按你的接口构造模块、灌入权重，再返回结果。

这样设计的好处是测试与实现解耦——你可以任意组织文件、命名类、设计构造参数，只要在 adapter 里完成转换，测试就能跑。代价是 adapter 成了唯一需要跟着实现一起维护的测试文件：改了接口却忘了改 adapter，测试会以"导入失败"或"参数不匹配"的形式失败，而不是算法错误。

`adapters.py` 本身不包含任何算法逻辑，只做三件事：**延迟导入**（函数体内 import，避免测试收集阶段就因某模块报错而整体失败）、**构造与灌权重**、**转发调用**。

### 映射关系

| Adapter 函数 | 对应实现 | 覆盖测试 |
| --- | --- | --- |
| `run_train_bpe` | `BPE_Tokenizer_train.train_bpe` | `test_train_bpe.py` |
| `get_tokenizer` | `Tokenizer.BPETokenizer` | `test_tokenizer.py` |
| `run_linear` / `run_embedding` | `Linear_Embedding_model.LinearModule` / `EmbeddingModule` | `test_model.py` |
| `run_rmsnorm` | `RMSNorm.RMSNorm` | `test_model.py` |
| `run_swiglu` / `run_silu` | `SwiGLU.SwiGLU` | `test_model.py` |
| `run_rope` | `run_rope.RoPE` | `test_model.py` |
| `run_scaled_dot_product_attention` | `Scaled_dot_product_attention` | `test_model.py` |
| `run_multihead_self_attention` | `Causal_MultiHead_Self_Attention` | `test_model.py` |
| `run_multihead_self_attention_with_rope` | `Causal_MultiHead_Self_Attention_with_rope` | `test_model.py` |
| `run_transformer_block` | `Transformer_Block.TransformerBlock` | `test_model.py` |
| `run_transformer_lm` | `Transformer_LM.TransformerLM` | `test_model.py` |
| `run_softmax` / `run_cross_entropy` | `softmax.softmax` / `Cross_Entropy_Loss` | `test_nn_utils.py` |
| `run_gradient_clipping` | `Gradient_Clipping.GradientClip` | `test_nn_utils.py` |
| `get_adamw_cls` | `AdamW.AdamW` | `test_optimizer.py` |
| `run_get_lr_cosine_schedule` | `Lr_Cosine_Schedule.CosineSchedule` | `test_optimizer.py` |
| `run_save_checkpoint` / `run_load_checkpoint` | `CheckPoint.save_checkpoint` / `load_checkpoint` | `test_serialization.py` |
| `run_get_batch` | `Data_Loader.DataLoader` | `test_data.py` |

### 维护 adapter 的几条经验

**接口变了，先看 adapter。** 给模块加参数、改构造签名、重命名类，第一反应应该是检查 adapter 里对应的 `from ... import ...` 与调用处。测试报 `ImportError` 或 `TypeError` 时，问题往往在这里，而不是算法。

**别把逻辑写进 adapter。** adapter 里出现的任何计算都会绕过正式实现，造成"测试通过但训练不对"。判断标准很简单：删掉 adapter 里的某段逻辑，正式代码是否还能独立跑通。

**注意 dtype 与 device 的传递。** 测试会用 `bf16`、`cuda` 等组合调用，adapter 构造模块时要把 `weights.device`、`weights.dtype` 传下去，否则会出现"测试在 CPU 上过、训练在 GPU 上炸"。

**快照测试失败先看数值范围。** `_snapshots/` 里存的是参考输出，报错信息通常是形状不匹配或最大绝对误差。先确认形状，再定位是缩放因子、mask 语义还是初始化差异。

---

## 推荐学习路线

代码与笔记一一对应，按下列顺序阅读可以顺着"数据 → 模型 → 训练 → 推理"的主线推进：

| 阶段 | 阅读 | 对应实现 |
| --- | --- | --- |
| 0. 前置 | `Python和PyTorch基础.md` | — |
| 1. 文本与字节 | `ASCII_Unicode_UTF-8.md` | — |
| 2. 分词 | `BPE算法原理.md` | `BPE_Tokenizer_train.py`、`Tokenizer.py` |
| 3. 线性层与嵌入 | `Embedding_Linear.md` | `Linear_Embedding_model.py` |
| 4. 归一化 | `RMSNorm.md` | `RMSNorm.py` |
| 5. 前馈网络 | `SwiGLU.md` | `SwiGLU.py` |
| 6. 注意力 | `Attention.md` → `缩放点积注意力.md` → `Causal_Multi-Head_self-attention.md` | `Causal_MultiHead_Self_Attention*.py` |
| 7. 位置编码 | `Causal_Multi-Head_self-attention.md`（RoPE 章节） | `run_rope.py` |
| 8. 组装 | `Transformer_Block.md` → `Transformer 模块.md` → `完整语言模型架构的组装.md` | `Transformer_Block.py`、`Transformer_LM.py` |
| 9. 损失函数 | `Cross_Entropy_Loss.md` | `Cross_Entropy_Loss.py`、`softmax.py` |
| 10. 优化器 | `AdamW.md` | `AdamW.py` |
| 11. 学习率 | `学习率调度：Cosine Schedule.md` | `Lr_Cosine_Schedule.py` |
| 12. 训练稳定性 | `Gradient Clipping.md` | `Gradient_Clipping.py` |
| 13. 数据管道 | `Data Loader.md` | `Data_Loader.py` |
| 14. 持久化 | `CheckPoint.md` | `CheckPoint.py` |
| 15. 端到端 | `端到端训练和模型推理.md` | `main_train.py`、`Inference.py`、`sample.py` |

建议的节奏：每个阶段先读笔记里的公式与形状推演，再对照代码确认实现细节，最后跑 `uv run pytest tests/<对应测试> -x`。测试通过只说明"和参考实现一致"，理解为什么要这样写，才算真正完成这一节。

---

## 常见问题

**BPE 训练跑完屏幕上什么都没有？**
脚本只在开头打印提示，没有结束语。产物在 `data/TinyStoriesV2-GPT4-train/`，看 `merges.txt` 行数即可确认。

**每次启动都从中间开始训练？**
说明 `--out_dir` 下的 `ckpt.pt` 已存在，训练脚本默认续训。换个目录或删除该文件即可从头开始。

**显存不足（OOM）？**
先降 `--batch_size`，再降 `--context_length`，最后才动 `--d_model`。训练脚本里 `main_train.py` 没有梯度累积，批大小直接决定显存峰值。

**Windows 提示 `wget` 不是内部或外部命令？**
用上文的 `Invoke-WebRequest` 写法，或改用系统自带的 `curl.exe`。

**测试全过，但训练 loss 不降？**
优先检查三处：`--vocab_size` 是否大于等于数据中最大 token id、`.bin` 是否按 `uint16` 写入且未被截断、学习率是否过大导致发散（先把 `--lr` 降到 1e-4 试一轮）。

**验证 loss 明显高于训练 loss？**
小模型在 TinyStories 上过拟合是正常现象；差距持续扩大时考虑加数据量、降训练步数或增大 `weight_decay`。

---



---

## 说明

- 作者：晓来云绕天
- 本项目为个人学习实现，非课程官方仓库；作业说明见 `cs336_assignment1_basics.pdf`。
- 代码仅供学习参考，请遵循 CS336 的学术诚信政策，不要直接用于提交作业。
- 原始课程代码版权归 Stanford University 所有，许可条款见 [LICENSE](LICENSE)。
- 训练数据来自 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)，请遵守其数据集许可。
