"""把原始文本编码成 uint16 的 token 二进制文件，供训练时 memmap 随机切片。

为什么要先落盘成 .bin：
    训练每一步都要随机切一段连续的 token 当样本。如果每次都现场做 BPE 编码，
    编码速度（纯 Python 约 1.5 MB/s）会完全盖过 GPU，训练循环基本空转。
    先编码一次、之后 memmap 按页读，是最省事的做法。

文件格式：
    裸 uint16 小端序，没有头部、没有元数据。用的时候直接
        np.memmap(path, dtype=np.uint16, mode="r")
    这也是讲义和官方 tokenizer 的约定，别自己加 .npy 头——
    加了之后 memmap 读出来的是 npy 的魔数和形状，不是 token。

用法：
    # 官方 TinyStories tokenizer 是 GPT-2 风格（vocab.json 是 str->str），
    # 需要先转成 {id: bytes}，脚本里的 load_gpt2_style_tokenizer 负责这一步
    python -m cs336_basics.prepare_data \\
        --text  data/TinyStoriesV2-GPT4-valid.txt \\
        --out   data/valid.bin \\
        --vocab data/TinyStoriesV2-GPT4-train/vocab.json \\
        --merges data/TinyStoriesV2-GPT4-train/merges.txt

    # 训练集很大时用 --max_mb 限制处理量，先跑通再放开
    python -m cs336_basics.prepare_data --text data/TinyStoriesV2-GPT4-train.txt \\
        --out data/train.bin --max_mb 200 ...
"""

import argparse
import os
import time

import numpy as np

from cs336_basics.Tokenizer import BPETokenizer


def load_gpt2_style_tokenizer(vocab_path, merges_path, special_tokens=("<|endoftext|>",)):
    """读 GPT-2 格式的 vocab.json + merges.txt，返回 BPETokenizer。

    真正的解析逻辑在 BPETokenizer.from_files 里（讲义要求的那个类方法），
    这里只是一层薄封装，把默认的特殊 token 补上，免得调用方每次都要写。

    参数:
        vocab_path: json 文件，形如 {"0": "Ā", ..., "9999": "<|endoftext|>"}，
                    即 id -> byte-level 字符串。
        merges_path: 每行两列，行号就是合并优先级。
        special_tokens: 不参与切分的特殊标记；官方词表里 <|endoftext|> 已经占了
                        最后一个 id（9999），传它是为了编码时整体保留、不被拆碎。

    返回:
        可直接 encode/decode 的 BPETokenizer 实例。
    """
    return BPETokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=list(special_tokens),
    )


def tokenize_file(in_path, out_path, tokenizer, max_chars=None, flush_tokens=20_000_000, log_every=200_000):
    """把文本文件流式编码成 uint16 二进制，返回写入的 token 数。

    参数:
        in_path: 输入文本，utf-8。按行读，不整文件加载——
                 2 GB 的 TinyStories 全读进来会把内存吃满。
        out_path: 输出 .bin。父目录必须已存在。
        tokenizer: 带 encode_iterable 的分词器。
        max_chars: 最多处理多少个字符（None 表示整个文件）。
                   用来先跑一小段验证流程，或者控制语料规模。
        flush_tokens: 攒够多少 token 落一次盘。太小会频繁 IO，
                      太大则内存里堆一个很大的 list。
        log_every: 每处理多少字符打印一次进度。

    返回:
        int，实际写出的 token 数。

    复杂度:
        时间由 BPE 编码主导，纯 Python 实现约 1.5 MB/s；
        空间 O(flush_tokens)，因为只在内存里攒一小批就写走。
    """
    written = 0
    consumed_chars = 0
    buffer = []          # 攒 token 用，元素是 int
    last_log = 0
    t0 = time.time()

    def flush(fh):
        nonlocal written, buffer
        if not buffer:
            return
        # 转 uint16：词表只有 10000 个 token，uint16 足够，
        # 相比 int64 省 4 倍磁盘和 IO
        np.asarray(buffer, dtype=np.uint16).tofile(fh)
        written += len(buffer)
        buffer = []

    with open(in_path, encoding="utf-8") as fin, open(out_path, "wb") as fout:
        for line in fin:
            if max_chars is not None and consumed_chars >= max_chars:
                break
            consumed_chars += len(line)

            # encode_iterable 内部就是逐行编码，返回一个生成器，
            # 这里再按行取出来是为了能随时中断（max_chars）和打进度
            for tid in tokenizer.encode_iterable([line]):
                buffer.append(tid)
            if len(buffer) >= flush_tokens:
                flush(fout)

            if consumed_chars - last_log >= log_every:
                last_log = consumed_chars
                elapsed = time.time() - t0
                print(
                    f"  {consumed_chars/1e6:7.2f} M 字符 -> {(written+len(buffer))/1e6:7.2f} M token"
                    f"  ({consumed_chars/1e6/max(elapsed,1e-9):.2f} MB/s)",
                    flush=True,
                )

        flush(fout)

    dt = time.time() - t0
    print(f"完成: {consumed_chars/1e6:.2f} M 字符 -> {written/1e6:.2f} M token，耗时 {dt:.1f}s")
    return written


def main():
    p = argparse.ArgumentParser(description="文本 -> uint16 token 二进制")
    p.add_argument("--text", required=True, help="输入文本文件（utf-8）")
    p.add_argument("--out", required=True, help="输出 .bin 路径")
    p.add_argument("--vocab", default="data/TinyStoriesV2-GPT4-train/vocab.json")
    p.add_argument("--merges", default="data/TinyStoriesV2-GPT4-train/merges.txt")
    p.add_argument("--max_mb", type=float, default=None, help="最多处理多少 MB 字符，默认全量")
    p.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])
    args = p.parse_args()

    # 用 abspath 兜底：--out 写成 "train.bin"（不带目录）时 dirname 是空串，
    # 直接 makedirs("") 会抛 FileNotFoundError。
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"加载 tokenizer: {args.vocab}")
    tok = load_gpt2_style_tokenizer(args.vocab, args.merges, tuple(args.special_tokens))
    max_chars = int(args.max_mb * 1_000_000) if args.max_mb else None

    n = tokenize_file(args.text, args.out, tok, max_chars=max_chars)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"写出 {args.out}: {n/1e6:.2f} M token, {size_mb:.1f} MB (uint16)")


if __name__ == "__main__":
    main()
