"""加载训练好的 checkpoint，用 prompt 生成文本。

和 main_train.py 是一对：那边产出 ckpt_final.pt，这边读它做推理。
模型结构必须和训练时**完全一致**（层数、维度、头数、d_ff、vocab），
否则 load_state_dict 会因 key 或 shape 对不上直接报错——这也是为什么
这些超参在下面既有默认值、又允许命令行覆盖。

用法：
    uv run python -m cs336_basics.sample \
        --ckpt data/ckpt_tinystories/ckpt_final.pt \
        --prompt "Once upon a time" \
        --max_tokens 150 --temperature 0.8 --top_p 0.95
"""

import argparse

import torch

from cs336_basics.Inference import decode_token
from cs336_basics.main_train import TransformerLM
from cs336_basics.prepare_data import load_gpt2_style_tokenizer


def build_model(args, device):
    """按训练时的超参重建模型，再把权重灌进去。

    注意构造完还要 .to(device)：SwiGLU 内部用的是不带 device 的 nn.Linear，
    光给构造函数传 device 管不到它。
    """
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    # map_location="cpu"：先读进内存，再由 .to(device) 搬过去。
    # 直接读成 cuda 在单卡机器上没区别，多卡时会全压到 0 号卡。
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("iteration", -1)


def main():
    p = argparse.ArgumentParser(description="从 checkpoint 采样生成文本")
    p.add_argument("--ckpt", default="data/ckpt_tinystories/ckpt_final.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_tokens", type=int, default=150, help="最多新生成多少 token")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None, help="固定种子可复现同一条输出")

    # 下面这些必须和训练时一致，改动会直接 load 失败
    p.add_argument("--vocab", default="data/TinyStoriesV2-GPT4-train/vocab.json")
    p.add_argument("--merges", default="data/TinyStoriesV2-GPT4-train/merges.txt")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = load_gpt2_style_tokenizer(args.vocab, args.merges)
    model, iteration = build_model(args, args.device)
    print(f"checkpoint: {args.ckpt} (iteration {iteration})")

    # 特殊 token 的 id：官方词表里 <|endoftext|> 正好是最后一个（9999）。
    # 从词表里反查而不是写死，换了词表也不会错。
    eos_id = next(
        (i for i, b in tokenizer.vocab.items() if b == b"<|endoftext|>"),
        None,
    )

    prompt_ids = tokenizer.encode(args.prompt)
    out = decode_token(
        prompt_ids,
        model,
        max_tokens_to_generate=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        eos_token_id=eos_id,
        context_length=args.context_length,
    )

    print("-" * 70)
    print(tokenizer.decode(out[0].tolist()))
    print("-" * 70)
    print(f"prompt {len(prompt_ids)} token -> 总计 {out.shape[1]} token")


if __name__ == "__main__":
    main()
