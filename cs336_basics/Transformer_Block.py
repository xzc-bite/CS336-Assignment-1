import torch
import torch.nn as nn

from cs336_basics.Causal_MultiHead_Self_Attention_with_rope import CausalMultiHeadAttentionWithRoPE
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    """
    Transformer 的一个基本层，pre-norm 结构。

    数据流（括号里是形状，B=batch，T=序列长）：

        x (B, T, d_model)
          ├─► RMSNorm ─► 因果多头注意力（带 RoPE） ─► (+) ─► y (B, T, d_model)
          │                                            ▲
          └──────────────────── 残差 ──────────────────┘

        y (B, T, d_model)
          ├─► RMSNorm ─► SwiGLU（升到 d_ff 再压回） ─► (+) ─► z (B, T, d_model)
          │                                            ▲
          └──────────────────── 残差 ──────────────────┘

    输入输出形状完全一样，所以这个块可以原样堆 N 层。

    "pre-norm" 指的是归一化放在子层之前、残差分支内部：
    主干上流的始终是未经归一化的原始数值，深层堆叠时尺度不会失控。
    代价是最后一层之后要补一个 RMSNorm，这个由外层的 LM 负责。

    Args:
        d_model: 隐藏维度。进出都是这个宽度，不存在"输入维度"的概念。
        n_heads: 注意力头数，必须整除 d_model，每个头分到 d_model // n_heads 维。
        d_ff:    SwiGLU 中间层宽度，通常 4 × d_model（LLaMA 用 8/3 倍）。
        max_seq_len: RoPE 位置表预计算长度；输入序列超过它会查表越界。
        theta:   RoPE 频率底数，常用 10000。
        attn_q/k/v/o_proj_weight: 注意力四个投影，(d_model, d_model)。
        ln1_weight / ln2_weight: 两个 RMSNorm 的 gain，(d_model,)。
        ffn_w1_weight: (d_ff, d_model)    ffn_w2_weight: (d_model, d_ff)
        ffn_w3_weight: (d_ff, d_model)
        device: 参数放在哪个设备上。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 attn_q_proj_weight: torch.Tensor, attn_k_proj_weight: torch.Tensor,
                 attn_v_proj_weight: torch.Tensor, attn_o_proj_weight: torch.Tensor,
                 ln1_weight: torch.Tensor, ln2_weight: torch.Tensor,
                 ffn_w1_weight: torch.Tensor, ffn_w2_weight: torch.Tensor,
                 ffn_w3_weight: torch.Tensor, device=None):
        super(TransformerBlock, self).__init__()

        # 权重是外部传进来的（评测就是从一个 state_dict 里拆出来喂进来），
        # 这里直接挂成普通属性。
        # 注意：普通 Tensor 不会被注册成 Parameter / buffer，
        # 所以 model.to("cuda") 不会搬它们，state_dict() 里也看不到。
        # 当前用法（每次新建模块、全程 CPU）没问题；
        # 如果要拿去真正训练，应该换成 nn.Parameter 或 register_buffer。
        self.attn_q_proj_weight = attn_q_proj_weight
        self.attn_k_proj_weight = attn_k_proj_weight
        self.attn_v_proj_weight = attn_v_proj_weight
        self.attn_o_proj_weight = attn_o_proj_weight

        self.ln1_weight = ln1_weight
        self.ln2_weight = ln2_weight

        self.ffn_w1_weight = ffn_w1_weight
        self.ffn_w2_weight = ffn_w2_weight
        self.ffn_w3_weight = ffn_w3_weight

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device

        # 两个 RMSNorm：一个在注意力前，一个在 FFN 前。eps 用讲义的默认 1e-5。
        # 用 load_state_dict 而不是直接 .weight = ...：
        # 前者会校验形状（对不上立刻报错），也不会把 Parameter 替换成普通 Tensor。
        self.rms_norm1 = RMSNorm(d_model, eps=1e-5, device=device)
        self.rms_norm1.load_state_dict({"weight": self.ln1_weight})

        self.rms_norm2 = RMSNorm(d_model, eps=1e-5, device=device)
        self.rms_norm2.load_state_dict({"weight": self.ln2_weight})

        # FFN：内部先把 d_model 升到 d_ff，乘完门控再压回 d_model。
        # 键名要和模块自己的参数名对上（w1.weight / w2.weight / w3.weight），
        # 写错会报 unexpected key，不会静默跳过。
        self.swiglu = SwiGLU(d_model, d_ff)
        self.swiglu.load_state_dict({
            "w1.weight": self.ffn_w1_weight,
            "w2.weight": self.ffn_w2_weight,
            "w3.weight": self.ffn_w3_weight,
        })

        # 注意力：RoPE 在内部完成，位置信息通过 forward 的 token_positions 传入
        self.causal_multi_head_attention = CausalMultiHeadAttentionWithRoPE(
            d_model, n_heads, max_seq_len, theta, device
        )

    def forward(self, in_features: torch.Tensor):
        # in_features: (B, T, d_model)
        # 整段前向一次算完，不复用 KV cache，所以位置就是从 0 开始的连续编号。
        # T 超过 max_seq_len 时，RoPE 查表会越界，这是硬约束。
        token_positions = torch.arange(in_features.shape[1], device=in_features.device)

        # ---- 子层一：归一化 → 注意力 → 残差 ----
        # 归一化在残差分支里面，主干上的数值不被改动
        x1 = self.rms_norm1(in_features)                 # (B, T, d_model)
        x1 = self.causal_multi_head_attention(
            x1,
            self.attn_q_proj_weight, self.attn_k_proj_weight,
            self.attn_v_proj_weight, self.attn_o_proj_weight,
            token_positions,
        )                                                # (B, T, d_model)
        # 残差加的是子层输入（in_features），不是归一化后的 x1——
        # 加错这一步，梯度主干就断了，深层会明显训不动
        x1 = x1 + in_features                            # (B, T, d_model)

        # ---- 子层二：归一化 → FFN → 残差 ----
        # 输入是 x1（已经加过一次残差的结果），不是最开始的 in_features
        x2 = self.rms_norm2(x1)                          # (B, T, d_model)
        x2 = self.swiglu(x2)                             # 内部升到 d_ff 再压回，形状不变
        out = x2 + x1                                    # (B, T, d_model)

        return out
