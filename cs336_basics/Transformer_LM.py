"""按外部 state_dict 前向的语言模型（给对拍/推理加载用）。

⚠️ 这个版本**不能训练**：权重以普通 Tensor 存放，且每层用 load_state_dict 拷进子模块
（拷贝过程 detach），所以 model.parameters() 是空的、梯度回传不到这些张量。
要训练请用 cs336_basics/main_train.py 里的 TransformerLM——零件相同，
但所有参数都注册成 nn.Parameter。改结构时两边都要看一眼，别只改一边。
"""

import torch
import torch.nn as nn

from cs336_basics.Transformer_Block import TransformerBlock
from cs336_basics.Linear_Embedding_model import LinearModule, EmbeddingModule
from cs336_basics.RMSNorm import RMSNorm


class TransformerLM(nn.Module):
    """
    decoder-only 语言模型：若干 TransformerBlock 串起来，两头接嵌入层和输出层。

    整条数据流（B = batch，T = 序列长度）：

        token id      (B, T)                int64，取值 [0, vocab_size)
          │ 查表  token_embeddings.weight   (vocab_size, d_model)
          ▼
        向量          (B, T, d_model)
          │ × num_layers 个 TransformerBlock（层内形状不变）
          ▼
        向量          (B, T, d_model)
          │ ln_final                        (d_model,)
          ▼
        向量          (B, T, d_model)
          │ lm_head                         (vocab_size, d_model)
          ▼
        logits        (B, T, vocab_size)    每个位置对全词表的原始打分

    输出故意不做 softmax：交叉熵损失要的是 logits，
    推理阶段再按需要自己 softmax 或采样。

    Args:
        vocab_size:     词表大小，决定嵌入表行数和输出维度。
        context_length: 最大上下文长度，同时被当成 RoPE 位置表的长度。
                        输入的 T 不能超过它，否则 RoPE 查表越界。
        d_model:        隐藏维度，所有中间张量的最后一维。
        num_layers:     TransformerBlock 的层数。
        num_heads:      每层注意力头数，必须整除 d_model。
        d_ff:           FFN 中间层宽度，一般 4 × d_model。
        rope_theta:     RoPE 频率底数，常用 10000。
        weights:        参考实现的 state_dict，键名形如
                        "token_embeddings.weight"、"layers.0.attn.q_proj.weight"、
                        "ln_final.weight"、"lm_head.weight"。
    """

    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, rope_theta: float,
                 weights: dict[str, torch.Tensor]):
        super().__init__()
        # 超参数全部记下来，forward 里建模块要用
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # 权重字典原样存着。普通 dict 不会被当成参数，
        # 所以 model.parameters() / state_dict() 里都看不到它——
        # 这个类目前是"每次前向重建模块 + 现场加载权重"的用法。
        self.weights = weights

    def forward(self, in_indices):
        # in_indices: (B, T) 整数张量，dtype 必须是 int64（PyTorch 索引要求）
        # T 允许小于 context_length（测试里有截断输入的用例），
        # 所以后面取位置时要用实际长度，不能写死 context_length。
        embedding_module = EmbeddingModule(self.vocab_size, self.d_model, device=None)
        # EmbeddingModule 的参数名叫 embedding_matrix，
        # 所以 state_dict 的键是这个，而不是 nn.Embedding 的 "weight"
        embedding_module.load_state_dict({"embedding_matrix": self.weights["token_embeddings.weight"]})
        # (B, T) -> (B, T, d_model)，每个 id 换成 d_model 维的一行
        embedding = embedding_module(in_indices)

        # 逐层过 Block。每层从 weights 里取自己那 9 份权重，
        # 层号直接拼进键名——层号写错不会报错，只会静默算错，改代码时要盯住 layer。
        for layer in range(self.num_layers):
            attn_q_proj_weight = self.weights[f"layers.{layer}.attn.q_proj.weight"]      # (d_model, d_model)
            attn_k_proj_weight = self.weights[f"layers.{layer}.attn.k_proj.weight"]
            attn_v_proj_weight = self.weights[f"layers.{layer}.attn.v_proj.weight"]
            attn_o_proj_weight = self.weights[f"layers.{layer}.attn.output_proj.weight"]  # 键叫 output_proj
            ln1_weight = self.weights[f"layers.{layer}.ln1.weight"]                       # (d_model,)
            ln2_weight = self.weights[f"layers.{layer}.ln2.weight"]
            ffn_w1_weight = self.weights[f"layers.{layer}.ffn.w1.weight"]                 # (d_ff, d_model)
            ffn_w2_weight = self.weights[f"layers.{layer}.ffn.w2.weight"]                 # (d_model, d_ff)
            ffn_w3_weight = self.weights[f"layers.{layer}.ffn.w3.weight"]                 # (d_ff, d_model)

            # 位置参数顺序必须和 TransformerBlock.__init__ 完全一致：
            # (d_model, n_heads, d_ff, max_seq_len, theta, 4×attn, 2×ln, 3×ffn, device)
            # 其中 context_length 传进去当 max_seq_len 用，决定 RoPE 位置表长度
            transformer_block = TransformerBlock(
                self.d_model, self.num_heads, self.d_ff, self.context_length, self.rope_theta,
                attn_q_proj_weight, attn_k_proj_weight, attn_v_proj_weight, attn_o_proj_weight,
                ln1_weight, ln2_weight, ffn_w1_weight, ffn_w2_weight, ffn_w3_weight,
                device=None,
            )
            # (B, T, d_model) -> (B, T, d_model)：层内形状不变，只有数值被更新
            embedding = transformer_block(embedding)

        transformer_block_out = embedding

        # pre-norm 结构里每层都是"先归一化再进子层"，主干上流的是没归一化的数值，
        # 所以最后一次归一化必须在这里补，否则输出量级会随层数漂移
        rms_norm = RMSNorm(self.d_model, eps=1e-5, device=None)
        rms_norm.load_state_dict({"weight": self.weights["ln_final.weight"]})
        out_norm = rms_norm(transformer_block_out)          # (B, T, d_model)，形状不变

        # 输出投影：把 d_model 维压成词表大小，得到每个位置对每个词的打分
        # LinearModule 的参数直接叫 W（不是 nn.Linear 子模块），所以键是 "W"
        linear_module = LinearModule(self.d_model, self.vocab_size, device=None)
        linear_module.load_state_dict({"W": self.weights["lm_head.weight"]})
        out_linear = linear_module(out_norm)               # (B, T, d_model) -> (B, T, vocab_size)

        # 输出停在 logits。训练时交叉熵直接吃它（内部会做 log-softmax），
        # 推理时再按需 softmax / 采样——提前 softmax 反而会让交叉熵的数值不稳定
        return out_linear
