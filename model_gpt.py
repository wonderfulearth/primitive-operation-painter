import torch
import torch.nn as nn
from torch.nn import functional as F

from token_layout import TOKEN_LAYOUT


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0, f"d_model {d_model} 必须能被 n_head {n_head} 整除！"
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.n_head = n_head
        self.d_model = d_model

        # QK-Norm 机制 (大模型防飞坡必备)
        self.q_ln = nn.LayerNorm(d_model // n_head)
        self.k_ln = nn.LayerNorm(d_model // n_head)

    def forward(self, x, past_key_value=None, use_cache=False):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_ln(q)
        k = self.k_ln(k)

        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_length = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        if past_length == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            # 增量解码时只有一个查询位置，所有 K/V 都属于当前或历史位置。
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # 支持一次追加多个 token 时的带偏移因果掩码。
            query_positions = torch.arange(T, device=x.device).unsqueeze(1)
            key_positions = torch.arange(past_length + T, device=x.device).unsqueeze(0)
            causal_mask = key_positions <= (past_length + query_positions)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        if use_cache:
            return y, (k, v)
        return y

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
    def forward(self, x, past_key_value=None, use_cache=False):
        attn_output = self.attn(self.ln_1(x), past_key_value, use_cache)
        if use_cache:
            attn_output, present_key_value = attn_output
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        if use_cache:
            return x, present_key_value
        return x

# ==========================================
# 逐 token 自回归几何 GPT
# ==========================================
class GeometrizeGPT(nn.Module):
    def __init__(
        self,
        vocab_size=TOKEN_LAYOUT.vocab_size,
        d_model=832,
        n_layer=12,
        n_head=16,
        max_context_len=2304,
        max_position_embeddings=2304,
    ):
        """逐 token 预测几何绘制序列。"""
        super().__init__()
        if max_context_len <= 0 or max_position_embeddings <= 0:
            raise ValueError("上下文长度和位置嵌入容量必须为正整数")
        if max_context_len > max_position_embeddings:
            raise ValueError(
                f"max_context_len={max_context_len} 不能超过位置嵌入容量 "
                f"max_position_embeddings={max_position_embeddings}"
            )
        self.max_context_len = max_context_len
        self.max_position_embeddings = max_position_embeddings
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_position_embeddings, d_model)
        self.h = nn.ModuleList([GPTBlock(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * len(self.h)) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        """
        Args:
            input_ids: [batch_size, sequence_length] 的整型 token 张量。
            past_key_values: 可选的逐层 (key, value) 缓存，仅用于推理增量解码。
            use_cache: 为 True 时，额外返回本轮更新后的 KV cache。
        Returns:
            use_cache=False 时返回 logits；否则返回 (logits, past_key_values)。
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids 必须是二维 [B, T] 张量，收到形状 {tuple(input_ids.shape)}"
            )

        _, seq_len = input_ids.shape
        if past_key_values is not None and len(past_key_values) != len(self.h):
            raise ValueError(
                f"past_key_values 应有 {len(self.h)} 层，实际收到 {len(past_key_values)} 层"
            )
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(2)
        if past_length + seq_len > self.max_context_len:
            raise ValueError(
                f"缓存长度 {past_length} + 当前长度 {seq_len} 超过 "
                f"max_context_len={self.max_context_len}"
            )

        positions = torch.arange(
            past_length, past_length + seq_len, dtype=torch.long, device=input_ids.device
        )
        x = self.wte(input_ids) + self.wpe(positions).unsqueeze(0)

        present_key_values = []
        for layer_idx, block in enumerate(self.h):
            layer_past = None if past_key_values is None else past_key_values[layer_idx]
            if use_cache:
                x, layer_present = block(x, layer_past, use_cache=True)
                present_key_values.append(layer_present)
            else:
                x = block(x, layer_past, use_cache=False)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if use_cache:
            return logits, tuple(present_key_values)
        return logits
