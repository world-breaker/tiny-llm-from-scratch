"""从零实现的 decoder-only Transformer（不引入任何 HuggingFace 层 / 预训练权重）。

包含
----
* **RMSNorm**：均方根归一化（去掉均值中心化，省一次减法、训练更稳）。
* **RoPE**：旋转位置编码，按 head 维度对 q/k 做旋转，实现相对位置感知的注意力。
* **因果自注意力**：可选 **GQA**（多 query 头共享 kv 头），省 KV cache 显存。
* **SwiGLU FFN**：``silu(gate) * up`` 再线性降维，比标准 FFN 效果更好、参数更省。
* **embedding 与 lm_head 权重共享**：小词表 + 小模型时这是最划算的一招。
* **残差投影缩放初始化**（GPT-2 风格 1/sqrt(2L)）：让深一点的堆叠也能稳定训练。
* **梯度检查点**：显存不够时按层重算激活，显存换算力。

所有实现只依赖 ``torch.nn`` / ``torch.nn.functional``，不 import 任何 transformers 组件。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    """模型结构配置。"""

    vocab_size: int = 16000
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2          # == n_head 即 MHA；< n_head 即 GQA
    d_ff: int = 1024            # SwiGLU 中间维
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class RMSNorm(nn.Module):
    """RMSNorm：``x / sqrt(mean(x^2) + eps) * weight``。"""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(dtype)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 RoPE 的 cos/sin 表，形状 (seq_len, head_dim/2)。"""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (T, D/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 (B, H, T, D) 的 q/k 应用旋转位置编码。"""
    T = x.size(-2)
    cos = cos[:T].to(x.dtype)
    sin = sin[:T].to(x.dtype)
    # (T, D/2) -> (1, 1, T, D/2) -> 偶/奇拆分
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    """因果自注意力，支持 GQA（n_kv_head < n_head）。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if past_kv is not None:
            # 增量解码：把历史 kv 接上
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)
        # prefill（无 past）走 is_causal；增量解码只有 1 个 query，可见全部历史，无需 mask
        causal = past_kv is None
        dropout_p = self.dropout if self.training else 0.0
        if attn_mask is not None:
            # 显式 mask（padding + 因果），此时不能再用 is_causal
            try:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, enable_gqa=(self.n_rep > 1), dropout_p=dropout_p
                )
            except (TypeError, RuntimeError):
                if self.n_rep > 1:
                    k = k.repeat_interleave(self.n_rep, dim=1)
                    v = v.repeat_interleave(self.n_rep, dim=1)
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
                )
        elif self.n_rep > 1:
            # GQA：把 kv 头复制 n_rep 份（用 enable_gqa 让 SDPA 内部做，省显存）
            try:
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=causal, enable_gqa=True, dropout_p=dropout_p,
                )
            except (TypeError, RuntimeError):
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=causal, dropout_p=dropout_p,
                )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, dropout_p=dropout_p,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.o_proj(y), new_kv


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络：``down(silu(gate(x)) * up(x))``。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block：x + Attn(RMSNorm(x))，再 x + FFN(RMSNorm(x))。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, kv = self.attn(self.attn_norm(x), cos, sin, past_kv, attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv


class TinyTransformer(nn.Module):
    """decoder-only 预训练语言模型。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # 权重共享：lm_head 直接复用 embedding 的参数
            self.lm_head.weight = self.tok_emb.weight
        self.grad_checkpointing = False
        self.apply(self._init_weights)
        # 残差输出投影按 1/sqrt(2L) 缩放，提升堆叠稳定性
        resid_scale = 1.0 / math.sqrt(2.0 * cfg.n_layer)
        for block in self.blocks:
            with torch.no_grad():
                block.attn.o_proj.weight.mul_(resid_scale)
                block.ffn.down_proj.weight.mul_(resid_scale)

    # -- 初始化 ------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # -- 前向 --------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        attn_mask: torch.Tensor | None = None,
    ):
        """idx: (B, T) -> logits: (B, T, vocab)。

        ``use_cache=True`` 时额外返回每层的 (k, v) 缓存，供自回归生成使用。
        增量解码约定：``past`` 非空时 ``T`` 必须为 1（逐 token 解码）。
        """
        B, T = idx.shape
        offset = 0
        if past is not None and len(past) > 0:
            offset = past[0][0].size(2)
        cos, sin = self._rope_for(offset + T, idx.device)
        cos = cos[offset:offset + T]
        sin = sin[offset:offset + T]
        x = self.emb_drop(self.tok_emb(idx))
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            pkv = past[i] if past is not None and i < len(past) else None
            if self.grad_checkpointing and self.training:
                x, kv = checkpoint(block, x, cos, sin, pkv, attn_mask, use_reentrant=False)
            else:
                x, kv = block(x, cos, sin, pkv, attn_mask)
            if use_cache:
                new_cache.append(kv)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if use_cache:
            return logits, new_cache
        return logits

    def _rope_for(self, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_rope_cache", None)
        if cache is None or cache[0].size(0) < T or cache[0].device != device:
            cos, sin = build_rope_cache(
                max(T, self.cfg.max_seq_len), self.cfg.head_dim, self.cfg.rope_theta, device
            )
            self._rope_cache = (cos, sin)
        return self._rope_cache

    # -- 工具 --------------------------------------------------------------
    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def parameter_breakdown(self) -> dict:
        emb = self.tok_emb.weight.numel()
        total = self.num_parameters()
        per_layer = []
        for block in self.blocks:
            per_layer.append(sum(p.numel() for p in block.parameters()))
        return {
            "total": total,
            "embedding": emb,
            "non_embedding": total - emb,
            "per_layer": per_layer,
            "layers": self.cfg.n_layer,
        }

    def enable_gradient_checkpointing(self, flag: bool = True) -> None:
        self.grad_checkpointing = flag

    def save_config(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_config_file(cls, path: str | Path) -> "TinyTransformer":
        cfg = ModelConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        return cls(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
