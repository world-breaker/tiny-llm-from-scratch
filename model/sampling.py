"""采样策略：温度 / top-k / top-p（nucleus），支持固定随机种子以便复现。

三种控制手段的作用不同，通常一起用：

* **temperature**：缩放 logits。T<1 让分布更尖（更保守、更"像训练语料"）；T>1 更随机；
  T→0 等价于贪心解码。
* **top-k**：只在概率最高的 k 个 token 里采样，硬性砍掉长尾。
* **top-p（nucleus）**：按概率降序累加，取累计概率刚超过 p 的最小集合，
  集合大小随分布自适应 —— 分布尖时集合小，分布平时集合大，比 top-k 更自然。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """nucleus 过滤：保留累计概率 >= p 的最小 token 集合。"""
    if p is None or p <= 0.0 or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    # 找到第一个累计概率超过 p 的位置，其后（含）都保留
    remove = cum - probs > p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """从 (B, V) logits 采样出下一个 token，返回 (B,) 的 LongTensor。"""
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    logits = top_k_filter(logits, top_k or 0)
    logits = top_p_filter(logits, top_p if top_p is not None else 1.0)
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    if probs.sum(dim=-1, keepdim=True).min().item() <= 0:
        # 极端情况：全被 mask 掉了 -> 退回 argmax
        return logits.argmax(dim=-1)
    idx = torch.multinomial(probs, num_samples=1, generator=generator)
    return idx.squeeze(-1)
