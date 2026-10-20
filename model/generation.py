"""自回归生成引擎（带 KV cache、左侧 padding、批量采样）。

设计要点
--------
* **KV cache**：prefill 一次算完整 prompt，之后每步只算 1 个 token，复杂度 O(T) 而不是 O(T^2)。
* **左侧 padding + 显式 attention mask**：批量生成时不同 prompt 长度不同，左补 pad 会让
  模型看到假的 token，所以必须传 mask 把 padding 位置屏蔽掉。
* **采样可复现**：``seed`` 固定时，同一 prompt + 同一参数 -> 同一输出（测试里验证这点）。
"""

from __future__ import annotations

import torch

from model.sampling import sample_next


def _causal_mask(T_q: int, T_k: int, offset: int, device) -> torch.Tensor:
    """(T_q, T_k) 的下三角因果 mask（key 的绝对位置从 offset 开始）。"""
    q_pos = torch.arange(T_q, device=device).unsqueeze(1) + offset
    k_pos = torch.arange(T_k, device=device).unsqueeze(0)
    return k_pos <= q_pos


@torch.no_grad()
def generate_batch(
    model,
    prompt_ids: list[list[int]],
    max_new_tokens: int = 60,
    temperature: float = 0.9,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    eos_id: int | None = None,
    pad_id: int = 0,
) -> list[list[int]]:
    """批量生成。返回每个 prompt 对应的 **新生成** 的 token id 列表。"""
    if not prompt_ids:
        return []
    device = torch.device(device)
    model.eval()
    B = len(prompt_ids)
    maxlen = max(len(p) for p in prompt_ids)
    # 左侧 padding
    inp = torch.full((B, maxlen), pad_id, dtype=torch.long)
    valid = torch.zeros((B, maxlen), dtype=torch.bool)
    for i, p in enumerate(prompt_ids):
        inp[i, maxlen - len(p):] = torch.tensor(p, dtype=torch.long)
        valid[i, maxlen - len(p):] = True

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    attn = valid.to(device).unsqueeze(1).unsqueeze(2)          # (B,1,1,T)
    causal = _causal_mask(maxlen, maxlen, 0, device)            # (T,T)
    mask = attn & causal.unsqueeze(0)                           # (B,1,T,T)
    logits, cache = model(inp.to(device), past=None, use_cache=True, attn_mask=mask)
    next_logits = logits[:, -1, :]

    # 有效位置掩码（decode 时每步往后加一列：新生成的 token 总是可见的）
    valid_full = valid.to(device)

    out: list[list[int]] = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        tok = sample_next(next_logits, temperature, top_k, top_p, gen)
        tok = torch.where(finished, torch.full_like(tok, pad_id), tok)
        for i in range(B):
            if not finished[i]:
                out[i].append(int(tok[i].item()))
        if eos_id is not None:
            finished = finished | (tok == eos_id)
        if bool(finished.all()):
            break
        # 增量解码：只喂最后一个 token
        valid_full = torch.cat(
            [valid_full, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1
        )
        step_mask = valid_full.unsqueeze(1).unsqueeze(2)        # (B,1,1,T_k)
        logits, cache = model(tok.view(B, 1), past=cache, use_cache=True, attn_mask=step_mask)
        next_logits = logits[:, -1, :]
    return out
