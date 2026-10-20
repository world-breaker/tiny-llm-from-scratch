"""训练与采样单测：一步训练 loss 会降、梯度裁剪生效、采样策略行为正确且可复现。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.sampling import sample_next, top_k_filter, top_p_filter  # noqa: E402
from model.transformer import ModelConfig, TinyTransformer  # noqa: E402


def tiny_config(**kw) -> ModelConfig:
    base = dict(vocab_size=64, d_model=32, n_layer=2, n_head=4, n_kv_head=2,
                d_ff=64, max_seq_len=32)
    base.update(kw)
    return ModelConfig(**base)


def test_one_training_step_reduces_loss():
    """在同一个 batch 上多步优化，loss 必须下降（训练确实在学）。"""
    torch.manual_seed(0)
    cfg = tiny_config()
    m = TinyTransformer(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95))
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    def loss_now() -> float:
        with torch.no_grad():
            logits = m(x)
            return F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(), y.reshape(-1)).item()

    first = loss_now()
    for _ in range(15):
        opt.zero_grad(set_to_none=True)
        logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    last = loss_now()
    assert last < first, f"训练后 loss 没有下降: {first:.4f} -> {last:.4f}"


def test_grad_clip_limits_norm():
    """梯度裁剪后，全局范数不超过阈值。"""
    torch.manual_seed(1)
    cfg = tiny_config()
    m = TinyTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    m(x).float().pow(2).mean().backward()
    total = torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
    assert float(total) <= 0.5 + 1e-5


def test_overfit_single_batch():
    """能在单个 batch 上过拟合（loss 接近 0），说明优化链路是通的。"""
    torch.manual_seed(2)
    cfg = tiny_config(vocab_size=32)
    m = TinyTransformer(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = torch.randint(0, cfg.vocab_size, (1, 8))
    losses = []
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_top_k_filter_keeps_only_k_largest():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    out = top_k_filter(logits, 2)
    finite = torch.isfinite(out[0]).nonzero().flatten().tolist()
    assert finite == [1, 4]


def test_top_p_filter_keeps_nucleus():
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
    out = top_p_filter(logits, 0.5)
    assert torch.isfinite(out[0, 0])
    assert not torch.isfinite(out[0, 1])


def test_temperature_zero_is_greedy():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    out = sample_next(logits, temperature=0.0)
    assert int(out[0]) == 1


def test_sampling_reproducible_with_same_seed():
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = sample_next(logits, temperature=1.0, top_k=20, top_p=0.9, generator=g1)
    b = sample_next(logits, temperature=1.0, top_k=20, top_p=0.9, generator=g2)
    assert torch.equal(a, b)


def test_sampling_different_seed_gives_different_results():
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    a = sample_next(logits, temperature=1.5, top_k=50, top_p=1.0, generator=g1)
    b = sample_next(logits, temperature=1.5, top_k=50, top_p=1.0, generator=g2)
    assert not torch.equal(a, b)


def test_lower_temperature_is_more_deterministic():
    """低温采样应集中在同一个 token 上（熵更低）。"""
    torch.manual_seed(0)
    logits = torch.randn(1, 200) * 0.5
    counts = []
    for t in (0.1, 2.0):
        picks = []
        for s in range(40):
            g = torch.Generator().manual_seed(s)
            picks.append(int(sample_next(logits, temperature=t, generator=g)[0]))
        counts.append(len(set(picks)))
    assert counts[0] < counts[1], f"低温应该更集中: {counts}"


def test_checkpoint_save_load_restores_weights(tmp_path):
    from train import set_seed  # 复用训练脚本里的种子设置

    set_seed(7)
    cfg = tiny_config()
    m = TinyTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    before = m(x).detach().clone()
    ck = {"model": m.state_dict(), "config": cfg.to_dict(), "step": 5, "tokens_seen": 100}
    p = tmp_path / "ck.pt"
    torch.save(ck, p)
    loaded = torch.load(p, map_location="cpu", weights_only=False)
    m2 = TinyTransformer(ModelConfig.from_dict(loaded["config"]))
    m2.load_state_dict(loaded["model"])
    assert torch.allclose(before, m2(x), atol=1e-6)
    assert loaded["step"] == 5


def test_lr_schedule_warmup_then_cosine():
    from train import build_lr_lambda

    fn = build_lr_lambda(warmup=10, total=100, min_ratio=0.1)
    assert fn(0) < 0.2                       # warmup 起点很小
    assert abs(fn(9) - 1.0) < 1e-9          # warmup 结束到 1.0
    assert abs(fn(100) - 0.1) < 1e-9        # cosine 收敛到 min_ratio
    assert fn(10) > fn(50) > fn(99)         # 单调下降
