"""模型单测：形状、参数量、权重共享、因果性、梯度能流、KV cache 一致性。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.transformer import ModelConfig, TinyTransformer, build_rope_cache  # noqa: E402


def tiny_config(**kw) -> ModelConfig:
    base = dict(
        vocab_size=128, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
        d_ff=128, max_seq_len=64,
    )
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shape():
    cfg = tiny_config()
    m = TinyTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 17))
    y = m(x)
    assert y.shape == (3, 17, cfg.vocab_size)


def test_parameter_count_and_breakdown():
    cfg = tiny_config()
    m = TinyTransformer(cfg)
    bd = m.parameter_breakdown()
    total = sum(p.numel() for p in m.parameters())
    assert bd["total"] == total
    assert bd["embedding"] == cfg.vocab_size * cfg.d_model
    assert bd["non_embedding"] == total - bd["embedding"]
    assert len(bd["per_layer"]) == cfg.n_layer


def test_weight_tying():
    """embedding 与 lm_head 共享同一份参数。"""
    cfg = tiny_config()
    m = TinyTransformer(cfg)
    assert m.lm_head.weight is m.tok_emb.weight


def test_no_untied_duplication_when_tying_enabled():
    cfg = tiny_config(tie_embeddings=True)
    m = TinyTransformer(cfg)
    unique = {id(p) for p in m.parameters()}
    assert len(unique) == len(list(m.parameters()))


def test_causality_future_tokens_do_not_affect_past():
    """因果性核心测试：改动位置 k 之后的 token，位置 <=k 的 logits 必须完全不变。"""
    torch.manual_seed(0)
    cfg = tiny_config()
    m = TinyTransformer(cfg).eval()
    T = 24
    x = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        base = m(x)
    # 改动后半段
    x2 = x.clone()
    x2[0, 12:] = (x2[0, 12:] + 7) % cfg.vocab_size
    with torch.no_grad():
        mod = m(x2)
    assert torch.allclose(base[0, :12], mod[0, :12], atol=1e-5), "未来 token 影响了过去的输出！"
    # 后半段应该确实变了（否则说明改动没生效，测试是假的）
    assert not torch.allclose(base[0, 12:], mod[0, 12:], atol=1e-5)


def test_causality_with_gqa_and_mha_both():
    for n_kv in (4, 2):
        torch.manual_seed(1)
        cfg = tiny_config(n_kv_head=n_kv)
        m = TinyTransformer(cfg).eval()
        x = torch.randint(0, cfg.vocab_size, (1, 16))
        with torch.no_grad():
            base = m(x)
            x2 = x.clone()
            x2[0, 8:] = (x2[0, 8:] + 3) % cfg.vocab_size
            mod = m(x2)
        assert torch.allclose(base[0, :8], mod[0, :8], atol=1e-5)


def test_gradients_flow_to_all_parameters():
    """梯度能流：每个可训练参数都必须拿到非 None 的梯度。"""
    cfg = tiny_config()
    m = TinyTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(x)
    loss = out.float().pow(2).mean()
    loss.backward()
    missing = []
    for name, p in m.named_parameters():
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
            missing.append(name)
    assert not missing, f"这些参数没有有效梯度: {missing}"


def test_embedding_grad_is_nonzero():
    cfg = tiny_config()
    m = TinyTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    m(x).float().pow(2).mean().backward()
    assert m.tok_emb.weight.grad.abs().sum().item() > 0


def test_rope_shapes_and_rotation_property():
    cos, sin = build_rope_cache(16, 8, theta=10000.0)
    assert cos.shape == (16, 4)
    assert sin.shape == (16, 4)
    assert torch.allclose(cos[0], torch.ones(4), atol=1e-6)
    assert torch.allclose(sin[0], torch.zeros(4), atol=1e-6)


def test_rmsnorm_normalizes():
    from model.transformer import RMSNorm

    n = RMSNorm(8)
    x = torch.randn(2, 3, 8) * 10
    y = n(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_kv_cache_matches_full_forward():
    """增量解码（KV cache）必须与一次性整段前向给出同样的下一步 logits。"""
    torch.manual_seed(2)
    cfg = tiny_config()
    m = TinyTransformer(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        full = m(ids)
        # 先 prefill 前 9 个
        logits, cache = m(ids[:, :9], past=None, use_cache=True)
        # 再喂第 10 个
        logits2, cache = m(ids[:, 9:10], past=cache, use_cache=True)
    assert torch.allclose(full[0, 9], logits2[0, 0], atol=1e-4)


def test_parameter_count_under_budget_for_project_config():
    """项目实际配置必须在 20M 参数预算内。"""
    cfg = ModelConfig(vocab_size=16000, d_model=384, n_layer=6, n_head=6,
                      n_kv_head=2, d_ff=1024, max_seq_len=512)
    m = TinyTransformer(cfg)
    assert m.num_parameters() < 20_000_000, f"参数量超标: {m.num_parameters()}"


def test_config_roundtrip_json(tmp_path):
    cfg = tiny_config()
    m = TinyTransformer(cfg)
    p = tmp_path / "cfg.json"
    m.save_config(p)
    m2 = TinyTransformer.from_config_file(p)
    assert m2.cfg.to_dict() == cfg.to_dict()


def test_generation_is_deterministic_with_seed():
    from model.generation import generate_batch

    torch.manual_seed(3)
    cfg = tiny_config()
    m = TinyTransformer(cfg).eval()
    prompts = [[1, 2, 3], [4, 5]]
    a = generate_batch(m, prompts, max_new_tokens=8, temperature=0.9, top_k=10,
                        top_p=0.9, seed=42, device=torch.device("cpu"))
    b = generate_batch(m, prompts, max_new_tokens=8, temperature=0.9, top_k=10,
                        top_p=0.9, seed=42, device=torch.device("cpu"))
    assert a == b
    c = generate_batch(m, prompts, max_new_tokens=8, temperature=0.9, top_k=10,
                        top_p=0.9, seed=43, device=torch.device("cpu"))
    assert a != c


def test_generation_padding_does_not_change_short_prompt_result():
    """批量生成里，左侧 padding 不应影响该条 prompt 的结果。"""
    from model.generation import generate_batch

    torch.manual_seed(4)
    cfg = tiny_config()
    m = TinyTransformer(cfg).eval()
    alone = generate_batch(m, [[7, 8, 9]], max_new_tokens=6, temperature=0.0,
                           seed=11, device=torch.device("cpu"))[0]
    together = generate_batch(m, [[7, 8, 9], [1, 2, 3, 4, 5]], max_new_tokens=6,
                              temperature=0.0, seed=11, device=torch.device("cpu"))[0]
    assert alone == together
