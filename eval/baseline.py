"""评测：模型 vs 诚实基线。

基线
----
* **均匀分布基线**：``P(t) = 1/V``。任何模型只要 loss 低于 ``ln(V)`` 就比它强。
  这是最弱的下限，用来证明「模型确实学到了东西，而不是碰运气」。
* **一元（unigram）基线**：只统计每个 token 的边缘频率，不看上下文。
* **二元（bigram）基线**：``P(t_{i+1} | t_i)``，加 alpha 平滑。
  这是「只看上一个 token」的 n-gram 模型 —— 比均匀分布强得多，是本文档的主要对手。

**关键诚实性约束**
1. 基线在 **训练集** 上统计，再在 **验证集** 上评测；不能拿验证集统计自己（那是作弊）。
2. 模型与所有基线在 **完全相同的 token 位置** 上评测（同一批窗口、同样的预测个数），
   loss 才有可比性。
3. bigram 只有 1 个 token 的上下文，而模型有 512 个。这个差距必须写进报告，
   不能拿「模型赢了一个很弱的对手」当成大结论。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.token_dataset import TokenStream  # noqa: E402
from model.transformer import ModelConfig, TinyTransformer  # noqa: E402


# ---------------------------------------------------------------------------
# 评测窗口（模型与基线共用，保证可比）
# ---------------------------------------------------------------------------
def build_eval_windows(stream: TokenStream, batch_size: int, seq_len: int, n_batches: int) -> np.ndarray:
    """取验证集最前面的 n_batches 个不重叠窗口 -> (N, seq_len+1)。"""
    out = []
    start = 0
    for _ in range(n_batches):
        need = batch_size * (seq_len + 1)
        if start + need > stream.n:
            break
        chunk = np.array(stream.data[start:start + need], dtype=np.int64)
        out.append(chunk.reshape(batch_size, seq_len + 1))
        start += need
    if not out:
        raise ValueError("验证集长度不足以取一个窗口")
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 基线统计
# ---------------------------------------------------------------------------
def build_unigram(stream: TokenStream, vocab_size: int, max_tokens: int = 0) -> np.ndarray:
    """统计训练集里每个 token 的出现次数（分块 np.bincount，内存友好）。"""
    counts = np.zeros(vocab_size, dtype=np.int64)
    limit = stream.n if max_tokens <= 0 else min(stream.n, max_tokens)
    CH = 1 << 21
    for s in range(0, limit, CH):
        chunk = np.asarray(stream.data[s:min(s + CH, limit)], dtype=np.int64)
        counts += np.bincount(chunk, minlength=vocab_size)
    return counts


def build_bigram(stream: TokenStream, vocab_size: int, max_tokens: int = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """统计训练集 bigram：返回 (去重后的 key 数组, 对应计数)。

    key = a * V + b（int64），分块统计 + 逐块 unique + 最后合并，
    避免把上千万个 pair 放进 Python dict（那会吃掉几百 MB 内存）。
    """
    limit = stream.n if max_tokens <= 0 else min(stream.n, max_tokens)
    CH = 1 << 21
    key_chunks: list[np.ndarray] = []
    cnt_chunks: list[np.ndarray] = []
    for s in range(0, limit - 1, CH):
        e = min(s + CH + 1, limit)
        chunk = np.asarray(stream.data[s:e], dtype=np.int64)
        keys = chunk[:-1] * vocab_size + chunk[1:]
        u, c = np.unique(keys, return_counts=True)
        key_chunks.append(u)
        cnt_chunks.append(c)
    if not key_chunks:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    all_keys = np.concatenate(key_chunks)
    all_counts = np.concatenate(cnt_chunks)
    order = np.argsort(all_keys, kind="stable")
    k = all_keys[order]
    c = all_counts[order]
    uniq, start_idx = np.unique(k, return_index=True)
    sums = np.add.reduceat(c, start_idx)
    return uniq, sums


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def evaluate_model(model: TinyTransformer, windows: np.ndarray, device: torch.device,
                   batch_size: int) -> tuple[float, float]:
    model.eval()
    total, n = 0.0, 0
    V = model.cfg.vocab_size
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            w = torch.from_numpy(windows[i:i + batch_size]).to(device)
            x, y = w[:, :-1], w[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1))
            total += loss.item()
            n += 1
    loss = total / max(1, n)
    return loss, math.exp(min(20.0, loss))


def _pair_arrays(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = windows[:, :-1].reshape(-1)
    y = windows[:, 1:].reshape(-1)
    return x, y


def evaluate_unigram(windows: np.ndarray, unigram: np.ndarray, vocab_size: int,
                     alpha: float = 1.0) -> tuple[float, float]:
    x, y = _pair_arrays(windows)
    total = unigram.sum()
    p = (unigram[y] + alpha) / (total + alpha * vocab_size)
    loss = float(-np.log(p).mean())
    return loss, math.exp(min(20.0, loss))


def evaluate_bigram(windows: np.ndarray, uniq: np.ndarray, counts: np.ndarray,
                    unigram: np.ndarray, vocab_size: int, alpha: float = 1.0
                    ) -> tuple[float, float]:
    x, y = _pair_arrays(windows)
    keys = x * vocab_size + y
    if len(uniq) == 0:
        p = (alpha) / (unigram[x] + alpha * vocab_size)
    else:
        idx = np.searchsorted(uniq, keys)
        idx_c = np.clip(idx, 0, len(uniq) - 1)
        hit = uniq[idx_c] == keys
        cnt = np.where(hit, counts[idx_c], 0)
        p = (cnt + alpha) / (unigram[x] + alpha * vocab_size)
    loss = float(-np.log(p).mean())
    return loss, math.exp(min(20.0, loss))


def evaluate_uniform(windows: np.ndarray, vocab_size: int) -> tuple[float, float]:
    loss = math.log(vocab_size)
    return loss, float(vocab_size)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="tiny-llm-from-scratch 评测（模型 vs 基线）")
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints" / "latest.pt"))
    ap.add_argument("--train-bin", default=str(REPO_ROOT / "data" / "processed" / "train.bin"))
    ap.add_argument("--val-bin", default=str(REPO_ROOT / "data" / "processed" / "val.bin"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=1.0, help="n-gram 平滑系数")
    ap.add_argument("--max-stats-tokens", type=int, default=0,
                    help="统计基线时最多用多少个训练 token（0=全部）")
    ap.add_argument("--skip-model", action="store_true", help="只算基线（没训完也能跑）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_stream = TokenStream(args.train_bin, args.seq_len)
    val_stream = TokenStream(args.val_bin, args.seq_len)
    vocab_size = 16000

    windows = build_eval_windows(val_stream, args.batch_size, args.seq_len, args.eval_batches)
    n_pred = windows.shape[0] * args.seq_len
    print(f"[eval] 评测窗口 {windows.shape[0]} 个, 共 {n_pred} 个预测位置 "
          f"(模型与基线完全一致)")

    t0 = time.time()
    print("[eval] 统计基线（训练集）...")
    unigram = build_unigram(train_stream, vocab_size, args.max_stats_tokens)
    uniq, counts = build_bigram(train_stream, vocab_size, args.max_stats_tokens)
    print(f"[eval] 基线统计完成: {len(uniq):,} 个不同 bigram, 用时 {time.time()-t0:.1f}s")

    results = {
        "eval_protocol": {
            "val_windows": int(windows.shape[0]),
            "seq_len": args.seq_len,
            "predicted_tokens": int(n_pred),
            "note": "模型与所有基线在同一批 token 位置、同样多的预测上评测",
            "baseline_stats_source": "训练集（不是验证集，避免自评作弊）",
        },
        "vocab_size": vocab_size,
    }

    u_loss, u_ppl = evaluate_uniform(windows, vocab_size)
    results["uniform"] = {"val_loss": u_loss, "val_ppl": u_ppl,
                          "desc": "P(t)=1/V，完全不看数据"}
    ug_loss, ug_ppl = evaluate_unigram(windows, unigram, vocab_size, args.alpha)
    results["unigram"] = {"val_loss": ug_loss, "val_ppl": ug_ppl,
                          "desc": "只看当前 token 的边缘频率"}
    bg_loss, bg_ppl = evaluate_bigram(windows, uniq, counts, unigram, vocab_size, args.alpha)
    results["bigram"] = {"val_loss": bg_loss, "val_ppl": bg_ppl,
                         "desc": "P(t_{i+1}|t_i)，只看上一个 token，add-alpha 平滑"}

    model_info = None
    if not args.skip_model:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ModelConfig.from_dict(ck.get("config", {}))
        model = TinyTransformer(cfg).to(device)
        model.load_state_dict(ck["model"])
        m_loss, m_ppl = evaluate_model(model, windows, device, args.batch_size)
        results["model"] = {
            "val_loss": m_loss, "val_ppl": m_ppl,
            "params": model.num_parameters(),
            "step": int(ck.get("step", 0)),
            "tokens_seen": int(ck.get("tokens_seen", 0)),
            "context_window": args.seq_len,
        }
        model_info = results["model"]
        print(f"[eval] 模型 val_loss={m_loss:.4f} val_ppl={m_ppl:.2f}")

    if model_info:
        results["comparison"] = {
            "model_better_than_uniform": model_info["val_ppl"] < results["uniform"]["val_ppl"],
            "model_better_than_unigram": model_info["val_ppl"] < results["unigram"]["val_ppl"],
            "model_better_than_bigram": model_info["val_ppl"] < results["bigram"]["val_ppl"],
            "ppl_ratio_vs_bigram": round(results["bigram"]["val_ppl"] / model_info["val_ppl"], 3),
            "ppl_ratio_vs_uniform": round(results["uniform"]["val_ppl"] / model_info["val_ppl"], 3),
            "loss_gap_vs_bigram_nats": round(results["bigram"]["val_loss"] - model_info["val_loss"], 4),
        }
        print("[eval] 对比:", json.dumps(results["comparison"], ensure_ascii=False))

    results["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out_dir / "eval_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Markdown 表格 ---------------------------------------------------
    lines = [
        "# 评测报告",
        "",
        f"生成时间：{results['generated_at']}",
        "",
        "## 评测协议",
        "",
        f"- 验证集窗口数：{results['eval_protocol']['val_windows']}（每窗 {args.seq_len} token）",
        f"- 评测预测位置总数：{results['eval_protocol']['predicted_tokens']}",
        f"- 基线统计来源：{results['eval_protocol']['baseline_stats_source']}",
        "- 模型与所有基线在**完全相同**的 token 位置上评测。",
        "",
        "## 结果",
        "",
        "| 方法 | 验证集 loss | 验证集 PPL | 说明 |",
        "|---|---|---|---|",
        f"| 均匀分布基线 | {results['uniform']['val_loss']:.4f} | {results['uniform']['val_ppl']:.2f} | {results['uniform']['desc']} |",
        f"| 一元基线 | {results['unigram']['val_loss']:.4f} | {results['unigram']['val_ppl']:.2f} | {results['unigram']['desc']} |",
        f"| 二元(bigram)基线 | {results['bigram']['val_loss']:.4f} | {results['bigram']['val_ppl']:.2f} | {results['bigram']['desc']} |",
    ]
    if model_info:
        lines.append(
            f"| **本项目模型** | **{model_info['val_loss']:.4f}** | **{model_info['val_ppl']:.2f}** | "
            f"{model_info['context_window']} token 上下文, {model_info['params']:,} 参数 |")
        c = results["comparison"]
        lines += [
            "",
            "## 结论",
            "",
            f"- 模型 vs 均匀分布：PPL 改善 **{c['ppl_ratio_vs_uniform']:.1f}×**",
            f"- 模型 vs bigram 基线：PPL 改善 **{c['ppl_ratio_vs_bigram']:.2f}×**"
            f"（loss 低 {c['loss_gap_vs_bigram_nats']:.3f} nats）",
            "",
            "> **边界说明**：bigram 只有 1 个 token 的上下文，本模型有 "
            f"{model_info['context_window']} 个，两者信息量不对等。",
            "> 「优于 bigram」证明的是模型确实学到了语言结构并利用了长上下文，",
            "> 不代表它已经接近真实语料的分布水平。",
        ]
    (out_dir / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[eval] 报告写入 {out_dir / 'eval_report.json'} / {out_dir / 'eval_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
