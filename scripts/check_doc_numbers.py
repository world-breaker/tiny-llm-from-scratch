# -*- coding: utf-8 -*-
"""核对 README 里引用的数字与 results/ 下的产物是否一致。

★ 为什么要有这个脚本：README 的数字是「引用」结果产物的。如果产物更新了而 README
  没改，README 就在说谎——而且没人会发现。这个脚本把两者对上，对不上就报错。

用法：
    python scripts/check_doc_numbers.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"

#: 产物 → 期望出现在 README 里的字符串
def variants(value: str) -> List[str]:
    """同一个数的可接受写法：原样 / 去千分位 / 加千分位 / 去尾零 / 整数化。

    README 里人手写的数字格式不统一（「16,000.00」/「16000.0」/「71,875」），
    判据不能因为格式差异就报漂移 —— 但**值本身**必须一致。
    """
    out = {value}
    try:
        f = float(value)
    except ValueError:
        return list(out)
    out.add(f"{f}")
    out.add(f"{f:,.2f}")
    out.add(f"{f:,.1f}")
    out.add(f"{f:,.0f}")
    out.add(f"{f:.2f}")
    out.add(f"{f:.1f}")
    out.add(f"{f:.4f}")
    out.add(f"{int(f)}" if f == int(f) else f"{f}")
    out.add(f"{int(f):,}" if f == int(f) else f"{f:,.2f}")
    return [v for v in out if v]


def expected() -> List[Tuple[str, str]]:
    if not RESULTS.exists():
        return []
    out: List[Tuple[str, str]] = []
    _last_elapsed = 0.0
    _log = list(csv.DictReader((RESULTS / "train_log.csv").read_text(encoding="utf-8").splitlines()))
    if _log:
        _last_elapsed = float(_log[-1].get("elapsed_s") or 0.0)

    curve = json.loads((RESULTS / "curve_summary.json").read_text(encoding="utf-8"))
    out += [
        ("实际训练步数", f"{curve['steps']:,}"),
        ("累计 token", f"{curve['tokens_seen']:,}"),
        ("峰值显存 GiB", f"{curve['mem_peak_gib']:.2f}"),
        ("参数量", f"{curve['params']:,}"),
        ("最佳 val loss", f"{curve['best_val_loss']:.4f}"),
        ("最佳 val PPL", f"{curve['best_val_ppl']:.2f}"),
        # 墙钟直接取训练日志末行的 elapsed_s（不要用四舍五入后的 wall_hours 再乘回去，
        # 那样会差 0.2 秒 → 假报漂移）
        ("墙钟（分钟）", f"{_last_elapsed / 60:.1f}"),
        ("墙钟（秒）", f"{_last_elapsed:.1f}"),
    ]

    ev = json.loads((RESULTS / "eval_report.json").read_text(encoding="utf-8"))
    m = ev["model"]
    out += [
        ("模型 val_loss", f"{m['val_loss']:.4f}"),
        ("模型 val_ppl", f"{m['val_ppl']:.2f}"),
        ("uniform PPL", f"{ev['uniform']['val_ppl']:.2f}"),
        ("unigram PPL", f"{ev['unigram']['val_ppl']:.2f}"),
        ("bigram PPL", f"{ev['bigram']['val_ppl']:.2f}"),
        ("vs uniform 倍数", f"{ev['comparison']['ppl_ratio_vs_uniform']:.1f}"),
        ("vs bigram 倍数", f"{ev['comparison']['ppl_ratio_vs_bigram']:.2f}"),
        ("vs bigram nats", f"{ev['comparison']['loss_gap_vs_bigram_nats']:.3f}"),
    ]

    # 训练日志末行：train loss 首末
    log = list(csv.DictReader((RESULTS / "train_log.csv").read_text(encoding="utf-8").splitlines()))
    if log:
        out += [
            ("首行 train loss", f"{float(log[0]['train_loss']):.4f}"),
            ("末行 train loss", f"{float(log[-1]['train_loss']):.4f}"),
            ("末行吞吐", f"{int(float(log[-1]['tok_per_s'])):,}"),
        ]

    tok = json.loads((RESULTS / "tokenizer_stats.json").read_text(encoding="utf-8"))
    if "vocab_size" in tok:
        out.append(("词表大小", f"{tok['vocab_size']:,}"))
    return out


def main() -> int:
    if not README.exists():
        print("[错误] 找不到 README.md", file=sys.stderr)
        return 1
    readme = README.read_text(encoding="utf-8")
    pairs = expected()
    if not pairs:
        print("[错误] results/ 里没有产物，先跑训练与评测", file=sys.stderr)
        return 1

    missing: List[str] = []
    for label, value in pairs:
        if not any(v in readme for v in variants(value)):
            missing.append(f"{label}: {value}")

    print(f"核对 {len(pairs)} 个数字（来源：results/）")
    if missing:
        print("\n❌ README 里没有这些值（或值已过期）：")
        for item in missing:
            print(f"   - {item}")
        return 1
    print("✅ README 的数字与结果产物一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
