"""训练 BPE 分词器：读取 data/processed/train.txt，产出 tokenizer/artifacts/tokenizer.json。

用法::

    python tokenizer/train_bpe.py --train data/processed/train.txt --vocab-size 16000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tokenizer.bpe import build_tokenizer  # noqa: E402


def iter_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line


def main() -> int:
    ap = argparse.ArgumentParser(description="训练纯 Python BPE 分词器")
    ap.add_argument("--train", default=str(REPO_ROOT / "data" / "processed" / "train.txt"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "tokenizer" / "artifacts"))
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--max-word-types", type=int, default=120_000)
    ap.add_argument("--min-merges", type=int, default=1000)
    ap.add_argument("--max-chars", type=int, default=12_000_000,
                    help="用于学习合并规则的字符数上限（0=全量）；合并规则对规模不敏感，"
                         "限制它是为了让词频表内存可控（本机整机空闲内存 < 1GB）")
    ap.add_argument("--stats-out", default=str(REPO_ROOT / "results" / "tokenizer_stats.json"))
    args = ap.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"找不到训练语料: {train_path}；请先运行 data/prepare.py", file=sys.stderr)
        return 1

    t0 = time.time()
    tok, stats = build_tokenizer(
        iter_lines(train_path),
        target_vocab=args.vocab_size,
        max_word_types=args.max_word_types,
        min_merges=args.min_merges,
        max_chars=args.max_chars,
    )
    stats["train_seconds"] = round(time.time() - t0, 2)
    stats["train_file"] = train_path.name

    out_dir = Path(args.out_dir)
    path = tok.save(out_dir)
    print(f"[bpe] 分词器已保存: {path}  (vocab={len(tok)})")

    # ---- 自检：往返无损 + UNK 率（抽样自训练语料）-------------------------
    n_tok = 0
    n_unk = 0
    n_chars = 0
    n_checked = 0
    n_mismatch = 0
    samples = []
    with train_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            ids, unk = tok.encode_with_unk(line)
            n_tok += len(ids)
            n_unk += unk
            n_chars += len(line)
            if i % 500 == 0:
                n_checked += 1
                if tok.decode(ids) != line:
                    n_mismatch += 1
                if len(samples) < 5:
                    samples.append({"text": line[:60], "tokens": tok.decode(ids)[:60]})
    stats["encode_tokens"] = n_tok
    stats["encode_chars"] = n_chars
    stats["unk_tokens"] = n_unk
    stats["unk_rate"] = round(n_unk / n_tok, 8) if n_tok else 0.0
    stats["tokens_per_char"] = round(n_tok / n_chars, 4) if n_chars else 0.0
    stats["roundtrip_checked_lines"] = n_checked
    stats["roundtrip_mismatches"] = n_mismatch
    stats["roundtrip_lossless"] = n_mismatch == 0
    stats["samples"] = samples

    print(f"[bpe] tokens/char={stats['tokens_per_char']}  unk_rate={stats['unk_rate']}  "
          f"往返校验: {n_checked} 行, 不一致 {n_mismatch} 行")
    for s in samples[:3]:
        print("   ", s["text"][:40], "->", s["tokens"][:40])

    stats_path = Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bpe] 统计写入 {stats_path}")
    print(f"[bpe] 总用时 {stats['train_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
