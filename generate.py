"""生成脚本：批量采样 + 样例落盘。

用法::

    python generate.py --ckpt checkpoints/latest.pt --num 8 --max-new-tokens 80
    python generate.py --prompt "维基百科" --temperature 0.7 --top-k 40 --top-p 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from model.generation import generate_batch  # noqa: E402
from model.transformer import ModelConfig, TinyTransformer  # noqa: E402
from tokenizer.bpe import load_tokenizer  # noqa: E402

DEFAULT_PROMPTS = [
    "维基百科是",
    "中国位于亚洲东部，是",
    "人工智能是指",
    "宇宙是",
    "足球是一项",
    "音乐是一种",
    "长城是",
    "物理学是研究",
]


def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ck.get("config", {}))
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def main() -> int:
    ap = argparse.ArgumentParser(description="tiny-llm-from-scratch 生成样例")
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints" / "latest.pt"))
    ap.add_argument("--tokenizer-dir", default=str(REPO_ROOT / "tokenizer" / "artifacts"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "generations.json"))
    ap.add_argument("--out-text", default=str(REPO_ROOT / "results" / "generations.txt"))
    ap.add_argument("--prompt", default=None, help="单个 prompt（不给就用内置的默认 prompt 列表）")
    ap.add_argument("--num", type=int, default=8, help="生成条数")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer_dir)
    model, cfg, ck = load_model(Path(args.ckpt), device)
    print(f"[gen] 模型 {cfg.to_dict()} | step={ck.get('step')} | device={device}")

    if args.prompt:
        prompts = [args.prompt] * args.num
    else:
        base = DEFAULT_PROMPTS
        prompts = [base[i % len(base)] for i in range(args.num)]

    prompt_ids = [tok.encode(p) for p in prompts]
    t0 = time.time()
    outs: list[list[int]] = []
    for i in range(0, len(prompt_ids), args.batch_size):
        chunk = prompt_ids[i:i + args.batch_size]
        outs.extend(generate_batch(
            model, chunk,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed + i,          # 每批一个种子 -> 可复现
            device=device,
            eos_id=tok.eos_id,
            pad_id=tok.pad_id,
        ))
    dt = time.time() - t0

    samples = []
    for prompt, ids in zip(prompts, outs):
        text = tok.decode(ids)
        samples.append({
            "prompt": prompt,
            "completion": text,
            "prompt_tokens": len(tok.encode(prompt)),
            "completion_tokens": len(ids),
        })

    payload = {
        "ckpt": Path(args.ckpt).name,
        "step": int(ck.get("step", 0)),
        "tokens_seen": int(ck.get("tokens_seen", 0)),
        "params": model.num_parameters(),
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
        },
        "decode_seconds": round(dt, 2),
        "samples": samples,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    for i, s in enumerate(samples, 1):
        lines.append(f"--- 样例 {i} ---")
        lines.append(f"[prompt] {s['prompt']}")
        lines.append(f"[续写 ] {s['completion']}")
        lines.append("")
    Path(args.out_text).write_text("\n".join(lines), encoding="utf-8")

    for i, s in enumerate(samples, 1):
        print(f"--- 样例 {i} ---")
        print(f"[prompt] {s['prompt']}")
        print(f"[续写 ] {s['completion']}")
    print(f"\n[gen] {len(samples)} 条样例写入 {out_path} / {args.out_text}（{dt:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
