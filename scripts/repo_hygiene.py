# -*- coding: utf-8 -*-
"""仓库卫生自证：公开仓库里不该出现的东西，一处都不许有。

检查四件事：
  1) 已跟踪文件里没有**绝对路径**（本机目录结构不该进公开仓）
  2) 没有敏感串（真名 / 邮箱 / 密钥形态）
  3) 没有大文件入库（模型权重、语料、token 流、checkpoint）
  4) 关键产物**确实**入库了（否则「没入库」会被误当成「没生成」）

★ 为什么写成 Python 而不是 bash：本机 Git Bash 不可用（`bash` 不在 PATH），
  而 CI 跑在 ubuntu 上。Python 版两边都能跑、也能本地验证。

用法：
    python scripts/repo_hygiene.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SELF = "scripts/repo_hygiene.sh"  # 本脚本的旧 shell 版（若还在仓里）也含示例路径

SENSITIVE: List[Tuple[str, str]] = [
    ("真名", r"田梦龙"),
    ("OpenAI 形态密钥", r"sk-[A-Za-z0-9]{16,}"),
    ("Google 形态密钥", r"AIza[A-Za-z0-9_-]{20,}"),
    ("邮箱", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
]
ABS_PATH = r"[A-Za-z]:\\"
MAX_FILE_BYTES = 2_000_000

REQUIRED_ARTIFACTS = [
    "results/train_log.csv",
    "results/val_log.csv",
    "results/eval_report.md",
    "results/eval_report.json",
    "results/tokenizer_stats.json",
    "results/training_curve.png",
    "results/generations.txt",
    "results/curve_summary.json",
]


def tracked() -> List[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def text_files(files: List[str]):
    """只对「像文本」的文件做内容检查（跳过二进制，避免乱码误判）。"""
    for rel in files:
        p = ROOT / rel
        if not p.is_file():
            continue
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:  # 二进制
            continue
        try:
            yield rel, raw.decode("utf-8")
        except UnicodeDecodeError:
            continue


def main() -> int:
    files = tracked()
    fail = 0

    print("== 1) 绝对路径 ==")
    hits = []
    for rel, body in text_files(files):
        if rel == SELF:
            continue
        for m in re.finditer(ABS_PATH, body):
            line = body[:m.start()].count("\n") + 1
            hits.append(f"{rel}:{line}: {body.splitlines()[line-1].strip()[:90]}")
    if hits:
        print("❌ 发现绝对路径：")
        for h in hits[:20]:
            print("   " + h)
        fail = 1
    else:
        print("✅ 无绝对路径")

    print("== 2) 敏感串 ==")
    for label, pat in SENSITIVE:
        found = []
        for rel, body in text_files(files):
            if rel == SELF:
                continue
            for m in re.finditer(pat, body):
                line = body[:m.start()].count("\n") + 1
                found.append(f"{rel}:{line}")
        if found:
            print(f"❌ 命中敏感串 [{label}]：{found[:8]}")
            fail = 1
    if fail == 0:
        print("✅ 无敏感串")

    print("== 3) 大文件入库 ==")
    big = []
    for rel in files:
        p = ROOT / rel
        if p.is_file() and p.stat().st_size > MAX_FILE_BYTES:
            big.append(f"{rel} ({p.stat().st_size} B)")
    if big:
        print("❌ 以下已跟踪文件超过 2MB（模型/语料/token 流不该入库）：")
        for b in big[:20]:
            print("   " + b)
        fail = 1
    else:
        print("✅ 没有大文件入库")

    print("== 4) 关键产物确实入库了 ==")
    for f in REQUIRED_ARTIFACTS:
        ok = f in files
        print(f"  {'✓' if ok else '✗'} {f}")
        if not ok:
            fail = 1

    print()
    if fail:
        print("❌ 仓库卫生自证未通过")
        return 1
    print("✅ 仓库卫生自证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
