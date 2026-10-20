"""把训练曲线画成 PNG：train loss / val loss / 困惑度 / 吞吐 / 显存。

用法::

    python scripts/plot_training.py --results-dir results --out results/training_curve.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示器环境（服务器/CI）必须先切后端
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 图上要画中文 → 得有中文字体；没有就**退回英文标签**（而不是画出一堆方框）。
#: 2026-10-21 实测踩过：容器/最小化 Windows 上没有 CJK 字体时，matplotlib 会
#: 打印 "Glyph ... missing from font(s) DejaVu Sans" 并把汉字画成豆腐块。
# 字体路径用 %WINDIR% 拼出来（源码里不出现盘符绝对路径 —— 公开仓库不该带本机路径）
_WIN_FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
_CJK_CANDIDATES = [
    str(_WIN_FONTS / "msyh.ttc"),
    str(_WIN_FONTS / "simhei.ttf"),
    str(_WIN_FONTS / "Deng.ttf"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

_CJK_FONT: str | None = None
for _p in _CJK_CANDIDATES:
    if Path(_p).exists():
        try:
            font_manager.fontManager.addfont(_p)
            _CJK_FONT = font_manager.FontProperties(fname=_p).get_name()
            break
        except Exception:  # noqa: BLE001 —— 字体读不了就试下一个
            continue

if _CJK_FONT:
    plt.rcParams["font.sans-serif"] = [_CJK_FONT]
plt.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII，避免字体缺 U+2212


def L(zh: str, en: str) -> str:
    """有中文字体就用中文标签，没有就用英文（保证图在任何机器上都不出豆腐块）。"""
    return zh if _CJK_FONT else en


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description="画训练曲线")
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "training_curve.png"))
    args = ap.parse_args()

    rd = Path(args.results_dir)
    train = read_csv(rd / "train_log.csv")
    val = read_csv(rd / "val_log.csv")
    if not train:
        print("没有训练日志，先跑 train.py")
        return 1

    t_step = [int(r["step"]) for r in train]
    t_loss = [float(r["train_loss"]) for r in train]
    t_mem = [float(r["mem_peak_gb"]) for r in train]
    t_tps = [float(r["tok_per_s"]) for r in train]
    t_lr = [float(r["lr"]) for r in train]

    v_step = [int(r["step"]) for r in val] if val else []
    v_loss = [float(r["val_loss"]) for r in val] if val else []
    v_ppl = [float(r["val_ppl"]) for r in val] if val else []

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(L("tiny-llm-from-scratch 训练曲线", "tiny-llm-from-scratch training curve"),
                 fontsize=15, fontweight="bold")

    # 1) loss
    ax = axes[0][0]
    ax.plot(t_step, t_loss, color="#4C78A8", lw=1.4, label=L("train loss (每步)", "train loss (per step)"))
    if v_step:
        ax.plot(v_step, v_loss, color="#F58518", lw=2.0, marker="o", ms=3.5, label="val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nats/token)")
    ax.set_title(L("训练 / 验证损失", "train / val loss"))
    ax.grid(alpha=0.3)
    ax.legend()

    # 2) 困惑度
    ax = axes[0][1]
    if v_step:
        ax.plot(v_step, v_ppl, color="#E45756", lw=2.0, marker="o", ms=3.5)
        ax.set_yscale("log")
        ax.set_title(L(f"验证集困惑度 (最佳 {min(v_ppl):.2f})", f"val PPL (best {min(v_ppl):.2f})"))
    else:
        ax.text(0.5, 0.5, L("无验证数据", "no val data"), ha="center", va="center", transform=ax.transAxes)
        ax.set_title(L("验证集困惑度", "val PPL"))
    ax.set_xlabel("step")
    ax.set_ylabel("PPL")
    ax.grid(alpha=0.3)

    # 3) 吞吐
    ax = axes[1][0]
    ax.plot(t_step, t_tps, color="#54A24B", lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("tokens/s")
    ax.set_title(L(f"训练吞吐 (峰值 {max(t_tps):,.0f} tok/s)", f"training throughput (peak {max(t_tps):,.0f} tok/s)"))
    ax.grid(alpha=0.3)

    # 4) 显存 + 学习率
    ax = axes[1][1]
    ax.plot(t_step, t_mem, color="#B279A2", lw=1.6, label=L("显存峰值 (GiB)", "peak memory (GiB)"))
    ax.set_xlabel("step")
    ax.set_ylabel(L("显存峰值 (GiB)", "peak memory (GiB)"), color="#B279A2")
    ax.set_title(L(f"显存峰值 (最大 {max(t_mem):.2f} GiB)", f"peak memory (max {max(t_mem):.2f} GiB)"))
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t_step, t_lr, color="#9D755D", lw=1.0, ls="--", label=L("学习率", "learning rate"))
    ax2.set_ylabel("learning rate", color="#9D755D")
    ax2.set_yscale("log")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"[plot] 训练曲线写入 {out}")

    # 顺手把关键数字落一份 json，方便 README / 报告引用
    summary = {}
    sp = rd / "train_summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
    # tokens_seen / wall_hours / params 的来源优先级：
    #   ① train_summary.json（训练正常收尾时写的，字段最全）
    #   ② 退回到训练日志末行（tokens_seen / elapsed_s）与评测报告里的参数量
    #   —— 训练被中断时 train_summary.json 可能不存在，此时**不能留空**（那等于
    #      「没测」而不是「没记录」），能从已有产物推出来就推出来，推不出才写 null。
    last_row = train[-1] if train else {}
    tokens_seen = summary.get("tokens_seen")
    if tokens_seen is None and last_row.get("tokens_seen"):
        tokens_seen = int(last_row["tokens_seen"])
    wall_hours = summary.get("wall_hours")
    if wall_hours is None and last_row.get("elapsed_s"):
        wall_hours = round(float(last_row["elapsed_s"]) / 3600.0, 4)
    params = (summary.get("params") or {}).get("total")
    if params is None:
        er = rd / "eval_report.json"
        if er.exists():
            try:
                _er = json.loads(er.read_text(encoding="utf-8")) or {}
                params = _er.get("model_params") or (_er.get("model") or {}).get("params")
            except Exception:  # noqa: BLE001
                params = None
    compact = {
        "final_train_loss": t_loss[-1] if t_loss else None,
        "min_train_loss": min(t_loss) if t_loss else None,
        "final_val_loss": v_loss[-1] if v_loss else None,
        "best_val_loss": min(v_loss) if v_loss else None,
        "best_val_ppl": min(v_ppl) if v_ppl else None,
        "max_tokens_per_s": max(t_tps) if t_tps else None,
        "mem_peak_gib": max(t_mem) if t_mem else None,
        "steps": t_step[-1] if t_step else None,
        "tokens_seen": tokens_seen,
        "wall_hours": wall_hours,
        "params": params,
    }
    (rd / "curve_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[plot]", json.dumps(compact, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
