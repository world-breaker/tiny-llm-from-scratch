"""训练脚本：AdamW + warmup/cosine + 梯度裁剪 + 梯度累积 + bf16 autocast + 梯度检查点。

工程要点
--------
* **显存自适应**：先用目标 micro-batch 做一次「试跑」，OOM 就先开梯度检查点、再把
  micro-batch 减半，必要时继续减半到 1；OOM 发生在梯度累积中途时会 **丢弃本轮已累积的梯度**
  并用更小的 micro-batch 重来，保证梯度不被污染。
* **断点续跑**：每 ``--save-every`` 步存 ``checkpoints/latest.pt``（含优化器 / 调度器 /
  RNG 状态），``--resume`` 原地续训；长任务可以随时中断。
* **确定性**：固定随机种子，验证集用固定顺序窗口，保证 val loss 跨 step 可比。
* **可观测**：每 ``--log-every`` 步写 ``results/train_log.csv``，周期性验证写
  ``results/val_log.csv``，结束时写 ``results/train_summary.json``（含显存峰值、耗时）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data.token_dataset import TokenStream  # noqa: E402
from model.transformer import ModelConfig, TinyTransformer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_lambda(warmup: int, total: int, min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        prog = min(1.0, max(0.0, prog))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return fn


def csv_append(path: Path, header: list[str], row: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        if is_new:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")


class OOM(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="tiny-llm-from-scratch 训练脚本")
    ap.add_argument("--train-bin", default=str(REPO_ROOT / "data" / "processed" / "train.bin"))
    ap.add_argument("--val-bin", default=str(REPO_ROOT / "data" / "processed" / "val.bin"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--ckpt-dir", default=str(REPO_ROOT / "checkpoints"))
    ap.add_argument("--model-config", default=None, help="模型配置 JSON（可选）")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-kv-head", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16, help="每个优化器 step 的样本数（有效 batch）")
    ap.add_argument("--micro-batch", type=int, default=None, help="每次前向的样本数（默认 = batch-size）")
    ap.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default=None, help="checkpoint 路径（'auto' = 用 checkpoints/latest.pt）")
    ap.add_argument("--init-from", default=None, help="只加载权重（不恢复优化器）")
    ap.add_argument("--grad-checkpoint", action="store_true", help="强制开启梯度检查点")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--max-hours", type=float, default=2.0, help="墙钟时间上限（小时），到了自动收尾")
    args = ap.parse_args()

    if args.micro_batch is None:
        args.micro_batch = args.batch_size
    if args.grad_accum < 1:
        args.grad_accum = 1
    # 有效 batch = micro_batch * grad_accum，必须等于目标 batch_size
    if args.micro_batch * args.grad_accum != args.batch_size:
        args.grad_accum = max(1, args.batch_size // args.micro_batch)
        print(f"[cfg] micro_batch={args.micro_batch} -> grad_accum={args.grad_accum} "
              f"(有效 batch={args.micro_batch * args.grad_accum})")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    # ---- 模型 ----------------------------------------------------------
    if args.model_config:
        cfg = ModelConfig.from_dict(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    else:
        cfg = ModelConfig(
            vocab_size=args.vocab_size, d_model=args.d_model, n_layer=args.n_layer,
            n_head=args.n_head, n_kv_head=args.n_kv_head, d_ff=args.d_ff,
            max_seq_len=args.seq_len,
        )
    model = TinyTransformer(cfg).to(device)
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing(True)
    bd = model.parameter_breakdown()
    print(f"[model] params={bd['total']:,} (emb={bd['embedding']:,}, "
          f"non-emb={bd['non_embedding']:,}, {bd['layers']} 层, 每层 "
          f"{sum(bd['per_layer']):,})")
    print(f"[model] config={cfg.to_dict()}")

    # ---- 数据 ----------------------------------------------------------
    train_stream = TokenStream(args.train_bin, args.seq_len)
    val_stream = TokenStream(args.val_bin, args.seq_len)
    print(f"[data] train tokens={train_stream.n_tokens:,}  val tokens={val_stream.n_tokens:,}")
    gen = torch.Generator().manual_seed(args.seed)

    # ---- 优化器 / 调度器 ------------------------------------------------
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8, fused=(device.type == "cuda"),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_steps, args.max_steps, args.min_lr_ratio)
    )

    # ---- 断点续跑 -------------------------------------------------------
    start_step = 0
    tokens_seen = 0
    if args.resume:
        ckpt_path = (ckpt_dir / "latest.pt") if args.resume == "auto" else Path(args.resume)
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
                scheduler.load_state_dict(ck["scheduler"])
            start_step = int(ck.get("step", 0))
            tokens_seen = int(ck.get("tokens_seen", 0))
            if "rng" in ck:
                random.setstate(ck["rng"]["python"])
                np.random.set_state(ck["rng"]["numpy"])
                torch.set_rng_state(ck["rng"]["torch"])
                if device.type == "cuda" and ck["rng"].get("cuda") is not None:
                    torch.cuda.set_rng_state(ck["rng"]["cuda"])
            print(f"[resume] 从 {ckpt_path} 恢复: step={start_step} tokens_seen={tokens_seen}")
        else:
            print(f"[resume] 找不到 {ckpt_path}，从头开始")
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        print(f"[init] 从 {args.init_from} 载入权重")

    use_bf16 = (device.type == "cuda") and (not args.no_bf16) and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"[amp] autocast dtype = {amp_dtype}")

    # ---- 固定验证窗口（跨 step / 跨实现可比）-----------------------------
    val_batches = args.eval_batches

    def evaluate() -> float:
        """在固定的验证窗口上求平均 loss。

        窗口切分规则与 eval/baseline.py 完全一致（同样的 batch_size / seq_len / 起点），
        所以训练期看到的 val loss 与评测脚本跑出来的 val loss 是可比的。
        内部再按 eval_micro_bs 切片前向，避免一次性物化 (B, T, vocab) 的 fp32 logits。
        """
        model.eval()
        loss_sum = 0.0
        n_tok = 0
        eval_micro = 4
        with torch.no_grad():
            start = 0
            for _ in range(val_batches):
                need = args.batch_size * (args.seq_len + 1)
                if start + need > val_stream.n:
                    break
                chunk = np.array(val_stream.data[start:start + need], dtype=np.int64)
                w = torch.from_numpy(chunk).view(args.batch_size, args.seq_len + 1).to(device)
                for i in range(0, args.batch_size, eval_micro):
                    sub = w[i:i + eval_micro]
                    x, y = sub[:, :-1], sub[:, 1:]
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits.reshape(-1, cfg.vocab_size).float(), y.reshape(-1),
                            reduction="sum")
                    loss_sum += loss.item()
                    n_tok += y.numel()
                start += need
        model.train()
        return loss_sum / max(1, n_tok)

    # ---- 训练主循环 -----------------------------------------------------
    log_path = out_dir / "train_log.csv"
    val_log_path = out_dir / "val_log.csv"
    header = ["step", "train_loss", "lr", "grad_norm", "micro_bs", "tokens_seen",
              "elapsed_s", "tok_per_s", "mem_peak_gb"]
    val_header = ["step", "val_loss", "val_ppl", "elapsed_s"]
    if start_step == 0 and log_path.exists():
        log_path.unlink()
        if val_log_path.exists():
            val_log_path.unlink()

    micro_bs = args.micro_batch
    oom_events = 0
    best_val = float("inf")
    best_step = 0
    t_start = time.time()
    tokens_at_start = tokens_seen
    step_times: list[float] = []
    history = []
    model.train()

    def run_micro_steps(x_all, y_all, n_micro):
        """跑 n_micro 个 micro-batch 的前向/反向，累加梯度。返回平均 loss。"""
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for i in range(n_micro):
            xb = x_all[i * micro_bs:(i + 1) * micro_bs].to(device, non_blocking=True)
            yb = y_all[i * micro_bs:(i + 1) * micro_bs].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(), yb.reshape(-1))
            (loss / n_micro).backward()
            loss_sum += loss.item()
            del logits, loss
        return loss_sum / max(1, n_micro)

    step = start_step
    stopped_early = ""
    while step < args.max_steps:
        t_step = time.time()
        # 采样 batch_size 个窗口，拆成 micro-batch 组
        n_micro = max(1, args.batch_size // micro_bs)
        batch = train_stream.random_batch(args.batch_size, gen)
        x_all, y_all = batch[:, :-1], batch[:, 1:]

        need_retry = True
        while need_retry:
            try:
                loss_val = run_micro_steps(x_all, y_all, n_micro)
                need_retry = False
            except torch.cuda.OutOfMemoryError:
                oom_events += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if not model.grad_checkpointing:
                    print(f"[oom] step={step} 显存不足 -> 开启梯度检查点")
                    model.enable_gradient_checkpointing(True)
                elif micro_bs > 1:
                    micro_bs = max(1, micro_bs // 2)
                    n_micro = max(1, args.batch_size // micro_bs)
                    print(f"[oom] step={step} 显存不足 -> micro_batch 降到 {micro_bs}")
                else:
                    raise
                continue

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += args.batch_size * args.seq_len
        step += 1
        dt = time.time() - t_step
        step_times.append(dt)

        if step % args.log_every == 0 or step == args.max_steps:
            elapsed = time.time() - t_start
            tps = (tokens_seen - tokens_at_start) / max(1e-9, elapsed)
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            csv_append(log_path, header, [step, f"{loss_val:.6f}", f"{scheduler.get_last_lr()[0]:.6f}",
                                         f"{float(gnorm):.4f}", micro_bs, tokens_seen,
                                         f"{elapsed:.1f}", f"{tps:.0f}", f"{mem:.2f}"])
            print(f"[step {step:5d}/{args.max_steps}] loss={loss_val:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} gnorm={float(gnorm):.3f} "
                  f"mb={micro_bs} mem={mem:.2f}GiB tps={tps:.0f} "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

        if step % args.eval_every == 0 or step == args.max_steps:
            vloss = evaluate()
            vppl = math.exp(min(20.0, vloss))
            csv_append(val_log_path, val_header, [step, f"{vloss:.6f}", f"{vppl:.3f}",
                                                  f"{time.time() - t_start:.1f}"])
            history.append({"step": step, "val_loss": vloss, "val_ppl": vppl})
            print(f"[eval @ {step}] val_loss={vloss:.4f} val_ppl={vppl:.2f}", flush=True)
            if vloss < best_val:
                best_val = vloss
                best_step = step
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                            "step": step, "tokens_seen": tokens_seen,
                            "val_loss": vloss, "val_ppl": vppl},
                           ckpt_dir / "best.pt")
                print(f"[eval @ {step}] 新的最佳 val_loss，保存 checkpoints/best.pt", flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            ck = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "tokens_seen": tokens_seen,
                "config": cfg.to_dict(),
                "args": vars(args),
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                },
            }
            torch.save(ck, ckpt_dir / "latest.pt")
            torch.save({k: ck[k] for k in ("model", "config", "step", "tokens_seen")},
                       ckpt_dir / f"step_{step}.pt")
            # 只保留最近两个 step checkpoint
            for old in sorted(ckpt_dir.glob("step_*.pt")):
                if old.stem != f"step_{step}":
                    try:
                        old.unlink()
                    except OSError:
                        pass

        if (time.time() - t_start) / 3600.0 > args.max_hours:
            stopped_early = f"达到 --max-hours={args.max_hours}h 上限"
            print(f"[stop] {stopped_early}")
            break

    # ---- 收尾 -----------------------------------------------------------
    wall = time.time() - t_start
    mem_peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    final_val = evaluate()
    summary = {
        "steps_done": step,
        "max_steps": args.max_steps,
        "stopped_early": stopped_early,
        "tokens_seen": tokens_seen,
        "tokens_per_step": args.batch_size * args.seq_len,
        "effective_batch": args.batch_size,
        "micro_batch_final": micro_bs,
        "grad_accum": args.grad_accum,
        "grad_checkpointing": model.grad_checkpointing,
        "oom_events": oom_events,
        "final_val_loss": final_val,
        "final_val_ppl": math.exp(min(20.0, final_val)),
        "best_val_loss": best_val if best_val != float("inf") else None,
        "best_val_ppl": math.exp(min(20.0, best_val)) if best_val != float("inf") else None,
        "best_step": best_step,
        "wall_seconds": round(wall, 1),
        "wall_hours": round(wall / 3600, 2),
        "tokens_per_second": round((tokens_seen - tokens_at_start) / max(1e-9, wall), 1),
        "mem_peak_gib": round(mem_peak, 3),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "model_config": cfg.to_dict(),
        "params": model.parameter_breakdown(),
        "train_args": vars(args),
        "val_history": history,
        "avg_step_seconds": round(sum(step_times) / max(1, len(step_times)), 4),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("train_args", "val_history")},
                     ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
