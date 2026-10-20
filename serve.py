"""极简 HTTP 生成服务：**零新依赖**，只用 Python 标准库 ``http.server``。

为什么用 stdlib
--------------
本项目刻意不引入 web 框架（Flask/FastAPI/uvicorn），一是「零新依赖」是硬约束，
二是这个服务的存在意义是把「模型 checkpoint -> 可被调用的接口」这一步走通，
框架会把这一步的细节藏起来。

接口
----
* ``GET  /healthz``  -> ``{"status": "ok", ...}``
* ``POST /generate`` -> ``{"prompt": "...", "temperature": 0.8, "top_k": 50,
  "top_p": 0.95, "max_new_tokens": 80, "seed": 1234}``
  -> ``{"prompt": "...", "completion": "...", "completion_tokens": [...], ...}``

启动::

    python serve.py --ckpt checkpoints/latest.pt --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from model.generation import generate_batch  # noqa: E402
from model.transformer import ModelConfig, TinyTransformer  # noqa: E402
from tokenizer.bpe import load_tokenizer  # noqa: E402

MAX_BODY = 1 << 20  # 1MB 上限，防止超大请求


class Engine:
    """模型 + 分词器 + 生成（线程安全：生成用一把锁串起来）。"""

    def __init__(self, ckpt: Path, tokenizer_dir: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = load_tokenizer(tokenizer_dir)
        ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.cfg = ModelConfig.from_dict(ck.get("config", {}))
        self.model = TinyTransformer(self.cfg).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.step = int(ck.get("step", 0))
        self.tokens_seen = int(ck.get("tokens_seen", 0))
        self.params = self.model.num_parameters()
        self.lock = threading.Lock()
        self.started_at = time.time()

    def generate(self, prompt: str, max_new_tokens: int = 80, temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.95, seed: int = 1234) -> dict:
        prompt_ids = self.tok.encode(prompt) if prompt else [self.tok.bos_id]
        with self.lock:
            ids = generate_batch(
                self.model, [prompt_ids],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                device=self.device,
                eos_id=self.tok.eos_id,
                pad_id=self.tok.pad_id,
            )[0]
        return {
            "prompt": prompt,
            "completion": self.tok.decode(ids),
            "completion_tokens": ids,
            "prompt_token_count": len(prompt_ids),
            "sampling": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed,
            },
        }


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None  # type: ignore[assignment]
    server_version = "tiny-llm/0.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/healthz":
            self._send(200, {
                "status": "ok",
                "model_params": self.engine.params,
                "train_step": self.engine.step,
                "tokens_seen": self.engine.tokens_seen,
                "vocab_size": self.engine.cfg.vocab_size,
                "device": str(self.engine.device),
                "uptime_seconds": round(time.time() - self.engine.started_at, 1),
            })
        else:
            self._send(404, {"error": "not found", "path": self.path,
                             "available": ["/healthz", "/generate"]})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/generate":
            self._send(404, {"error": "not found", "path": self.path,
                             "available": ["/healthz", "/generate"]})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, {"error": "bad content-length"})
            return
        if length <= 0 or length > MAX_BODY:
            self._send(400, {"error": "empty or too large body"})
            return
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"invalid json: {e}"})
            return
        if not isinstance(req, dict):
            self._send(400, {"error": "body must be a json object"})
            return
        prompt = str(req.get("prompt", ""))
        try:
            out = self.engine.generate(
                prompt=prompt,
                max_new_tokens=int(req.get("max_new_tokens", 80)),
                temperature=float(req.get("temperature", 0.8)),
                top_k=int(req.get("top_k", 50)),
                top_p=float(req.get("top_p", 0.95)),
                seed=int(req.get("seed", 1234)),
            )
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return
        self._send(200, out)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="tiny-llm-from-scratch 极简 HTTP 生成服务")
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints" / "latest.pt"))
    ap.add_argument("--tokenizer-dir", default=str(REPO_ROOT / "tokenizer" / "artifacts"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    engine = Engine(Path(args.ckpt), Path(args.tokenizer_dir), args.device)
    Handler.engine = engine
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}  "
          f"(params={engine.params:,}, step={engine.step})")
    print("[serve]  GET /healthz   POST /generate")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 退出")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
