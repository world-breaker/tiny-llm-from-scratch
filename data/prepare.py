"""语料获取 / 清洗 / 切分流水线（流式）。

设计要点
--------
1. **全流程流式**：本机整机内存常态空闲 < 1GB，因此任何一步都不允许把整个语料读进内存。
   - wiki 路径：用 ``pyarrow.parquet.ParquetFile.iter_batches`` 按 batch 逐行读；
   - 清洗在生成器里逐行做，写文件边写边算 SHA256；
   - 统计量只累加标量，不保留文本。
2. **不依赖预训练权重**：只下载 Wikipedia 纯文本 parquet（数据集，不是模型）。
3. **两条路都支持**：
   - ``--source wiki``  ：维基百科 20231101.zh 固定分片（版本 + SHA256 写死在校验里）；
   - ``--source local`` ：本机已有中文文本（同级仓库的 README/文档/注释 + 工作区根目录下的
     HTML 说明文档），保证完全离线可跑（规模小，如实说明）。
4. 输出：``data/processed/{train.txt,val.txt}`` + ``stats.json``（字符数、样本数、SHA256、来源信息）。

用法::

    python data/prepare.py --source wiki  --max-chars 40000000
    python data/prepare.py --source local
    python data/prepare.py --tokenize          # 用已训练的分词器把语料编码成 .bin
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "processed"
DEFAULT_RAW = REPO_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# 维基语料来源：固定 commit + 固定分片 + 官方公布的 LFS SHA256（可自证）
# ---------------------------------------------------------------------------
WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.zh"
WIKI_COMMIT = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_SHARD = "train-00002-of-00006.parquet"
WIKI_SHARD_SIZE = 126796358
WIKI_SHARD_SHA256 = "ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f"
WIKI_URL = (
    f"https://huggingface.co/datasets/{WIKI_REPO}/resolve/{WIKI_COMMIT}"
    f"/{WIKI_CONFIG}/{WIKI_SHARD}"
)

# ---------------------------------------------------------------------------
# 清洗规则（维基文本 -> 干净自然语言段落）
# ---------------------------------------------------------------------------
_RE_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_FILE = re.compile(r"\[\[(?:File|Image|文件|档案):[^\]]*\]\]", re.I)
_RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_RE_BRACE = re.compile(r"\{\{.*?\}\}", re.S)
_RE_EXTLINK = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s*([^\]]*)\]")
_RE_BOLD = re.compile(r"'{2,5}")
_RE_LIST = re.compile(r"^[\*#:;]+\s*", re.M)
_RE_CITE = re.compile(r"\[\d+\]|\[[a-z]\]|[\u00b7•▪]|\[[^\]]{0,40}\]")
_RE_HDR = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
_RE_WS = re.compile(r"[ \t\u00a0\u3000]+")
_RE_BLANK = re.compile(r"\n{3,}")
# 至少要有一个汉字或一个 ASCII 字母，否则整行丢弃
_RE_HAS_TEXT = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]")
_RE_CJK_PUNCT = re.compile(r"[\u3000-\u303f\uff00-\uffef]")


def clean_wiki_text(text: str) -> str:
    """把维基纯文本压成自然语言段落。"""
    if not text:
        return ""
    t = text
    t = _RE_REF.sub(" ", t)
    t = _RE_TABLE.sub(" ", t)
    t = _RE_BRACE.sub(" ", t)
    t = _RE_FILE.sub(" ", t)
    t = _RE_EXTLINK.sub(r" \1 ", t)
    t = re.sub(r"\[\[([^\[\]|]*)\|([^\[\]]*)\]\]", r"\2", t)   # [[a|b]] -> b
    t = re.sub(r"\[\[([^\[\]]*)\]\]", r"\1", t)                  # [[a]]  -> a
    t = _RE_BOLD.sub("", t)
    t = _RE_LIST.sub("", t)
    t = _RE_HDR.sub(r"\1", t)
    t = _RE_CITE.sub(" ", t)
    t = _RE_HTML_TAG.sub(" ", t)
    t = t.replace(" ", " ").replace("", "")
    t = _RE_WS.sub(" ", t)
    out_lines = []
    for line in t.split("\n"):
        line = line.strip()
        if not line or not _RE_HAS_TEXT.search(line):
            continue
        if len(_RE_CJK_PUNCT.sub("", line)) < 2:   # 只有标点的行丢掉
            continue
        out_lines.append(line)
    t = "\n".join(out_lines)
    t = _RE_BLANK.sub("\n\n", t)
    return t.strip()


def clean_plain_text(text: str) -> str:
    """本地退路语料：去 HTML 标签 + 压缩空白，保留可读正文。"""
    t = text
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", t)
    t = _RE_HTML_TAG.sub("\n", t)
    t = html_mod.unescape(t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    out_lines = []
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        if not _RE_HAS_TEXT.search(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# 流式读取源
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_wiki_shard(raw_dir: Path, proxy: str | None, force: bool = False) -> Path:
    """下载固定分片（带 SHA256 校验），命中缓存则直接复用。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / WIKI_SHARD
    if dest.exists() and not force:
        print(f"[wiki] 命中本地缓存: {dest.name}")
        return dest
    opener_kwargs = {}
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(handler)
    else:
        opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "tiny-llm-from-scratch/0.1 (corpus fetch)")]
    print(f"[wiki] 下载 {WIKI_URL}")
    h = hashlib.sha256()
    total = 0
    t0 = time.time()
    tmp = dest.with_suffix(".part")
    with opener.open(WIKI_URL, timeout=180) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            total += len(chunk)
            if total % (16 << 20) < (1 << 20):
                print(f"  {total / 1e6:.1f} MB / {WIKI_SHARD_SIZE / 1e6:.1f} MB", flush=True)
    got = h.hexdigest()
    if got != WIKI_SHARD_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 不匹配: got={got} expected={WIKI_SHARD_SHA256}")
    tmp.replace(dest)
    print(f"[wiki] 下载完成 {total/1e6:.1f}MB, SHA256 校验通过, 用时 {time.time()-t0:.1f}s")
    return dest


def iter_wiki_articles(path: Path) -> Iterator[str]:
    """逐行流式吐出 parquet 里的 article 文本（不整表载入内存）。"""
    import pyarrow.parquet as pq  # 延迟导入：local 路径不需要

    pf = pq.ParquetFile(path)
    cols = [c for c in ("text", "title") if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        data = batch.to_pydict()
        texts = data.get("text") or []
        titles = data.get("title") or []
        for i, t in enumerate(texts):
            if not t:
                continue
            title = titles[i] if i < len(titles) else ""
            if title:
                yield f"{title}\n{t}"
            else:
                yield t


LOCAL_TEXT_SUFFIXES = {".md", ".txt", ".rst", ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml", ".toml"}


def iter_local_texts(root: Path) -> Iterator[str]:
    """扫描本地已有中文文本：同级 projects 目录下的仓库 + 工作区根目录的 HTML 文档。

    路径全部由 ``root`` 推导，脚本里不写死任何绝对路径。
    """
    seen_files: list[Path] = []
    projects_dir = root / "projects"
    if projects_dir.is_dir():
        for repo in sorted(projects_dir.iterdir()):
            if not repo.is_dir() or repo.name == "tiny-llm-from-scratch":
                continue
            for p in sorted(repo.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix.lower() in LOCAL_TEXT_SUFFIXES or p.suffix.lower() == ".html":
                    if any(part in {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
                           for part in p.parts):
                        continue
                    if p.stat().st_size > 4 * 1024 * 1024:
                        continue
                    seen_files.append(p)
    # 工作区根目录下的 HTML 说明文档
    for p in sorted(root.glob("*.html")):
        seen_files.append(p)
    print(f"[local] 命中本地文件 {len(seen_files)} 个")
    for p in seen_files:
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = p.relative_to(root) if p.is_relative_to(root) else p.name
        yield f"### 文件: {rel}\n{raw}"


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------
def write_split(
    chunks: Iterable[str],
    out_dir: Path,
    val_ratio: float,
    max_chars: int,
    seed: int = 1234,
) -> dict:
    """把清洗后的文本按行写入 train/val 两份文件（流式写 + 边写边算 SHA256）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    import random

    rng = random.Random(seed)
    h_train = hashlib.sha256()
    h_val = hashlib.sha256()
    n_train = n_val = 0
    total = 0
    t0 = time.time()
    with train_path.open("w", encoding="utf-8", newline="\n") as ft, \
            val_path.open("w", encoding="utf-8", newline="\n") as fv:
        for text in chunks:
            if not text:
                continue
            # 逐行写入，保证行数足够做切分，也保证不把整块文本留在内存
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if total + len(line) > max_chars:
                    break
                total += len(line) + 1
                if rng.random() < val_ratio:
                    fv.write(line + "\n")
                    h_val.update((line + "\n").encode("utf-8"))
                    n_val += 1
                else:
                    ft.write(line + "\n")
                    h_train.update((line + "\n").encode("utf-8"))
                    n_train += 1
            if total >= max_chars:
                break
            if n_train % 20000 == 0 and n_train:
                print(f"  已写入 {total/1e6:.2f}M 字符 (train={n_train}, val={n_val})", flush=True)
    print(f"[切分] 完成: {total/1e6:.2f}M 字符, train={n_train} 行, val={n_val} 行, "
          f"用时 {time.time()-t0:.1f}s")
    return {
        "train_lines": n_train,
        "val_lines": n_val,
        "total_chars": total,
        "train_sha256": h_train.hexdigest(),
        "val_sha256": h_val.hexdigest(),
        "train_bytes": train_path.stat().st_size,
        "val_bytes": val_path.stat().st_size,
    }


def build_wiki_chunks(path: Path, max_chars: int) -> Iterator[str]:
    produced = 0
    for raw in iter_wiki_articles(path):
        cleaned = clean_wiki_text(raw)
        if cleaned:
            yield cleaned
            produced += len(cleaned)
            if produced >= max_chars * 1.05:
                return


def build_local_chunks(root: Path) -> Iterator[str]:
    for raw in iter_local_texts(root):
        cleaned = clean_plain_text(raw)
        if cleaned:
            yield cleaned


# ---------------------------------------------------------------------------
# tokenize 阶段
# ---------------------------------------------------------------------------
def tokenize_stage(args) -> int:
    """用训练好的分词器把 train/val 文本编码成 uint16 token 流（流式、可 memmap 读）。"""
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT))
    from tokenizer.bpe import load_tokenizer  # noqa: E402

    tok = load_tokenizer(args.tokenizer_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"vocab_size": len(tok), "splits": {}}
    for split in ("train", "val"):
        src = out_dir / f"{split}.txt"
        dst = out_dir / f"{split}.bin"
        n_tok = 0
        n_unk = 0
        n_chars = 0
        buf: list[int] = []
        t0 = time.time()
        with dst.open("wb") as f:
            with src.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    ids, n_unk_line = tok.encode_with_unk(line)
                    buf.extend(ids)
                    n_unk += n_unk_line
                    n_tok += len(ids)
                    n_chars += len(line)
                    if len(buf) >= (1 << 20):
                        f.write(np.asarray(buf, dtype=np.uint16).tobytes())
                        buf = []
            if buf:
                f.write(np.asarray(buf, dtype=np.uint16).tobytes())
        stats["splits"][split] = {
            "tokens": n_tok,
            "chars": n_chars,
            "unk_tokens": n_unk,
            "unk_rate": (n_unk / n_tok) if n_tok else 0.0,
            "tokens_per_char": (n_tok / n_chars) if n_chars else 0.0,
            "sha256": sha256_file(dst),
            "seconds": round(time.time() - t0, 2),
        }
        print(f"[tokenize] {split}: {n_tok} tokens, unk_rate={stats['splits'][split]['unk_rate']:.6f}, "
              f"{stats['splits'][split]['tokens_per_char']:.3f} tok/char, "
              f"{stats['splits'][split]['seconds']}s")
    stats_path = out_dir / "token_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[tokenize] 统计写入 {stats_path}")
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="tiny-llm-from-scratch 语料流水线")
    ap.add_argument("--source", choices=["wiki", "local"], default="wiki",
                    help="语料来源：wiki=中文维基固定分片；local=本机已有中文文本（离线退路）")
    ap.add_argument("--max-chars", type=int, default=40_000_000, help="最大字符数")
    ap.add_argument("--val-ratio", type=float, default=0.005, help="验证集比例")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or "http://127.0.0.1:7897")
    ap.add_argument("--local-root", default=None,
                    help="local 退路语料的根目录（默认由仓库位置推导）")
    ap.add_argument("--tokenize", action="store_true", help="用已有分词器把文本编码成 .bin")
    ap.add_argument("--tokenizer-dir", default=str(REPO_ROOT / "tokenizer" / "artifacts"))
    args = ap.parse_args()

    if args.tokenize:
        return tokenize_stage(args)

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    t0 = time.time()
    source_info: dict = {"source": args.source}

    if args.source == "wiki":
        shard = download_wiki_shard(raw_dir, args.proxy)
        actual_sha = sha256_file(shard)
        source_info.update({
            "dataset": WIKI_REPO,
            "config": WIKI_CONFIG,
            "commit": WIKI_COMMIT,
            "shard": WIKI_SHARD,
            "url": WIKI_URL,
            "shard_sha256": actual_sha,
            "shard_sha256_expected": WIKI_SHARD_SHA256,
            "shard_bytes": shard.stat().st_size,
            "license": "CC BY-SA 4.0 (Wikipedia)",
        })
        chunks = build_wiki_chunks(shard, args.max_chars)
    else:
        root = Path(args.local_root) if args.local_root else REPO_ROOT.parent.parent
        source_info.update({
            # 只记录目录名，不记录绝对路径（产物里不留本机路径）
            "root_name": root.name,
            "note": "本机已有中文文本：同级 projects 仓库的 README/文档/注释 + 工作区根目录 HTML 文档",
            "license": "本地自有内容",
        })
        chunks = build_local_chunks(root)

    split_info = write_split(chunks, out_dir, args.val_ratio, args.max_chars)
    source_info["split"] = split_info
    source_info["prepare_seconds"] = round(time.time() - t0, 2)
    source_info["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(source_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 统计写入 {stats_path}")
    print(json.dumps(source_info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
