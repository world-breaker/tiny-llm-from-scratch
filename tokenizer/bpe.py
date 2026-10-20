"""纯 Python 实现的 BPE 分词器（不依赖 `tokenizers` / `sentencepiece` 任何预训练资源）。

为什么自己写
------------
规格要求「分词器也从零训」「不依赖 tokenizers 库」。这里用标准的 **BPE（Byte Pair
Encoding）** 算法，唯一的工程改动是：把「词」的定义从「空格分隔」换成 **字符类别分段**
（汉字串 / 拉丁串 / 数字串 / 单个其它字符），这样中文才能在没有分词器的情况下被正确合并。

核心概念
--------
* **base token**：语料中出现的每个字符（汉字按字起），保证 **encode→decode 往返无损**。
* **merge**：(a, b) -> "ab" 的合并规则，按学习顺序排名；编码时按排名贪心合并。
* **UNK**：语料中没出现过的字符（模型无法表示），统计其比例。

关键实现取舍
------------
1. **训练期只学合并规则**（全局 pair→rank），不学词表条目；因此罕见词类型即使没被统计到，
   编码时依然会套用学到的合并规则，且仍然能逐字回退到 base token —— **不会破坏往返无损**。
2. **合并统计用「词类型频次表」而非原始语料**，并对词类型数设上限；这让纯 Python 能在
   有限内存（本机整机空闲 < 1GB）下完成训练。
3. **增量式 pair 统计**：维护 ``pair -> 出现该 pair 的词下标集合``，每轮合并只重算受影响的词，
   复杂度从 O(轮数 × 全词表) 降到接近 O(总 pair 实例数)。
"""

from __future__ import annotations

import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# 字符一律 intern：语料里同一个字出现几十万次，intern 后所有词类型共享同一批 str 对象，
# 词频表内存从数百 MB 降到几十 MB（本机整机空闲内存 < 1GB，这是关键取舍）。
_intern = sys.intern

# ---------------------------------------------------------------------------
# 字符类别
# ---------------------------------------------------------------------------
_HAN_RANGES = (
    (0x4E00, 0x9FFF),    # CJK 统一表意文字
    (0x3400, 0x4DBF),    # 扩展 A
    (0xF900, 0xFAFF),    # 兼容表意文字
    (0x20000, 0x2A6DF),  # 扩展 B
    (0x2A700, 0x2EBEF),  # 扩展 C-F
    (0x2F800, 0x2FA1F),  # 兼容补充
)
_LATIN = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_DIGIT = set("0123456789")
_SPACE = set(" \t\n\r\v\f\u00a0\u3000")

MAX_WORD_CHARS = 32  # 超长汉字串切块上限（维基正文中极少触发）


def _is_han(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _HAN_RANGES:
        if lo <= cp <= hi:
            return True
        if cp < lo:
            break
    return False


def pre_tokenize(text: str) -> Iterator[list[str]]:
    """把文本切成「词」（每个词是字符列表）。

    规则：
    * 连续汉字  -> 一个词（超过 ``MAX_WORD_CHARS`` 则切块）
    * 连续拉丁字母 -> 一个词
    * 连续数字  -> 一个词
    * 其它字符（标点、空白、符号）-> 单字词
    """
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if _is_han(ch):
            j = i
            while j < n and _is_han(text[j]) and (j - i) < MAX_WORD_CHARS:
                j += 1
            yield [_intern(c) for c in text[i:j]]
            i = j
        elif ch in _LATIN:
            j = i
            while j < n and text[j] in _LATIN:
                j += 1
            yield [_intern(c) for c in text[i:j]]
            i = j
        elif ch in _DIGIT:
            j = i
            while j < n and text[j] in _DIGIT:
                j += 1
            yield [_intern(c) for c in text[i:j]]
            i = j
        else:
            yield [_intern(ch)]
            i += 1


# ---------------------------------------------------------------------------
# BPE 训练
# ---------------------------------------------------------------------------
def train_bpe(
    word_freq: dict[tuple[str, ...], int],
    num_merges: int,
    verbose: bool = False,
) -> list[tuple[str, str]]:
    """在词类型频次表上学习 BPE 合并规则，返回按学习顺序排列的 pair 列表。

    增量算法：``pair_counts`` 维护 pair 频次，``pair_idx`` 维护 pair -> 词下标集合，
    每轮合并只重算受影响词；用堆（惰性失效）取当前最高频 pair，并列时按 pair 字典序
    取小者，保证 **训练结果完全可复现**。
    """
    words = [list(w) for w in word_freq]
    freqs = [word_freq[w] for w in word_freq]
    pair_counts: Counter = Counter()
    pair_idx: dict[tuple[str, str], set[int]] = defaultdict(set)

    for wi, w in enumerate(words):
        f = freqs[wi]
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            pair_counts[p] += f
            pair_idx[p].add(wi)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges: list[tuple[str, str]] = []
    log_every = max(1, num_merges // 10)

    for step in range(num_merges):
        # 取频次最高的 pair（并列取字典序最小）
        best = None
        while heap:
            negc, p = heapq.heappop(heap)
            cur = pair_counts.get(p)
            if cur is not None and cur == -negc and cur > 0:
                best = p
                break
        if best is None:
            break
        merged = best[0] + best[1]
        merges.append(best)
        affected = list(pair_idx.pop(best, ()))
        for wi in affected:
            w = words[wi]
            f = freqs[wi]
            old = [(w[i], w[i + 1]) for i in range(len(w) - 1)]
            new_w: list[str] = []
            k = 0
            while k < len(w):
                if k + 1 < len(w) and w[k] == best[0] and w[k + 1] == best[1]:
                    new_w.append(merged)
                    k += 2
                else:
                    new_w.append(w[k])
                    k += 1
            new = [(new_w[i], new_w[i + 1]) for i in range(len(new_w) - 1)]
            new_set = set(new)
            for p in old:
                c = pair_counts.get(p)
                if c is None:
                    continue
                c -= f
                if c <= 0:
                    del pair_counts[p]
                    pair_idx.pop(p, None)
                else:
                    pair_counts[p] = c
                    if p not in new_set:
                        pair_idx[p].discard(wi)
            words[wi] = new_w
            for p in new:
                c = pair_counts.get(p, 0) + f
                pair_counts[p] = c
                pair_idx[p].add(wi)
                heapq.heappush(heap, (-c, p))
        if len(heap) > 2 * len(pair_counts) + 200_000:
            heap = [(-c, p) for p, c in pair_counts.items()]
            heapq.heapify(heap)
        if verbose and (step + 1) % log_every == 0:
            print(f"  merge {step+1}/{num_merges}  best={merged!r}  freq={-heap[0][0] if heap else 0}"
                  f"  pairs={len(pair_counts)}", flush=True)
    return merges


# ---------------------------------------------------------------------------
# 分词器
# ---------------------------------------------------------------------------
class BPETokenizer:
    """训练好的 BPE 分词器。"""

    def __init__(
        self,
        tokens: Sequence[str],
        merges: Sequence[tuple[str, str]],
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self.tokens: list[str] = list(tokens)
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.merges: list[tuple[str, str]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.ranks: dict[tuple[str, str], int] = {m: i for i, m in enumerate(self.merges)}
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        self.unk_id: int = self.special_tokens.get("<unk>", 0)
        self.pad_id: int = self.special_tokens.get("<pad>", 0)
        self.bos_id: int = self.special_tokens.get("<bos>", 1)
        self.eos_id: int = self.special_tokens.get("<eos>", 2)
        self._cache: dict[tuple[str, ...], list[str]] = {}
        self._cache_limit = 400_000

    # -- 基础属性 ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    # -- 编码 -------------------------------------------------------------
    def _bpe_word(self, word: tuple[str, ...]) -> list[str]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        chars = list(word)
        if len(chars) > 1:
            while len(chars) > 1:
                best_rank = None
                best_i = -1
                for i in range(len(chars) - 1):
                    r = self.ranks.get((chars[i], chars[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank = r
                        best_i = i
                if best_i < 0:
                    break
                chars[best_i:best_i + 2] = [chars[best_i] + chars[best_i + 1]]
        if len(self._cache) < self._cache_limit:
            self._cache[word] = chars
        return chars

    def encode_with_unk(self, text: str) -> tuple[list[int], int]:
        """返回 (token ids, UNK 个数)。"""
        ids: list[int] = []
        unk = 0
        for word in pre_tokenize(text):
            for piece in self._bpe_word(tuple(word)):
                tid = self.token_to_id.get(piece)
                if tid is None:
                    tid = self.unk_id
                    unk += 1
                ids.append(tid)
        return ids, unk

    def encode(self, text: str) -> list[int]:
        return self.encode_with_unk(text)[0]

    # -- 解码 -------------------------------------------------------------
    def decode(self, ids: Iterable[int]) -> str:
        special_ids = set(self.special_tokens.values())
        out: list[str] = []
        for i in ids:
            if i in special_ids:
                continue
            if 0 <= i < len(self.tokens):
                out.append(self.tokens[i])
        return "".join(out)

    def id_to_token(self, i: int) -> str:
        return self.tokens[i] if 0 <= i < len(self.tokens) else self.special_tokens.get("<unk>", "<unk>")

    # -- 存取 -------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "tiny-llm-bpe-v1",
            "special_tokens": self.special_tokens,
            "tokens": self.tokens,
            "merges": [list(m) for m in self.merges],
        }
        path = d / "tokenizer.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        d = Path(directory)
        path = d / "tokenizer.json" if d.is_dir() else d
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            tokens=payload["tokens"],
            merges=[tuple(m) for m in payload["merges"]],
            special_tokens=payload.get("special_tokens", {}),
        )


def load_tokenizer(directory: str | Path) -> BPETokenizer:
    return BPETokenizer.load(directory)


# ---------------------------------------------------------------------------
def build_tokenizer(
    text_lines: Iterable[str],
    target_vocab: int = 16000,
    max_word_types: int = 120_000,
    min_merges: int = 1000,
    max_chars: int = 0,
    verbose: bool = True,
) -> tuple[BPETokenizer, dict]:
    """从文本行训练一个 BPE 分词器。

    返回 (tokenizer, stats)。统计里含 UNK 率、压缩率等可写入 README 的实测数字。

    ``max_chars > 0`` 时只用语料的前 ``max_chars`` 个字符来学合并规则
    （合并规则对语料规模不敏感，但词频表内存随规模线性增长；这是本机的内存取舍）。
    """
    special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    word_freq: Counter = Counter()
    char_freq: Counter = Counter()
    n_chars = 0            # 参与「合并规则学习」的字符数
    n_chars_total = 0      # 语料总字符数（字符统计永远扫全量）
    n_words = 0
    for line in text_lines:
        line = line.strip()
        if not line:
            continue
        n_chars_total += len(line)
        # 字符统计必须覆盖 **全部** 语料：base token 少一个字符，
        # 该字符就会被编码成 UNK，解码时被丢掉 —— 往返无损就破了。
        char_freq.update(line)
        if max_chars and n_chars >= max_chars:
            continue
        n_chars += len(line)
        for w in pre_tokenize(line):
            word_freq[tuple(w)] += 1
            n_words += 1
    if verbose:
        print(f"[bpe] 语料 {n_chars_total} 字符 (合并学习用 {n_chars}), {n_words} 词次, "
              f"{len(word_freq)} 种词类型, {len(char_freq)} 种字符")

    # 训练用词类型上限（按频次取前 N，保证可复现：同频按词本身排序）
    if len(word_freq) > max_word_types:
        ranked = sorted(word_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        word_freq = Counter(dict(ranked[:max_word_types]))
        if verbose:
            print(f"[bpe] 词类型裁剪到 {len(word_freq)}")

    base_chars = sorted(char_freq.items(), key=lambda kv: (-kv[1], kv[0]))
    n_base = len(base_chars) + len(special)
    num_merges = max(min_merges, target_vocab - n_base)
    if verbose:
        print(f"[bpe] base token {n_base} 个, 计划学习 {num_merges} 条合并规则")

    merges = train_bpe(dict(word_freq), num_merges, verbose=verbose)

    tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    tokens.extend(ch for ch, _ in base_chars)
    for a, b in merges:
        tokens.append(a + b)
    tok = BPETokenizer(tokens, merges, special)

    stats = {
        "n_chars": n_chars,
        "n_chars_total": n_chars_total,
        "n_words": n_words,
        "n_word_types": len(word_freq),
        "n_base_chars": len(base_chars),
        "n_merges": len(merges),
        "vocab_size": len(tok),
        "target_vocab": target_vocab,
    }
    return tok, stats
