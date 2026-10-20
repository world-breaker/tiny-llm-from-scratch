"""分词器单测：往返无损、UNK 率、词表规模、特殊 token、增量合并。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tokenizer.bpe import BPETokenizer, pre_tokenize, train_bpe  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "tokenizer" / "artifacts"


def test_pre_tokenize_class_split():
    """汉字串 / 拉丁串 / 数字串 分别成词，其它字符单独成词。"""
    words = list(pre_tokenize("中国abc123，测试"))
    assert ["中", "国"] in words
    assert ["a", "b", "c"] in words
    assert ["1", "2", "3"] in words
    assert ["，"] in words
    # 汉字串不跨标点断开
    assert ["测", "试"] in words


def test_pre_tokenize_long_han_run_split():
    """超长汉字串按 MAX_WORD_CHARS 切块，不会无限增长。"""
    from tokenizer.bpe import MAX_WORD_CHARS

    words = list(pre_tokenize("中" * 100))
    assert all(len(w) <= MAX_WORD_CHARS for w in words)
    assert sum(len(w) for w in words) == 100


def _train_tiny() -> BPETokenizer:
    """在极小语料上训一个分词器，用于单测（秒级）。

    注意：base token 来自训练语料的字符集合，所以语料里出现过的所有字符
    （包括标点）都必须留在行里，否则那些字符会变成 UNK、往返就会丢字。
    """
    from tokenizer.bpe import build_tokenizer

    lines = (_CORPUS.replace("。", "。\n")).splitlines()
    tok, _ = build_tokenizer(lines, target_vocab=300, max_word_types=5000, min_merges=20,
                             verbose=False)
    return tok


# 训练语料：必须覆盖下面所有测试样例里用到的字符
_CORPUS = (
    "中国是一个国家，首都在北京。上海是最大的城市。"
    "北京大学是中国的著名大学，清华大学也是。"
    "人工智能是研究智能的分支，机器学习是重要方法。"
    "混合 mixed 文本 text 123 是一个测试。"
    "第一个人第一次来到中国，测试结束。"
    "人工智能与机器学习，结尾部分!符号@和#也有。abc abc abc !"
) * 20


def test_roundtrip_is_lossless():
    """核心契约：encode -> decode 必须等于原文。"""
    tok = _train_tiny()
    samples = [
        "中国是一个国家，首都在北京。",
        "上海是最大的城市",
        "人工智能与机器学习",
        "abc 123 !@#",
        "混合 mixed 文本 text 123",
        "第一个人第一次来到中国",
    ]
    for s in samples:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, f"往返不一致: {s!r} -> {tok.decode(ids)!r}"


def test_roundtrip_on_long_text():
    tok = _train_tiny()
    s = ("北京大学是中国的著名大学。" * 30) + "结尾"
    assert tok.decode(tok.encode(s)) == s


def test_unk_rate_is_zero_for_known_text():
    """语料内字符不应该出现 UNK。"""
    tok = _train_tiny()
    ids, unk = tok.encode_with_unk("中国是一个国家")
    assert unk == 0
    assert all(i != tok.unk_id for i in ids)


def test_unk_appears_for_characters_outside_corpus():
    """语料外的字符会退化成 UNK（这是设计行为，不是 bug）。"""
    tok = _train_tiny()
    ids, unk = tok.encode_with_unk("中国🦄")
    assert unk == 1
    assert tok.unk_id in ids


def test_vocab_size_in_range():
    """产出的分词器词表落在规格要求的 8k~16k 区间（若 artifacts 已生成）。"""
    if not (ARTIFACT_DIR / "tokenizer.json").exists():
        pytest.skip("尚未训练分词器 artifacts")
    tok = BPETokenizer.load(ARTIFACT_DIR)
    assert 8000 <= len(tok) <= 16000


def test_saved_tokenizer_roundtrip():
    """保存 / 载入后行为一致（磁盘往返）。"""
    if not (ARTIFACT_DIR / "tokenizer.json").exists():
        pytest.skip("尚未训练分词器 artifacts")
    tok = BPETokenizer.load(ARTIFACT_DIR)
    s = "维基百科是自由的百科全书"
    assert tok.decode(tok.encode(s)) == s


def test_train_bpe_is_deterministic():
    """同样的输入 -> 同样的合并规则（可复现）。"""
    from collections import Counter

    words = Counter()
    for _ in range(5):
        for w in pre_tokenize("中国人民银行行长"):
            words[tuple(w)] += 1
    a = train_bpe(dict(words), 5)
    b = train_bpe(dict(words), 5)
    assert a == b
    assert len(a) == 5


def test_merges_are_applied_by_rank():
    """学到的合并规则确实在编码时被套用（合并后的 token 在词表里）。"""
    tok = _train_tiny()
    ids = tok.encode("北京大学")
    pieces = [tok.id_to_token(i) for i in ids]
    assert len(pieces) < len("北京大学")  # 发生了合并
    assert "".join(pieces) == "北京大学"


def test_special_tokens_are_skipped_on_decode():
    tok = _train_tiny()
    ids = [tok.bos_id] + tok.encode("测试") + [tok.eos_id]
    assert tok.decode(ids) == "测试"