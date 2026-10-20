"""把 token 流（uint16 二进制）当作 memmap 随机窗口采样器使用。

为什么用 memmap 而不是把 token 读进内存
---------------------------------------
本机整机内存常态空闲 < 1GB，语料编码后 token 数是千万级；memmap 让操作系统按页换入，
Python 侧常驻内存只有「当前 batch」那点。
"""

from __future__ import annotations

import numpy as np
import torch


class TokenStream:
    """一段连续的 token 流，支持随机窗口采样。"""

    def __init__(self, path: str, seq_len: int) -> None:
        self.path = path
        self.seq_len = seq_len
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n = int(self.data.shape[0])
        if self.n <= seq_len:
            raise ValueError(f"token 流太短: {self.n} <= seq_len+1")

    def __len__(self) -> int:
        return self.n

    @property
    def n_tokens(self) -> int:
        return self.n

    def window(self, start: int) -> torch.Tensor:
        """取 [start, start+seq_len+1) 的 token，返回长度 seq_len+1 的 LongTensor。"""
        chunk = np.array(self.data[start:start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk)

    def random_batch(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        """随机采 batch_size 个连续窗口 -> (batch_size, seq_len+1)。"""
        out = torch.empty((batch_size, self.seq_len + 1), dtype=torch.long)
        max_start = self.n - self.seq_len - 1
        for i in range(batch_size):
            start = int(torch.randint(0, max_start, (1,), generator=generator).item())
            out[i] = self.window(start)
        return out

    def sequential_batches(self, batch_size: int, n_batches: int) -> torch.Tensor:
        """固定顺序取前 n_batches 个窗口（验证集用，保证跨 step / 跨实现可比）。"""
        out = torch.empty((batch_size, self.seq_len + 1), dtype=torch.long)
        for b in range(n_batches):
            start = b * batch_size * self.seq_len
            if start + self.seq_len + 1 > self.n:
                break
            for i in range(batch_size):
                out[b, i] = self.window(start + i * self.seq_len)
        return out
