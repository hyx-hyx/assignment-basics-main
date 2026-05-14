import json
import re
from functools import lru_cache
from typing import Iterable, Iterator

from cs336_basics.BPE import BpeTrain


# 缓存字符编码结果
@lru_cache(maxsize=65536)
def _encode_char(c: str) -> bytes:
    """缓存单个字符的编码结果"""
    return c.encode(encoding="utf-8", errors="surrogateescape")


# 缓存子串编码元组
@lru_cache(maxsize=65536)
def _encode_tuple(substr: str) -> tuple:
    """缓存整个子串的编码元组"""
    return tuple(_encode_char(c) for c in substr)


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self._vocab_rev = {}
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self._merges_cache = {}

        # Your tokenizer should also support user provided special tokens (appending them to the vocabulary if they
        # aren’t already there).
        if special_tokens:
            for st in special_tokens:
                if st.encode() not in vocab.values():
                    vocab[len(vocab)] = st

        for k, v in vocab.items():
            self._vocab_rev[v] = k

        for m in self.merges:
            merge_byte = bytes(m[0] + m[1])
            self._merges_cache[merge_byte] = self._vocab_rev[merge_byte]

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding="utf-8") as vf:
            file_vocab = {v: k for k, v in json.load(vf).items()}
        with open(vocab_filepath, 'r', encoding="utf-8") as mf:
            file_merge = []
            for line in mf:
                [left, right] = line.split(' ')
                file_merge.append(
                    (left.encode("utf-8"), right.encode("utf-8")))
        return Tokenizer(file_vocab, file_merge, special_tokens)

    def encode(self, text: str) -> list[int]:
        encode_list = []
        text_byte_list = []
        delimiters = []

        # 支持用户自定义special_tokens
        if self.special_tokens:
            # 这个排序是为了解决重叠special_tokens，例如special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]
            # 优先匹配"<|endoftext|><|endoftext|>"
            self.special_tokens = sorted(self.special_tokens, reverse=True)
            pattern = '|'.join(map(re.escape, self.special_tokens))
            delimiters = re.findall(pattern, text)

            text_seg_list = re.split(pattern, text)
            for text_seg in text_seg_list:
                pre_tokenization_text = self._encode_pre_tokenization(text_seg)
                text_byte_list.append(pre_tokenization_text)
        else:
            pre_tokenization_text = self._encode_pre_tokenization(text)
            text_byte_list.append(pre_tokenization_text)

        for byte_list in text_byte_list:
            # 遍历每个预分词字符串
            for bytes_str in byte_list:
                # 重要优化：先合并为单字节，如果匹配到单字节，则直接加入编码列表
                single_byte = b"".join(b for b in bytes_str)
                if single_byte in self._vocab_rev.keys():
                    encode_list.append(self._vocab_rev[single_byte])
                    continue

                # 如果匹配到多字节，则逐个进行合并
                single_byte_list = []
                for byte_idx in bytes_str:
                    # 一个字节可能由多个bytes组成，需要逐个进行合并
                    for item in byte_idx:
                        single_byte_list.append(bytes([item]))
                idx = 0
                len_single_byte_list = len(single_byte_list)
                while idx < len_single_byte_list:
                    merge_flg = False
                    merge_true_byte = b''
                    for merge_len in range(2, len_single_byte_list - idx + 1):
                        single_byte = b''.join(
                            single_byte_list[idx:idx + merge_len])
                        if single_byte in self._merges_cache and len(merge_true_byte) < len(single_byte):
                            merge_flg = True
                            merge_true_byte = single_byte
                    if merge_flg:
                        encode_list.append(self._merges_cache[merge_true_byte])
                        idx += len(merge_true_byte)
                    else:
                        encode_list.append(
                            self._vocab_rev[single_byte_list[idx]])
                        idx += 1

            # 如果有分隔符，要在后面追加分隔符的token_id
            if len(delimiters) > 0:
                encode_list.append(self._vocab_rev[delimiters[0].encode()])
            delimiters = delimiters[1:]
        return encode_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            yield from self.encode(it)

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[i] for i in ids).decode("utf-8", errors='replace')

    def _encode_pre_tokenization(self, text: str):
        """
        Tokenizer.encode 使用的 预分词器
        """
        pre_tokenization_text = []
        # 使用预编译的正则表达式
        for m in BpeTrain.PRE_TOKENIZATION_PATTERN.finditer(text):
            substr = m.group()
            # 直接从缓存获取或计算编码元组
            str_encode = _encode_tuple(substr)
            # 更新计数
            pre_tokenization_text.append(str_encode)
        return pre_tokenization_text
