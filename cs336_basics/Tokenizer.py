import json
from typing import Iterable, Iterator

from cs336_basics.BPE import pre_tokenization


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self._vocab_rev = {}
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # Your tokenizer should also support user provided special tokens (appending them to the vocabulary if they
        # aren’t already there).
        if special_tokens:
            for st in special_tokens:
                if st not in vocab.values():
                    vocab[len(vocab)] = st

        for k, v in vocab.items():
            self._vocab_rev[v] = k

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding="utf-8") as vf:
            file_vocab = {v: k for k, v in json.load(vf).items()}
        with open(vocab_filepath, 'r', encoding="utf-8") as mf:
            file_merge = []
            for line in mf:
                [left, right] = line.split(' ')
                file_merge.append((left.encode("utf-8"), right.encode("utf-8")))
        return Tokenizer(file_vocab, file_merge, special_tokens)

    def encode(self, text: str) -> list[int]:
        encode_list = []
        single_byte_list = []
        byte_list, _ = pre_tokenization(text)

        # 遍历每个预分词字符串
        for bytes_str in byte_list:
            # 先合并为单字节，如果匹配到单字节，则直接加入编码列表
            single_byte = b"".join(b for b in bytes_str)
            if single_byte in self._vocab_rev.keys():
                encode_list.append(self._vocab_rev[single_byte])
                continue
            # 如果匹配到多字节，则逐个进行合并
            for byte_idx in bytes_str:
                if self.special_tokens and byte_idx in self.special_tokens:
                    encode_list.append(self._vocab_rev[byte_idx])

                #一个字节可能由多个bytes组成，需要逐个进行合并
                for item in byte_idx:
                    single_byte_list.append(bytes([item]))
                idx = 0
                while idx < len(single_byte_list) - 1:
                    s1, s2 = single_byte_list[idx], single_byte_list[idx + 1]
                    for m in self.merges:
                        if s1 == m[0] and s2 == m[1]:
                            idx += 1
                            s1 = s1 + s2
                            if idx + 1 < len(single_byte_list):
                                s2 = single_byte_list[idx + 1]
                            else:
                                break
                    if s1 + s2 in self.merges:
                        s = s1 + s2
                    else:
                        s = s1
                    encode_list.append(self._vocab_rev[s])
                    idx += 1
                if idx == len(single_byte_list) - 1:
                    encode_list.append(self._vocab_rev[single_byte_list[idx]])
        return encode_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            with open(it, 'r') as f:
                yield from self.encode(f)

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[i] for i in ids).decode("utf-8", errors='replace')
