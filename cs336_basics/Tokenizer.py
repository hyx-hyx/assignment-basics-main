import json
from typing import Iterable, Iterator

from cs336_basics.BPE import pre_tokenization


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # Your tokenizer should also support user provided special tokens (appending them to the vocabulary if they
        # aren’t already there).
        for st in special_tokens:
            if st not in vocab.values():
                vocab[len(vocab)] = st

        for k, v in vocab.items():
            self._vocab_rev[v] = k

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r') as vf:
            file_vocab = {v: k for k, v in json.load(vf)}
        with open(vocab_filepath, 'r') as mf:
            file_merge = []
            for line in mf:
                [left, right] = line.split(' ')
                file_merge.append((bytes(left), bytes(right)))
        return Tokenizer(file_vocab,file_merge,special_tokens)

    def encode(self, text: str) -> list[int]:
        encode_list = []
        byte_list, _ = pre_tokenization(text)
        for b in byte_list:
            if b in self.special_tokens:
                encode_list.append(self._vocab_rev[b])
            idx = 0
            while idx < len(b) - 1:
                s1, s2 = b[idx], b[idx + 1]
                for m in self.merges:
                    if s1 == m[0] and s2 == m[1]:
                        idx += 1
                        s1 = s1 + s2
                        if idx + 1 < len(b):
                            s2 = b[idx + 1]
                        else:
                            break
                if s1 + s2 in self.merges:
                    s = s1 + s2
                else:
                    s = s1
                encode_list.append(self._vocab_rev[s])
                idx += 1
            if idx == len(b) - 1:
                encode_list.append(self._vocab_rev[b[idx]])
        return encode_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            with open(it,'r') as f:
                yield from self.encode(f)

    def decode(self, ids: list[int]) -> str:
        out: str = ""
        for i in ids:
            out += bytes.decode(self.vocab[i], errors='\ufffd')
        return out
