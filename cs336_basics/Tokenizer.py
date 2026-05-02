import json
import re
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
                if st.encode() not in vocab.values():
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
        text_byte_list=[]
        delimiters=[]
        # 支持用户自定义special_tokens
        if self.special_tokens:
            pattern = '|'.join(map(re.escape, self.special_tokens))
            delimiters = re.findall(pattern, text)

            text_seg_list=re.split(pattern, text)
            for text_seg in text_seg_list:
                byte_list, _ = pre_tokenization(text_seg)
                text_byte_list.append(byte_list)
        else:
            byte_list, _ = pre_tokenization(text)
            text_byte_list.append(byte_list)
        for byte_list in text_byte_list:
            # 遍历每个预分词字符串
            for bytes_str in byte_list:
                # 先合并为单字节，如果匹配到单字节，则直接加入编码列表
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
                while idx < len(single_byte_list):
                    merge_flg = False
                    for m in self.merges:
                        merge_bytes = bytes(m[0] + m[1])
                        single_byte = b''.join(
                            single_byte_list[idx:idx + min(len(merge_bytes), len(single_byte_list))])
                        if merge_bytes == single_byte:
                            encode_list.append(self._vocab_rev[merge_bytes])
                            merge_flg = True
                            idx+=len(merge_bytes)
                            break

                    if not merge_flg:
                        encode_list.append(self._vocab_rev[single_byte_list[idx]])
                        idx += 1
            # 如果有分隔符，要在后面追加分隔符的token_id
            if len(delimiters)>0:
                encode_list.append(self._vocab_rev[delimiters[0].encode()])
                delimiters=delimiters[1:]
        return encode_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            with open(it, 'r') as f:
                yield from self.encode(f)

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[i] for i in ids).decode("utf-8", errors='replace')
