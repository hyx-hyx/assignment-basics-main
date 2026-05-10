from __future__ import annotations

import cProfile
import multiprocessing
import os
import pathlib
import pstats
import time
from collections import defaultdict
from functools import wraps, lru_cache
from io import StringIO

import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries


def profile_section(section_name):
    """分析函数内部特定部分的装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 创建性能分析器
            pr = cProfile.Profile()
            pr.enable()

            result = func(*args, **kwargs)

            pr.disable()

            # 获取统计信息
            s = StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats('cumulative')

            # 输出结果
            print(f"\n{'=' * 50}")
            print(f"性能分析 - {section_name}")
            print(f"{'=' * 50}")
            ps.print_stats(20)  # 显示前20行
            print(s.getvalue()[:1000])  # 只显示前1000字符
            return result

        return wrapper

    return decorator


# 缓存字符编码结果
@lru_cache(maxsize=65536)
def _encode_char(c: str) -> bytes:
    """缓存单个字符的编码结果"""
    return c.encode(encoding="utf-8")


# 缓存子串编码元组
@lru_cache(maxsize=65536)
def _encode_tuple(substr: str) -> tuple:
    """缓存整个子串的编码元组"""
    return tuple(_encode_char(c) for c in substr)


class BpeTrain():
    # 静态变量 预编译正则表达式
    PRE_TOKENIZATION_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], ):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def _pre_tokenization(self, text: str):
        """
        优化版本：减少重复编码操作，使用缓存
        """
        bytes_dict = {}
        # 使用预编译的正则表达式
        for m in BpeTrain.PRE_TOKENIZATION_PATTERN.finditer(text):
            substr = m.group()

            # 直接从缓存获取或计算编码元组
            str_encode = _encode_tuple(substr)
            # 更新计数
            bytes_dict[str_encode] = bytes_dict.get(str_encode, 0) + 1
        return bytes_dict

    def _merge(self, bytes_list: list, char_dict_list: dict, max_pair, pairs):
        # merge
        max_c1, max_c2 = max_pair

        # 获取所有需要查询的单个字节键
        keys = [bytes([b]) for b in (max_c1 + max_c2)]
        # 获取所有对应的集合
        sets = [char_dict_list.get(key, set()) for key in keys]
        # 计算交集
        if sets:
            re_pair_bytes_list = set.intersection(*sets)
        else:
            re_pair_bytes_list = set()

        for key in re_pair_bytes_list:
            idx, v = key
            k = bytes_list[idx]
            key_str = b''.join(k)
            c1_c2_str = b''.join([max_c1, max_c2])
            if c1_c2_str in key_str:
                t = []

                # 清除这个key对应的pairs
                index_k_end = len(k) - 1
                for index in range(0, index_k_end):
                    p = (k[index], k[index + 1])
                    pairs[p] -= v

                index = 0
                while index < index_k_end:
                    (c1, c2) = (k[index], k[index + 1])
                    if tuple([c1, c2]) == max_pair:
                        t.append(c1 + c2)
                        index += 2
                    else:
                        t.append(c1)
                        index += 1

                if index == index_k_end:
                    t.append(k[index])

                # 添加最新的pairs
                index_t_end = len(t) - 1
                for index in range(0, index_t_end):
                    (c1, c2) = (t[index], t[index + 1])
                    pairs[(c1, c2)] = pairs.get((c1, c2), 0) + v
                bytes_list[idx] = tuple(t)
        return bytes_list

    def train(self):

        with open(self.input_path, "rb") as f:
            num_processes = multiprocessing.cpu_count()
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            vocab = {}
            merges = []
            vocab_rev = set()
            for i in range(0, 256):
                vocab[i] = bytes([i])
                vocab_rev.add(bytes([i]))

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks += chunk.split("<|endoftext|>")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            with multiprocessing.Pool(num_processes) as pool:
                multi_bytes_dict = pool.map(self._pre_tokenization, chunks)

            # 这里先把所有的分块bytes_list进行合并，统计整体的pre_token的出现次数
            result = defaultdict(int)
            for d in multi_bytes_dict:
                for key, value in d.items():
                    result[key] += value
            all_bytes_dict = dict(result)
            bytes_list = list(all_bytes_dict.keys())
            value_list = list(all_bytes_dict.values())

            pairs = {}
            char_dict_list = defaultdict(set)
            for idx, b in enumerate(bytes_list):
                v = value_list[idx]
                for c1, c2 in zip(b, b[1:]):
                    t = tuple([c1, c2])
                    pairs[t] = pairs.get(t, 0) + v
                    char_dict_list[c1].add((idx, v))
                    char_dict_list[c2].add((idx, v))

            while len(vocab) < self.vocab_size - len(self.special_tokens):
                max_value = max(pairs.values())
                max_pair = max([b for b, v in pairs.items() if v == max_value])
                (c1, c2) = max_pair
                new_word = c1 + c2
                if new_word not in vocab_rev:
                    # 添加到vocab
                    merges.append((c1, c2))
                    vocab[len(vocab)] = new_word
                    vocab_rev.add(new_word)
                    bytes_list = self._merge(bytes_list, char_dict_list, max_pair, pairs)
                pairs[max_pair] = 0

            for st in self.special_tokens:
                vocab[len(vocab)] = st.encode()
            return vocab, merges


if __name__ == "__main__":
    # start = time.time()
    # FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "./tests/fixtures"
    # input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    # trainer = BpeTrain(input_path, 1000, ["<|endoftext|>"])
    # vocab, merges = trainer.train()
    # time = time.time() - start
    # print(vocab)
    # print(merges)
    # print(f"耗时: {time:.3f}秒")

    test_string = "hello! こんにちは!"
    PRE_TOKENIZATION_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    bytes_dict = {}
    # 使用预编译的正则表达式
    for m in BpeTrain.PRE_TOKENIZATION_PATTERN.finditer(test_string):
        substr = m.group()

        # 直接从缓存获取或计算编码元组
        str_encode = _encode_tuple(substr)
        # 更新计数
        bytes_dict[str_encode] = bytes_dict.get(str_encode, 0) + 1
    print(bytes_dict)
    for k in bytes_dict:
        if b'\xe3' in k:
            print(k)
