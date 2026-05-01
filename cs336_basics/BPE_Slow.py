import cProfile
import pstats
import time
from functools import wraps, lru_cache
from io import StringIO

import pathos
import regex as re

from tests.adapters import run_train_bpe

# 预编译正则表达式
PRE_TOKENIZATION_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


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
    return c.encode()


# 缓存子串编码元组
@lru_cache(maxsize=65536)
def _encode_tuple(substr: str) -> tuple:
    """缓存整个子串的编码元组"""
    return tuple(_encode_char(c) for c in substr)


def pre_tokenization(text: str) -> dict:
    """
    优化版本：减少重复编码操作，使用缓存
    """
    bytes_dict = {}
    # 使用预编译的正则表达式
    for m in PRE_TOKENIZATION_PATTERN.finditer(text):
        substr = m.group()

        # 直接从缓存获取或计算编码元组
        str_encode = _encode_tuple(substr)
        # 更新计数
        bytes_dict[str_encode] = bytes_dict.get(str_encode, 0) + 1
    return bytes_dict


def find_max_pair(bytes_dict: dict) -> (tuple, int):
    # find max_value
    pairs = {}
    for k, v in bytes_dict.items():
        for c1, c2 in zip(k, k[1:]):
            t = tuple([c1, c2])

            if t in pairs.keys():
                pairs[tuple([c1, c2])] += v
            else:
                pairs[tuple([c1, c2])] = v
    max_value = max(pairs.values())
    max_pair = max([k for k, v in pairs.items() if v == max_value])
    return max_pair


def merge(bytes_dict: dict, max_pair: (tuple, int)):
    # merge
    merged_bytes_dict = {}
    for k, v in bytes_dict.items():
        t = []
        index = 0

        while index < len(k) - 1:
            (c1, c2) = (k[index], k[index + 1])
            if tuple([c1, c2]) == max_pair:
                t.append(c1 + c2)
                index += 2
            else:
                t.append(c1)
                index += 1

        if index == len(k) - 1:
            t.append(k[index])
        merged_bytes_dict[tuple(t)] = v
    return merged_bytes_dict


if __name__ == "__main__":
    start = time.time()
    vocab, merges = run_train_bpe(r"../data/TinyStories/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"])
    print(vocab)
    print(merges)
    time = time.time() - start
    print(f"耗时: {time:.3f}秒")
