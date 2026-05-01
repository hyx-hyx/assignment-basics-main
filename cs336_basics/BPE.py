import cProfile
import pstats
import time
from functools import wraps, lru_cache
from io import StringIO

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


def pre_tokenization(text: str):
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
    return list(bytes_dict.keys()), list(bytes_dict.values())


def merge(bytes_list: list, char_dict_list: dict, max_pair, pairs):
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
            clear_pairs = []
            for index in range(0, len(k) - 1):
                p = (k[index], k[index + 1])
                pairs[p] -= v
                index += 1

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

            # 添加最新的pairs
            for index in range(0, len(t) - 1):
                (c1, c2) = (t[index], t[index + 1])
                pairs[(c1, c2)] = pairs.get((c1, c2), 0) + v
                index += 1
            bytes_list[idx] = tuple(t)
    return bytes_list


if __name__ == "__main__":
    # text = r"""low low low low low
    # lower lower widest widest widest
    # newest newest newest newest newest newest"""
    # # 初始化char_dict_list
    # bytes_list, value_list = pre_tokenization(text)
    # vocab = {}
    # vocab_rev = set()
    # merges = []
    # pairs = {}
    # char_dict_list = defaultdict(set)
    #
    # # get pairs
    # for idx, b in enumerate(bytes_list):
    #     v = value_list[idx]
    #     for c1, c2 in zip(b, b[1:]):
    #         t = tuple([c1, c2])
    #         if t in pairs.keys():
    #             pairs[t] += v
    #         else:
    #             pairs[t] = v
    #         char_dict_list[c1].add((idx, v))
    #         char_dict_list[c2].add((idx, v))
    #
    # for _ in range(0, 500):
    #     max_value = max(pairs.values())
    #     max_pair = max([k for k, v in pairs.items() if v == max_value])
    #     (c1, c2) = max_pair
    #     new_word = c1 + c2
    #     if new_word not in vocab_rev:
    #         # 添加到vocab
    #         merges.append((c1, c2))
    #         vocab[len(vocab)] = new_word
    #         vocab_rev.add(new_word)
    #         bytes_list = merge(bytes_list, char_dict_list, max_pair, pairs)
    #     pairs[max_pair] = 0
    #
    # print(vocab)
    # print(merges)

    start = time.time()
    vocab, merges = run_train_bpe(r"../data/TinyStories/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"])
    print(vocab)
    print(merges)
    time = time.time() - start
    print(f"耗时: {time:.3f}秒")
