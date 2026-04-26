import multiprocessing

import regex as re

from pretokenization_example import find_chunk_boundaries


def pre_tokenization(pat: str, text: str):
    bytes_dict = {}
    for m in re.finditer(pat, text):
        substr = m.group()
        str_encode = tuple(c.encode() for c in substr)
        if str_encode in bytes_dict.keys():
            bytes_dict[str_encode] += 1
        else:
            bytes_dict[str_encode] = 1
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


def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        num_processes = multiprocessing.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        vocab = {}
        merges = []
        for i in range(0, 256):
            vocab[i] = chr(i).encode()

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            bytes_dict = pre_tokenization(pat, chunk)

            for i in range(0, 20):
                max_pair = find_max_pair(bytes_dict)

                # 添加到vocab
                (c1, c2) = max_pair
                merges.append(max_pair)
                vocab[len(vocab)] = c1 + c2
                bytes_dict = merge(bytes_dict, max_pair)
        return vocab, merges


if __name__ == "__main__":
    vocab, merges = bpe_train(r"../data/TinyStories/TinyStoriesV2-GPT4-valid.txt", 30000, ["<|endoftext|>"])
    print(vocab)
    print(merges)

#     text = """Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
# Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
# Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
# They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit."""
#     pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#     bytes_dict = pre_tokenization(pat, text)
#     for i in range(0, 6):
#         max_pair = find_max_pair(bytes_dict)
#         print(f"max_pair={max_pair}")
#         bytes_dict = merge(bytes_dict, max_pair)
#         print(f"bytes_dict={bytes_dict}")
