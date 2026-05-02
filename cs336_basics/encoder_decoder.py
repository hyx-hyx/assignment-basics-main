import json

from BPE import pre_tokenization


def encode(text: str) -> list[int]:
    encode_list = []
    text = text.replace(" ", "Ġ")
    byte_list, _ = pre_tokenization(text)
    for b in byte_list:
        if b in special_tokens:
            encode_list.append(vocab_rev[b])
        idx = 0
        while idx < len(b) - 1:
            s1, s2 = b[idx], b[idx + 1]
            for m in merges:
                if s1 == m[0] and s2 == m[1]:
                    idx += 1
                    s1 = s1 + s2
                    if idx + 1 < len(b):
                        s2 = b[idx + 1]
                    else:
                        break
            if s1 + s2 in merges:
                s = s1 + s2
            else:
                s = s1
            encode_list.append(vocab_rev[s])
            idx += 1
        if idx == len(b) - 1:
            encode_list.append(vocab_rev[b[idx]])
    return encode_list


def decode(ids: list[int]) -> str:
    text=b''.join(vocab[i] for i in ids).decode("utf-8", errors='replace')
    return text.replace("Ġ"," ")


if __name__ == "__main__":
    with open("../tests/fixtures/gpt2_vocab.json",'r',encoding="utf-8") as vf:
        vocab_rev = {k.encode("utf-8"): v for k,v in json.load(vf).items()}
    with open("../tests/fixtures/gpt2_merges.txt",'r',encoding="utf-8") as mf:
        merges = []
        for line in mf:
            [left, right] = line.strip().split()
            merges.append((left.encode("utf-8"), right.encode("utf-8")))
    vocab = {v: k for k, v in vocab_rev.items()}
    special_tokens = [b'<|endoftext|>']
    in_str = "Hello, how are you?"
    output = encode(in_str)
    decode_output = decode(output)
    print(decode_output)
