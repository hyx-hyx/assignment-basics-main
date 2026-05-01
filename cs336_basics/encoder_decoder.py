from BPE import pre_tokenization


def encode(text: str) -> list[int]:
    vocab_rev = {}
    encode_list = []
    for k, v in vocab.items():
        vocab_rev[v] = k

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
    out: str = ""
    for i in ids:
        out += bytes.decode(vocab[i], errors='\ufffd')
    return out


if __name__ == "__main__":
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    special_tokens = [b'<|endoftext|>']
    in_str = "the cat ate"
    output = encode(in_str)
    decode_output = decode(output)
    print(output)
    print(decode_output)
