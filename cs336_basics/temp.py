def segment_by_vocab(s: str, vocab: dict) -> list[str]:
    if not vocab or not s:
        return list(s) if s else []

    # 将词汇表按长度降序排序
    vocab_sorted = sorted(set(vocab), key=len, reverse=True)

    result = []
    i = 0
    n = len(s)

    while i < n:
        matched = False

        # 尝试所有词汇，从最长开始
        for word in vocab_sorted:
            if s.startswith(word, i):
                result.append(word)
                i += len(word)
                matched = True
                break

        if not matched:
            result.append(s[i])
            i += 1

    return result