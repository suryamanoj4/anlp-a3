from collections import Counter


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def char_accuracy(preds, refs):
    total = 0
    correct = 0
    for pred, ref in zip(preds, refs):
        total += len(ref)
        for i, ch in enumerate(ref):
            if i < len(pred) and pred[i] == ch:
                correct += 1
    return correct / total if total else 0.0


def word_accuracy(preds, refs):
    if not refs:
        return 0.0
    hits = sum(1 for pred, ref in zip(preds, refs) if pred.split() == ref.split())
    return hits / len(refs)


def macro_prf(preds, refs):
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for pred, ref in zip(preds, refs):
        width = max(len(pred), len(ref))
        for i in range(width):
            a = pred[i] if i < len(pred) else None
            b = ref[i] if i < len(ref) else None
            if a is not None and a == b:
                tp[b] += 1
            else:
                if a is not None:
                    fp[a] += 1
                if b is not None:
                    fn[b] += 1
    classes = set(tp) | set(fp) | set(fn)
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        precision += p
        recall += r
        f1 += 2 * p * r / (p + r) if p + r else 0.0
    n = len(classes) or 1
    return precision / n, recall / n, f1 / n


def mean_edit_distance(preds, refs):
    if not refs:
        return 0.0
    return sum(edit_distance(p, r) for p, r in zip(preds, refs)) / len(refs)
