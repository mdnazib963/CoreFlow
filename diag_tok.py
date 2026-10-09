import os
import sys
from collections import Counter

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "data"))

from cam.m0_data import build_corpus
from bpe import Tokenizer

tok = Tokenizer.load("data/tokenizer.json")
counts = Counter(np.fromfile("data/train.bin", dtype=np.uint16).tolist())
n_tok = sum(counts.values())
print("train tokens {}  vocab covered {}/{}".format(
    n_tok, sum(1 for v in counts if counts[v] > 0), tok.vocab_size if hasattr(tok, "vocab_size") else "?"))


def first_id(prefix, answer):
    full = tok.encode(prefix + answer)
    pre = tok.encode(prefix)
    if full[:len(pre)] == pre:
        return full[len(pre)] if len(full) > len(pre) else None
    if len(pre) and full[:len(pre) - 1] == pre[:-1]:
        return full[len(pre) - 1]
    return None


def report(tag, items):
    freqs = []
    zero = 0
    unseen_tok = 0
    total_tok = 0
    for pre, ans in items:
        ids = tok.encode(ans)
        total_tok += len(ids)
        unseen_tok += sum(1 for i in ids if counts.get(i, 0) == 0)
        i = first_id(pre, ans)
        if i is None:
            continue
        c = counts.get(i, 0)
        freqs.append(c)
        zero += (c == 0)
    freqs.sort()
    if not freqs:
        print(tag, "no data")
        return
    print("{:<14} n={:<5} first never-seen={:>5.1%} | answer tok never-seen={:>5.1%} "
          "| freq p50={:<5}".format(
              tag, len(freqs), zero / len(freqs), unseen_tok / max(1, total_tok),
              freqs[len(freqs) // 2]))


for _ns in ("legacy", "ood"):
    store, facts, questions, meta = build_corpus(seed=42, ns=_ns)
    m0 = []
    for q in questions:
        if q["type"] == "2hop":
            m0.append(("Q: {}\nA: ".format(q["q"]), q["answer"]))
    report("m0 " + _ns, m0)

import json as _json
for name in ("s3.ood", "s2.ood", "s3.id", "s2.id"):
    items = []
    path = "data/{}.jsonl".format(name)
    if not os.path.exists(path):
        continue
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        txt = _json.loads(line).get("text", "")
        if "\nA: " not in txt:
            continue
        pre, ans = txt.rsplit("\nA: ", 1)
        if ans and ans != "I don't know.":
            items.append((pre + "\nA: ", ans))
        if len(items) >= 1000:
            break
    report(name, items)

i = first_id("Q: What does a brixd35 eat?\nA: ", "drotb662")
print("example gold first id", i, "freq", counts.get(i, 0), "repr", repr(tok.decode([i])))
for cand in (283, 499):
    print("  pred id", cand, "freq", counts.get(cand, 0), "repr", repr(tok.decode([cand])))
for sid in (1974, 125, 1208):
    print("  gold id", sid, "freq", counts.get(sid, 0), "repr", repr(tok.decode([sid])))
