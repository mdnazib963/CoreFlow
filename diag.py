import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/content")

from eval import load_ckpt, render, greedy, score
from bpe import Tokenizer

tok = Tokenizer.load("/content/tokenizer.json")
model, cfg, st = load_ckpt("/content/run/best.pt", torch.device("cuda"))
model.eval()

tasks = [json.loads(l) for l in open("/content/tasks_val.jsonl", encoding="utf-8")][:3]
for t in tasks:
    pr = render(t["chain"], t["q"], 2)
    print("=" * 70)
    print("PROMPT:", repr(pr))
    p = tok.encode(pr)
    x = torch.tensor([p], dtype=torch.long, device="cuda")
    with torch.no_grad():
        lg = model(x)
    pr_top = lg[0, -1].softmax(-1).topk(8)
    print("gold answer:", repr(t["answer"]), "gold_ids:", tok.encode(t["answer"]))
    print("top8:", [(int(i), tok.decode([int(i)]), round(float(v), 4))
                    for v, i in zip(pr_top.values, pr_top.indices)])
    print("greedy:", repr(greedy(model, tok, pr, torch.device("cuda"))))


def eval_pairs(pairs, n, tag):
    first = exact = nll = ntok = 0
    for pref, ans in pairs[:n]:
        sc = score(model, tok, pref, ans, torch.device("cuda"))
        if sc is None:
            continue
        first += int(sc["first"])
        nll += sc["nll"]
        ntok += sc["ntok"]
        g = greedy(model, tok, pref, torch.device("cuda")).strip()
        exact += int(g == ans.strip())
        if exact and exact <= 2:
            print("  HIT:", repr(pref[-60:]), "->", repr(g))
    n2 = min(n, len(pairs))
    print("{}: n={} first_tok={:.4f} exact={:.4f} ans_ppl={:.3f}".format(
        tag, n2, first / max(1, n2), exact / max(1, n2),
        math.exp(nll / max(1, ntok))))
    return first, exact, n2


id_pairs = []
for name in ("s3.id.jsonl", "s2.id.jsonl"):
    for line in open("/content/" + name, encoding="utf-8"):
        txt = json.loads(line)["text"]
        if "\nA: " in txt:
            pref, ans = txt.rsplit("\nA: ", 1)
            id_pairs.append((pref + "\nA: ", ans))
        elif "\nAnswer: " in txt:
            pref, ans = txt.rsplit("\nAnswer: ", 1)
            id_pairs.append((pref + "\nAnswer: ", ans))

ood_pairs = []
for t in tasks:
    for v in range(3):
        ood_pairs.append((render(t["chain"], t["q"], v), t["answer"]))

ood_all = [json.loads(l) for l in open("/content/tasks_val.jsonl", encoding="utf-8")]
ood_pairs_all = []
for t in ood_all:
    for v in range(3):
        ood_pairs_all.append((render(t["chain"], t["q"], v), t["answer"]))

print("=" * 70)
eval_pairs(id_pairs, 300, "ID  (TRAIN_NS, s2/s3.id)")
eval_pairs(ood_pairs_all, 300, "OOD (TEST_NS, tasks_val)")
