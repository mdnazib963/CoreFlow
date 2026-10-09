import json
import re
import sys

import torch

sys.path.insert(0, "/content")

from eval import load_ckpt, render, greedy, pair_ids
from bpe import Tokenizer

tok = Tokenizer.load("/content/tokenizer.json")
dev = torch.device("cuda")
model, cfg, st = load_ckpt("/content/run3/best.pt", dev)
model.eval()


def run(pref, ans, max_new=24):
    p_ids, _ = pair_ids(tok, pref, ans)
    return greedy(model, tok, p_ids, dev, max_new).strip()


id_pairs = []
for name in ("s3.id.jsonl", "s2.id.jsonl"):
    for line in open("/content/" + name, encoding="utf-8"):
        txt = json.loads(line)["text"]
        for sep in ("\nA: ", "\nAnswer: "):
            if sep in txt:
                pr, ans = txt.rsplit(sep, 1)
                id_pairs.append((pr + sep, ans))
                break

ood = [json.loads(l) for l in open("/content/tasks_val.jsonl", encoding="utf-8")]
train_foods = sorted({a for _, a in id_pairs})
test_foods = sorted({t["answer"] for t in ood})
train_ents = []
for pref, _ in id_pairs:
    m = re.search(r"does a (\S+) eat\?", pref)
    if m:
        train_ents.append(m.group(1))
test_ents = []
for t in ood:
    m = re.search(r"does a (\S+) eat\?", t["q"])
    if m:
        test_ents.append(m.group(1))
print("train_foods", len(train_foods), "test_foods", len(test_foods),
      "train_ents", len(train_ents), "test_ents", len(test_ents))


def cell(ent_pool, food_pool, tag, n=60):
    follow = refuse = wrong = 0
    ex = []
    for i, (pref, ans) in enumerate(id_pairs):
        if follow + refuse + wrong >= n:
            break
        if ans not in pref:
            continue
        m = re.search(r"does a (\S+) eat\?", pref)
        if not m:
            continue
        old_ent = m.group(1)
        decoy_f = food_pool[(i * 13 + 7) % len(food_pool)]
        if decoy_f == ans:
            decoy_f = food_pool[(i * 13 + 8) % len(food_pool)]
        new_ent = ent_pool[(i * 17 + 3) % len(ent_pool)]
        if new_ent == old_ent:
            new_ent = ent_pool[(i * 17 + 4) % len(ent_pool)]
        p2 = re.sub(r"\b" + re.escape(old_ent) + r"\b", new_ent, pref)
        p2 = p2.replace(ans, decoy_f)
        got = run(p2, decoy_f)
        if got == decoy_f:
            follow += 1
            if len(ex) < 3:
                ex.append(("OK", new_ent, decoy_f, got))
        elif got == "I don't know.":
            refuse += 1
        else:
            wrong += 1
            if len(ex) < 3:
                ex.append(("WRONG", new_ent, decoy_f, got))
    tot = max(1, follow + refuse + wrong)
    print("{:26s} follow={:2d} refuse={:2d} wrong={:2d}  ({:.0%})".format(
        tag, follow, refuse, wrong, follow / tot))
    for e, a, g, gt in ex:
        print("      {} ent={} decoy={!r} got={!r}".format(e, a, g, gt))


cell(train_ents, train_foods, "TRAIN ent + TRAIN food")
cell(train_ents, test_foods, "TRAIN ent + TEST food")
cell(test_ents, train_foods, "TEST  ent + TRAIN food")
cell(test_ents, test_foods, "TEST  ent + TEST  food")
