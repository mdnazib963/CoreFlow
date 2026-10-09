import json
import sys

import torch

sys.path.insert(0, "/content")

from eval import load_ckpt, render, greedy, pair_ids
from bpe import Tokenizer

tok = Tokenizer.load("/content/tokenizer.json")
dev = torch.device("cuda")
model, cfg, st = load_ckpt("/content/run/best.pt", dev)
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
print("id", len(id_pairs), "train_foods", len(train_foods), "test_foods", len(test_foods))
assert not (set(train_foods) & set(test_foods)), "namespaces overlap!"


def counterfactual(pairs, pool, n=80, tag=""):
    follow = refuse = wrong = 0
    for i, (pref, ans) in enumerate(pairs[:n]):
        if ans not in pref:
            continue
        decoy = pool[(i * 13 + 7) % len(pool)]
        if decoy == ans:
            decoy = pool[(i * 13 + 8) % len(pool)]
        got = run(pref.replace(ans, decoy), decoy)
        if got == decoy:
            follow += 1
        elif got == "I don't know.":
            refuse += 1
        else:
            wrong += 1
            if wrong <= 5:
                print("    wrong: decoy={!r} got={!r}".format(decoy, got))
    tot = follow + refuse + wrong
    print("{}: follow={} refuse={} wrong={} / {}  ({:.1%} follow)".format(
        tag, follow, refuse, wrong, tot, follow / max(1, tot)))


counterfactual(id_pairs, train_foods, 80, "Q=known food=KNOWN ")
counterfactual(id_pairs, test_foods, 80, "Q=known food=UNKNOWN")

print("---- copy probe: does tail survive? ----")
for i, (pref, ans) in enumerate(id_pairs[:6]):
    decoy = test_foods[(i * 13 + 7) % len(test_foods)]
    got = run(pref.replace(ans, decoy), decoy)
    print("   decoy={!r} got={!r}".format(decoy, got))
