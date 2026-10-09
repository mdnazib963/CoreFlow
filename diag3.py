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
foods = sorted({t["answer"] for t in ood} | {a for _, a in id_pairs})
print("id_pairs", len(id_pairs), "ood", len(ood), "foods", len(foods))

follow = memor = other = 0
N = 60
for i, (pref, ans) in enumerate(id_pairs[:N]):
    decoy = foods[(i * 7 + 3) % len(foods)]
    if decoy == ans:
        decoy = foods[(i * 7 + 4) % len(foods)]
    if ans not in pref:
        continue
    new_pref = pref.replace(ans, decoy)
    got = run(new_pref, decoy)
    if got == decoy:
        follow += 1
    elif got == ans:
        memor += 1
    else:
        other += 1
        if other <= 4:
            print("  other: gold={} decoy={} got={!r}".format(ans, decoy, got))
print("ID counterfactual (block says DECOY): follow={} memorize={} other={} / {}".format(
    follow, memor, other, N))

print("---- OOD outputs ----")
c = {}
for t in ood[:20]:
    pref = render(t["chain"], t["q"], 2)
    got = run(pref, t["answer"])
    c[got] = c.get(got, 0) + 1
    print("  q={} gold={} got={!r}".format(t["q"], t["answer"], got))
print("OOD distinct outputs:", len(c))

print("---- OOD counterfactual ----")
f2 = m2 = o2 = 0
for i, t in enumerate(ood[:N]):
    decoy = foods[(i * 11 + 5) % len(foods)]
    if decoy == t["answer"]:
        decoy = foods[(i * 11 + 6) % len(foods)]
    ch = [x.replace(t["answer"], decoy) for x in t["chain"]]
    pref = render(ch, t["q"], 2)
    got = run(pref, decoy)
    if got == decoy:
        f2 += 1
    elif got == t["answer"]:
        m2 += 1
    else:
        o2 += 1
        if o2 <= 4:
            print("  other: gold={} decoy={} got={!r}".format(t["answer"], decoy, got))
print("OOD counterfactual: follow={} memorize={} other={} / {}".format(f2, m2, o2, N))
