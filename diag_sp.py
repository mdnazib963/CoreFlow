import json
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "data"))

from bpe import Tokenizer
from eval import greedy, load_ckpt, pair_ids, render
from cam.m0_data import build_corpus
from cam.m0_run import CoreFlowBackend

tok = Tokenizer.load("data/tokenizer.json")
dev = torch.device("cpu")
model, _, _ = load_ckpt("run/best.pt", dev)
model.eval()

store, facts, questions, meta = build_corpus(seed=42, ns="ood")
is_of, eat_of = {}, {}
for f in facts:
    if f["relation"] == "is" and f["text"].startswith("a "):
        is_of[f["subject"]] = f["object"]
    elif f["relation"] == "eats":
        eat_of[f["subject"]] = f["object"]

prompts = []
for q in [x for x in questions if x["type"] == "2hop"][:8]:
    ent = q["q"].split()[3]
    cat = is_of[ent]
    mem = ["a {} is a {}".format(ent, cat), "{}s eat {}".format(cat, eat_of[cat])]
    prompts.append((render(mem, q["q"], 2), q["answer"]))

with open("data/tasks_val.jsonl", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i >= 8:
            break
        t = json.loads(line)
        prompts.append((render(t["chain"], t["q"], 2), t["answer"]))


def last_ids(ids, n=4):
    return [(i, repr(tok.decode([i]))) for i in ids[-n:]]


ok_raw = ok_pair = 0
for pr, ans in prompts:
    raw = tok.encode(pr)
    p_ids, a_ids = pair_ids(tok, pr, ans)
    g_raw = greedy(model, tok, raw, dev, 24).strip()
    g_pair = greedy(model, tok, p_ids, dev, 24).strip()
    ok_raw += int(g_raw == ans.strip())
    ok_pair += int(g_pair == ans.strip())
    print("gold={:<14} raw={:<14} pair={:<14} same_ids={}".format(
        ans, repr(g_raw), repr(g_pair), raw == p_ids))
    if raw != p_ids:
        print("   raw  tail", last_ids(raw))
        print("   pair tail", last_ids(p_ids))

print("\nraw-prompt exact {}/{} | pair_ids-prompt exact {}/{}".format(
    ok_raw, len(prompts), ok_pair, len(prompts)))
