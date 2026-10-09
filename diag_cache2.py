import os
import sys

import torch

sys.path.insert(0, os.getcwd())

from eval import greedy, load_ckpt, pair_ids, render
from bpe import Tokenizer
from cam.m0_data import build_corpus
from cam.m0_run import CoreFlowBackend

tok = Tokenizer.load("data/tokenizer.json")
model, cfg, _ = load_ckpt("run/best.pt", torch.device("cpu"))
model.eval()
be = CoreFlowBackend("run/best.pt", "data/tokenizer.json", threads=1,
                     max_new=24, variant=2)

store, facts, questions, meta = build_corpus(seed=42, ns="ood")
is_of, eat_of = {}, {}
for f in facts:
    if f["relation"] == "is" and f["text"].startswith("a "):
        is_of[f["subject"]] = f["object"]
    elif f["relation"] == "eats":
        eat_of[f["subject"]] = f["object"]

prompts = []
for q in [x for x in questions if x["type"] == "2hop"][:5]:
    ent = q["q"].split()[3]
    cat = is_of[ent]
    mem = ["a {} is a {}".format(ent, cat), "{}s eat {}".format(cat, eat_of[cat])]
    prompts.append((render(mem, q["q"], 2), q["answer"]))

tasks = []
with open("data/tasks_val.jsonl", encoding="utf-8") as fh:
    for line in fh:
        tasks.append(__import__("json").loads(line))
        if len(tasks) >= 5:
            break
for t in tasks:
    prompts.append((render(t["chain"], t["q"], 2), t["answer"]))


def logits_first(prompt):
    ids = tok.encode(prompt)
    with torch.no_grad():
        full = model(torch.tensor([ids]))[0, -1]
    st = model.init_cache(1)
    with torch.no_grad():
        cached = model(torch.tensor([ids]), st, 0)[0, -1]
    return int(full.argmax()), int(cached.argmax()), float((full - cached).abs().max()), ids


same_step = 0
same_gen = 0
for i, (pr, ans) in enumerate(prompts):
    a_full, a_cache, diff, ids = logits_first(pr)
    g_full = greedy(model, tok, ids, torch.device("cpu"), 24).strip()
    g_cache = be.generate(pr).strip()
    same_step += int(a_full == a_cache)
    same_gen += int(g_full == g_cache)
    print("[{}] first-step full={} cache={} | dlogit={:.2e}".format(
        i, a_full, a_cache, diff))
    print("     gold={!r}".format(ans))
    print("     full={!r}".format(g_full))
    print("     cache={!r}".format(g_cache))

print("\nfirst-step agree {}/{} | greedy agree {}/{}".format(
    same_step, len(prompts), same_gen, len(prompts)))
