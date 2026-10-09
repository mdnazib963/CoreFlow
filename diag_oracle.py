import os
import random
import sys

sys.path.insert(0, os.getcwd())

from cam.m0_data import build_corpus
from cam.m0_run import CoreFlowBackend, food_tokens, has_evidence, is_correct
from cam.read import format_block, retrieve

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
V = int(sys.argv[2]) if len(sys.argv) > 2 else 2

store, facts, questions, meta = build_corpus(seed=42, ns="ood")
qs = [q for q in questions if q["type"] == "2hop"][:N]
foods = food_tokens(store)
be = CoreFlowBackend("run/best.pt", "data/tokenizer.json", threads=1,
                     max_new=24, variant=V)

texts = [f["text"] for f in facts]
is_of, eat_of = {}, {}
for f in facts:
    if f["relation"] == "is" and f["text"].startswith("a "):
        is_of[f["subject"]] = f["object"]
    elif f["relation"] == "eats":
        eat_of[f["subject"]] = f["object"]

rng = random.Random(0)


def wrap(mem):
    rng.shuffle(mem)
    return "[MEM]\n" + "\n".join("- {} (src: user)".format(l) for l in mem) + "\n[/MEM]"


def oracle_block(ent, cat):
    mem = ["a {} is a {}".format(ent, cat), "{}s eat {}".format(cat, eat_of[cat])]
    banned = [ent, cat]
    if rng.random() < 0.55:
        nd = rng.randrange(0, 5)
        used, tries = set(mem), 0
        while len(mem) < 2 + nd and tries < 30:
            tries += 1
            t = texts[rng.randrange(len(texts))]
            if t in used or any(b and b in t for b in banned):
                continue
            used.add(t)
            mem.append(t)
    return wrap(mem)


def chain_block(ent, cat):
    return wrap(["a {} is a {}".format(ent, cat),
                 "{}s eat {}".format(cat, eat_of[cat])])


def bare_block(cat):
    return wrap(["{}s eat {}".format(cat, eat_of[cat])])


res = {}
for tag in ("chain2", "oracle", "bare", "ret2", "ret4"):
    res[tag] = []

for q in qs:
    ent = q["q"].split()[3]
    cat = is_of[ent]
    gold = q["answer"]
    variants = {
        "chain2": chain_block(ent, cat),
        "oracle": oracle_block(ent, cat),
        "bare": bare_block(cat),
        "ret2": format_block(retrieve(store, q["q"], k=2), limit=2),
        "ret4": format_block(retrieve(store, q["q"], k=4), limit=4),
    }
    for tag, blk in variants.items():
        pred = be.generate(be.build_prompt(blk, q["q"]))
        res[tag].append({
            "ev": has_evidence(blk, gold),
            "ok": is_correct(pred, gold, foods),
            "pred": pred.strip(),
            "gold": gold,
            "blk": blk,
            "q": q["q"],
        })

print("variant {}  n={}".format(V, len(qs)))
for tag, rows in res.items():
    ev = sum(r["ev"] for r in rows) / len(rows)
    acc = sum(r["ok"] for r in rows) / len(rows)
    print("  {:<8} evidence={:>5.1%} acc={:>5.1%}".format(tag, ev, acc))

for tag in ("chain2", "ret4", "oracle"):
    print("\n--- {} (first 6) ---".format(tag))
    for r in res[tag][:6]:
        print("ok={} gold={:<12} pred={:<14} | {}".format(
            r["ok"], r["gold"], r["pred"], r["blk"].replace("\n", " / ")))
