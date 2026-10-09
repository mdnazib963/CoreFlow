import json
import os
import sys

sys.path.insert(0, os.getcwd())

from cam.m0_data import build_corpus
from cam.m0_run import CoreFlowBackend, food_tokens, is_correct
from cam.read import format_block, retrieve

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

be = CoreFlowBackend("run/best.pt", "data/tokenizer.json", threads=1,
                     max_new=24, variant=2)
store, facts, questions, meta = build_corpus(seed=42, ns="ood")
foods = food_tokens(store)

items = []
for line in open("data/s3.ood.jsonl", encoding="utf-8"):
    txt = json.loads(line)["text"]
    if "\nA: " not in txt:
        continue
    pre, ans = txt.rsplit("\nA: ", 1)
    if ans == "I don't know.":
        continue
    items.append((pre, ans))
    if len(items) >= N:
        break


def split_pre(pre):
    if "[/MEM]" in pre:
        blk = pre.split("[/MEM]")[0] + "[/MEM]"
        rest = pre.split("[/MEM]", 1)[1]
    else:
        blk, rest = "", pre
    if "\nQ: " in rest:
        q = rest.split("\nQ: ", 1)[1].split("\nA: ", 1)[0]
    else:
        q = rest.split("\nQuestion: ", 1)[1].split("\nAnswer: ", 1)[0]
    return blk, q


def acc(rows):
    return sum(r[1] for r in rows) / max(1, len(rows))


with_blk = []
no_blk = []
for pre, ans in items:
    blk, q = split_pre(pre)
    p1 = be.generate(pre + "\nA: ")
    p2 = be.generate(be.build_prompt("", q))
    with_blk.append((ans, is_correct(p1, ans, foods), p1.strip(), ans))
    no_blk.append((ans, is_correct(p2, ans, foods), p2.strip(), ans))

print("s3.ood  n={}  original-prompt={:>5.1%}  no-block(recall)={:>5.1%}".format(
    len(items), acc(with_blk), acc(no_blk)))
for r in with_blk[:6]:
    print("   blk ok={} gold={:<14} pred={}".format(r[1], r[3], r[2]))
print("   --- no block ---")
for r in no_blk[:6]:
    print("   rec ok={} gold={:<14} pred={}".format(r[1], r[3], r[2]))

is_of, eat_of = {}, {}
for f in facts:
    if f["relation"] == "is" and f["text"].startswith("a "):
        is_of[f["subject"]] = f["object"]
    elif f["relation"] == "eats":
        eat_of[f["subject"]] = f["object"]

m0_blk = []
m0_no = []
for q in [x for x in questions if x["type"] == "2hop"][:N]:
    ent = q["q"].split()[3]
    cat = is_of[ent]
    lines = ["a {} is a {}".format(ent, cat), "{}s eat {}".format(cat, eat_of[cat])]
    blk = "[MEM]\n" + "\n".join("- {} (src: user)".format(l) for l in lines) + "\n[/MEM]"
    p1 = be.generate(be.build_prompt(blk, q["q"]))
    p2 = be.generate(be.build_prompt("", q["q"]))
    m0_blk.append((q["answer"], is_correct(p1, q["answer"], foods), p1.strip(), q["answer"]))
    m0_no.append((q["answer"], is_correct(p2, q["answer"], foods), p2.strip(), q["answer"]))

print("\nm0 ood  n={}  chain2-block={:>5.1%}  no-block(recall)={:>5.1%}".format(
    len(m0_blk), acc(m0_blk), acc(m0_no)))
for r in m0_blk[:6]:
    print("   blk ok={} gold={:<14} pred={}".format(r[1], r[3], r[2]))
print("   --- no block ---")
for r in m0_no[:6]:
    print("   rec ok={} gold={:<14} pred={}".format(r[1], r[3], r[2]))
