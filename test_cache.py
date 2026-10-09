import json
import os
import sys

import torch

HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
        else os.getcwd())
for _p in (HERE, os.path.join(HERE, "data")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from eval import load_ckpt, render, pair_ids, greedy
from bpe import Tokenizer

STOP = (2, 13, 0, 1)
torch.manual_seed(0)
dev = torch.device("cpu")
model, cfg, _ = load_ckpt("run/best.pt", dev)
model.eval()
tok = Tokenizer.load("data/tokenizer.json")

L = 96
ids = torch.randint(0, cfg["vocab"], (1, L))
with torch.no_grad():
    full = model(ids)
    st = model.init_cache(1)
    pf = model(ids, st, 0)
print("prefill max abs diff: {:.3e}".format(float((full - pf).abs().max())))

with torch.no_grad():
    st = model.init_cache(1)
    model(ids[:, :40], st, 0)
    step_d = 0.0
    for i in range(40, L):
        got = model(ids[:, i:i + 1], st, i)
        ref = model(ids[:, :i + 1])
        step_d = max(step_d, float((got[0, -1] - ref[0, -1]).abs().max()))
print("streamed step max abs diff: {:.3e}".format(step_d))


def gen_cached(model, ids, max_new=24):
    st = model.init_cache(1)
    x = torch.tensor([ids], dtype=torch.long)
    out = []
    with torch.no_grad():
        lg = model(x, st, 0)
        nxt = int(lg[0, -1].argmax())
        p = len(ids)
        while True:
            if nxt in STOP:
                break
            out.append(nxt)
            if len(out) >= max_new:
                break
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
            lg = model(x[:, -1:], st, p)
            p += 1
            nxt = int(lg[0, -1].argmax())
    return out


tasks = [json.loads(l) for l in open("data/tasks_val.jsonl", encoding="utf-8")][:40]
mismatch = 0
total = 0
for t in tasks:
    for v in range(3):
        pr = render(t["chain"], t["q"], v)
        p_ids, a_ids = pair_ids(tok, pr, t["answer"])
        ref = greedy(model, tok, p_ids, dev, 24)
        got = tok.decode(gen_cached(model, p_ids, 24))
        total += 1
        if ref.strip() != got.strip():
            mismatch += 1
            if mismatch <= 3:
                print("MISMATCH ref={!r} got={!r}".format(ref, got))
print("greedy cache vs full: {} prompts, mismatches={}".format(total, mismatch))
