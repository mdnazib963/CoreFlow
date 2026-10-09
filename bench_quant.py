import json
import os
import sys
import time

import torch
import torch.nn as nn

HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
        else os.getcwd())
for _p in (HERE, os.path.join(HERE, "data")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from eval import load_ckpt, render, pair_ids
from bench_cpu import sustained, gen_cached
from bpe import Tokenizer

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.manual_seed(0)
dev = torch.device("cpu")

model, cfg, _ = load_ckpt("run/best.pt", dev)
model.eval()
tok = Tokenizer.load("data/tokenizer.json")
NT = int(os.environ.get("NQT", "40"))
tasks = [json.loads(l) for l in open("data/tasks_val.jsonl", encoding="utf-8")][:NT]


def acc(m):
    ok = 0
    for t in tasks:
        for v in range(3):
            p = render(t["chain"], t["q"], v)
            p_ids, _ = pair_ids(tok, p, t["answer"])
            got = tok.decode(gen_cached(m, tok, p_ids, 24))
            if got.strip() == t["answer"].strip():
                ok += 1
    return round(100 * ok / (3 * len(tasks)), 2)


def report(tag, m, seconds):
    print("  running {} sustained {:.0f}s ...".format(tag, seconds), flush=True)
    sus, sec, n, p99 = sustained(m, tok, [render(t["chain"], t["q"], 2)
                                          for t in tasks], 1, seconds)
    print("  running {} accuracy ...".format(tag), flush=True)
    a = acc(m)
    print("{:<10} sustained {:>6} tok/s  p99 {:>6} ms  exact {:>6}%".format(
        tag, sus, p99, a), flush=True)
    return {"tag": tag, "sustained": sus, "p99_ms": p99, "exact": a}


out = {"fp32": report("fp32", model, 20)}

for tag, mods in (("int8_linear", {nn.Linear}), ("int8_lin_conv", {nn.Linear, nn.Conv1d})):
    try:
        q = torch.ao.quantization.quantize_dynamic(model, mods, dtype=torch.qint8)
        q.eval()
        out[tag] = report(tag, q, 40)
    except Exception as e:
        print(tag, "failed:", e, flush=True)

with open("run/bench_quant.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("wrote run/bench_quant.json", flush=True)
