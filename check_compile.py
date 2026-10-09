import json
import os
import sys

import torch

HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
        else os.getcwd())
for _p in (HERE, os.path.join(HERE, "data")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from eval import load_ckpt, render, pair_ids
from bench_cpu import gen_cached
from bpe import Tokenizer
from model import warm_mlp

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.manual_seed(0)

model, cfg, _ = load_ckpt("run/best.pt", torch.device("cpu"))
model.eval()
warm_mlp(model)
cm = torch.compile(model, dynamic=False)
tok = Tokenizer.load("data/tokenizer.json")
tasks = [json.loads(l) for l in open("data/tasks_val.jsonl", encoding="utf-8")][:40]

ok_e = ok_c = diff = 0
for t in tasks:
    for v in range(3):
        p = render(t["chain"], t["q"], v)
        p_ids, _ = pair_ids(tok, p, t["answer"])
        ge = tok.decode(gen_cached(model, tok, p_ids, 24))
        gc = tok.decode(gen_cached(cm, tok, p_ids, 24))
        ok_e += ge.strip() == t["answer"].strip()
        ok_c += gc.strip() == t["answer"].strip()
        diff += ge.strip() != gc.strip()
n = 3 * len(tasks)
print("eager exact {:.1f}%  compiled exact {:.1f}%  eager!=compiled {}".format(
    100 * ok_e / n, 100 * ok_c / n, diff))
