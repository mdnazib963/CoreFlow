import os
import sys
import time

import torch

HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
        else os.getcwd())
for _p in (HERE, os.path.join(HERE, "data")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from eval import load_ckpt
from model import warm_mlp

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.manual_seed(0)


def bench(f, n=20):
    f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t0) / n * 1000


model, cfg, _ = load_ckpt("run/best.pt", torch.device("cpu"))
model.eval()
warm_mlp(model)
cm = torch.compile(model, dynamic=False)
ids = torch.randint(0, cfg["vocab"], (1, 65))
one = torch.randint(0, cfg["vocab"], (1, 1))

with torch.inference_mode():
    st = model.init_cache(1)
    e_pre = bench(lambda: model(ids, st, 0))
    st = cm.init_cache(1)
    c_pre = bench(lambda: cm(ids, st, 0))
    st = model.init_cache(1)
    model(ids, st, 0)
    e_dec = bench(lambda: model(one, st, 70), 300)
    st = cm.init_cache(1)
    cm(ids, st, 0)
    c_dec = bench(lambda: cm(one, st, 70), 300)

print("prefill eager {:.2f} ms | compiled {:.2f} ms | {:.2f}x".format(
    e_pre, c_pre, e_pre / c_pre))
print("decode  eager {:.3f} ms | compiled {:.3f} ms | {:.2f}x".format(
    e_dec, c_dec, e_dec / c_dec))

import torch._dynamo.utils as du
c = du.counters
print("stats:", dict(c.get("stats", {})))
print("graph_breaks:", dict(c.get("graph_break", {})))
print("frames:", dict(c.get("frames", {})))
