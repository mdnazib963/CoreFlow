import os
import sys
import time

import torch

HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
        else os.getcwd())
for _p in (HERE, os.path.join(HERE, "data")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from eval import load_ckpt, render

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.manual_seed(0)

model, cfg, _ = load_ckpt("run/best.pt", torch.device("cpu"))
model.eval()
ids = torch.randint(0, cfg["vocab"], (1, 1))

try:
    with torch.no_grad():
        st = model.init_cache(1)
        tr = torch.jit.trace(model, (ids, st, 7), strict=False)
    print("traced ok", flush=True)
except Exception as e:
    print("TRACE FAILED:", type(e).__name__, e, flush=True)
    sys.exit(0)

with torch.no_grad():
    st = model.init_cache(1)
    ref = model(ids, st, 7)
    st2 = model.init_cache(1)
    got = tr(ids, st2, 7)
print("traced vs eager max diff: {:.3e}".format(float((ref - got).abs().max())))

N = 500
with torch.no_grad():
    st = model.init_cache(1)
    model(ids, st, 7)
    t0 = time.perf_counter()
    for _ in range(N):
        model(ids, st, 7)
    e1 = (time.perf_counter() - t0) / N * 1000
    st2 = model.init_cache(1)
    tr(ids, st2, 7)
    t0 = time.perf_counter()
    for _ in range(N):
        tr(ids, st2, 7)
    e2 = (time.perf_counter() - t0) / N * 1000
print("eager  {:.3f} ms/step".format(e1))
print("jit    {:.3f} ms/step".format(e2))
print("speedup {:.2f}x".format(e1 / e2))
