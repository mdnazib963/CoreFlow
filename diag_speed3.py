import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
        else os.getcwd())
for _p in (HERE, os.path.join(HERE, "data")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from eval import load_ckpt

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.manual_seed(0)
N = 2000


def t(fn, n=N):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


model, cfg, _ = load_ckpt("run/best.pt", torch.device("cpu"))
model.eval()
for _p_ in model.parameters():
    _p_.requires_grad_(False)
x = torch.randn(1, 1, 384)
h = model.blocks[0].n1(x)
buf = torch.randn(1, 384, 3)
w = model.blocks[0].dw.weight
print("{:<40} {:>9}".format("op", "ms"))

hv = torch.cat([buf, h.transpose(1, 2)], dim=2)
print("{:<40} {:>9.4f}".format("conv path cat+conv1d+T", t(lambda: (
    model.blocks[0].dw(torch.cat([buf, h.transpose(1, 2)], dim=2)).transpose(1, 2)))))
print("{:<40} {:>9.4f}".format("elementwise t=1 (mul+sum)", t(lambda: (
    (hv * w.squeeze(1)).sum(-1, keepdim=True).transpose(1, 2)))))
r1 = model.blocks[0].dw(torch.cat([buf, h.transpose(1, 2)], dim=2)).transpose(1, 2)
r2 = (hv * w.squeeze(1)).sum(-1, keepdim=True).transpose(1, 2)
print("  conv equivalence max diff: {:.3e}".format(float((r1 - r2).abs().max())))

at = model.blocks[11]
kv = torch.randn(1, 2, 32, 64)
print("{:<40} {:>9.4f}".format("repeat_interleave", t(lambda: kv.repeat_interleave(3, 1))))

lg_head_w = model.wte.weight
with torch.no_grad():
    print("{:<40} {:>9.4f}".format("head fp32", t(lambda: F.linear(h, lg_head_w))))
    w16 = lg_head_w.to(torch.bfloat16)
    h16 = h.to(torch.bfloat16)
    print("{:<40} {:>9.4f}".format("head bf16", t(lambda: F.linear(h16, w16))))

ids = torch.randint(0, 4096, (1, 1))
st = model.init_cache(1)
with torch.no_grad():
    print("{:<40} {:>9.4f}".format("fwd no_grad", t(lambda: model(ids, st, 7), 500)))
with torch.inference_mode():
    print("{:<40} {:>9.4f}".format("fwd inference_mode",
                                   t(lambda: model(ids, st, 7), 500)))
