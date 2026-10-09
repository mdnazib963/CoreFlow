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
N = 1000


def t(fn, n=N):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


model, cfg, _ = load_ckpt("run/best.pt", torch.device("cpu"))
model.eval()
x = torch.randn(1, 1, 384)
w = torch.randn(384)
n1 = model.blocks[0].n1

print("{:<38} {:>9}".format("op", "ms"))
print("{:<38} {:>9.4f}".format("manual rms", t(lambda: n1(x))))
print("{:<38} {:>9.4f}".format("torch.rms_norm", t(lambda: torch.rms_norm(x, (384,), w, 1e-6))))

mlp = model.blocks[0].mlp
w12 = torch.cat([mlp.f1.weight, mlp.f2.weight], 0)
print("{:<38} {:>9.4f}".format("mlp f1+f2 (2 matmul)", t(lambda: (mlp.f1(x), mlp.f2(x)))))
print("{:<38} {:>9.4f}".format("mlp fused w12 (1 matmul)", t(lambda: F.linear(x, w12))))
print("{:<38} {:>9.4f}".format("mlp full", t(lambda: mlp(x))))

at = model.blocks[11]
print("{:<38} {:>9.4f}".format("qkv linear", t(lambda: at.qkv(x))))
print("{:<38} {:>9.4f}".format("out linear", t(lambda: at.out(x))))

h = torch.randn(1, 384, 80)
buf = torch.zeros(1, 384, 3)
print("{:<38} {:>9.4f}".format("conv cat+dw+transpose", t(lambda: model.blocks[0].dw(
    torch.cat([buf, h], dim=2)).transpose(1, 2))))

kv = torch.randn(1, 2, 64, 64)
print("{:<38} {:>9.4f}".format("repeat_interleave", t(lambda: kv.repeat_interleave(3, 1))))

ids = torch.randint(0, 4096, (1, 1))
print("{:<38} {:>9.4f}".format("embedding lookup", t(lambda: model.wte(ids))))

lg = torch.randn(1, 4096)
print("{:<38} {:>9.4f}".format("argmax+int", t(lambda: int(lg[0].argmax()))))
print("{:<38} {:>9.4f}".format("head linear", t(lambda: F.linear(n1(x), model.wte.weight))))
print("{:<38} {:>9.4f}".format("tanh softcap", t(lambda: 30 * torch.tanh(lg / 30))))

for name, fn in (("no_grad", torch.no_grad), ("inference_mode", torch.inference_mode)):
    with fn():
        print("{:<38} {:>9.4f}".format("full forward " + name,
                                       t(lambda: model(ids, model.init_cache(1), 7), 300)))
with torch.no_grad():
    st = model.init_cache(1)
    print("{:<38} {:>9.4f}".format("full forward (reused cache)",
                                   t(lambda: model(ids, st, 7), 300)))
