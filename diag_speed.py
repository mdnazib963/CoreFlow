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

model, cfg, _ = load_ckpt("run/best.pt", torch.device("cpu"))
model.eval()
B, T = 1, 1
N = 300


def t(fn, n=N):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


ids = torch.randint(0, cfg["vocab"], (B, T))
st = model.init_cache(B)

with torch.no_grad():
    parts = []
    parts.append(("full forward (cache, t=1)", t(lambda: model(ids, st, 7))))

    x = model.wte(ids)
    parts.append(("embedding", t(lambda: model.wte(ids))))

    blk = model.blocks[0]
    y = x
    parts.append(("convblock[0]", t(lambda: blk(y, st[0], 7))))
    parts.append(("  n1", t(lambda: blk.n1(y))))
    h = blk.n1(y)
    parts.append(("  cat", t(lambda: torch.cat([st[0]["buf"], h.transpose(1, 2)], dim=2))))
    hv = torch.cat([st[0]["buf"], h.transpose(1, 2)], dim=2)
    parts.append(("  dw conv", t(lambda: blk.dw(hv))))
    c = blk.dw(hv).transpose(1, 2)
    parts.append(("  gate*mul*mix", t(lambda: blk.mix(F.silu(blk.gate(h)) * c))))
    parts.append(("  n2", t(lambda: blk.n2(y))))
    n2 = blk.n2(y)
    parts.append(("  mlp", t(lambda: blk.mlp(n2))))
    mlp = blk.mlp(n2)
    parts.append(("    f1", t(lambda: blk.mlp.f1(n2))))
    f1 = blk.mlp.f1(n2)
    parts.append(("    f2", t(lambda: blk.mlp.f2(n2))))
    parts.append(("    silu*mul", t(lambda: F.silu(f1) * blk.mlp.f2(n2))))
    s = F.silu(f1) * blk.mlp.f2(n2)
    parts.append(("    f3", t(lambda: blk.mlp.f3(s))))

    at = model.blocks[11]
    parts.append(("attnblock[11]", t(lambda: at(y, st[11], 7))))
    parts.append(("  n1", t(lambda: at.n1(y))))
    hh = at.n1(y)
    parts.append(("  qkv", t(lambda: at.qkv(hh))))
    q, k, v = at.qkv(hh).split([384, 128, 128], -1)
    q = q.view(1, 1, 6, 64).transpose(1, 2)
    k = k.view(1, 1, 2, 64).transpose(1, 2)
    v = v.view(1, 1, 2, 64).transpose(1, 2)
    cos = at.cos[7:8]
    sin = at.sin[7:8]
    from model import apply_rope
    parts.append(("  rope x2", t(lambda: (
        apply_rope(q, cos.to(q.dtype), sin.to(q.dtype)),
        apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))))))
    qr = apply_rope(q, cos.to(q.dtype), sin.to(q.dtype))
    kr = apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))
    ke = kr.repeat_interleave(3, 1)
    ve = v.repeat_interleave(3, 1)
    parts.append(("  repeat_interleave x2", t(lambda: (
        kr.repeat_interleave(3, 1), v.repeat_interleave(3, 1)))))
    parts.append(("  sdpa t=1", t(lambda: F.scaled_dot_product_attention(qr, ke, ve))))
    o = F.scaled_dot_product_attention(qr, ke, ve)
    parts.append(("  out proj", t(lambda: at.out(o.transpose(1, 2).reshape(1, 1, 384)))))
    parts.append(("  final linear+softcap", t(lambda: (
        model.softcap * torch.tanh(F.linear(model.nf(x), model.wte.weight) / model.softcap)))))

print("PROFILING (per call, ms, 1 thread, t=1 decode step)")
print("{:<32} {:>10}".format("component", "ms"))
for name, ms in parts:
    print("{:<32} {:>10.3f}".format(name, ms))
