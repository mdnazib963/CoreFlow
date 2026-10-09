import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import CoreFlowLM, count_params

L3 = 4096 * 1024
BUDGET = L3 // 2
CTX = 1024
VOCAB = 4096
K = 4


def cost(d, dff, nconv, natt, heads, kv):
    m = CoreFlowLM(VOCAB, d, dff, nconv, natt, heads, kv, CTX)
    p = count_params(m)
    hd = d // heads
    scales = (VOCAB + nconv * (5 * d + dff) + natt * 5 * d) * 4
    header = 66 + 32 * (1 + nconv * 8 + natt * 7 + 1)
    kvb = natt * 2 * CTX * kv * (d // heads)
    kvs = natt * 2 * CTX * kv * 4
    cbuf = nconv * d * (K - 1) * 4
    rope = CTX * (hd // 2) * 8
    scratch = (8 * d + d + 2 * kv * (d // heads) + 2 * dff + CTX + VOCAB + K) * 4
    fileb = p + scales + header
    ws = fileb + kvb + kvs + cbuf + rope + scratch
    return m, p, fileb, ws


rows = []
grid = itertools.product(
    range(112, 257, 16), (160, 192, 224, 256, 320, 384),
    (3, 4, 5, 6), (1, 2, 3), (4, 6, 8), (1, 2))
for d, dff, nconv, natt, heads, kv in grid:
    if d % heads or heads % kv:
        continue
    n = nconv + natt
    if natt and (n // natt) < 1:
        continue
    m, p, fileb, ws = cost(d, dff, nconv, natt, heads, kv)
    rows.append((p, ws, d, dff, nconv, natt, heads, kv, fileb))

ok = [r for r in rows if r[1] <= BUDGET]
ok.sort(key=lambda r: -r[0])
print("budget {} B  |  {} configs fit".format(BUDGET, len(ok)))
print("{:>9} {:>9} {:>6} | {:>4} {:>4} {:>4} {:>3} {:>4} {:>3}".format(
    "params", "workset", "%L3", "d", "dff", "ncv", "nat", "hd", "kv"))
for r in ok[:12]:
    p, ws, d, dff, nconv, natt, heads, kv, fileb = r
    print("{:>9} {:>9} {:>6.1f} | {:>4} {:>4} {:>4} {:>3} {:>4} {:>3}".format(
        p, ws, 100 * ws / L3, d, dff, nconv, natt, heads, kv))

best = ok[0] if ok else None
if best:
    p, ws, d, dff, nconv, natt, heads, kv, fileb = best
    m = CoreFlowLM(VOCAB, d, dff, nconv, natt, heads, kv, CTX)
    print("\nbest fit: d={} dff={} nconv={} natt={} heads={} kv={}".format(
        d, dff, nconv, natt, heads, kv))
    print("  params {:.4f}M  file {:.1f} KB  working set {:.1f} KB = {:.1f}% of L3"
          .format(p / 1e6, fileb / 1024, ws / 1024, 100 * ws / L3))
    print("  slack to budget {:.1f} KB".format((BUDGET - ws) / 1024))
    print("  blocks:", [type(b).__name__ for b in m.blocks])
