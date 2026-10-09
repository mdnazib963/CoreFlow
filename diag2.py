import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/content")

from eval import load_ckpt
from bpe import Tokenizer

tok = Tokenizer.load("/content/tokenizer.json")
model, cfg, st = load_ckpt("/content/run/best.pt", torch.device("cuda"))
model.eval()
ctx = cfg["ctx"]

ids = np.fromfile("/content/val.bin", dtype=np.uint16).tolist()
print("val tokens:", len(ids))

probe = tok.encode("A: ")
print("probe ids:", probe, [tok.decode([i]) for i in probe])

hits = []
for i in range(len(ids) - 20):
    if ids[i:i + len(probe)] == probe:
        hits.append(i + len(probe) - 1)
print("A: hits:", len(hits))

acc = nll = n = 0
samples = []
for j in hits:
    end = j + 1
    start = max(0, end - ctx)
    x = torch.tensor([ids[start:end]], dtype=torch.long, device="cuda")
    with torch.no_grad():
        lg = model(x)
    pred = int(lg[0, -1].argmax())
    gold = ids[end]
    acc += int(pred == gold)
    nll += float(F.cross_entropy(lg[0, -1:],
                                 torch.tensor([gold], device="cuda")))
    n += 1
    if len(samples) < 8:
        samples.append((tok.decode(ids[start:end])[-40:],
                        tok.decode([gold]), tok.decode([pred])))

print("in-dist A: first-token acc={:.4f}  nll={:.3f}  ppl={:.2f}".format(
    acc / max(1, n), nll / max(1, n), math.exp(nll / max(1, n))))
for s, g, p in samples:
    print("  ctx=...%r gold=%r pred=%r" % (s, g, p))

probe2 = tok.encode("\nQ: ")
hits2 = []
for i in range(len(ids) - 20):
    if ids[i:i + len(probe2)] == probe2:
        hits2.append(i + len(probe2) - 1)
print("Q: hits:", len(hits2))
acc2 = n2 = 0
for j in hits2:
    end = j + 1
    start = max(0, end - ctx)
    x = torch.tensor([ids[start:end]], dtype=torch.long, device="cuda")
    with torch.no_grad():
        lg = model(x)
    acc2 += int(int(lg[0, -1].argmax()) == ids[end])
    n2 += 1
print("in-dist after-Q next-token acc={:.4f} n={}".format(
    acc2 / max(1, n2), n2))
