import json
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cam.m0_data import build_corpus
from cam.read import format_block, retrieve
from eval import load_ckpt
from bpe import Tokenizer
from model import warm_mlp

torch.set_num_threads(1)
torch.manual_seed(0)

model, cfg, _ = load_ckpt("run/best.pt", torch.device("cpu"))
model.eval()
warm_mlp(model)
tok = Tokenizer.load("data/tokenizer.json")
STOP = {2, 13, 0, 1}


def gen(prompt, max_new=24):
    ids = tok.encode(prompt)
    st = model.init_cache(1)
    x = torch.tensor([ids], dtype=torch.long)
    out = []
    with torch.inference_mode():
        lg = model(x, st, 0)
        nxt = int(lg[0, -1].argmax())
        p = len(ids)
        while len(out) < max_new:
            if nxt in STOP:
                break
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
            lg = model(x[:, -1:], st, p)
            p += 1
            nxt = int(lg[0, -1].argmax())
    return tok.decode(out), ids, out


store, facts, questions, meta = build_corpus(seed=42)
q2 = [q for q in questions if q["type"] == "2hop"][:3]
for q in q2:
    block = format_block(retrieve(store, q["q"], k=8), limit=8)
    prompt = block + "\nQ: " + q["q"] + "\nA: "
    pred, ids, out = gen(prompt)
    print("=" * 70)
    print("PROMPT:")
    print(prompt)
    print("gold   :", q["answer"])
    print("pred   :", pred)
    print("gold ids :", tok.encode(q["answer"]))
    print("pred ids :", out)
    print("prompt len tokens:", len(ids))

print("=" * 70)
print("contrast: s3.ood (training-format, TEST_NS)")
n = ok = 0
for line in open("data/s3.ood.jsonl", encoding="utf-8"):
    if n >= 20:
        break
    d = json.loads(line)
    txt = d["text"]
    pre, ans = txt.rsplit("\nA: ", 1)
    pred, _, _ = gen(pre + "\nA: ")
    ok += pred.strip() == ans.strip()
    n += 1
    if n <= 3:
        print("-" * 50)
        print(pre)
        print("gold:", ans, "| pred:", pred)
print("s3.ood 20-sample exact: {}/{}".format(ok, n))
