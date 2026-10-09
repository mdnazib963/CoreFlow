import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eval import load_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="run/p0_best.pt")
    ap.add_argument("--val", default="data/val.bin")
    ap.add_argument("--windows", type=int, default=80)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    model, cfg, st = load_ckpt(a.ckpt, torch.device("cpu"))
    model.eval()

    raw = open(a.val, "rb").read()
    toks = torch.frombuffer(memoryview(raw), dtype=torch.uint16).to(torch.int64)

    tot = 0.0
    cnt = 0
    wn = 0
    with torch.no_grad():
        for w in range(a.windows):
            base = w * a.seq
            if base + a.seq + 1 > len(toks):
                break
            window = toks[base: base + a.seq]
            x = window[:-1].unsqueeze(0)
            y = window[1:].unsqueeze(0)
            logits = model(x)
            nll = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            tot += float(nll)
            cnt += y.numel()
            wn += 1

    out = {
        "windows": wn,
        "seq": a.seq,
        "ntok": cnt,
        "nll": tot / cnt,
        "ppl": float(torch.tensor(tot / cnt).exp()),
        "step": st.get("step"),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
