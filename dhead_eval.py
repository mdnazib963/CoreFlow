import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from eval import load_ckpt
from train import windows, batch_at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="run/best.pt")
    p.add_argument("--val", default="data/val.bin")
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--depth", type=int, default=0)
    a = p.parse_args()
    if a.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(a.device)
    model, cfg, _ = load_ckpt(a.ckpt, device)
    if not getattr(model, "dhead", 0):
        raise SystemExit("checkpoint has no dhead")
    if a.depth:
        model.dhead_depth = a.depth
    data = np.fromfile(a.val, dtype=np.uint16)
    st = windows(data, a.seq)
    model.eval()
    acc, ntok, ftot, dtot, cnt = 0, 0, 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, min(len(st), a.limit * a.batch), a.batch):
            x, y = batch_at(data, st[i:i + a.batch], a.seq)
            x = x.to(device)
            y = y.to(device)
            lg, dl = model(x, draft=True)
            arg = lg.argmax(-1)
            acc += int((dl.argmax(-1) == arg).sum())
            ntok += arg.numel()
            ftot += float(F.cross_entropy(lg.view(-1, lg.size(-1)),
                                          y.view(-1), reduction="sum"))
            dtot += float(F.cross_entropy(dl.view(-1, dl.size(-1)),
                                          y.view(-1), reduction="sum"))
            cnt += y.numel()
    out = dict(ckpt=a.ckpt, acceptance=round(acc / max(1, ntok), 4),
               final_ppl=round(math.exp(ftot / max(1, cnt)), 4),
               draft_ppl=round(math.exp(dtot / max(1, cnt)), 4),
               depth=int(model.dhead_depth), rot=cfg.get("dhead_rot"),
               tokens=cnt)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
