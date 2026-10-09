import argparse
import json
import os
import subprocess

import numpy as np
import torch

from eval import load_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="run/p3/run/best.pt")
    p.add_argument("--i8", default="phase0/model_p3.i8")
    p.add_argument("--stages", default="phase0/stages_p3.json")
    p.add_argument("--bin", default="data/val.bin")
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--match", type=float, default=0.98)
    a = p.parse_args()

    exe = os.path.join("phase0", "stage_rt.exe")
    out = subprocess.run(
        [exe, a.i8, a.stages, "draftprobe", str(a.N), a.bin, str(a.steps)],
        capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise SystemExit("draftprobe failed: " + out.stderr[-500:])
    line = out.stdout.strip().splitlines()[-1]
    c = json.loads(line)
    pairs = c["pairs"]

    data = np.fromfile(a.bin, dtype=np.uint16)
    model, cfg, _ = load_ckpt(a.ckpt, torch.device("cpu"))
    model.eval()
    model.dhead_depth = a.depth
    steps = min(a.steps, len(data) - 2)
    x = torch.from_numpy(data[:steps + 1].astype(np.int64))
    with torch.no_grad():
        lg, dl = model(x.unsqueeze(0), draft=True)
    t_arg = dl[0].argmax(-1).numpy()
    t_conf = torch.softmax(dl[0], -1).max(-1).values.numpy()

    n = ok = 0
    conf_diff = 0.0
    tgt_ok = 0
    conf_match, conf_mism = [], []
    for pos, target, cd, cconf in pairs:
        if pos >= steps:
            continue
        n += 1
        hit = int(cd) == int(t_arg[pos])
        if hit:
            ok += 1
            conf_match.append(float(t_conf[pos]))
        else:
            conf_mism.append(float(t_conf[pos]))
        if int(cd) == int(target):
            tgt_ok += 1
        conf_diff += abs(float(cconf) - float(t_conf[pos]))
    res = dict(
        steps=n, depth=a.depth,
        token_match=round(ok / max(1, n), 5),
        torch_draft_ppl_proxy=round(tgt_ok / max(1, n), 5),
        conf_mean_absdiff=round(conf_diff / max(1, n), 5),
        torch_conf_mean=round(float(t_conf[:n].mean()), 5),
        mismatches=len(conf_mism),
        mism_conf_mean=round(float(np.mean(conf_mism)) if conf_mism else 0.0, 5),
        match_conf_mean=round(float(np.mean(conf_match)) if conf_match else 0.0, 5),
        threshold=a.match, pass_=bool(ok / max(1, n) >= a.match),
    )
    res["pass"] = res.pop("pass_")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
