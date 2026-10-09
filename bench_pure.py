import json
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
from bpe import Tokenizer

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.manual_seed(0)


def pure_decode(model, tok, prompt, seconds=60.0, threads=1):
    torch.set_num_threads(threads)
    ids = tok.encode(prompt)
    st = model.init_cache(1)
    x = torch.tensor([ids], dtype=torch.long)
    lat = []
    with torch.inference_mode():
        lg = model(x, st, 0)
        p = len(ids)
        t0 = time.perf_counter()
        nxt = int(lg[0, -1].argmax())
        while time.perf_counter() - t0 < seconds:
            if p >= model.ctx - 8:
                tp = time.perf_counter()
                keep = x[:, -(model.ctx - 256):]
                st = model.init_cache(1)
                lg = model(keep, st, 0)
                x = keep
                p = x.size(1)
                nxt = int(lg[0, -1].argmax())
                t0 += time.perf_counter() - tp
                continue
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
            ta = time.perf_counter()
            lg = model(x[:, -1:], st, p)
            p += 1
            nxt = int(lg[0, -1].argmax())
            lat.append(time.perf_counter() - ta)
    el = time.perf_counter() - t0
    lat.sort()
    n = len(lat)
    return {
        "tok_per_s": round(n / max(el, 1e-9), 1),
        "tokens": n,
        "sec": round(el, 1),
        "mean_ms": round(sum(lat) / n * 1000, 2) if n else 0,
        "p50_ms": round(lat[n // 2] * 1000, 2) if n else 0,
        "p99_ms": round(lat[int(0.99 * (n - 1))] * 1000, 2) if n else 0,
        "prefill_ms": None,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="run/best.pt")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--threads", default="1")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--mode", default="")
    p.add_argument("--out", default="run/bench_pure.json")
    a = p.parse_args()

    tok = Tokenizer.load("data/tokenizer.json")
    model, cfg, _ = load_ckpt(a.ckpt, torch.device("cpu"))
    model.eval()
    if a.compile:
        from model import warm_mlp
        warm_mlp(model)
        model = torch.compile(model, dynamic=False,
                              mode=a.mode or None)
    tasks = [json.loads(l) for l in open("data/tasks_val.jsonl", encoding="utf-8")]
    prompt = render(tasks[0]["chain"], tasks[0]["q"], 2)
    nprompt = len(tok.encode(prompt))

    out = {"prompt_tokens": nprompt, "ctx": cfg["ctx"], "compile": a.compile,
           "mode": a.mode, "results": []}
    print("prompt {} tokens".format(nprompt), flush=True)
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)

    if a.compile:
        tw = time.time()
        with torch.inference_mode():
            for _ in range(3):
                model(x, model.init_cache(1), 0)
        pure_decode(model, tok, prompt, 15.0, int(a.threads.split(",")[0]))
        print("compile warmup {:.0f}s".format(time.time() - tw), flush=True)

    pf = {}
    for t in [int(v) for v in a.threads.split(",")]:
        torch.set_num_threads(t)
        st = model.init_cache(1)
        with torch.inference_mode():
            for _ in range(3):
                model(x, st, 0)
            t0 = time.perf_counter()
            for _ in range(10):
                model(x, model.init_cache(1), 0)
            pf[t] = round((time.perf_counter() - t0) / 10 * 1000, 2)
        r = pure_decode(model, tok, prompt, a.seconds, t)
        r["threads"] = t
        r["prefill_ms"] = pf[t]
        r["prefill_tok_s"] = round(nprompt / (pf[t] / 1000), 1)
        print("{:>3} thr | decode {:>6} tok/s | mean {:>6} ms | p50 {:>6} ms | "
              "p99 {:>6} ms | prefill {:>6} ms ({:>6} tok/s)".format(
                  t, r["tok_per_s"], r["mean_ms"], r["p50_ms"], r["p99_ms"],
                  pf[t], r["prefill_tok_s"]), flush=True)
        out["results"].append(r)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
