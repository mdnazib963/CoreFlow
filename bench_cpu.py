import argparse
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

from eval import load_ckpt, render, pair_ids, greedy
from bpe import Tokenizer


def prefill_bench(model, ids_list, threads):
    torch.set_num_threads(threads)
    with torch.no_grad():
        n = 0
        t0 = time.perf_counter()
        for ids in ids_list:
            x = torch.tensor([ids], dtype=torch.long)
            model(x)
            n += len(ids)
    el = time.perf_counter() - t0
    return round(n / max(el, 1e-9), 1)


def gen_cached(model, tok, ids, max_new):
    st = model.init_cache(1)
    x = torch.tensor([ids], dtype=torch.long)
    out = []
    with torch.no_grad():
        lg = model(x, st, 0)
        nxt = int(lg[0, -1].argmax())
        p = len(ids)
        while True:
            if nxt in (2, 13, 0, 1):
                break
            out.append(nxt)
            if len(out) >= max_new:
                break
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
            lg = model(x[:, -1:], st, p)
            p += 1
            nxt = int(lg[0, -1].argmax())
    return out


def decode_bench(model, tok, prompts, threads, max_new):
    torch.set_num_threads(threads)
    n = 0
    t0 = time.perf_counter()
    for pr in prompts:
        n += len(gen_cached(model, tok, tok.encode(pr), max_new))
    el = time.perf_counter() - t0
    return round(n / max(el, 1e-9), 1), round(el, 2)


def sustained(model, tok, prompts, threads, seconds):
    torch.set_num_threads(threads)
    lat = []
    toks = 0
    i = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ids = tok.encode(prompts[i % len(prompts)])
        i += 1
        st = model.init_cache(1)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            lg = model(x, st, 0)
            nxt = int(lg[0, -1].argmax())
            p = len(ids)
            while True:
                if nxt in (2, 13, 0, 1):
                    break
                ta = time.perf_counter()
                x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
                lg = model(x[:, -1:], st, p)
                p += 1
                nxt = int(lg[0, -1].argmax())
                lat.append(time.perf_counter() - ta)
                toks += 1
                if time.perf_counter() - t0 >= seconds:
                    break
    el = time.perf_counter() - t0
    lat.sort()
    p99 = lat[int(0.99 * (len(lat) - 1))] if lat else 0.0
    return (round(toks / max(el, 1e-9), 1), round(el, 1), toks,
            round(p99 * 1000, 2))


def stream_bench(model, ids, threads, ctx=1024, windows=120):
    torch.set_num_threads(threads)
    with torch.no_grad():
        n = 0
        t0 = time.perf_counter()
        for i in range(0, min(len(ids) - 1, windows * ctx), ctx):
            w = ids[i:i + ctx + 1]
            if len(w) < 2:
                break
            x = torch.tensor([w[:-1]], dtype=torch.long)
            model(x)
            n += len(w) - 1
    el = time.perf_counter() - t0
    return round(n / max(el, 1e-9), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="run/best.pt")
    p.add_argument("--tasks", default="data/tasks_val.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--threads", default="1,2,4,8")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--prompts", type=int, default=60)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--out", default="run/bench.json")
    a = p.parse_args()

    threads = [int(t) for t in a.threads.split(",")]
    tok = Tokenizer.load(a.tokenizer)
    model, cfg, _ = load_ckpt(a.ckpt, torch.device("cpu"))
    model.eval()
    torch.manual_seed(0)
    if a.compile:
        model = torch.compile(model, dynamic=False)

    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")][:a.prompts]
    prs = [render(t["chain"], t["q"], 2) for t in tasks]
    pids = [tok.encode(s) for s in prs]

    if a.compile:
        tw = time.time()
        gen_cached(model, tok, tok.encode(prs[0]), 8)
        print("compile warmup {:.1f}s".format(time.time() - tw))

    import numpy as np
    stream_ids = np.fromfile("data/val.bin", dtype=np.uint16).tolist()

    print("params {:.3f}M  ctx {}  vocab {}".format(
        sum(x.numel() for x in model.parameters()) / 1e6, cfg["ctx"], cfg["vocab"]))
    print("{:>8} {:>14} {:>14} {:>14} {:>18} {:>12}".format(
        "threads", "prefill_tok/s", "decode_tok/s", "stream_tok/s",
        "sustained_tok/s", "p99_ms"))
    out = {"params_m": round(sum(x.numel() for x in model.parameters()) / 1e6, 3),
           "ctx": cfg["ctx"], "max_new": a.max_new, "seconds": a.seconds,
           "compile": a.compile, "results": []}
    for t in threads:
        pf = prefill_bench(model, pids, t)
        dc, sec = decode_bench(model, tok, prs, t, a.max_new)
        st = stream_bench(model, stream_ids, t)
        sus, sus_sec, sus_n, p99 = sustained(model, tok, prs, t, a.seconds)
        print("{:>8} {:>14} {:>14} {:>14} {:>18} {:>12}".format(
            t, pf, dc, st, sus, p99))
        out["results"].append({"threads": t, "prefill": pf, "decode": dc,
                               "stream": st, "decode_sec": sec,
                               "sustained": sus, "sustained_sec": sus_sec,
                               "sustained_tokens": sus_n, "p99_ms": p99})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
