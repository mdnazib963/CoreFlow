import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import CoreFlowLM, count_params


def log(step, **kw):
    print(json.dumps({"step": step, **kw}), flush=True)


def load(p):
    return np.fromfile(p, dtype=np.uint16)


def windows(data, seq):
    n = (len(data) - seq - 1) // seq
    return np.arange(n, dtype=np.int64) * seq


def batch_at(data, starts, seq):
    s = starts[:, None] + np.arange(seq + 1, dtype=np.int64)
    buf = data[s]
    x = torch.from_numpy(buf[:, :-1].astype(np.int64))
    y = torch.from_numpy(buf[:, 1:].astype(np.int64))
    return x, y


def val_ppl(model, data, seq, batch, device, limit=20, want_draft=False):
    model.eval()
    st = windows(data, seq)
    tot, cnt, acc, ntok = 0.0, 0, 0, 0
    with torch.no_grad():
        for i in range(0, min(len(st), limit * batch), batch):
            x, y = batch_at(data, st[i:i + batch], seq)
            x = x.to(device)
            y = y.to(device)
            out = model(x, draft=True) if want_draft else model(x)
            if want_draft:
                lg, dl = out
                arg = lg.argmax(-1)
                acc += int((dl.argmax(-1) == arg).sum())
                ntok += arg.numel()
            else:
                lg = out
            nll = F.cross_entropy(lg.view(-1, lg.size(-1)), y.view(-1), reduction="sum")
            tot += float(nll)
            cnt += y.numel()
    model.train()
    ppl = math.exp(tot / max(1, cnt))
    if want_draft:
        return ppl, acc / max(1, ntok)
    return ppl


def lr_at(step, total, base, warm, min_ratio):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return min_ratio * base + (1 - min_ratio) * base * 0.5 * (1 + math.cos(math.pi * p))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train.bin")
    p.add_argument("--val", default="data/val.bin")
    p.add_argument("--out", default="run")
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--dff", type=int, default=512)
    p.add_argument("--nconv", type=int, default=8)
    p.add_argument("--natt", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--kv", type=int, default=2)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--softcap", type=float, default=30.0)
    p.add_argument("--stage-dropout", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--zloss", type=float, default=1e-4)
    p.add_argument("--warm", type=int, default=200)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dhead", type=int, default=0)
    p.add_argument("--dhead-depth", type=int, default=3)
    p.add_argument("--dhead-ff", type=int, default=64)
    p.add_argument("--dhead-lambda", type=float, default=0.3)
    p.add_argument("--dhead-rot", type=int, default=1)
    p.add_argument("--qat", type=int, default=0)
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, "latest.pt")
    data = load(a.train)
    val = load(a.val)
    st = windows(data, a.seq)
    steps_per_epoch = max(1, len(st) // a.batch)
    total = steps_per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)

    if a.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(a.device)
    print("device: {}".format(device), flush=True)

    model = CoreFlowLM(a.vocab, a.d, a.dff, a.nconv, a.natt, a.heads, a.kv,
                       a.ctx, a.softcap, a.stage_dropout,
                       a.dhead, a.dhead_depth, a.dhead_ff,
                       a.dhead_rot, a.qat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=a.wd)
    step, tokens, best = 0, 0, float("inf")
    if a.resume and os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step, tokens, best = s["step"], s["tokens"], s["best"]
        print("resumed step={} tokens={}".format(step, tokens), flush=True)

    cfg = {k: getattr(a, k) for k in vars(a)}
    cfg["params"] = count_params(model)
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print("params {:.3f}M  windows {}  steps/epoch {}  total {}".format(
        count_params(model) / 1e6, len(st), steps_per_epoch, total), flush=True)

    model.train()
    rng = np.random.default_rng(a.seed + 1)
    order = rng.permutation(st)
    pos = 0
    t0 = time.perf_counter()
    tok_acc, t_mark = 0, t0
    history = os.path.join(a.out, "log.jsonl")
    last_val = float("inf")

    while step < total:
        if pos + a.batch > len(order):
            order = rng.permutation(st)
            pos = 0
        lr = lr_at(step, total, a.lr, a.warm, a.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch_at(data, order[pos:pos + a.batch], a.seq)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pos += a.batch

        out = model(x)
        if a.dhead:
            lg, dl = out
        else:
            lg = out
        ce = F.cross_entropy(lg.view(-1, lg.size(-1)), y.view(-1))
        z = torch.logsumexp(lg, dim=-1)
        loss = ce + a.zloss * (z * z).mean()
        ce_d = None
        if a.dhead:
            ce_d = F.cross_entropy(dl.view(-1, dl.size(-1)), y.view(-1))
            loss = loss + a.dhead_lambda * ce_d
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()

        step += 1
        tokens += y.numel()
        tok_acc += y.numel()

        if step % 50 == 0 or step == 1:
            el = time.perf_counter() - t_mark
            tps = tok_acc / max(el, 1e-9)
            tok_acc, t_mark = 0, time.perf_counter()
            done = time.perf_counter() - t0
            eta = done / step * (total - step)
            kw = dict(loss=loss.item(), ce=ce.item(),
                      lr="{:.2e}".format(lr), gnorm=float(gnorm),
                      tok_s=round(tps), elapsed=round(done), eta=round(eta))
            if ce_d is not None:
                kw["ce_d"] = round(ce_d.item(), 4)
            log(step, **kw)

        do_eval = (step % a.eval_every == 0) or (step == total)
        if do_eval:
            r = val_ppl(model, val, a.seq, a.batch, device,
                        want_draft=bool(a.dhead))
            if a.dhead:
                vp, acc = r
            else:
                vp, acc = r, None
            last_val = vp
            rec = {"step": step, "tokens": tokens, "val_ppl": round(vp, 4),
                   "lr": lr, "elapsed": round(time.perf_counter() - t0)}
            if acc is not None:
                rec["accept"] = round(acc, 4)
            with open(history, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            kw = {"val_ppl": round(vp, 4)}
            if acc is not None:
                kw["accept"] = round(acc, 4)
            log(step, **kw)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "tokens": tokens, "best": min(best, vp),
                        "config": cfg}, ckpt)
            if vp < best:
                best = vp
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "tokens": tokens, "best": best,
                            "config": cfg}, os.path.join(a.out, "best.pt"))

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "tokens": tokens, "best": min(best, last_val),
                "config": cfg}, ckpt)
    log(step, done=True, val_ppl=round(last_val, 4), best=round(best, 4),
        elapsed=round(time.perf_counter() - t0))


if __name__ == "__main__":
    main()
