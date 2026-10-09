import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
        else "/content")
for _p in (HERE, os.path.join(HERE, "data")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from model import CoreFlowLM, count_params
from bpe import EOS, Tokenizer

NEWLINE = 3 + 10


def load_ckpt(path, device):
    s = torch.load(path, map_location=device, weights_only=False)
    c = s["config"]
    m = CoreFlowLM(c["vocab"], c["d"], c["dff"], c["nconv"], c["natt"],
                   c["heads"], c["kv"], c["ctx"], c["softcap"],
                   c.get("stage_dropout", 0.0),
                   c.get("dhead", 0), c.get("dhead_depth", 3),
                   c.get("dhead_ff", 64), c.get("dhead_rot", 0),
                   c.get("qat", 0))
    m.load_state_dict(s["model"])
    m.to(device).eval()
    return m, c, s


def pair_ids(tok, prompt, answer):
    full = tok.encode(prompt + answer)
    pre = tok.encode(prompt)
    if len(pre) and full[:len(pre)] == pre:
        return full[:len(pre)], full[len(pre):]
    if len(pre) > 1 and full[:len(pre) - 1] == pre[:-1]:
        return full[:len(pre) - 1], full[len(pre) - 1:]
    if not len(pre):
        return [], full
    return pre, full[len(pre):]


def render(mem_lines, q, variant):
    block = "[MEM]\n" + "\n".join("- " + l + " (src: user)" for l in mem_lines) + "\n[/MEM]"
    if variant == 0:
        return block + "\nQuestion: " + q + "\nAnswer: "
    if variant == 1:
        return "Read the memory block, then answer the question.\n" + block + "\nQ: " + q + "\nA: "
    return block + "\nQ: " + q + "\nA: "


def greedy(model, tok, ids, device, max_new=24):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for _ in range(max_new):
            nxt = int(model(x)[0, -1].argmax())
            if nxt in (EOS, NEWLINE, 0, 1):
                break
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return tok.decode(out)


def score(model, p_ids, a_ids, device):
    full = p_ids + a_ids
    if len(full) < 3 or not len(a_ids):
        return None
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        lg = model(x)
        nll = F.cross_entropy(lg[0], y[0], reduction="none")
    start = len(p_ids) - 1
    seg = nll[start:]
    if seg.numel() == 0:
        return None
    first = int(lg[0, start].argmax()) == int(y[0, start])
    return {"nll": float(seg.sum()), "ntok": int(seg.numel()), "first": first}


def lm_ppl(model, tok, texts, ctx, device, limit_windows=400):
    ids = []
    for t in texts:
        ids.extend(tok.encode(t))
        ids.append(EOS)
    tot, cnt, n = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(ids) - 1, ctx):
            if n >= limit_windows:
                break
            w = ids[i:i + ctx + 1]
            if len(w) < 2:
                break
            x = torch.tensor([w[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([w[1:]], dtype=torch.long, device=device)
            lg = model(x)
            tot += float(F.cross_entropy(lg[0], y[0], reduction="sum"))
            cnt += y.numel()
            n += 1
    return math.exp(tot / max(1, cnt)), n


def bench_decode(model, tok, prompts, device, threads):
    torch.set_num_threads(threads)
    gen, raw = 0, 0
    t0 = time.perf_counter()
    for pr in prompts:
        s = greedy(model, tok, tok.encode(pr), device)
        gen += max(1, len(s))
        raw += len(pr)
    el = time.perf_counter() - t0
    return {"prompts": len(prompts), "sec": round(el, 2),
            "tok_s": round(gen / max(el, 1e-9), 1),
            "prompt_toks_s": round(raw / max(el, 1e-9), 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="run/best.pt")
    p.add_argument("--tasks", default="data/tasks_val.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--val", default="data/val.bin")
    p.add_argument("--ood", default="data/s2.ood.jsonl")
    p.add_argument("--ood2", default="data/s3.ood.jsonl")
    p.add_argument("--id", default="data/s2.id.jsonl")
    p.add_argument("--id2", default="data/s3.id.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new", type=int, default=24)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no-greedy", action="store_true")
    p.add_argument("--bench", type=int, default=0)
    a = p.parse_args()

    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                          else (a.device if a.device != "auto" else "cpu"))
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    tok = Tokenizer.load(a.tokenizer)
    model, cfg, st = load_ckpt(a.ckpt, device)
    print("ckpt={} step={} device={} params={:.3f}M".format(
        a.ckpt, st.get("step"), device, count_params(model) / 1e6), flush=True)

    tasks = []
    with open(a.tasks, encoding="utf-8") as fh:
        for line in fh:
            tasks.append(json.loads(line))
    if a.limit:
        tasks = tasks[:a.limit]

    nll, ntok, first, exact, total = 0.0, 0, 0, 0, 0
    per_variant = {v: [0, 0] for v in range(a.variants)}
    t0 = time.perf_counter()
    for i, t in enumerate(tasks):
        for v in range(a.variants):
            pr = render(t["chain"], t["q"], v)
            p_ids, a_ids = pair_ids(tok, pr, t["answer"])
            sc = score(model, p_ids, a_ids, device)
            if sc is None:
                continue
            nll += sc["nll"]
            ntok += sc["ntok"]
            first += int(sc["first"])
            total += 1
            per_variant[v][1] += 1
            if not a.no_greedy:
                g = greedy(model, tok, p_ids, device, a.max_new).strip()
                ok = g == t["answer"].strip()
                exact += int(ok)
                per_variant[v][0] += int(ok)
                if i == 0 and v == a.variants - 1:
                    print("sanity: gold={!r} got={!r}".format(t["answer"], g), flush=True)
        if (i + 1) % 200 == 0:
            print("  {} / {}".format(i + 1, len(tasks)), flush=True)
    el = time.perf_counter() - t0

    out = {
        "ckpt": a.ckpt,
        "step": st.get("step"),
        "device": str(device),
        "tasks": len(tasks),
        "variants": a.variants,
        "scored": total,
        "first_tok_acc": round(first / max(1, total), 4),
        "greedy_exact_acc": round(exact / max(1, total), 4) if not a.no_greedy else None,
        "answer_ppl": round(math.exp(nll / max(1, ntok)), 4),
        "answer_nll": round(nll / max(1, ntok), 4),
        "per_variant_exact": {str(v): round(c[0] / max(1, c[1]), 4)
                              for v, c in per_variant.items()} if not a.no_greedy else None,
        "eval_sec": round(el, 1),
    }

    id_pairs = []
    for f in (a.id, a.id2):
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                txt = json.loads(line)["text"]
                for sep in ("\nA: ", "\nAnswer: "):
                    if sep in txt:
                        pr, ans = txt.rsplit(sep, 1)
                        id_pairs.append((pr + sep, ans))
                        break
    if id_pairs:
        if a.limit:
            id_pairs = id_pairs[:a.limit]
        fi = ex = tt = nn = 0
        for pr, ans in id_pairs:
            p_ids, a_ids = pair_ids(tok, pr, ans)
            sc = score(model, p_ids, a_ids, device)
            if sc is None:
                continue
            fi += int(sc["first"])
            tt += sc["ntok"]
            nn += sc["nll"]
            if not a.no_greedy:
                ex += int(greedy(model, tok, p_ids, device, a.max_new).strip() == ans.strip())
        out["id_n"] = len(id_pairs)
        out["id_first_tok_acc"] = round(fi / max(1, len(id_pairs)), 4)
        out["id_greedy_exact"] = round(ex / max(1, len(id_pairs)), 4) if not a.no_greedy else None
        out["id_answer_ppl"] = round(math.exp(nn / max(1, tt)), 4)

    if os.path.exists(a.val):
        ids = np.fromfile(a.val, dtype=np.uint16).tolist()
        with torch.no_grad():
            tot, cnt = 0.0, 0
            for i in range(0, min(len(ids) - 1, 400 * cfg["ctx"]), cfg["ctx"]):
                w = ids[i:i + cfg["ctx"] + 1]
                if len(w) < 2:
                    break
                x = torch.tensor([w[:-1]], dtype=torch.long, device=device)
                y = torch.tensor([w[1:]], dtype=torch.long, device=device)
                lg = model(x)
                tot += float(F.cross_entropy(lg[0], y[0], reduction="sum"))
                cnt += y.numel()
        out["val_ppl"] = round(math.exp(tot / max(1, cnt)), 4)

    ood_texts = []
    for f in (a.ood, a.ood2):
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    ood_texts.append(json.loads(line)["text"])
    if ood_texts:
        v, _ = lm_ppl(model, tok, ood_texts, cfg["ctx"], device)
        out["ood_ppl"] = round(v, 4)

    if a.bench:
        pr = [render(t["chain"], t["q"], 2) for t in tasks[:min(len(tasks), a.bench)]]
        out["decode_bench"] = bench_decode(model, tok, pr, device, a.threads)

    print(json.dumps(out, indent=2), flush=True)
    with open(os.path.join(os.path.dirname(a.ckpt) or ".", "eval.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
