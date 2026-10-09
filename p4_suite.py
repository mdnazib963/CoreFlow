import argparse
import json
import struct
import sys

import torch

import eval as ev

STOP = {0, 1, 2, ev.NEWLINE}


def greedy_ids(model, ids, device, max_new=24):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for _ in range(max_new):
            nxt = int(model(x)[0, -1].argmax())
            if nxt in STOP:
                break
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return out


def cmd_build(a):
    from bpe import Tokenizer
    tok = Tokenizer.load(a.tokenizer)
    device = torch.device("cpu")
    torch.manual_seed(0)
    model, cfg, st = ev.load_ckpt(a.ckpt, device)
    tasks = []
    with open(a.tasks, encoding="utf-8") as fh:
        for line in fh:
            tasks.append(json.loads(line))
    if a.limit:
        tasks = tasks[:a.limit]
    recs = []
    meta = []
    idx = 0
    with open(a.meta, "w", encoding="utf-8") as mf:
        for t in tasks:
            for v in range(a.variants):
                pr = ev.render(t["chain"], t["q"], v)
                p_ids, a_ids = ev.pair_ids(tok, pr, t["answer"])
                tids = greedy_ids(model, p_ids, device, a.max_new)
                recs.append((p_ids, a_ids, tids))
                meta.append({"idx": idx, "prompt": pr, "answer": t["answer"],
                             "a_ids": a_ids, "t_ids": tids,
                             "p_len": len(p_ids)})
                mf.write(json.dumps(meta[-1]) + "\n")
                idx += 1
    with open(a.bin_out, "wb") as f:
        f.write(struct.pack("<I", len(recs)))
        for p_ids, a_ids, tids in recs:
            f.write(struct.pack("<III", len(p_ids), len(a_ids), len(tids)))
            f.write(struct.pack("<{}H".format(len(p_ids)), *p_ids))
            f.write(struct.pack("<{}H".format(len(a_ids)), *a_ids))
            f.write(struct.pack("<{}H".format(len(tids)), *tids))
    maxp = max(len(r[0]) for r in recs)
    out = {"records": len(recs), "tasks": len(tasks),
           "variants": a.variants, "max_prompt_len": maxp,
           "ctx": cfg["ctx"],
           "ctx_ok": maxp + 24 + 1 <= cfg["ctx"],
           "bin_out": a.bin_out, "meta": a.meta}
    print(json.dumps(out, indent=2))
    return 0


def cmd_compare(a):
    from bpe import Tokenizer
    tok = Tokenizer.load(a.tokenizer)
    meta = []
    with open(a.meta, encoding="utf-8") as fh:
        for line in fh:
            meta.append(json.loads(line))
    with open(a.out, "rb") as f:
        buf = f.read()
    (np,) = struct.unpack_from("<I", buf, 0)
    off = 4
    recs = {}
    for _ in range(np):
        idx, first, ne = struct.unpack_from("<III", buf, off)
        off += 12
        ids = list(struct.unpack_from("<{}H".format(ne), buf, off)) if ne else []
        off += 2 * ne
        recs[idx] = (first, ids)
    first_ok = exact = pex = 0
    elem = tot = 0
    skipped = 0
    n = 0
    for m in meta:
        if m["idx"] not in recs:
            continue
        first, ids = recs[m["idx"]]
        a_ids, tids = m["a_ids"], m["t_ids"]
        if m["p_len"] + 25 > 1024:
            skipped += 1
        if a_ids:
            first_ok += int(first == a_ids[0])
        g = tok.decode(ids).strip()
        exact += int(g == m["answer"].strip())
        n += 1
        pex += int(ids == tids)
        k = max(len(ids), len(tids))
        for i in range(k):
            x = ids[i] if i < len(ids) else None
            y = tids[i] if i < len(tids) else None
            elem += int(x is not None and x == y)
            tot += 1
    out = {"prompts": n, "skipped": skipped,
           "first_tok_acc": round(first_ok / max(1, n), 4),
           "greedy_exact_acc": round(exact / max(1, n), 4),
           "torch_parity_exact": round(pex / max(1, n), 4),
           "torch_token_match": round(elem / max(1, tot), 4),
           "records_out": np}
    print(json.dumps(out, indent=2))
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ckpt", default="run/p3/run/best.pt")
    b.add_argument("--tasks", default="data/tasks_val.jsonl")
    b.add_argument("--tokenizer", default="data/tokenizer.json")
    b.add_argument("--bin-out", default="phase0/suite_prompts.bin")
    b.add_argument("--meta", default="phase0/suite_meta.jsonl")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--variants", type=int, default=3)
    b.add_argument("--max-new", type=int, default=24)
    c = sub.add_parser("compare")
    c.add_argument("--out", default="phase0/suite_out.bin")
    c.add_argument("--meta", default="phase0/suite_meta.jsonl")
    c.add_argument("--tokenizer", default="data/tokenizer.json")
    c.add_argument("--report", default="")
    a = p.parse_args()
    if a.cmd == "build":
        return cmd_build(a)
    return cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
