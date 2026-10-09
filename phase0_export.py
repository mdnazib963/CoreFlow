import argparse
import json
import os
import struct
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eval import load_ckpt

MAGIC = 0x30465043
VERSION = 1
K_I8_ROWS = 0
K_F32 = 1

TAG_WTE, TAG_N1, TAG_N2, TAG_DW, TAG_GATE, TAG_MIX = 0, 1, 2, 3, 4, 5
TAG_F1, TAG_F2, TAG_F3, TAG_QKV, TAG_OUT, TAG_NF = 6, 7, 8, 9, 10, 11
TAG_DN1, TAG_DM1, TAG_DM2, TAG_DM3, TAG_DN2 = 12, 13, 14, 15, 16


def q_rows(w):
    w = w.detach().float().cpu()
    if w.dim() == 1:
        return w.numpy(), None
    rows = w.reshape(w.shape[0], -1)
    amax = rows.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.clamp((rows / scale[:, None]).round(), -127, 127).to(torch.int8)
    return q.reshape(w.shape).numpy(), scale.numpy().astype("<f4")


def sec(kind, tag, arr, scale=None):
    return {"kind": kind, "tag": tag, "rows": int(arr.shape[0]),
            "cols": int(arr.shape[1]) if arr.ndim > 1 else 1,
            "arr": arr, "scale": scale}


def collect(model):
    secs = []
    secs.append(sec(K_I8_ROWS, TAG_WTE, *q_rows(model.wte.weight)))
    for blk in model.blocks:
        if type(blk).__name__ == "ConvBlock":
            secs.append(sec(K_F32, TAG_N1, blk.n1.weight.detach().float().cpu().numpy()[:, None]))
            secs.append(sec(K_F32, TAG_N2, blk.n2.weight.detach().float().cpu().numpy()[:, None]))
            secs.append(sec(K_I8_ROWS, TAG_DW, *q_rows(blk.dw.weight)))
            secs.append(sec(K_I8_ROWS, TAG_GATE, *q_rows(blk.gate.weight)))
            secs.append(sec(K_I8_ROWS, TAG_MIX, *q_rows(blk.mix.weight)))
            secs.append(sec(K_I8_ROWS, TAG_F1, *q_rows(blk.mlp.f1.weight)))
            secs.append(sec(K_I8_ROWS, TAG_F2, *q_rows(blk.mlp.f2.weight)))
            secs.append(sec(K_I8_ROWS, TAG_F3, *q_rows(blk.mlp.f3.weight)))
        else:
            secs.append(sec(K_F32, TAG_N1, blk.n1.weight.detach().float().cpu().numpy()[:, None]))
            secs.append(sec(K_F32, TAG_N2, blk.n2.weight.detach().float().cpu().numpy()[:, None]))
            secs.append(sec(K_I8_ROWS, TAG_QKV, *q_rows(blk.qkv.weight)))
            secs.append(sec(K_I8_ROWS, TAG_OUT, *q_rows(blk.out.weight)))
            secs.append(sec(K_I8_ROWS, TAG_F1, *q_rows(blk.mlp.f1.weight)))
            secs.append(sec(K_I8_ROWS, TAG_F2, *q_rows(blk.mlp.f2.weight)))
            secs.append(sec(K_I8_ROWS, TAG_F3, *q_rows(blk.mlp.f3.weight)))
    secs.append(sec(K_F32, TAG_NF, model.nf.weight.detach().float().cpu().numpy()[:, None]))
    if getattr(model, "dhead", 0):
        dn1 = sec(K_F32, TAG_DN1, model.dn1.weight.detach().float().cpu().numpy()[:, None])
        dn1["cols"] = int(model.dhead_depth)
        secs.append(dn1)
        secs.append(sec(K_I8_ROWS, TAG_DM1, *q_rows(model.dm.f1.weight)))
        secs.append(sec(K_I8_ROWS, TAG_DM2, *q_rows(model.dm.f2.weight)))
        secs.append(sec(K_I8_ROWS, TAG_DM3, *q_rows(model.dm.f3.weight)))
        secs.append(sec(K_F32, TAG_DN2, model.dn2.weight.detach().float().cpu().numpy()[:, None]))
    return secs


HDR = struct.Struct("<13I f 2I 2x")
SEC = struct.Struct("<4I 2Q")


def write_model(path, model, cfg, spread=0):
    secs = collect(model)
    hdr_size = HDR.size
    table_size = SEC.size * len(secs)
    payload0 = hdr_size + table_size
    stride = spread // len(secs) if spread else 0
    table, chunks = [], []
    cur = payload0
    for i, s in enumerate(secs):
        a = s["arr"]
        raw = a.tobytes()
        if stride:
            doff = payload0 + i * stride
            if doff < cur:
                raise RuntimeError("stride too small for section {}".format(i))
        else:
            doff = cur
        soff = doff + len(raw)
        if s["kind"] == K_I8_ROWS:
            raw2 = s["scale"].tobytes()
            cur = max(cur, soff + len(raw2))
            table.append(SEC.pack(s["kind"], s["rows"], s["cols"], s["tag"], doff, soff))
            chunks.append((doff, raw))
            chunks.append((soff, raw2))
        else:
            cur = max(cur, doff + len(raw))
            table.append(SEC.pack(s["kind"], s["rows"], s["cols"], s["tag"], doff, 0))
            chunks.append((doff, raw))

    c = model.ctx if hasattr(model, "ctx") else cfg.get("ctx", 1024)
    hdr = HDR.pack(MAGIC, VERSION, cfg["vocab"], cfg["d"], cfg["dff"],
                   cfg["nconv"], cfg["natt"], cfg["heads"], cfg["kv"],
                   c, len(model.blocks), len(secs), cfg["k"],
                   float(cfg["softcap"]), hdr_size, table_size)
    body = bytearray(cur - payload0)
    for off, data in chunks:
        body[off - payload0: off - payload0 + len(data)] = data
    with open(path, "wb") as fh:
        fh.write(hdr)
        for e in table:
            fh.write(e)
        fh.write(body)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--spread", type=int, default=0,
                    help="pad layout so payload spans this many bytes")
    a = ap.parse_args()

    model, cfg, st = load_ckpt(a.ckpt, torch.device("cpu"))
    cfg = dict(cfg)
    cfg["k"] = 4
    size = write_model(a.out, model, cfg, a.spread)
    man = {k: cfg.get(k) for k in ("vocab", "d", "dff", "nconv", "natt",
                                   "heads", "kv", "ctx", "softcap")}
    man["k"] = 4
    if getattr(model, "dhead", 0):
        man["dhead"] = 1
        man["dhead_depth"] = int(model.dhead_depth)
        man["dhead_ff"] = int(model.dm.f1.out_features)
    man["params"] = int(sum(p.numel() for p in model.parameters()))
    man["bytes"] = int(size)
    man["step"] = st.get("step")
    man["ckpt"] = a.ckpt
    mp = a.manifest or (os.path.splitext(a.out)[0] + ".json")
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    print(json.dumps(man, indent=2))
    print("wrote {} ({:.1f} KB)".format(a.out, size / 1024))


if __name__ == "__main__":
    main()
