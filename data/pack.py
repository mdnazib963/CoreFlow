import argparse
import json
import os
import random
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)

from bpe import EOS, Tokenizer

STREAMS = ("s1", "s2", "s3")


def load_jsonl_chunks(path, tok):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            ids = tok.encode(json.loads(line)["text"])
            if ids:
                ids.append(EOS)
                out.append(array("H", ids))
    return out


def load_s1_chunks(path, tok, target=1600):
    out = []
    buf = array("H")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    for para in text.split("\n\n"):
        ids = tok.encode(para.strip())
        if not ids:
            continue
        buf.extend(ids)
        buf.append(EOS)
        if len(buf) >= target:
            out.append(buf)
            buf = array("H")
    if len(buf) > 32:
        out.append(buf)
    return out


def concat(chunks):
    total = sum(len(c) for c in chunks)
    out = array("H", bytes(2 * total))
    i = 0
    for c in chunks:
        out[i:i + len(c)] = c
        i += len(c)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frac-s1", type=float, default=1.0)
    p.add_argument("--frac-s2", type=float, default=1.0)
    p.add_argument("--frac-s3", type=float, default=1.0)
    p.add_argument("--val-tokens", type=int, default=120000)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    tok = Tokenizer.load(os.path.join(HERE, "tokenizer.json"))
    print("tokenizer: {} merges".format(len(tok.merges)), flush=True)

    rng = random.Random(a.seed)

    s1 = load_s1_chunks(os.path.join(HERE, "s1.train.txt"), tok)
    print("s1: {} chunks, {} tokens".format(len(s1), sum(len(c) for c in s1)), flush=True)
    s2 = load_jsonl_chunks(os.path.join(HERE, "s2.train.jsonl"), tok)
    print("s2: {} chunks, {} tokens".format(len(s2), sum(len(c) for c in s2)), flush=True)
    s3 = load_jsonl_chunks(os.path.join(HERE, "s3.train.jsonl"), tok)
    print("s3: {} chunks, {} tokens".format(len(s3), sum(len(c) for c in s3)), flush=True)

    def take(chunks, frac):
        if frac >= 1.0:
            return list(chunks)
        rng.shuffle(chunks)
        n = max(1, int(len(chunks) * frac))
        return chunks[:n]

    s1 = take(s1, a.frac_s1)
    s2 = take(s2, a.frac_s2)
    s3 = take(s3, a.frac_s3)
    print("kept: s1={}  s2={}  s3={}".format(
        sum(len(c) for c in s1), sum(len(c) for c in s2), sum(len(c) for c in s3)), flush=True)

    chunks = s1 + s2 + s3
    rng.shuffle(chunks)
    train = concat(chunks)
    print("train: {} tokens ({:.2f} MB)".format(len(train), len(train) * 2 / 1048576), flush=True)
    with open(os.path.join(HERE, "train.bin"), "wb") as fh:
        fh.write(train.tobytes())

    val_chunks = []
    val_chunks += load_s1_chunks(os.path.join(HERE, "s1.val.txt"), tok, target=800)
    val_chunks += load_jsonl_chunks(os.path.join(HERE, "s2.id.jsonl"), tok)
    val_chunks += load_jsonl_chunks(os.path.join(HERE, "s3.id.jsonl"), tok)
    val_chunks += load_jsonl_chunks(os.path.join(HERE, "s2.ood.jsonl"), tok)
    val_chunks += load_jsonl_chunks(os.path.join(HERE, "s3.ood.jsonl"), tok)
    rng.shuffle(val_chunks)
    val = concat(val_chunks)
    if len(val) > a.val_tokens:
        val = val[:a.val_tokens]
    with open(os.path.join(HERE, "val.bin"), "wb") as fh:
        fh.write(val.tobytes())
    print("val: {} tokens".format(len(val)), flush=True)

    meta = {
        "train_tokens": len(train),
        "val_tokens": len(val),
        "s1_chunks": len(s1),
        "s2_chunks": len(s2),
        "s3_chunks": len(s3),
        "frac_s1": a.frac_s1,
        "frac_s2": a.frac_s2,
        "frac_s3": a.frac_s3,
        "vocab": 259 + len(tok.merges),
        "estimated_train_hours_434_toks": round(len(train) / 434 / 3600, 2),
    }
    with open(os.path.join(HERE, "pack_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
