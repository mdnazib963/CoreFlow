import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAD, BOS, EOS = 0, 1, 2
BYTE_OFF = 3
SHIFT = 20

PAT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


class Tokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {}
        self.table = {}
        for r, (a, b) in enumerate(self.merges):
            self.ranks[(a, b)] = r
            self.table[(a, b)] = BYTE_OFF + 256 + r
        self._pieces = {i: bytes([i - BYTE_OFF]) for i in range(BYTE_OFF, BYTE_OFF + 256)}

    def piece(self, tok):
        p = self._pieces.get(tok)
        if p is not None:
            return p
        a, b = self.merges[tok - BYTE_OFF - 256]
        p = self.piece(a) + self.piece(b)
        self._pieces[tok] = p
        return p

    def encode_chunk(self, ids):
        if len(ids) < 2:
            return list(ids)
        ids = list(ids)
        while len(ids) > 1:
            best_rank = None
            best_i = -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i < 0:
                break
            ids[best_i:best_i + 2] = [self.table[(ids[best_i], ids[best_i + 1])]]
        return ids

    def encode(self, text, add_specials=False):
        out = []
        for chunk in PAT.findall(text):
            ids = [BYTE_OFF + b for b in chunk.encode("utf-8")]
            out.extend(self.encode_chunk(ids))
        if add_specials:
            return [BOS] + out + [EOS]
        return out

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            if i < BYTE_OFF:
                continue
            buf.extend(self.piece(i))
        return buf.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "vocab_size": BYTE_OFF + 256 + len(self.merges),
                       "specials": {"pad": PAD, "bos": BOS, "eos": EOS},
                       "pattern": PAT.pattern, "merges": [list(m) for m in self.merges]},
                      fh)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(d["merges"])


def build_sample(total):
    parts = []
    with open(os.path.join(HERE, "s1.train.txt"), "rb") as fh:
        parts.append(fh.read(total // 2))
    for name, share in (("s2.train.jsonl", total // 4), ("s3.train.jsonl", total // 4)):
        buf = bytearray()
        with open(os.path.join(HERE, name), "r", encoding="utf-8") as fh:
            for line in fh:
                buf += json.loads(line)["text"].encode("utf-8")
                buf += b"\n\n"
                if len(buf) >= share:
                    break
        parts.append(bytes(buf))
    return b"".join(parts)


def flatten(sample, sep=EOS):
    ids = []
    append = ids.append
    for chunk in PAT.findall(sample.decode("utf-8", errors="ignore")):
        for b in chunk.encode("utf-8"):
            append(BYTE_OFF + b)
        append(sep)
    return np.array(ids, dtype=np.int32)


def train(sample, vocab_size, sep=EOS, log_every=256):
    seq = flatten(sample, sep=sep)
    total_raw = int((seq != sep).sum())
    n_merges = vocab_size - (BYTE_OFF + 256)
    merges = []
    mask_a2 = (1 << SHIFT) - 1
    for rank in range(n_merges):
        if seq.size < 2:
            break
        valid = (seq[:-1] != sep) & (seq[1:] != sep)
        key = np.where(valid, (seq[:-1].astype(np.int64) << SHIFT) | seq[1:], -1)
        uk, cnt = np.unique(key, return_counts=True)
        keepk = uk >= 0
        uk, cnt = uk[keepk], cnt[keepk]
        maxc = int(cnt.max()) if cnt.size else 0
        if maxc < 2:
            break
        k = int(uk[cnt == maxc].max())
        a, b = k >> SHIFT, k & mask_a2
        mask = (seq[:-1] == a) & (seq[1:] == b)
        if a == b:
            seen = np.cumsum(mask) - mask
            sel = mask & (seen % 2 == 0)
        else:
            sel = mask
        idx = np.flatnonzero(sel)
        keep = np.ones(seq.size, dtype=bool)
        keep[idx + 1] = False
        vals = seq.copy()
        vals[idx] = BYTE_OFF + 256 + rank
        seq = vals[keep]
        merges.append([a, b])
        if log_every and rank % log_every == 0:
            live = int((seq != sep).sum())
            print("merge {:4d}/{}  live={}  compression={:.2f} B/tok".format(
                rank, n_merges, live, total_raw / max(1, live)), flush=True)
    return merges


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--sample", type=int, default=1500000)
    p.add_argument("--out", type=str, default=os.path.join(HERE, "tokenizer.json"))
    a = p.parse_args()
    sample = build_sample(a.sample)
    print("sample: {:.2f} MB".format(len(sample) / 1048576), flush=True)
    merges = train(sample, a.vocab)
    tok = Tokenizer(merges)
    tok.save(a.out)
    enc = tok.encode("The quick brown fox jumps over 1234 lazy dogs.")
    raw = len("The quick brown fox jumps over 1234 lazy dogs.")
    print("merges: {}  vocab: {}".format(len(merges), BYTE_OFF + 256 + len(merges)))
    print("roundtrip:", tok.decode(enc))
    print("test: {} chars -> {} tokens = {:.2f} B/tok".format(raw, len(enc), raw / len(enc)))
    enc2 = tok.encode(sample.decode("utf-8", errors="ignore")[:200000])
    print("sample slice: {:.2f} B/tok".format(200000 / max(1, len(enc2))))
    print("saved:", a.out)


if __name__ == "__main__":
    main()
