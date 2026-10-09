import glob
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", re.I)
END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
KEEP = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
           " .,;:'\"-!?()[]{}/\n")


def clean_text(text):
    m = START.search(text)
    if m:
        text = text[m.end():]
    m = END.search(text)
    if m:
        text = text[:m.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            out.append("")
            continue
        good = sum(1 for ch in s if ch in KEEP)
        if good / len(s) < 0.80:
            continue
        if len(s) > 3 and s.isupper() and not s.endswith("."):
            continue
        out.append(s)
    return "\n".join(out).strip()


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.txt")))
    rng = random.Random(0)
    rng.shuffle(files)
    val_books = set(os.path.basename(f) for f in files[:3])
    stats = {"train": [], "val": []}
    handles = {
        "train": open(os.path.join(HERE, "s1.train.txt"), "w", encoding="utf-8"),
        "val": open(os.path.join(HERE, "s1.val.txt"), "w", encoding="utf-8"),
    }
    try:
        for f in files:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                text = clean_text(fh.read())
            split = "val" if os.path.basename(f) in val_books else "train"
            if len(text) < 20000:
                continue
            handles[split].write(text + "\n\n")
            stats[split].append((os.path.basename(f), len(text)))
    finally:
        for fh in handles.values():
            fh.close()
    for split in ("train", "val"):
        total = sum(n for _, n in stats[split])
        print("{}: {} books, {} chars, {:.2f} MB".format(
            split, len(stats[split]), total, total / 1048576))


if __name__ == "__main__":
    main()
