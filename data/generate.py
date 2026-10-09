import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = "abcdefghijklmnopqrstuvwxyz"

TRAIN_NS = {
    "ent": ["flur", "plix", "qvath", "zorb", "meln", "drak", "tivk", "gonk",
            "wexl", "lumo", "brix", "nash", "skel", "yurg", "trov", "krel",
            "dorn", "hexa"],
    "cat": ["bix", "wox", "zell", "quarn", "murt", "fenk", "dosk", "larn",
            "pyr", "vash", "trog", "mirn"],
    "food": ["zorn", "plim", "gruk", "tark", "muls", "xebl", "quib", "droth",
             "farn", "hask", "pree", "ylm"],
    "place": ["arka", "osta", "vira", "kalm", "lund", "setha", "mora", "drel",
              "nimo", "zeph", "ilva", "tarn"],
    "fam": ["glip", "hornz", "rapt", "ursi", "cetra", "vorn"],
}

TEST_NS = {
    "ent": ["hexon", "krynn", "vosh", "thul", "mirg", "zant", "wroth",
            "plath", "cruv", "snell", "yirr", "quoz", "brell", "fenro",
            "galdr", "drupp"],
    "cat": ["skarn", "vorp", "gnash", "yelm", "tazik", "brevo", "koltr",
            "shumn"],
    "food": ["krekk", "floom", "zibit", "drull", "qwap", "vint", "shlom",
             "prax"],
    "place": ["jorra", "mikk", "sarno", "tulka", "pelki", "ornsk", "havla",
              "zeema"],
    "fam": ["bront", "skorr", "veldi", "pakh"],
}


BANNED_SYLLS = set()
for _ns in (TEST_NS,):
    for _vals in _ns.values():
        BANNED_SYLLS.update(_vals)

FOOD_SYLLS = set()


def make_name(rng, sylls, used):
    for _ in range(200):
        n = (sylls[rng.randrange(len(sylls))] + ALPHA[rng.randrange(26)] +
             str(rng.randrange(10, 10000)))
        if n not in used:
            used.add(n)
            return n
    raise RuntimeError("namespace exhausted for {}".format(sylls[0]))


def make_food(rng, used):
    for _ in range(4000):
        n = rng.choice((3, 4, 4, 5, 5, 6, 6))
        s = "".join(ALPHA[rng.randrange(26)] for _ in range(n))
        if s in FOOD_SYLLS or s in BANNED_SYLLS:
            continue
        FOOD_SYLLS.add(s)
        name = s + ALPHA[rng.randrange(26)] + str(rng.randrange(10, 10000))
        used.add(name)
        return name
    raise RuntimeError("food syllable space exhausted")


def chain(rng, ns, used):
    return {
        "ent": make_name(rng, ns["ent"], used),
        "cat": make_name(rng, ns["cat"], used),
        "food": make_food(rng, used),
        "place": make_name(rng, ns["place"], used),
        "fam": make_name(rng, ns["fam"], used),
    }


def facts_of(c):
    return [
        "a {} is a {}".format(c["ent"], c["cat"]),
        "{}s eat {}".format(c["cat"], c["food"]),
        "{}s live in {}".format(c["cat"], c["place"]),
        "{}s are a kind of {}".format(c["cat"], c["fam"]),
    ]


def pick_decoys(rng, decoys, banned, k):
    out = []
    tries = 0
    while len(out) < k and tries < 30:
        tries += 1
        f = decoys[rng.randrange(len(decoys))]
        if f in out:
            continue
        if any(b and b in f for b in banned):
            continue
        out.append(f)
    return out


def other_index(rng, size, i):
    j = rng.randrange(size)
    return (j + 1) % size if j == i else j


def s2_example(rng, c, decoys, unknown):
    facts = facts_of(c)
    order = list(range(4))
    rng.shuffle(order)
    passage = ". ".join(facts[i] for i in order) + "."
    roll = rng.random()
    if roll < 0.32:
        return passage
    if unknown is not None:
        q, a = "What does a {} eat?".format(unknown), "I don't know."
    elif roll < 0.56:
        q, a = "What does a {} eat?".format(c["ent"]), c["food"]
    elif roll < 0.70:
        q, a = "What do {}s eat?".format(c["cat"]), c["food"]
    elif roll < 0.84:
        q, a = "Where do {}s live?".format(c["cat"]), c["place"]
    else:
        q, a = "What kind of thing is a {}?".format(c["ent"]), c["cat"]
    return passage + "\nQ: " + q + "\nA: " + a


def s3_render(rng, mem_lines, q, a):
    block = "[MEM]\n" + "\n".join("- " + l + " (src: user)" for l in mem_lines) + "\n[/MEM]"
    v = rng.randrange(3)
    if v == 0:
        return block + "\nQuestion: " + q + "\nAnswer: " + a
    if v == 1:
        return ("Read the memory block, then answer the question.\n" + block +
                "\nQ: " + q + "\nA: " + a)
    return block + "\nQ: " + q + "\nA: " + a


def s3_example(rng, c, decoys, unknown):
    f = facts_of(c)
    banned = [c["ent"], c["cat"]]
    if unknown:
        banned.append(unknown)
    roll = rng.random()
    if unknown is not None and roll < 0.22:
        mem = [f[0], f[1]]
        rng.shuffle(mem)
        mem += pick_decoys(rng, decoys, banned, rng.randrange(0, 3))
        return s3_render(rng, mem, "What does a {} eat?".format(unknown),
                         "I don't know.")
    if roll < 0.45:
        mem = [f[0], f[1]]
        mem += pick_decoys(rng, decoys, banned, rng.randrange(0, 5) if rng.random() < 0.55 else 0)
        rng.shuffle(mem)
        return s3_render(rng, mem, "What does a {} eat?".format(c["ent"]),
                         c["food"])
    if roll < 0.65:
        mem = [f[1], f[2]]
        mem += pick_decoys(rng, decoys, banned, rng.randrange(0, 4) if rng.random() < 0.4 else 0)
        rng.shuffle(mem)
        return s3_render(rng, mem, "What do {}s eat?".format(c["cat"]),
                         c["food"])
    if roll < 0.85:
        mem = [f[1]]
        mem += pick_decoys(rng, decoys, banned, rng.randrange(0, 4) if rng.random() < 0.4 else 0)
        rng.shuffle(mem)
        return s3_render(rng, mem, "Where do {}s live?".format(c["cat"]),
                         c["place"])
    mem = [f[0]]
    mem += pick_decoys(rng, decoys, banned, rng.randrange(0, 4) if rng.random() < 0.5 else 0)
    rng.shuffle(mem)
    return s3_render(rng, mem, "What kind of thing is a {}?".format(c["ent"]),
                     c["cat"])


def build(ns, n_s2, n_s3, seed, pool=40000, decoy_pool=400):
    rng = random.Random(seed)
    used = set()
    size = min(max(n_s2, n_s3) + 64, pool)
    chains = [chain(rng, ns, used) for _ in range(size)]
    decoy_facts = []
    for i in range(min(decoy_pool, size // 2)):
        decoy_facts.extend(facts_of(chains[i]))
    s2, s3, tasks = [], [], []
    for i in range(n_s2):
        c = dict(chains[i % size])
        c["food"] = make_food(rng, used)
        unknown = None
        if rng.random() < 0.08:
            unknown = chains[other_index(rng, size, i % size)]["ent"]
        s2.append({"text": s2_example(rng, c, decoy_facts, unknown)})
    for i in range(n_s3):
        c = dict(chains[(i * 3 + 1) % size])
        c["food"] = make_food(rng, used)
        unknown = None
        if rng.random() < 0.10:
            unknown = chains[other_index(rng, size, (i * 3 + 1) % size)]["ent"]
        s3.append({"text": s3_example(rng, c, decoy_facts, unknown)})
    for i in range(min(1000, size)):
        c = dict(chains[i])
        c["food"] = make_food(rng, used)
        tasks.append({
            "q": "What does a {} eat?".format(c["ent"]),
            "answer": c["food"],
            "chain": [facts_of(c)[0], facts_of(c)[1]],
        })
    return s2, s3, tasks


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s2", type=int, default=130000)
    p.add_argument("--s3", type=int, default=95000)
    p.add_argument("--val", type=int, default=2000)
    p.add_argument("--id", type=int, default=2000)
    p.add_argument("--tasks", type=int, default=1000)
    a = p.parse_args()

    tr_s2, tr_s3, _ = build(TRAIN_NS, a.s2, a.s3, seed=101)
    id_s2, id_s3, _ = build(TRAIN_NS, a.id, a.id, seed=202)
    ood_s2, ood_s3, _ = build(TEST_NS, a.val, a.val, seed=303)
    _, _, tasks = build(TEST_NS, a.tasks, a.tasks, seed=404)

    write_jsonl(os.path.join(HERE, "s2.train.jsonl"), tr_s2)
    write_jsonl(os.path.join(HERE, "s3.train.jsonl"), tr_s3)
    write_jsonl(os.path.join(HERE, "s2.id.jsonl"), id_s2)
    write_jsonl(os.path.join(HERE, "s3.id.jsonl"), id_s3)
    write_jsonl(os.path.join(HERE, "s2.ood.jsonl"), ood_s2)
    write_jsonl(os.path.join(HERE, "s3.ood.jsonl"), ood_s3)
    write_jsonl(os.path.join(HERE, "tasks_val.jsonl"), tasks)

    with open(os.path.join(HERE, "namespace.json"), "w", encoding="utf-8") as fh:
        json.dump({"train": TRAIN_NS, "test": TEST_NS}, fh, ensure_ascii=False, indent=2)

    for name in ("s2.train", "s3.train", "s2.id", "s3.id", "s2.ood", "s3.ood", "tasks_val"):
        path = os.path.join(HERE, name + ".jsonl")
        chars = 0
        n = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                n += 1
                chars += len(line)
        print("{}: {} rows, {:.2f} MB text".format(name, n, chars / 1048576))


if __name__ == "__main__":
    main()
