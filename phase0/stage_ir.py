import json, struct, sys, itertools

HDR = struct.Struct("<13I f 2I 2x")
SEC = struct.Struct("<4I 2Q")
K_I8, K_F32 = 0, 1
TAGS = ["WTE","N1","N2","DW","GATE","MIX","F1","F2","F3","QKV","OUT","NF",
        "DN1","DM1","DM2","DM3","DN2"]

def read_model(path):
    buf = open(path, "rb").read()
    v = HDR.unpack_from(buf, 0)
    (magic, ver, vocab, d, dff, nconv, natt, heads, kvh, ctx,
     nblocks, nsections, k) = v[:13]
    softcap, sections_off, table_size = v[13], v[14], v[15]
    assert magic == 0x30465043, hex(magic)
    secs = []
    off = sections_off
    for _ in range(nsections):
        kind, rows, cols, tag, data_off, scale_off = SEC.unpack_from(buf, off)
        off += SEC.size
        data_len = rows * cols if kind == K_I8 else rows * 4
        scale_len = rows * 4 if kind == K_I8 else 0
        secs.append(dict(kind=kind, rows=rows, cols=cols, tag=TAGS[tag],
                         data_off=data_off, data_len=data_len,
                         scale_off=scale_off if kind == K_I8 else None,
                         scale_len=scale_len))
    return buf, dict(vocab=vocab, d=d, dff=dff, nconv=nconv, natt=natt,
                     heads=heads, kv_heads=kvh, ctx=ctx, nblocks=nblocks,
                     k=k, softcap=softcap), secs

def sec_range(s):
    r = [dict(off=s["data_off"], len=s["data_len"])]
    if s["scale_len"]:
        r.append(dict(off=s["scale_off"], len=s["scale_len"]))
    return r

def sec_entry(s):
    e = dict(tag=s["tag"], kind="i8" if s["kind"] == K_I8 else "f32",
             rows=s["rows"], cols=s["cols"], bytes=s["data_len"] + s["scale_len"],
             ranges=sec_range(s))
    return e

def build_units(hdr, secs):
    it = iter(secs)
    units = []
    wte = next(it)
    assert wte["tag"] == "WTE"
    for bi in range(hdr["nblocks"]):
        n1 = next(it); n2 = next(it)
        first = next(it)
        if first["tag"] == "DW":
            mats = [first, next(it), next(it)]
        else:
            assert first["tag"] == "QKV"
            mats = [first, next(it)]
        f1 = next(it); f2 = next(it); f3 = next(it)
        kind = "attn" if mats[0]["tag"] == "QKV" else "conv"
        owned = [n1, n2] + mats + [f1, f2, f3]
        cost = sum(s["rows"] * s["cols"] if s["kind"] == K_I8 else s["rows"]
                   for s in owned)
        if kind == "attn":
            cost += (hdr["ctx"] // 2) * 2 * hdr["d"]
        units.append(dict(unit_id=len(units), role="block", block=bi,
                          kind=kind, blocks=[bi], embed=False, head=False,
                          sections=[sec_entry(s) for s in owned],
                          reads=[], cost_mac=cost,
                          bytes=sum(s["data_len"] + s["scale_len"] for s in owned)))
    nf = next(it)
    assert nf["tag"] == "NF"
    rest = list(it)
    dhead = None
    if rest:
        assert [s["tag"] for s in rest] == ["DN1","DM1","DM2","DM3","DN2"], \
            [s["tag"] for s in rest]
        dhead = dict(depth=rest[0]["cols"],
                     bytes=sum(s["data_len"] + s["scale_len"] for s in rest),
                     sections=[sec_entry(s) for s in rest])
    head_cost = nf["rows"] + wte["rows"] * wte["cols"]
    units.insert(0, dict(unit_id=0, role="embed", block=None, kind="embed",
                         blocks=[], embed=True, head=False, sections=[],
                         reads=[sec_entry(wte)], cost_mac=hdr["d"], bytes=0))
    for i, u in enumerate(units):
        u["unit_id"] = i
    units.append(dict(unit_id=len(units), role="head", block=None, kind="head",
                      blocks=[], embed=False, head=True,
                      sections=[sec_entry(nf), sec_entry(wte)], reads=[],
                      cost_mac=head_cost,
                       bytes=nf["data_len"] + nf["scale_len"]
                             + wte["data_len"] + wte["scale_len"]))
    return units, dhead

def partition(units, n):
    if n > len(units):
        return None
    best = None
    for cuts in itertools.combinations(range(1, len(units)), n - 1):
        bounds = (0,) + cuts + (len(units),)
        groups = [(bounds[i], bounds[i + 1] - 1) for i in range(n)]
        costs = [sum(u["cost_mac"] for u in units[lo:hi + 1]) for lo, hi in groups]
        key = (max(costs), sum(c * c for c in costs))
        if best is None or key < best[0]:
            best = (key, groups, costs)
    return best

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "model_185.i8"
    out = sys.argv[2] if len(sys.argv) > 2 else "stages.json"
    buf, hdr, secs = read_model(model)
    units, dhead = build_units(hdr, secs)
    nparts = {}
    for n in range(1, len(units) + 1):
        r = partition(units, n)
        if r:
            _, groups, costs = r
            nparts[str(n)] = [dict(lo=lo, hi=hi, units=list(range(lo, hi + 1)),
                                   cost_mac=c)
                              for (lo, hi), c in zip(groups, costs)]
    ir = dict(
        ir_version=1,
        model=model,
        model_bytes=len(buf),
        arch=hdr,
        max_d=hdr["d"],
        chain=[u["unit_id"] for u in units],
        units=units,
        dhead=dhead,
        partitions=nparts,
        edges="chain",
        note="weights own one home: sections touched only by owning stage core; "
             "embed reads wte (home=unit7 head). costs are estimated MACs "
             "(attn kv term assumes ctx/2).",
    )
    with open(out, "w") as f:
        json.dump(ir, f, indent=1)
    total = sum(u["cost_mac"] for u in units)
    print(f"units={len(units)} model_bytes={len(buf)}")
    for n, p in nparts.items():
        groups = ",".join(f"[{g['lo']}-{g['hi']}]:{g['cost_mac']}"
                          for g in p)
        mx = max(g["cost_mac"] for g in p)
        print(f"N={n}: {groups}  max/avg={mx / (total / int(n)):.2f}")

if __name__ == "__main__":
    main()
