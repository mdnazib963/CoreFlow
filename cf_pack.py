import struct
import sys

MAGIC = b"CORECF01"


def pack(out_path, entries):
    n = len(entries)
    hdr = 16 + 24 * n
    blobs = []
    off = hdr
    table = []
    for tag, data in entries:
        off = (off + 63) & ~63
        table.append((tag, off, len(data)))
        blobs.append((off, data))
        off += len(data)
    out = bytearray(MAGIC + struct.pack("<II", 1, n))
    for tag, o, s in table:
        out += tag.encode().ljust(8, b"\0") + struct.pack("<QQ", o, s)
    end = hdr
    for o, data in blobs:
        if o > len(out):
            out += b"\0" * (o - len(out))
        out += data
        end = o + len(data)
    with open(out_path, "wb") as f:
        f.write(out[:end])
    return table


def read_cf(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != MAGIC:
        raise SystemExit("bad magic")
    ver, n = struct.unpack_from("<II", buf, 8)
    if ver != 1:
        raise SystemExit("bad version")
    out = {}
    for i in range(n):
        e = 16 + 24 * i
        tag = buf[e:e + 8].rstrip(b"\0").decode()
        off, sz = struct.unpack_from("<QQ", buf, e + 8)
        out[tag] = buf[off:off + sz]
    return out


def main(argv):
    if len(argv) < 5:
        print("usage: cf_pack.py <out.cf> <model.i8> <stages.json> <manifest.json>")
        return 1
    out, model_p, ir_p, man_p = argv[1:5]
    entries = [
        ("manifest", open(man_p, "rb").read()),
        ("ir", open(ir_p, "rb").read()),
        ("model", open(model_p, "rb").read()),
    ]
    table = pack(out, entries)
    back = read_cf(out)
    for tag, data in entries:
        if back.get(tag) != data:
            raise SystemExit(f"roundtrip FAIL: {tag}")
    import os
    print(f"packed {out} ({os.path.getsize(out)} B) " +
          " ".join(f"{t}={s}@{o}" for t, o, s in table) + " roundtrip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
