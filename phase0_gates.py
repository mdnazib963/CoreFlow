import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "phase0", "phase0.exe")
L3_KB = 4096
BUDGET = L3_KB * 1024 // 2


def run_exe(model, *args, timeout=300):
    out = subprocess.run([EXE, model] + [str(a) for a in args],
                         capture_output=True, text=True, timeout=timeout)
    line = out.stdout.strip().splitlines()[-1]
    return json.loads(line)


def run_parity(ckpt, windows, seq, threads=4):
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "phase0_parity.py"),
         "--ckpt", ckpt, "--windows", str(windows), "--seq", str(seq),
         "--threads", str(threads)],
        capture_output=True, text=True, timeout=900)
    line = out.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phase0/model.i8")
    ap.add_argument("--ckpt", default="run/p0_best.pt")
    ap.add_argument("--val", default="data/val.bin")
    ap.add_argument("--out", default="phase0/gates.json")
    ap.add_argument("--bench-sec", type=float, default=60)
    ap.add_argument("--windows", type=int, default=80)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--skip-evict", action="store_true")
    a = ap.parse_args()

    res = {"model": a.model, "ckpt": a.ckpt, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    res["info"] = run_exe(a.model, "info")
    res["stream"] = run_exe(a.model, "stream", 2)
    res["clean"] = run_exe(a.model, "bench", a.val, a.bench_sec, 65, 0, 64,
                           timeout=int(a.bench_sec) + 120)
    if not a.skip_evict:
        res["evict_write"] = run_exe(a.model, "bench", a.val, a.bench_sec, 65, 1, 64,
                                     timeout=int(a.bench_sec) + 120)
        res["evict_read"] = run_exe(a.model, "bench", a.val, a.bench_sec, 65, 2, 64,
                                    timeout=int(a.bench_sec) + 120)
    res["c_ppl"] = run_exe(a.model, "ppl", a.val, a.windows, a.seq,
                           timeout=600)
    res["torch_ppl"] = run_parity(a.ckpt, a.windows, a.seq)

    info = res["info"]
    clean = res["clean"]
    gate1 = clean["tok_s"] >= 50 and clean["sec"] >= a.bench_sec * 0.99
    gate2_ws = info["fit_half_l3"]
    ev = res.get("evict_read") or res.get("evict_write")
    slowdown = (clean["tok_s"] / ev["tok_s"]) if ev else None
    gate2 = gate2_ws and (slowdown is not None and slowdown >= 1.5)
    cp, tp = res["c_ppl"]["ppl"], res["torch_ppl"]["ppl"]
    rel = abs(cp - tp) / tp
    gate3 = rel <= 0.01

    res["verdict"] = {
        "gate1_speed": {"pass": bool(gate1), "tok_s": clean["tok_s"],
                        "p99_ms": clean["p99_ms"], "sec": clean["sec"]},
        "gate2_cache": {"pass": bool(gate2),
                        "ws_bytes": info["working_set"],
                        "ws_pct_l3": info["ws_pct_l3"],
                        "evict_slowdown": slowdown,
                        "dram_gb_s": res["stream"]["gb_s"]},
        "gate3_parity": {"pass": bool(gate3), "c_ppl": cp, "torch_ppl": tp,
                         "delta": cp - tp, "rel": rel},
        "all_pass": bool(gate1 and gate2 and gate3),
    }

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    v = res["verdict"]
    print("model={} ws={} B ({:.1f}% L3)".format(
        a.model, info["working_set"], info["ws_pct_l3"]))
    print("gate1 speed : {}  {:.1f} tok/s  p99 {:.2f} ms  ({:.0f}s)".format(
        "PASS" if v["gate1_speed"]["pass"] else "FAIL",
        v["gate1_speed"]["tok_s"], v["gate1_speed"]["p99_ms"],
        v["gate1_speed"]["sec"]))
    print("gate2 cache : {}  ws<=1/2 L3:{}  evict slowdown {}x  (DRAM stream {:.1f} GB/s)".format(
        "PASS" if v["gate2_cache"]["pass"] else "FAIL", gate2_ws,
        "{:.2f}".format(slowdown) if slowdown else "n/a",
        v["gate2_cache"]["dram_gb_s"]))
    print("gate3 parity: {}  C {:.6f} vs torch {:.6f}  rel {:.4f}%".format(
        "PASS" if v["gate3_parity"]["pass"] else "FAIL", cp, tp, rel * 100))
    print("ALL PASS" if v["all_pass"] else "SOME FAILED")
    print("wrote {}".format(a.out))


if __name__ == "__main__":
    main()
