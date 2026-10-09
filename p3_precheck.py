import json
import subprocess
import sys

import torch

sys.path.insert(0, ".")
from model import CoreFlowLM
from phase0_export import write_model


def make(dff):
    cfg = dict(vocab=4096, d=144, dff=dff, nconv=4, natt=2, heads=8, kv=1,
               ctx=1024, softcap=30.0, stage_dropout=0.1, dhead=1,
               dhead_depth=3, dhead_ff=64, dhead_rot=1, qat=0, k=4)
    m = CoreFlowLM(cfg["vocab"], cfg["d"], cfg["dff"], cfg["nconv"],
                   cfg["natt"], cfg["heads"], cfg["kv"], cfg["ctx"],
                   cfg["softcap"], cfg["stage_dropout"], cfg["dhead"],
                   cfg["dhead_depth"], cfg["dhead_ff"], cfg["dhead_rot"],
                   cfg["qat"])
    return m, cfg


out = {}
for dff in (400, 396, 392):
    m, cfg = make(dff)
    ck = f"phase0/pre_p{dff}.pt"
    torch.save({"model": m.state_dict(), "config": cfg, "step": 0}, ck)
    i8 = f"phase0/model_pre{dff}.i8"
    size = write_model(i8, m, cfg, 0)
    info = json.loads(subprocess.run(
        ["phase0\\phase0.exe", i8, "info"], capture_output=True, text=True).stdout)
    st = f"phase0/stages_pre{dff}.json"
    r = subprocess.run([sys.executable, "phase0\\stage_ir.py", i8, st],
                       capture_output=True, text=True)
    uni = {}
    for line in r.stdout.splitlines():
        if line.startswith("N=4:"):
            uni["n4"] = line.split("max/avg=")[1]
    ir = json.load(open(st))
    parts = ir["partitions"]["4"]
    dh = ir.get("dhead")
    out[dff] = dict(export_bytes=size, info=info, uniformity_n4=float(uni["n4"]),
                    dhead_bytes=dh["bytes"] if dh else 0,
                    dhead_depth=dh["depth"] if dh else None,
                    n4_costs=[g["cost_mac"] for g in parts])
print(json.dumps(out, indent=1))
