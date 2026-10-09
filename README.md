# CoreFlow

**A 1.9 MB int8 language model that decodes 3,400+ tokens/second on a single CPU core — trained from scratch in pure PyTorch, served by a dependency-free C11 runtime. No Hugging Face wrappers, no llama.cpp, no GPU.**

Target spec was *">50 tok/s on one core."* The 12.6M variant hits 346 tok/s (6.9× the goal) despite being 333% of L3; the cache-fit 1.9M model — sized so its weights live in L3 — does **3,405 tok/s sustained, single core** (68×), with 99.4% token-level parity against the PyTorch reference.

| Metric | Result |
|---|---|
| Single-core decode (15 s sustained) | **3,405 tok/s** · p50 0.261 ms · p99 0.704 ms |
| Single-core decode (G5 median) | **3,609 tok/s** (2.4× over float kernel baseline) |
| 4 streams × 4 tokens in flight (60 s) | **5,997 tok/s** |
| Model size on disk | **1,965,250 B** (1.91 M params, int8) |
| Working set vs L3 | **49.8% of 4 MB L3** — cache-resident by design |
| C vs PyTorch perplexity | 26.8407 vs 26.8188 — **0.082% rel. delta** |
| Token-level parity vs PyTorch | **99.41%** match (first-token 98.7%) |
| Isolated cache-residency benefit | **1.15×** (clflush cold vs warm, position-matched) |

All numbers are reproducible from the JSON records in [`phase0/`](phase0/) — every claim below links to its source file.

---

## Demo

```console
> phase0\build_p0.bat
> phase0\phase0.exe phase0\model_p3.i8 info
{"file_bytes":1965250,"d":144,"dff":396,"nconv":4,"natt":2,"heads":8,"kv":1,
 "ctx":1024,"vocab":4096,"working_set":2090710,"l3":4096,"ws_pct_l3":49.8,
 "fit_half_l3":true}

> phase0\phase0.exe phase0\model_p3.i8 bench data\val.bin 15.0 65
{"mode":"clean","tokens":51077,"sec":15.0,"tok_s":3405.11,"p50_ms":0.261,
 "p99_ms":0.704,"prefill_tok_s":6126}

> phase0\phase0.exe phase0\model_p3.i8 stream 2.0
{"mode":"stream","sec":2.003,"passes":247,"gb_s":16.55}
```

Every forward pass streams the model's full ~1.97 MB working set, so 3,405 tok/s means **~6.7 GB/s of weight reads — served out of L3, not DRAM**. The proof is in the eviction tests: force a concurrent process to pollute L3 (`evict_read` mode in the same binary) and the identical model drops to 1,681 tok/s — a 2.03× penalty you only pay when the cache stops working (`gates_p5.json`). The `stream` command measures your box's raw DRAM bandwidth instead (128 MB sequential scan — 16.4 GB/s on the reference machine), so you can compare the two directly.

---

## Quickstart

**Requirements:** Windows x64, MSVC Build Tools 2022 (C11), Python 3.10+ (training/eval only — the runtime has zero dependencies).

```bat
:: build the runtime (AVX2 int8 kernels)
phase0\build_p0.bat

:: sanity check: model metadata + cache-fit verdict
phase0\phase0.exe phase0\model_p3.i8 info

:: sustained decode benchmark (tokens/sec on your machine)
phase0\phase0.exe phase0\model_p3.i8 bench data\val.bin 15.0 65

:: raw DRAM bandwidth probe (128 MB scan — compare against the 6.7 GB/s
:: of weight reads the model actually needs from cache)
phase0\phase0.exe phase0\model_p3.i8 stream 2.0
```

Expected: `tok_s` in the thousands, `fit_half_l3: true`, `gb_s` well above what your DRAM sustains for random access. Full benchmark suite (accuracy gates + eviction tests + multi-stream):

```bat
phase0\build_rt.bat
phase0\stage_rt.exe phase0\model_p3.i8 phase0\stages_p3.json bench 1
```

---

## How it works

**1. Size the model to the cache, then never leave it.** The cache-fit law: keep the int8 weight footprint ≤ ½ of L3 (here 1.97 MB against 4 MB L3) and decode becomes a cache-streaming problem instead of a DRAM problem. Measured isolated benefit of residency: 1.15× (cold vs warm, weights clflushed, position-matched — `cold_warm_summary_p5.json`). More importantly it removes the 2× run-to-run variance that plagues non-resident models: the 12.6M variant (333% of L3) swings 275–540 tok/s with memory contention; the cache-fit model holds p99 under 1 ms (`bench126_N1_p5_r*.json`).

**2. Hybrid architecture, O(1) decode state.** 4 gated short-conv blocks + 2 grouped-query attention blocks (kv=1, 8 heads), RMSNorm, RoPE, softcapped logits — conv-heavy so the KV cache stays tiny (1×d per layer, not seq_len×d), vocab 4096 so the LM head doesn't dominate. Spec: [`BRIEF.md`](BRIEF.md).

**3. Integer kernels, hand-written C11.** Per-input int8 activation quant amortized across all rows, sign-extend dot (`vpmovsxbw` + `vpmaddwd`, 4-accumulator 64-wide), AVX2 vectorized attention scores/values, degree-7 polynomial exp for softmax, row prefetch. Float fallback retained for wide layers. 1,494 → 3,696 tok/s single-core through the kernel push alone (`phase5_results.json` → `profile_us_per_token`).

**4. Parity is a gate, not a hope.** The C runtime must match PyTorch: perplexity delta ≤ 1% (measured 0.082%), greedy-token match 99.41%, plus determinism + ASan suites. All four Phase-5 gates pass in [`phase5_results.json`](phase0/phase5_results.json).

---

## Measured results

### Single core (N=1, 60 s sustained)

| Model | tok/s | Notes |
|---|---|---|
| cache-fit p3 (1.9 MB) | **3,444** | median, fits ½ L3 |
| 12.6M (12.8 MB, 333% of L3) | 346 | median of 3; range 275–540 tracks DRAM contention |
| float-kernel baseline | 1,494 | Phase-5 starting point |

### Multi-stream (60 s, median of 3 runs — `scaling60_*_p5.json`)

Streams and tokens-in-flight are how you beat single-core pipeline serialization (one token in flight = flat N≤4 ceiling):

| Config | tok/s | vs same-N K=1 | vs N=1 K=1 |
|---|---|---|---|
| N2 K2 | 5,048 | 1.38× | 1.47× |
| **N4 K4** | **5,997** | 1.72× | **1.74×** |
| N8 K8 | 4,069 | 1.77× | 1.18× |
| N4 K8 | 5,744 | 1.64× | — (K=N saturates) |

### Gates (Phase 5, all pass)

| Gate | Verdict |
|---|---|
| G5-a accuracy | C ppl 26.8407 vs torch 26.8188 (0.082%), token match 99.41% |
| G5-b speed | 3,609 tok/s median, 2.42× vs float baseline |
| G5-c cache-fit law | ws 49.8% of L3; 2.03× under concurrent eviction pressure; isolated benefit 1.15× |
| G5-d determinism / ASan | clean |

---

## Training from scratch

Pure PyTorch, no HF model/transformers dependency — data pipeline (`data/`: Gutenberg texts → `clean.py` → synthetic task streams `generate.py` → BPE `bpe.py` → `pack.py`), training (`train.py`), eval (`eval.py`), int8 export (`phase0_export.py`). Corpus build takes ~80 s; then:

```bash
python data/clean.py && python data/generate.py && python data/pack.py   # ~80 s, writes data/train.bin
python train.py --train data/train.bin --val data/val.bin --out run \
  --d 144 --dff 396 --nconv 4 --natt 2 --heads 8 --kv 1 --ctx 1024 --qat 10000
python phase0_export.py --ckpt run/best.pt --out phase0/model.i8
phase0\phase0.exe phase0\model.i8 info
```

GPU wrappers for Colab (`run_p0.ps1`, `run_full.ps1`) are included; the exported int8 artifact is what the C runtime loads. Evaluations (`eval.py`, `p4_suite.py`) cover perplexity, greedy parity, and task suites.

---

## Repository layout

```
model.py            hybrid conv+attention SLM (PyTorch reference)
train.py            from-scratch trainer (stage dropout, QAT-lite int8)
eval.py             perplexity / greedy parity / decode bench
phase0_export.py    checkpoint -> int8 .i8 + manifest
phase0/             C11 runtime: cf_model.c (kernels), phase0.c (CLI),
                    stage_rt.c (multi-stream runtime), build_*.bat,
                    *_results.json + gates (the actual measurements)
data/               corpus pipeline: raw texts, clean/generate/bpe/pack,
                    tokenizer + 240 KB validation slice
BRIEF.md            scientific brief: architecture & quantization rationale
```

---

## Roadmap

- **v1 (this):** cache-fit int8 kernels, single core, multi-stream N/K scaling, full parity gates — shipped.
- **v2 (CoreFlow-2):** 1.58-bit ternary weights (5× less bandwidth than int8), Mamba-style SSM blocks with O(1) state for unbounded context, MIMO decode (arithmetic-intensity ↑4×), elastic N-core runtime. No published CPU + MIMO + int8 result exists as of Oct 2026 — see [`BRIEF.md`](BRIEF.md).

---

## Star / contribute

Benchmarked something different on your hardware? Open an issue with your `phase0.exe ... bench` output — hardware diversity is the point. Docs gaps and portability work are good first issues.

## License

[MIT](LICENSE)
