# CoreFlow — Scientific Brief v1
### Question (user's spec): *An SLM that runs on ONE CPU core at >50 tok/s, from scratch, pure PyTorch, no HF wrappers. Unlock the future, compact, fast. No regurgitation.*
### Method: fresh online research (2026) + first-principles reasoning. No reliance on training memory alone.

---

## The 5 questions that decide everything

| # | Question | Why it matters |
|---|---|---|
| Q1 | What is the *physics floor*? | Tells us how much margin we actually have |
| Q2 | Transformer, SSM, conv, or hybrid? | Different state/KV → different cache & latency profile |
| Q3 | int8, int4, or 1.58-bit ternary? | Bandwidth ÷ latency ÷ dequant cost |
| Q4 | Does spec-decoding help on ONE core? | The plan assumed yes for N cores; single-core is different |
| Q5 | Do we need the full CoreFlow triangle for this spec? | If not → strip to minimum, ship sooner |

---

## Q1 — Physics floor (measured, not assumed)

```
Model: 20M params, int8 (1 byte/param)  → 20 MB weights
Weight streaming needed at 50 tok/s      = 20 MB × 50 = 1.0 GB/s
Measured single-core L3 read BW (Q1 docs) = 32–90 GB/s (Zen5, Genoa-X, Rome, SPR)

→ 50 tok/s needs only ~1–3% of L3 BW.   HUGE margin.

Compute: 20M MAC/token = 40 MFLOP
Single-core int8 VNNI ≈ 500–1300 GOPS
→ compute ceiling ~12,000–30,000 tok/s.     Compute is NOT the limit.

Memory: with SSM/conv (no KV growth), fixed state ~O(1). Fits.
```
**Verdict: 50 tok/s is not a physics problem. It is a *kernel-efficiency* problem.**
The number is below the achievable roof (~1000+ tok/s for 20M int8) by 20×.

## Q2 — Architecture: transformer vs SSM vs conv vs hybrid

| Arch | Per-token state | KV grows? | CPU decode char | 2026 evidence |
|---|---|---|---|---|
| Transformer | full KV | **YES** → +latency drift | every past K,V read | EACL'26: drafting overhead kills spec at small scale |
| Mamba SSM | O(1) fixed state | **NO** | ~2.5 ops/byte, memory-bound | BitMamba-2 1B: 27–53 tok/s; Mamba-3 raises arithmetic intensity 4× |
| Conv (gated) | fixed-size | **NO** | cheapest FLOPs/byte | LFM2: 2× faster than Qwen on CPU, conv-major |
| Linear attn (RWKV/RetNet/GDN) | O(1) | **NO** | similar to SSM | Mamba-3 comparison |
| **Hybrid conv + Mamba** | O(1) | **NO** | best of both | LFM2-8B / BitMamba both validate |

**The transformer's KV cache is the enemy of single-core decode.** At context L=512,
d=320, 12 layers: KV read per token = 512 × 2 × 320 × 12 × 1 byte ≈ **3.9 MB/token** —
**20% of the 20 MB weight stream.** It also grows forever. An SSM or conv model has
**fixed memory** and a **flat latency curve** (BitMamba-2: "speed is perfectly constant
regardless of sequence length"). For a *single-core, predictable-latency, cache-resident*
demo, **SSM/conv hybrid beats transformers on every axis that matters to the spec.**

## Q3 — Quantization: int8 vs int4 vs 1.58-bit ternary

| Quant | Bytes/param | Dequant cost on CPU | Kernel needed | Single-core tok/s headroom |
|---|---|---|---|---|
| fp16 | 2.0 | none | FMA | poor (40 MB stream) |
| **int8** | **1.0** | cheap scale-shift | **VNNI `vpdpbusd`** (native) | excellent |
| int4 | 0.5 | lookup/dequant | Q4_0-style LUT | excellent, but kernel brittle |
| **1.58-bit (ternary)** | **0.2** | **none — sign-flip + zero** | `vpmaddubsw` on {-1,0,1} | **best BDY** — 5× less BW than int8 |

**Ternary (BitNet b1.58) is the underrated single-core winner:** weights ∈ {-1,0,1} →
matmul becomes `add/subtract/skip`, no multiplier needed. BitMamba-2 1B runs at
**27–53 tok/s on x86 AVX2**, 255M at **82–113 tok/s**. A 20M ternary model is ~4 MB
weights → **4 MB × 50 = 200 MB/s** — trivial. This is the only path that makes a
20M model fit *L2* (typically 1–2 MB per core is too tight, but 32 MB L3 CCX on Zen5
absorbs it easily).

**Recommended: int8 for v1 (simplest, VNNI-native, well-understood), ternary as the
"unlock" research axis for v2.** Don't gate v1 on ternary's training maturity.

## Q4 — Does speculative decoding help on ONE core? (Nuanced.)

Plan assumed "spec pipeline = n tokens in flight" → wins on N cores. On ONE core:

- Decode 1 token: stream W bytes once, do 2·P FLOPs. Bandwidth-bound ⇒ time = W/BW.
- Verify k drafted tokens in one pass: stream W bytes **once**, do 2·P·k FLOPs.
  If still bandwidth-bound ⇒ **k tokens for the price of 1** → up to k× speedup.

**So spec-decode CAN help single-core** — but only while k·2P ≤ BW·(W/BW), i.e. compute
stays under the bandwidth time. For 20M, W≈20 MB, BW≈50 GB/s → BW time = 0.4 ms.
Compute at 500 GOPS: k=5 → 200 MFLOP → 0.4 ms. **So k≈4–5 is the bandwidth/compute
knee; beyond that, verify becomes compute-bound and gains stop.**

But the EACL'26 evidence is brutal at small scale:
- LM head = 17–33% of draft time (vocab-sized, not model-sized)
- 1B models often *slow down* (0.91×) because draft overhead isn't compensated
- LayerSkip without the recipe: slower than autoregressive

**Single-core spec-decode verdict:** *possible, narrow window (k≈4), only if vocab is
small (≤8K) and D-head is co-trained.* **Not a v1 dependency.** Treat as v2 experiment.

## Q5 — Do we need the full triangle for THIS spec?

User's spec = single core > 50 tok/s. Map the triangle onto it:

| Triangle piece | Needed for single-core >50 tok/s? |
|---|---|
| P1 cache-fit (int8 ≤ ½ L3) | **YES** — keep |
| P2 barrier-free stage dataflow | **NO** — one core = sequential, no barriers |
| P2 N-core elasticity | **NO** — spec is single-core |
| P3 self-spec D-head pipeline | **NO** (v2 maybe, see Q4) |
| Uniform stage cost | NO (no partitioning on 1 core) |
| Trained D-head | NO |
| QAT-lite int8 | **YES** — keep, cheap insurance |
| SSM/conv hybrid (NEW) | **YES** — replaces transformer, kills KV growth |

**→ Strip CoreFlow to its single-core essence. The N-core/barrier-free/spec-pipeline
parts are *CoreFlow-2*, the elastic extension.** Don't let the bigger vision block the
first falsifiable measurement.

---

## The scientific distillation

```
HYPOTHESIS:
  A ~20M-param hybrid (conv + SSM/Mamba-style) SLM, int8, O(1) state, no growing
  KV cache, small vocab (≤8K), VNNI kernel on a pinned single core, will decode
  at >50 tok/s with ≥99% weight reads served from L3 — and likely 300–1000 tok/s.

WHY THIS IS THE SHORTEST PATH:
  1. Physics says 50 tok/s needs 1 GB/s from L3 — 1–3% of available 32–90 GB/s.
  2. SSM/conv kills the KV cache → flat latency, O(1) memory, L3-resident.
  3. int8 VNNI is the boring, proven, fast single-core path (BitMamba-2 1B: 53 tok/s).
  4. Small vocab keeps the LM head — the #1 hidden cost at small scale — cheap.
  5. From-scratch PyTorch + LFM2/Mamba-3 recipes are published and reproducible.

WHAT WOULD FALSIFY IT (Phase 0):
  - DRAM READ bytes/token not ≈ 0  → cache-fit law broken on this hardware
  - tok/s < 50  → kernel or memory-layout problem (fixable, not architectural)
  - attention-KV growth dominates at ctx>512 → confirms the SSM/conv choice

THE GENUINELY FRESH AXIS (v2 unlock, not v1):
  Mamba-3 MIMO on CPU int8 VNNI. MIMO raises decode arithmetic intensity ~4×, so on
  a memory-bound CPU decode kernel it converts idle compute into useful cache reuse.
  No published CPU+MIMO+int8 result exists as of 2026-10-07. Combined with 1.58-bit
  ternary weights (BitNet b1.58), this is the "unlock": a 20M MIMO-Mamba at ternary
  weights ≈ 4 MB on disk, L3-resident, O(1) state, single-core, plausibly >500 tok/s.
```

## The minimal CoreFlow-1 spec (for the user's exact ask)

| Spec | Choice | Rationale |
|---|---|---|
| Architecture | Hybrid: gated short-conv + Mamba-2/3-style SSM blocks, SwiGLU, RMSNorm, RoPE-free (SSM/conv don't need positional enc) | O(1) state, no KV, CPU-friendly, proven (LFM2/Mamba) |
| Size | ~20M params (16–24M) | cache-fit at int8 ≤ ½ L3 |
| Precision | int8 weights (VNNI); fp16/bf16 activations; int8 KV/state | VNNI native, simple dequant |
| State | fixed SSM state + fixed conv window; **no growing KV** | O(1) memory, flat latency |
| Vocab | ≤ 8K | keeps LM head cheap (Q4's lesson) |
| Context design | 512–1024 | flat SSM latency makes this safe |
| Runtime | single Rust/C11 binary, one pinned core, no barriers | trivial for 1 core |
| Training | from-scratch PyTorch, LFM2/Mamba-3 recipe, layer/stage dropout, int8 QAT-lite | reproducible, no HF dependency |
| Bar | **>50 tok/s single core, DRAM READ ≈ 0 B/token** | the falsifiable Phase 0 gate |

## What we are NOT building in v1 (deferred to CoreFlow-2)

- Elastic N-core runtime (barrier-free SPSC, stage ownership)
- Self-spec D-head pipeline
- Uniform-stage-cost training for partition invariance
- Multi-socket / GB-LLC execution

## One-line answer to "can you do that for me?"

Yes. The fastest, most honest path to *>50 tok/s on one core* is a **~20M int8 SSM/conv
hybrid, O(1) state, small vocab, VNNI kernel, pinned core** — not the full elastic
triangle. The physics has 20–100× margin. Phase 0 is a one-afternoon kernel +
measurement. The elastic triangle is the v2 paper; this is the v1 falsification.

---

*Next action options:*
- **A.** Lock CoreFlow-1 spec → I write the Phase 0 benchmark harness (Rust or C, int8 GEMV, pinned core, PCM logging).
- **B.** Add MIMO-Mamba + ternary-weights as the explicit v2 research axis in the plan.
- **C.** Both: patch the plan to v1.1 (strip to single-core essence) + queue CoreFlow-2.
