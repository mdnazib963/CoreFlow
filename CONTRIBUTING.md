# Contributing to CoreFlow

Thanks for your interest in improving CoreFlow.

## Ground rules

- **Measurements over opinions.** Every performance claim in this repo is backed by a JSON record in `phase0/`. If you change a kernel or claim a speedup, include the benchmark output (`phase0.exe ... bench`, `stage_rt.exe ... bench`) before/after, with your CPU model.
- **Parity is a gate.** C runtime changes must keep perplexity delta vs PyTorch ≤ 1% and token match ≥ 99% (see `phase0/gates_p5.json` for the reference). Run `phase0_gates.py` before opening a PR that touches `cf_model.c`.
- **No code comments** in source files; explain reasoning in PR descriptions and commit messages.
- **Determinism + ASan matter.** If you touch memory handling, run the ASan build (`build_asan.bat`).

## Setting up

```bat
phase0\build_p0.bat
phase0\phase0.exe phase0\model_p3.i8 bench data\val.bin 15.0 65
```

Python side needs only `torch` (CPU is fine for eval; GPU for training).

## Good first issues

- **Reproduce the headline benchmark on your hardware** and post your `bench` JSON in an issue (different CPU generations are especially useful — the cache-fit claim should hold on any chip where the working set fits in L3).
- **Docs gaps**: anything in `README.md` you had to figure out yourself belongs in the README.
- **Portability**: Makefile/CMake for Linux/macOS builds of `phase0/` (currently MSVC `.bat` only).
- **Benchmark scripts**: shell wrappers for the multi-stream suite on non-Windows platforms.

## PR checklist

1. `phase0\build_p0.bat` compiles clean (no new warnings ideally).
2. Gates pass: parity ≤ 1%, determinism, ASan clean (if memory touched).
3. Include benchmark JSON evidence for any performance-related change.
4. Update `phase0/` result records if your change alters published numbers.
