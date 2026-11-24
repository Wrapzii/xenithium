# Kernel change validation

For changes to QSA, attention, KV cache, prefill, GDN, GEMM/FFN, or speculative verification kernels, use replay-first validation:

1. Select or add a relevant XRV fixture and run XRV correctness before relying on benchmarks.
2. Run repeatable baseline-versus-candidate performance on the same fixture and attach the JSON/Markdown result to the PR.
3. Use stateful replay for changes involving cross-layer state, request scheduling, prefix behavior, or a complete model step. XRV SMR is not implemented yet; use an approved existing flow and report that limitation.
4. Reserve long-context end-to-end benchmarks for final verification or when replay cannot cover the changed behavior.
5. If the fixture is missing or Intel GPU access is unavailable, report the gap explicitly. CPU or synthetic-CPU validation is not GPU validation. A successful XPU replay on synthetic fixtures is GPU-kernel validation only for the tested adapter, device, and toolchain; it does not establish TP2, production-input, full-model, or throughput results.
6. A changed kernel contract or new kernel must add/update fixture coverage, adapter support, and documentation in the same change.

Do not stop/restart the production server, claim production GPUs, load a 27B model, or run long-context prefill for fixture work unless the owner separately approves an isolated test window. Do not upload model weights, prompts, secrets, or sensitive captures to public CI.

Start with [docs/XRV.md](docs/XRV.md). See [docs/XRV-FOLLOW-UP.md](docs/XRV-FOLLOW-UP.md) for uncovered kernels and stateful replay status.
