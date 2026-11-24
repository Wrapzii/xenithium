# XRV implementation report

## Implemented

- Versioned fixture contracts: selector schema v1 and attention schema v2, little-endian arrays, SHA-256, dimensions/strides, quantization, causal/position and TP metadata, provenance, and independent-reference identity.
- Discoverable CLI for list, inspect, validate, capture, replay, compare, bench and same-fixture A/B.
- Production QSA selector adapter and production packed K8V4 sparse-attention plus merge adapter.
- Opt-in probe capture hooks for selector tensors and non-tree K8V4 sparse-attention boundaries. Real attention captures preserve packed cache tensors and save candidate output separately from the independent reference.
- Synthetic selector and K8V4 attention fixtures. The attention fixture covers a 513-block row with top-512 IDs, a short all-block row, partial open blocks, multiple slots, permuted page mappings, packed K8/V4, zero scales and an independent float64/BF16 reference.
- CPU tests include exact selector replay, fixture integrity/reference checks, mock K8V4 capture conversion, numeric mismatch diagnostics, A/B gating, path/checksum/scale failures and explicit proof that CPU reference mode is not a kernel benchmark.
- PR agent instructions, a validation template, CPU CI, manually gated B60 workflow, owner runner setup and a separate disabled full-model workflow.

## What the old loop can skip

The assignment's approximately 15-minute setup/run figure is the reported old-loop baseline; it was not independently timed during this implementation. After one approved capture, repeated adapter build/replay/compare/bench on that fixed tensor boundary no longer needs model weight loading or a long prefix prefill. The synthetic fixture path needs neither capture nor weights. Capture itself remains a model/GPU operation and synchronizes/copies tensors; capture timing must not be used.

The final 28-test CPU suite duration is recorded after the full final run below. This is a local observed duration, not a guarantee for other machines. The three checked-in fixtures occupy 4,257,871 bytes total; the K8V4 attention fixture is 1,754,280 bytes. The attention adapter allocates about 2.5 MiB of device buffers for that fixture from the declared tensor and partial-buffer dimensions, excluding runtime/library overhead. Real fixture storage depends on selected cache pages and remains outside git. The capture path has a configurable byte limit and preserves hashes for all fixture tensors.

## Correctness and performance status

The independent Python attention reference dequantizes packed K8/V4 and computes float64 softmax, with explicit BF16 rounding for the production gate path. Candidate output comparison reports mismatch count, first mismatch, samples, max absolute/relative/scaled error and tolerance violations. Discrete selector IDs/counts remain exact checks. Captured candidate output is never used as the golden.

The selector XPU adapter, attention XPU adapter, probe capture-hook translation unit, and full xenithium-probe target built successfully on IntelLLVM 2025.3.2. The two authorized synthetic XPU replays then completed successfully on a single Intel device using oneAPI 2025.3.2:

| Adapter | Comparison | Result |
| --- | --- | --- |
| QSA selector | Exact selected-ID comparison | Pass; 0 mismatches across 2,048 IDs |
| K8V4 sparse attention + merge | Declared numeric tolerance | Pass; 0 tolerance violations; 17/6,144 BF16 values differed bitwise |

These results establish execution and fixture-level correctness for the tested single-device adapter paths. They do not establish TP2 correctness or critical-path latency, correctness on production captures, full-model behavior, or end-to-end throughput. They also do not qualify the oneAPI 2026.2 CI toolchain. No performance claim was measured, and the oneDNN A/B was not run. The production restore and service checks were reported separately from these XRV replays. Replay logs remain in the authorized ai01 run directory and are not included in the repository. The capture hook adds an atomic callback-pointer check on the QSA decode host path when enabled; its disabled-path overhead has not been measured.


Tier 1 CPU CI runs Python/schemas/checksums/references and both synthetic replay flows. Tier 2 B60 remains manual and gated until an isolated runner is enrolled and explicitly authorized. No GPU runner has been enrolled by this implementation.

## Remaining work

Long-context score generation and prefill append, tree/split attention state, GDN, W4A16/FFN, DSpine verifier replay and stateful model replay remain explicit follow-ups in docs/XRV-FOLLOW-UP.md. Use full-model benchmarks as a deliberate final validation when a kernel replay cannot cover the changed behavior.
