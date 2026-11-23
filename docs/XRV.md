# Xenithium Replay Validation (XRV)

XRV is the replay-first workflow for kernel changes. Kernel Replay Validation (KRV) now has two direct adapters:

- qsa.select: production QSA top-k selection on captured scores; correctness is exact selected IDs/counts against a stable independent CPU reference.
- qsa.sparse_attention.k8v4: production packed K8/V4 sparse attention plus output merge on captured query, gate, block IDs, pages and packed cache; correctness is compared to an independent float64 reference with explicit BF16 tolerances.

Neither adapter loads the 27B model during replay. The checked-in fixtures are synthetic. The selector and attention adapters have compiled, but no Intel GPU has run them yet; CPU fixture/reference checks are not GPU-kernel validation. Attention coverage is non-tree decode; scoring, long-prefill append, GDN, FFN/GEMM, DSpine and full stateful model replay remain follow-up work.

## Five-minute CPU start

From the repository root:

    scripts/xrv list
    scripts/xrv inspect tests/data/xrv/qsa-select-synthetic
    scripts/xrv validate tests/data/xrv/qsa-select-synthetic
    scripts/xrv replay --fixture tests/data/xrv/qsa-select-synthetic --backend cpu --output /tmp/xrv-select
    scripts/xrv compare --fixture tests/data/xrv/qsa-select-synthetic --result /tmp/xrv-select

This runs the portable selector candidate and checks discrete outputs exactly. The fixture covers ties, stable ordering, padding, chunk/merge, and multi-row bounds.

For K8V4, CPU mode only recomputes the independent float64 reference; it does not execute the attention kernel:

    scripts/xrv validate tests/data/xrv/qsa-k8v4-sparse-synthetic
    scripts/xrv inspect tests/data/xrv/qsa-k8v4-sparse-synthetic
    scripts/xrv replay --fixture tests/data/xrv/qsa-k8v4-sparse-synthetic --backend cpu --output /tmp/xrv-attention-reference

The packed fixture includes a 513-block sparse row, a short all-block row, open causal tails, two slots, a permuted page table, K8/V4 bytes and scales, zero-scale cases, and explicit independent goldens. It establishes fixture/reference integrity only. A captured production kernel must be replayed with --backend xpu for kernel correctness.

## Build and replay production kernels on Intel XPU

    source scripts/oneapi-env.sh
    scripts/build-host.sh xenithium-xrv-qsa-select
    scripts/xrv replay --fixture tests/data/xrv/qsa-select-synthetic --backend xpu --warmup 5 --repeat 30 --output /tmp/xrv-select-xpu

    scripts/build-host.sh xenithium-xrv-qsa-sparse-attention
    scripts/xrv replay --fixture tests/data/xrv/qsa-k8v4-sparse-synthetic --backend xpu --warmup 5 --repeat 30 --output /tmp/xrv-attention-xpu

The attention adapter loads fixture tensors, preserves packed K8/V4 representation in the fixture and device cache, and directly calls the production sparse attention and merge kernels. It does not run score selection, QKV projections, the output projection, or model layers around attention. Device-event span covers attention plus merge; host wall time includes dispatch and wait. It reports median/p90/p99 and samples, hardware/runtime identity, warm-after-warmup cache state, and workload isolation. Cold-cache timing, frequency/power state, actual bytes read from the device, and concurrent workload are not measured; unknowns stay labeled. The adapter currently replays one TP rank per process. Two-rank correctness/latency must be run on both cards and reported on the critical path before claiming TP2 evidence.

A useful correctness command after an edit is:

    scripts/xrv replay --fixture tests/data/xrv/qsa-k8v4-sparse-synthetic --backend xpu --warmup 2 --repeat 10 --output /tmp/xrv-attention-candidate

A repeatable same-input timing uses:

    scripts/xrv bench --fixture tests/data/xrv/qsa-k8v4-sparse-synthetic --backend xpu --warmup 10 --repeat 100 --output /tmp/xrv-attention-bench

Only call a timing comparison credible when baseline and candidate use the same fixture, rank, device/runtime/compiler, cache state and isolation. Use scripts/xrv ab with separately built adapters for same-fixture A/B; it withholds speedup unless both outputs pass the declared output tolerance, hardware/runtime match and workload isolation is explicitly attested. Attention A/B records bitwise mismatch counts and scaled errors so tolerance passes are not presented as bitwise identity. Host lock/process checks belong in a trusted wrapper on an isolated runner. Never use a production GPU for unattended XRV work.

## Capture a real kernel boundary

Capture is explicit and may load weights or execute a model. XRV never starts the supplied command itself. Real captured Q/K/V state is derived from a request and must stay outside git and public artifacts. Output paths inside the repository are rejected.

Example for an approved, isolated capture from an actual model token stream:

    export XRV_FIXTURE_DIR=/mnt/xrv-private
    MODEL=/mnt/models/qwen38
    TOKENS=/mnt/private/corpus.ppl
    MODEL_SHA256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    scripts/xrv capture --kernel qsa.sparse_attention.k8v4 --output "$XRV_FIXTURE_DIR/qsa-attn-private" --fixture-id qsa-attn-private --min-bound 32768 --tp-rank 0 --tp-size 2 --model-fingerprint "$MODEL_SHA256" -- ./build/xenithium-probe model --model "$MODEL" --ppl "$TOKENS" --rows 1 --generate 1 --at 131072 --context 131072 --kv-bytes 1 --kv-v4 1

This is model/GPU-dependent; the full-model probe loads weights and may prefill a long corpus before the first matching decode call. Run it only in an approved isolated window. The XRV hook captures one non-tree K8V4 decode attention call on the selected rank, including Q, gate, logical positions, compact page mapping, packed K8/V4 pages/scales, selected IDs and candidate output. Capture synchronizes and copies tensors, so its timing is invalid. The CPU reference is generated independently from the candidate output. XRV saves a comparison in capture_observation/ and exits nonzero if the captured output fails its declared tolerance. The fixture records a sanitized command template and its digest; no prompt text or model weights are stored, but activations/cache remain sensitive.

Both selector and attention capture manifests distinguish prompt-derived full-model inputs from the synthetic qsa diagnostic probe. Selector capture copies only valid selected IDs and serializes unused padding as -1, rather than copying potentially stale scratch. Attention capture normalizes the documented single-slot cache layout (QsaCache.slot_pages == 0) into an explicit page-table size. The qsa diagnostic subcommand is also capture-capable but uses a random K/V history and synthetic chain activations, so XRV labels that origin as diagnostic rather than prompt-derived. Use the model subcommand with a caller-supplied token corpus when you need prompt-derived attention inputs.

After capture:

    scripts/xrv inspect "$XRV_FIXTURE_DIR/qsa-attn-private"
    scripts/xrv validate "$XRV_FIXTURE_DIR/qsa-attn-private"
    scripts/xrv replay --fixture "$XRV_FIXTURE_DIR/qsa-attn-private" --backend xpu --warmup 2 --repeat 10 --output /tmp/xrv-attn-captured-replay

Capture hook code lives in the probe and is not registered by the serving process. Captured files are tied to the exact model/runtime/kernel state described in the manifest. Do not put private captures in CI, PRs, logs, or a shared artifact bucket without an explicit secure distribution policy.

## Refresh synthetic fixtures

Generators refuse to overwrite existing fixture directories. Generate into a new temporary location, inspect the binary and manifest changes, run validation, then replace the checked-in fixture in a reviewed change:

    python3 tools/xrv/make_synthetic.py /tmp/xrv-qsa-select-new
    python3 tools/xrv/make_qsa_attention.py /tmp/xrv-qsa-attention-new
    scripts/xrv validate /tmp/xrv-qsa-attention-new

Never regenerate automatically after a correctness failure. Intentional golden changes require fixture-version review and an explanation.

## Stateful/full-model escalation

Kernel replay removes repeated model initialization and prefix fill for a captured boundary only. It does not restore cross-layer state or model scheduling. Stateful Model Replay (SMR) is not implemented; tracked scope is in docs/XRV-FOLLOW-UP.md. Changes affecting request scheduling, prefix behavior, recurrent state, speculative acceptance or a complete verifier step still need a separately approved stateful or full-model run. Do not call kernel-only speedups end-to-end tok/s gains.

Tier 1 also compiles the probe and both XPU adapters in an Intel oneAPI container without running them; this is a compile check only. See docs/XRV-CI-SETUP.md for runner setup and security. Current test duration, storage and validation limits are in docs/XRV-REPORT.md.
