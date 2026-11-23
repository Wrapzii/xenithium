# XRV adapter contract

An adapter replays one fixture operation directly against the implementation under test. It must not load a model or perform hidden expensive initialization unless the fixture explicitly declares stateful/full-model mode.

## qsa.select

xenithium-xrv-qsa-select --scores FILE --nblk FILE --rows N --stride N --bound N --chunk N --output DIR --warmup N --repeat N --device N --tp-rank N --tp-size N

Inputs are little-endian tensors from the manifest. Outputs are selected_ids.i32le and selected_counts.i32le; selection identity and padding are exact. The adapter reports event and host timing plus hardware/runtime metadata.

## qsa.sparse_attention.k8v4

xenithium-xrv-qsa-sparse-attention consumes --query, --gate, --positions, --slots, --page-table, packed --keys/--values, --key-scales, --value-scales, --selected-ids, --selected-counts, dimensions, TP/card identity, device, timing options and output directory. tools/xrv/qsa_attention.py builds the command from the fixture manifest.

Inputs remain packed as K8/V4 bytes in fixture files and device cache. The production kernel handles dequantization during attention. The adapter directly launches production sparse attention and merge using fixture-provided projected query/gate and selected IDs. It does not execute query projection, score generation/top-k, output projection, communication, prefill planning, tree decode, or state restore. It currently executes one TP rank per process; report each rank and use the critical rank for TP timing.

The C++ executable writes output.u16le and a JSON timing record on stdout. Device event span includes attention and merge; host wall includes launch and synchronization. H2D fixture setup and D2H final output transfer are excluded from event span. The Python runner writes result.json, binds it to the fixture manifest SHA-256, and compares output to the independent float64/BF16 reference. Numeric tolerances are in the fixture manifest; NaN/Inf fail.

## Requirements for future adapters

Adapters must verify dimensions, dtype/endianness, bounds, finite values, hardware target and allocation sizes; use fixed fixture inputs and separate warmups; report host and GPU timing distinctly with median/p90/p99 and raw samples; record compiler/runtime/driver/device and isolation; preserve mutated state when applicable; and fail on invalid execution. Same-fixture A/B must use identical execution settings and exact fixture fingerprints.

Discrete outputs (IDs, mappings, masks, token IDs and accepted tokens) require exact identity. Numeric outputs declare explicit abs/rel/scaled/ULP criteria and report max errors, mismatch count, first mismatch and samples. Never auto-rebaseline.
