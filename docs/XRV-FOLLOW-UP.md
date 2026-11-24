# XRV staged follow-up work

XRV KRV currently covers production QSA selection and K8V4 sparse attention plus merge. Two synthetic-fixture XPU replays passed on a single Intel device with oneAPI 2025.3.2: selector IDs matched exactly (0/2,048 mismatches), and attention had 0 declared-tolerance violations (17/6,144 BF16 bitwise differences). XPU correctness has therefore been exercised for these fixture-level paths. This does not prove production-capture correctness, TP2 behavior, complete model behavior, prefill gains, concurrency, or end-to-end speed. XPU performance and the oneDNN A/B have not been measured; oneAPI 2026.2 GPU qualification remains pending.

## Next KRV adapters

1. **Scoring and prefill append:** capture pooled keys, queries, score outputs, prefix state and positions. Compare 512/1024/2048/4096 appends against the same frozen prefix. Time score writes, top-k reads/selection, attention and complete chunk separately. Cover the dense and sparse prefill routes actually used by Xenithium.
2. **Split/tree attention:** current adapter covers non-tree decode and directly measures production attention plus merge. Add split partials, tree masks and acceptance-state interactions where those are part of the changed kernel contract.
3. **GDN recurrent and convolution state:** compare outputs and state after a step and after repeatable restore.
4. **W4A16 / FFN GEMM and DSpine verifier:** preserve packed-weight metadata, input rows, speculative tree words, accepted tokens, output IDs and mutated cache. Report milliseconds per committed token.
5. Every adapter should implement tools/xrv/adapter-contract.md and add deterministic CPU/reference fixtures before private real captures.

## Stateful Model Replay follow-up

SMR is a separate, not-yet-implemented mode. It must checkpoint every mutated state owner: K8V4 page-pool bytes and ownership, per-sequence page tables, QSA pooled keys/indexer sidecars, GDN convolution/recurrent state, sequence length and RoPE positions, request metadata, speculative tree/acceptance state, and scratch affecting repeatability. Snapshot identity includes model/runtime/compiler/driver/kernel revision and TP topology.

A safe first SMR change should restore and verify one isolated single-user operation, use dirty-range or copy-on-write restore only after proving equivalence, and demonstrate deterministic outputs plus state equality before expanding to C4. Persistent fixtures are not resident checkpoints; process rebuild invalidates in-memory snapshots. Do not claim SMR until restore is independently compared to a known-good baseline.

The gated full-model workflow remains disabled until this adapter exists. Until then, use approved existing probes and report their cost and limits separately.
