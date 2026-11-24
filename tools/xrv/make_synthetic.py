#!/usr/bin/env python3
"""Regenerate the deterministic, weight-free QSA selector smoke fixtures."""
from __future__ import annotations
import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import xrv


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def write_fixture(fixture_id: str, rows: int, stride: int, bound: int, chunk: int, nblk: list[int], score_fn, purpose: str) -> Path:
    out = ROOT / "tests/data/xrv" / fixture_id
    out.mkdir(parents=True, exist_ok=True)
    scores = [0] * (rows * stride)
    for r, n in enumerate(nblk):
        for i in range(n):
            scores[r * stride + i] = bits(score_fn(r, i))
    ids, counts = xrv.reference_select(scores, nblk, rows, stride, 512)
    payloads = {
        "scores.u32le": xrv.pack_values("uint32", scores),
        "nblk.i32le": xrv.pack_values("int32", nblk),
        "expected_ids.i32le": xrv.pack_values("int32", ids),
        "expected_counts.i32le": xrv.pack_values("int32", counts),
    }
    for name, data in payloads.items():
        (out / name).write_bytes(data)

    def tensor(name: str, dtype: str, shape: list[int], strides: list[int], role: str) -> dict:
        data = payloads[name]
        return {"file": name, "dtype": dtype, "shape": shape, "strides": strides, "byte_order": "little",
                "role": role, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    manifest = {
        "schema": xrv.SCHEMA, "schema_version": xrv.SCHEMA_VERSION, "fixture_id": fixture_id,
        "kernel": {"id": xrv.KERNEL, "version": 1, "implementation": "xenithium::kernels::qsa_select"},
        "model_fingerprint": {"kind": "synthetic", "sha256": None, "weights_included": False},
        "runtime_fingerprint": {"kind": "synthetic-fixture", "engine_revision": None, "driver": None, "compiler": None, "device": None},
        "tp": {"rank": 0, "size": 1},
        "operation": {"rows": rows, "stride": stride, "source_stride": stride, "bound": bound, "top_k": 512, "chunk": chunk,
                       "score_encoding": "nonnegative_float32_bits_as_uint32"},
        "quantization": {"score_encoding": "positive-float32-order-preserving-bits", "kv": "not-present-at-selector-boundary"},
        "position": {"kind": "selector-only", "start": None, "causal_mask": "already-applied-before-selector"},
        "masks": [], "mutable_state": [],
        "tensors": {
            "scores": tensor("scores.u32le", "uint32", [rows, stride], [stride, 1], "input"),
            "nblk": tensor("nblk.i32le", "int32", [rows], [1], "input"),
            "expected_ids": tensor("expected_ids.i32le", "int32", [rows, 512], [512, 1], "independent_reference"),
            "expected_counts": tensor("expected_counts.i32le", "int32", [rows], [1], "independent_reference"),
        },
        "reference": {"kind": "independent-host-reference", "independent": True,
                      "source": "xrv.reference_select: full stable ordering by descending uint32 score then ascending block ID; returned IDs ascending",
                      "candidate_output_is_not_golden": True},
        "provenance": {"kind": "synthetic", "generator": "tools/xrv/make_synthetic.py", "seed": "deterministic formula; no random state",
                        "contains_model_weights": False, "contains_prompt_data": False, "purpose": purpose},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    xrv.load_fixture(out)
    print(out)
    return out


write_fixture("qsa-select-synthetic", 4, 16384, 8192, 4096, [1, 512, 513, 8192],
              lambda row, i: 1.0 if row == 0 else float(5 if row == 2 else ((i * (7 if row == 1 else 17)) % (13 if row == 1 else 1024))),
              "selector chunk/merge, tie cutoff, and mixed candidate counts")

write_fixture("qsa-select-max-bound", 2, 262144, 262144, 16384, [0, 262144],
              lambda row, i: float((i * 37) % 65536),
              "zero-candidate row, maximum supported bound, and deterministic ties")
