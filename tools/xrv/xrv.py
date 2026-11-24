#!/usr/bin/env python3
"""Xenithium Replay Validation (XRV), portable fixture and replay frontend."""
from __future__ import annotations

import argparse
import array
import hashlib
import heapq
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

SCHEMA = "org.xenithium.xrv.fixture"
SCHEMA_VERSION = 1
KERNEL = "qsa.select"
ATTENTION_KERNEL = "qsa.sparse_attention.k8v4"
TOP_K = 512
MAX_BLOCKS = 262_144
MAX_FIXTURE_BYTES = 1 << 30
DTYPES = {"uint32": ("I", 4), "int32": ("i", 4), "float32": ("f", 4), "uint16": ("H", 2), "int8": ("b", 1), "uint8": ("B", 1)}

class XrvError(Exception):
    pass

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def pack_values(dtype: str, values: Any) -> bytes:
    code, _ = DTYPES[dtype]
    result = array.array(code, values)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()

def unpack_values(dtype: str, data: bytes) -> array.array:
    code, width = DTYPES[dtype]
    if len(data) % width:
        raise XrvError(f"{dtype} tensor byte count {len(data)} is not divisible by {width}")
    result = array.array(code)
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result

def tensor_info(path: Path, dtype: str, shape: list[int], strides: list[int], role: str) -> dict[str, Any]:
    return {"file": path.name, "dtype": dtype, "shape": shape, "strides": strides, "byte_order": "little", "role": role,
            "size_bytes": path.stat().st_size, "sha256": sha256(path)}

def safe_tensor_path(root: Path, desc: dict[str, Any]) -> Path:
    if not isinstance(desc, dict) or not isinstance(desc.get("file"), str):
        raise XrvError("tensor descriptor must contain a file name")
    rel = Path(desc["file"])
    if rel.is_absolute() or ".." in rel.parts:
        raise XrvError(f"unsafe tensor path: {desc['file']}")
    path = (root / rel).resolve(strict=True)
    if root.resolve() not in path.parents:
        raise XrvError(f"tensor escapes fixture directory: {desc['file']}")
    return path

def _num(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise XrvError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise XrvError(f"{name} must be >= {minimum}")
    return value

def load_fixture(path: str | Path, check_data: bool = True) -> tuple[Path, dict[str, Any]]:
    supplied = Path(path).expanduser()
    root = supplied if supplied.is_dir() else supplied.parent
    manifest_path = root / "manifest.json" if supplied.is_dir() else supplied
    if not manifest_path.is_file():
        raise XrvError(f"fixture manifest not found: {manifest_path}")
    root = manifest_path.parent.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise XrvError(f"cannot read fixture manifest: {e}") from e
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or type(manifest.get("schema_version")) is not int:
        raise XrvError(f"unsupported fixture schema; expected {SCHEMA}")
    if manifest.get("schema_version") == 2 and isinstance(manifest.get("kernel"), dict) and manifest["kernel"].get("id") == "qsa.sparse_attention.k8v4":
        try:
            import qsa_attention
            qsa_attention.validate_fixture(root, manifest, check_data=check_data)
        except qsa_attention.FixtureError as e:
            raise XrvError(str(e)) from e
        return root, manifest
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise XrvError(f"unsupported fixture schema version {manifest.get('schema_version')}")
    if not isinstance(manifest.get("fixture_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", manifest["fixture_id"]):
        raise XrvError("fixture_id must be a 1..128 character safe identifier using letters, digits, dot, underscore or hyphen")
    kernel = manifest.get("kernel")
    if not isinstance(kernel, dict) or kernel.get("id") != KERNEL:
        raise XrvError(f"this XRV build supports {KERNEL}; fixture kernel is {kernel}")
    _num(kernel.get("version"), "kernel.version", 1)
    op = manifest.get("operation")
    if not isinstance(op, dict):
        raise XrvError("operation object is required")
    rows = _num(op.get("rows"), "operation.rows", 1)
    stride = _num(op.get("stride"), "operation.stride", 1)
    bound = _num(op.get("bound"), "operation.bound", 1)
    top_k = _num(op.get("top_k"), "operation.top_k", 1)
    chunk = _num(op.get("chunk", 0), "operation.chunk", 0)
    if rows > 64 or stride > MAX_BLOCKS or bound > MAX_BLOCKS or bound > stride or top_k != TOP_K:
        raise XrvError("qsa.select supports rows<=64, stride/bound<=262144, bound<=stride, and top_k=512")
    if chunk and (chunk < 4096 or chunk > 32768):
        raise XrvError("operation.chunk must be 0 (engine default) or in [4096,32768]")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        raise XrvError("tensors object is required")
    required = {"scores": ("uint32", [rows, stride]), "nblk": ("int32", [rows]),
                "expected_ids": ("int32", [rows, TOP_K]), "expected_counts": ("int32", [rows])}
    for name, (dtype, shape) in required.items():
        desc = tensors.get(name)
        if not isinstance(desc, dict) or desc.get("dtype") != dtype or desc.get("shape") != shape or desc.get("byte_order") != "little":
            raise XrvError(f"tensor {name} must have dtype={dtype}, shape={shape}, byte_order=little")
    total_bytes = 0
    for name, desc in tensors.items():
        if not isinstance(name, str) or not isinstance(desc, dict):
            raise XrvError("every tensors entry must be a named descriptor object")
        dtype, shape, strides = desc.get("dtype"), desc.get("shape"), desc.get("strides")
        if not isinstance(dtype, str) or dtype not in DTYPES or desc.get("byte_order") != "little":
            raise XrvError(f"tensor {name} has unsupported dtype or byte_order")
        if not isinstance(shape, list) or not shape or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in shape):
            raise XrvError(f"tensor {name}.shape must be a list of positive dimensions")
        if not isinstance(strides, list) or len(strides) != len(shape) or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in strides):
            raise XrvError(f"tensor {name}.strides must contain one positive stride per dimension")
        expected_bytes = math.prod(shape) * DTYPES[dtype][1]
        declared_bytes = _num(desc.get("size_bytes"), f"tensor {name}.size_bytes", 0)
        total_bytes += expected_bytes
        if total_bytes > MAX_FIXTURE_BYTES:
            raise XrvError(f"fixture exceeds the {MAX_FIXTURE_BYTES}-byte validation limit")
        if declared_bytes != expected_bytes:
            raise XrvError(f"tensor {name} declared byte size {declared_bytes}, expected {expected_bytes}")
        if check_data:
            file = safe_tensor_path(root, desc)
            if file.stat().st_size != expected_bytes:
                raise XrvError(f"tensor {name} byte size mismatch: expected {expected_bytes}, found {file.stat().st_size}")
            digest = sha256(file)
            if desc.get("sha256") != digest:
                raise XrvError(f"tensor {name} SHA-256 mismatch: expected {desc.get('sha256')}, found {digest}")
    for field in ("model_fingerprint", "runtime_fingerprint", "tp", "quantization", "position", "masks", "mutable_state", "reference", "provenance"):
        if field not in manifest:
            raise XrvError(f"manifest is missing required field {field}")
    for field in ("model_fingerprint", "runtime_fingerprint", "tp", "quantization", "position", "reference", "provenance"):
        if not isinstance(manifest[field], dict):
            raise XrvError(f"manifest.{field} must be an object")
    for field in ("masks", "mutable_state"):
        if not isinstance(manifest[field], list):
            raise XrvError(f"manifest.{field} must be an array")
    tp = manifest["tp"]
    rank = _num(tp.get("rank"), "tp.rank", 0)
    size = _num(tp.get("size"), "tp.size", 1)
    if rank >= size:
        raise XrvError("tp.rank must be less than tp.size")
    if check_data:
        scores = unpack_values("uint32", safe_tensor_path(root, tensors["scores"]).read_bytes())
        nblk = unpack_values("int32", safe_tensor_path(root, tensors["nblk"]).read_bytes())
        for r, n in enumerate(nblk):
            if n < 0 or n > bound:
                raise XrvError(f"nblk[{r}]={n} is outside [0,bound={bound}]")
            off = r * stride
            for i, bits in enumerate(scores[off:off+n]):
                if bits & 0x80000000:
                    raise XrvError(f"scores[{r},{i}] is negative; qsa.select requires non-negative score bits")
                if ((bits >> 23) & 0xff) == 0xff:
                    raise XrvError(f"scores[{r},{i}] is NaN or infinity")
    ref = manifest["reference"]
    if ref.get("independent") is not True or not isinstance(ref.get("source"), str) or not ref["source"].strip():
        raise XrvError("reference must name an independent source; candidate output cannot silently become its own golden")
    return root, manifest

def verify_golden(root: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("kernel", {}).get("id") == "qsa.sparse_attention.k8v4":
        try:
            import qsa_attention
            qsa_attention.verify_golden(root, manifest)
            return
        except qsa_attention.FixtureError as e:
            raise XrvError(str(e)) from e
    op, ts = manifest["operation"], manifest["tensors"]
    scores = unpack_values("uint32", safe_tensor_path(root, ts["scores"]).read_bytes())
    nblk = unpack_values("int32", safe_tensor_path(root, ts["nblk"]).read_bytes())
    expected = reference_select(scores, nblk, op["rows"], op["stride"], op["top_k"])
    golden_ids = list(unpack_values("int32", safe_tensor_path(root, ts["expected_ids"]).read_bytes()))
    golden_counts = list(unpack_values("int32", safe_tensor_path(root, ts["expected_counts"]).read_bytes()))
    if expected[0] != golden_ids or expected[1] != golden_counts:
        first = next((i for i, (a, b) in enumerate(zip(expected[0], golden_ids)) if a != b), None)
        raise XrvError(f"independent golden does not match host reference; first selected_ids flat index={first}; intentional golden refresh requires review")

def reference_select(scores: list[int | float], nblk: list[int], rows: int, stride: int, top_k: int) -> tuple[list[int], list[int]]:
    ids = [-1] * (rows * top_k)
    counts = []
    for r in range(rows):
        n = nblk[r]
        start = r * stride
        order = sorted(range(n), key=lambda i: (-int(scores[start + i]), i))[:top_k]
        order.sort()
        counts.append(len(order))
        ids[r * top_k:r * top_k + len(order)] = order
    return ids, counts

def candidate_cpu(scores: list[int | float], nblk: list[int], rows: int, stride: int, top_k: int) -> tuple[list[int], list[int]]:
    ids = [-1] * (rows * top_k)
    counts = []
    for r in range(rows):
        n = nblk[r]
        start = r * stride
        picked = heapq.nlargest(min(n, top_k), range(n), key=lambda i: (int(scores[start + i]), -i))
        picked.sort()
        counts.append(len(picked))
        ids[r * top_k:r * top_k + len(picked)] = picked
    return ids, counts

def result_dir(path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()

def output_sha256(out: Path) -> dict[str, str]:
    return {"selected_ids": sha256(out / "selected_ids.i32le"),
            "selected_counts": sha256(out / "selected_counts.i32le")}

def write_result(out: Path, fixture_root: Path, manifest: dict[str, Any], backend: str, ids: list[int], counts: list[int], timing: dict[str, Any] | None = None) -> None:
    (out / "selected_ids.i32le").write_bytes(pack_values("int32", ids))
    (out / "selected_counts.i32le").write_bytes(pack_values("int32", counts))
    op = manifest["operation"]
    obj = {"xrv_result_version": 1, "fixture_id": manifest["fixture_id"], "fixture_manifest_sha256": sha256(fixture_root / "manifest.json"),
           "kernel_id": KERNEL, "backend": backend, "mode": "kernel-only", "status": "completed", "tp": manifest["tp"],
           "output_files": {"selected_ids": "selected_ids.i32le", "selected_counts": "selected_counts.i32le"},
           "output_sha256": output_sha256(out),
           "bytes": {"scores_read": op["rows"] * op["stride"] * 4, "nblk_read": op["rows"] * 4,
                     "selected_ids_written": op["rows"] * op["top_k"] * 4, "selected_counts_written": op["rows"] * 4},
           "timing": timing or {}, "host": {"platform": platform.platform(), "python": platform.python_version()},
           "created_utc": datetime.now(timezone.utc).isoformat()}
    (out / "result.json").write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def qsa_adapter() -> Path:
    env = os.environ.get("XRV_XPU_ADAPTER")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "build" / "xenithium-xrv-qsa-select"

def run_xpu(fixture_root: Path, manifest: dict[str, Any], out: Path, warmup: int = 0, repeat: int = 1, adapter: Path | None = None, device: int = 0) -> dict[str, Any]:
    exe = (adapter or qsa_adapter()).resolve()
    if not exe.is_file():
        raise XrvError(f"XPU adapter not found: {exe}; build Xenithium with the documented oneAPI/XPU toolchain or set XRV_XPU_ADAPTER")
    op, ts = manifest["operation"], manifest["tensors"]
    cmd = [str(exe), "--scores", str(safe_tensor_path(fixture_root, ts["scores"])), "--nblk", str(safe_tensor_path(fixture_root, ts["nblk"])),
           "--rows", str(op["rows"]), "--stride", str(op["stride"]), "--bound", str(op["bound"]), "--chunk", str(op.get("chunk", 0)),
           "--tp-rank", str(manifest["tp"]["rank"]), "--tp-size", str(manifest["tp"]["size"]), "--device", str(device),
           "--output", str(out), "--warmup", str(warmup), "--repeat", str(repeat)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode:
        raise XrvError(f"XPU adapter exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    try:
        meta = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise XrvError(f"XPU adapter did not return a JSON result: {proc.stdout}\n{proc.stderr}") from e
    if not (out / "selected_ids.i32le").is_file() or not (out / "selected_counts.i32le").is_file():
        raise XrvError("XPU adapter completed without output tensors")
    result = {"xrv_result_version": 1, "fixture_id": manifest["fixture_id"], "fixture_manifest_sha256": sha256(fixture_root / "manifest.json"),
              "kernel_id": KERNEL, "backend": "xpu", "mode": "kernel-only", "status": "completed", "output_files": {
                  "selected_ids": "selected_ids.i32le", "selected_counts": "selected_counts.i32le"},
              "output_sha256": output_sha256(out), "timing": meta.get("timing", {}),
              "hardware": meta.get("hardware", {}), "runtime": meta.get("runtime", {}),
              "workload_isolation": meta.get("workload_isolation", {"status": "unknown", "evidence": "not-reported"}),
              "tp": meta.get("tp", manifest["tp"]), "execution": {"warmup": warmup, "repeat": repeat, "device": device,
              "chunk": op.get("chunk", 0), "rows": op["rows"], "bound": op["bound"],
              "cache_state": "warm-device-resident inputs; host transfers excluded from device event span"},
              "bytes": {"scores_read": op["rows"] * op["stride"] * 4, "nblk_read": op["rows"] * 4,
                        "selected_ids_written": op["rows"] * op["top_k"] * 4, "selected_counts_written": op["rows"] * 4},
              "created_utc": datetime.now(timezone.utc).isoformat()}
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

def compare_result(fixture_root: Path, manifest: dict[str, Any], out: Path, write: bool = True) -> dict[str, Any]:
    result_path = out / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise XrvError(f"cannot read result envelope under {out}: {e}") from e
    if not isinstance(result, dict):
        raise XrvError("result.json must contain an object")
    expected_envelope = {"xrv_result_version": 1, "fixture_id": manifest["fixture_id"],
        "fixture_manifest_sha256": sha256(fixture_root / "manifest.json"), "kernel_id": KERNEL,
        "mode": "kernel-only", "status": "completed", "tp": manifest["tp"]}
    for key, expected in expected_envelope.items():
        if result.get(key) != expected:
            raise XrvError(f"result envelope {key} mismatch: expected {expected!r}, found {result.get(key)!r}")
    expected_files = {"selected_ids": "selected_ids.i32le", "selected_counts": "selected_counts.i32le"}
    if result.get("output_files") != expected_files:
        raise XrvError(f"result output_files must be exactly {expected_files}")
    checksums = result.get("output_sha256")
    if not isinstance(checksums, dict) or set(checksums) != set(expected_files):
        raise XrvError("result output_sha256 must contain selected_ids and selected_counts")
    for name, filename in expected_files.items():
        path = out / filename
        if not path.is_file():
            raise XrvError(f"result output missing: {filename}")
        if checksums.get(name) != sha256(path):
            raise XrvError(f"result output SHA-256 mismatch for {name}")
    ts, op = manifest["tensors"], manifest["operation"]
    exp_ids = unpack_values("int32", safe_tensor_path(fixture_root, ts["expected_ids"]).read_bytes())
    exp_counts = unpack_values("int32", safe_tensor_path(fixture_root, ts["expected_counts"]).read_bytes())
    got_ids_path, got_counts_path = out / "selected_ids.i32le", out / "selected_counts.i32le"
    if not got_ids_path.is_file() or not got_counts_path.is_file():
        raise XrvError(f"result outputs missing under {out}")
    got_ids, got_counts = unpack_values("int32", got_ids_path.read_bytes()), unpack_values("int32", got_counts_path.read_bytes())
    rows, k = op["rows"], op["top_k"]
    if len(got_ids) != rows * k or len(got_counts) != rows:
        raise XrvError(f"result shape mismatch: IDs={len(got_ids)} counts={len(got_counts)}, expected {rows*k} and {rows}")
    mismatch_count = 0
    samples: list[dict[str, Any]] = []
    first_mismatch = None
    def record(mismatch: dict[str, Any]) -> None:
        nonlocal mismatch_count, first_mismatch
        mismatch_count += 1
        if first_mismatch is None:
            first_mismatch = mismatch
        if len(samples) < 16:
            samples.append(mismatch)
    for r in range(rows):
        if got_counts[r] != exp_counts[r]:
            record({"tensor": "selected_counts", "row": r, "index": r, "expected": exp_counts[r], "actual": got_counts[r]})
        for j in range(k):
            expected, actual = exp_ids[r*k+j], got_ids[r*k+j]
            if expected != actual:
                record({"tensor": "selected_ids", "row": r, "index": j, "expected": expected, "actual": actual})
    report = {"fixture_id": manifest["fixture_id"], "kernel_id": KERNEL, "status": "pass" if mismatch_count == 0 else "fail",
              "comparison": "exact discrete identity", "mismatch_count": mismatch_count, "first_mismatch": first_mismatch,
              "sample_mismatches": samples, "rows": rows, "selected_ids_compared": rows * k,
              "expected_source": manifest["reference"].get("source"), "candidate_backend": result.get("backend")}
    if write:
        (out / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "comparison.md").write_text("# XRV comparison\n\n" + f"- Fixture: `{manifest['fixture_id']}`\n- Kernel: `{KERNEL}`\n- Status: **{report['status'].upper()}**\n- Gate: exact selected ID/count identity\n- Compared IDs: {report['selected_ids_compared']}\n- Mismatches: {report['mismatch_count']}\n" +
            (f"- First mismatch: row {report['first_mismatch']['row']} `{report['first_mismatch']['tensor']}` index {report['first_mismatch']['index']}: expected {report['first_mismatch']['expected']}, got {report['first_mismatch']['actual']}\n" if first_mismatch else ""), encoding="utf-8")
    return report

def percentile(values: list[float], p: float) -> float:
    v = sorted(values)
    return v[max(0, math.ceil(p * len(v)) - 1)]

def cmd_list(args: argparse.Namespace) -> int:
    roots = [Path(__file__).resolve().parents[2] / "tests/data/xrv"]
    if os.environ.get("XRV_FIXTURE_DIR"):
        roots.extend(Path(x).expanduser() for x in os.environ["XRV_FIXTURE_DIR"].split(os.pathsep) if x)
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for m in sorted(root.rglob("manifest.json")):
            if m in seen:
                continue
            seen.add(m)
            try:
                obj = json.loads(m.read_text(encoding="utf-8"))
                print(f"{obj.get('fixture_id','?')}\t{obj.get('kernel',{}).get('id','?')}\t{m.parent}")
            except Exception as e:
                print(f"INVALID\t{m}\t{e}")
    return 0

def cmd_inspect(args: argparse.Namespace) -> int:
    root, m = load_fixture(args.fixture, check_data=not args.no_hash)
    if args.json:
        print(json.dumps(m, indent=2, sort_keys=True))
    else:
        op = m["operation"]
        if m.get("kernel", {}).get("id") == "qsa.sparse_attention.k8v4":
            summary = f"rows={op['rows']} capacity={op['capacity']} pool_pages={op['pool_pages']} tiles/group={op.get('tiles_per_group',1)}"
        else:
            summary = f"rows={op['rows']} stride={op['stride']} bound={op['bound']} top_k={op['top_k']} chunk={op.get('chunk',0)}"
        print(f"Fixture: {m['fixture_id']}\nKernel: {m['kernel']['id']} v{m['kernel'].get('version')}\nProvenance: {m['provenance'].get('kind')}\n"
              f"Operation: {summary}\nTP: rank {m['tp']['rank']} / {m['tp']['size']}\nReference: {m['reference'].get('source')}\nPath: {root}")
        for name, d in m["tensors"].items():
            print(f"  {name}: {d['dtype']} {d['shape']} sha256={d['sha256']}")
    return 0

def cmd_validate(args: argparse.Namespace) -> int:
    root, m = load_fixture(args.fixture)
    verify_golden(root, m)
    kernel_id = m["kernel"]["id"]
    print(json.dumps({"status":"pass","golden_reference":"verified-exact-or-declared-numeric-tolerance","fixture_id":m["fixture_id"],
                      "kernel_id":kernel_id,"manifest_sha256":sha256(root/"manifest.json")},sort_keys=True))
    return 0

def cmd_replay(args: argparse.Namespace) -> int:
    root, m = load_fixture(args.fixture)
    verify_golden(root, m)
    out = result_dir(args.output)
    if m.get("kernel", {}).get("id") == "qsa.sparse_attention.k8v4":
        import qsa_attention
        if args.backend == "cpu":
            qsa_attention.cpu_reference_result(root, m, out)
        else:
            qsa_attention.run_xpu(root, m, out, args.warmup, args.repeat, args.device)
        report = qsa_attention.compare_output(root, m, out)
        print(json.dumps({"status":report["status"],"fixture_id":m["fixture_id"],"backend":args.backend,"output":str(out),"comparison":report},sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    if args.backend == "cpu":
        op, ts = m["operation"], m["tensors"]
        scores = unpack_values("uint32", safe_tensor_path(root, ts["scores"]).read_bytes())
        nblk = unpack_values("int32", safe_tensor_path(root, ts["nblk"]).read_bytes())
        t0 = time.perf_counter()
        ids, counts = candidate_cpu(scores, nblk, op["rows"], op["stride"], op["top_k"])
        elapsed = (time.perf_counter() - t0) * 1000
        write_result(out, root, m, "cpu-reference-candidate", ids, counts, {"host_wall_ms": elapsed, "device_event_ms": None, "measurement_kind": "portable_cpu_algorithm"})
    else:
        run_xpu(root, m, out, args.warmup, args.repeat, device=args.device)
    comparison = compare_result(root, m, out)
    print(json.dumps({"status":comparison["status"],"fixture_id":m["fixture_id"],"backend":args.backend,"output":str(out),"comparison":comparison},sort_keys=True))
    return 0 if comparison["status"] == "pass" else 1

def cmd_compare(args: argparse.Namespace) -> int:
    root, m = load_fixture(args.fixture)
    verify_golden(root, m)
    out = Path(args.result).expanduser().resolve()
    if m.get("kernel", {}).get("id") == "qsa.sparse_attention.k8v4":
        import qsa_attention
        report = qsa_attention.compare_output(root, m, out)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    report = compare_result(root, m, out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1

def cmd_bench(args: argparse.Namespace) -> int:
    root, m = load_fixture(args.fixture)
    verify_golden(root, m)
    out = result_dir(args.output)
    if m.get("kernel", {}).get("id") == "qsa.sparse_attention.k8v4":
        import qsa_attention
        if args.backend == "cpu":
            raise XrvError("CPU is the independent reference path, not a candidate attention kernel; use replay --backend cpu or bench --backend xpu")
        qsa_attention.run_xpu(root,m,out,args.warmup,args.repeat,args.device)
        report=qsa_attention.compare_output(root,m,out)
        result=json.loads((out/"result.json").read_text(encoding="utf-8"))
        timing=result.get("timing",{})
        measured=timing.get("device_event_ms") if args.backend=="xpu" else timing.get("host_wall_ms")
        label="SYCL device event span" if args.backend=="xpu" else "float64 CPU reference wall time (not a candidate kernel)"
        (out/"bench.md").write_text(f"# XRV K8V4 attention benchmark\n\n- Fixture: {m['fixture_id']}\n- Kernel: {m['kernel']['id']}\n- Correctness: {report['status']}\n- Timing: {label}\n- Warmup/repeat: {args.warmup}/{args.repeat}\n- Measurement: {json.dumps(measured,sort_keys=True)}\n",encoding="utf-8")
        print(json.dumps({"status":report["status"],"fixture_id":m["fixture_id"],"backend":args.backend,"timing":measured,"output":str(out)},sort_keys=True))
        return 0 if report["status"]=="pass" else 1
    if args.backend == "cpu":
        op, ts = m["operation"], m["tensors"]
        scores = unpack_values("uint32", safe_tensor_path(root, ts["scores"]).read_bytes())
        nblk = unpack_values("int32", safe_tensor_path(root, ts["nblk"]).read_bytes())
        for _ in range(args.warmup): candidate_cpu(scores,nblk,op["rows"],op["stride"],op["top_k"])
        samples=[]
        for _ in range(args.repeat):
            t=time.perf_counter(); ids,counts=candidate_cpu(scores,nblk,op["rows"],op["stride"],op["top_k"]); samples.append((time.perf_counter()-t)*1000)
        timing={"measurement_kind":"portable_cpu_algorithm","device_event_ms":None,"host_wall_ms":{"median":median(samples),"p90":percentile(samples,.90),"p99":percentile(samples,.99),"jitter_max_minus_min":max(samples)-min(samples),"samples":samples}}
        write_result(out,root,m,"cpu-reference-candidate",ids,counts,timing)
    else:
        run_xpu(root,m,out,args.warmup,args.repeat,device=args.device)
    comparison=compare_result(root,m,out)
    result=json.loads((out/"result.json").read_text(encoding="utf-8"))
    timing=result.get("timing",{})
    measured=timing.get("device_event_ms") if args.backend == "xpu" else timing.get("host_wall_ms")
    timing_label="SYCL event span" if args.backend == "xpu" else "portable CPU algorithm wall time"
    lines=["# XRV benchmark", "", f"- Fixture: {m['fixture_id']}", f"- Kernel: {KERNEL}", f"- Backend: {args.backend}",
           f"- Correctness: {comparison['status']} (exact discrete IDs/counts)", f"- Timing: {timing_label}",
           f"- Warmup/repeat: {args.warmup}/{args.repeat}", f"- Measurements: {json.dumps(measured,sort_keys=True)}"]
    if args.backend == "cpu": lines.append("- GPU performance claim: none; this is host-only algorithm timing.")
    else:
        lines.append(f"- Hardware: {json.dumps(result.get('hardware',{}),sort_keys=True)}")
        lines.append(f"- Runtime: {json.dumps(result.get('runtime',{}),sort_keys=True)}")
        lines.append("- Inputs: warm and device-resident; host transfers are outside the event span.")
    (out/"bench.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":comparison["status"],"fixture_id":m["fixture_id"],"backend":args.backend,"output":str(out),"comparison":comparison,"timing":measured},sort_keys=True))
    return 0 if comparison["status"] == "pass" else 1

def cmd_ab(args: argparse.Namespace) -> int:
    root, m = load_fixture(args.fixture)
    verify_golden(root, m)
    out = result_dir(args.output)
    if m.get("kernel", {}).get("id") == "qsa.sparse_attention.k8v4":
        import qsa_attention
        if any(out.iterdir()): raise XrvError(f"A/B output directory must be empty: {out}")
        base_dir,cand_dir=out/"baseline",out/"candidate";base_dir.mkdir();cand_dir.mkdir()
        base=qsa_attention.run_xpu(root,m,base_dir,args.warmup,args.repeat,args.device,args.baseline_adapter)
        cand=qsa_attention.run_xpu(root,m,cand_dir,args.warmup,args.repeat,args.device,args.candidate_adapter)
        bc=qsa_attention.compare_output(root,m,base_dir);cc=qsa_attention.compare_output(root,m,cand_dir)
        bt=base.get("timing",{}).get("device_event_ms",{});ct=cand.get("timing",{}).get("device_event_ms",{})
        bm,cm=bt.get("median"),ct.get("median")
        hw=base.get("hardware")==cand.get("hardware") and base.get("runtime")==cand.get("runtime")
        iso=lambda x:isinstance(x,dict) and x.get("status")=="caller-attested-idle" and x.get("evidence")=="process-check+exclusive-lock-attested"
        isolated=iso(base.get("workload_isolation")) and iso(cand.get("workload_isolation"))
        correct=bc["status"]=="pass" and cc["status"]=="pass"
        comparable=all(isinstance(x,(int,float)) and x>0 for x in (bm,cm)) and hw and isolated and correct
        report={"xrv_ab_version":1,"fixture_id":m["fixture_id"],"fixture_manifest_sha256":sha256(root/"manifest.json"),"kernel_id":m["kernel"]["id"],"mode":"kernel-only","tp":m["tp"],"warmup":args.warmup,"repeat":args.repeat,"status":"pass" if correct else "fail",
          "correctness":{"acceptance":"declared K8V4 numeric tolerance","tolerance":m["reference"]["tolerance"],"baseline":bc["status"],"candidate":cc["status"],"baseline_bf16_bitwise_mismatch_count":bc["bf16_bitwise_mismatch_count"],"candidate_bf16_bitwise_mismatch_count":cc["bf16_bitwise_mismatch_count"],"baseline_max_scaled_error":bc["max_scaled_error"],"candidate_max_scaled_error":cc["max_scaled_error"]},"timing":{"baseline_device_median_ms":bm,"candidate_device_median_ms":cm,
          "candidate_speedup":bm/cm if comparable else None,"candidate_change_percent":(bm-cm)/bm*100 if comparable else None,
          "hardware_runtime_match":hw,"workload_isolation_attested":isolated,"timing_comparable":comparable,
          "reason":"comparable" if comparable else "correctness, hardware/runtime identity or workload isolation gate failed"},"baseline_timing":base.get("timing",{}),"candidate_timing":cand.get("timing",{}),"hardware_runtime":{"baseline":{"hardware":base.get("hardware",{}),"runtime":base.get("runtime",{})},"candidate":{"hardware":cand.get("hardware",{}),"runtime":cand.get("runtime",{})}},"workload_isolation":{"baseline":base.get("workload_isolation",{}),"candidate":cand.get("workload_isolation",{})}}
        (out/"ab.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        change=f"{report['timing']['candidate_change_percent']:+.2f}%" if comparable else "not comparable"
        (out/"ab.md").write_text("\n".join([
          "# XRV K8V4 attention A/B", "", f"- Fixture: {m['fixture_id']}",
          f"- Correctness: baseline {bc['status']}, candidate {cc['status']} under declared numeric tolerance",
          f"- BF16 bitwise mismatches: baseline {bc['bf16_bitwise_mismatch_count']}; candidate {cc['bf16_bitwise_mismatch_count']}",
          f"- Max scaled error: baseline {bc['max_scaled_error']}; candidate {cc['max_scaled_error']}",
          f"- Tolerance: scaled <= {m['reference']['tolerance']['scaled_error_max']}; BF16 mismatch fraction <= {m['reference']['tolerance']['bf16_mismatch_fraction_max']}",
          f"- Device median: {bm} ms / {cm} ms", f"- Change: {change}", f"- Timing comparable: {comparable}",
        ])+"\n",encoding="utf-8")
        print(json.dumps(report,sort_keys=True))
        return 0 if correct else 1
    if any(out.iterdir()):
        raise XrvError(f"A/B output directory must be empty: {out}")
    baseline_dir, candidate_dir = out / "baseline", out / "candidate"
    baseline_dir.mkdir(); candidate_dir.mkdir()
    base = run_xpu(root, m, baseline_dir, args.warmup, args.repeat, Path(args.baseline_adapter), args.device)
    cand = run_xpu(root, m, candidate_dir, args.warmup, args.repeat, Path(args.candidate_adapter), args.device)
    base_check = compare_result(root, m, baseline_dir)
    cand_check = compare_result(root, m, candidate_dir)
    bt = base.get("timing", {}).get("device_event_ms", {})
    ct = cand.get("timing", {}).get("device_event_ms", {})
    bm, cm = bt.get("median"), ct.get("median")
    hw_match = base.get("hardware") == cand.get("hardware") and base.get("runtime") == cand.get("runtime")
    base_iso, cand_iso = base.get("workload_isolation", {}), cand.get("workload_isolation", {})
    isolation_attested = all(isinstance(w, dict) and w.get("status") == "caller-attested-idle" and
                             w.get("evidence") == "process-check+exclusive-lock-attested" for w in (base_iso, cand_iso))
    correctness_ok = base_check["status"] == "pass" and cand_check["status"] == "pass"
    timing_comparable = (isinstance(bm, (int, float)) and isinstance(cm, (int, float)) and bm > 0 and cm > 0
                         and hw_match and isolation_attested and correctness_ok)
    timing_reason = ("workload isolation was not attested for both runs" if not isolation_attested else
                     "hardware/runtime fingerprints differ" if not hw_match else
                     "one or both outputs failed exact correctness" if not correctness_ok else "comparable")
    report = {"xrv_ab_version": 1, "fixture_id": m["fixture_id"], "fixture_manifest_sha256": sha256(root / "manifest.json"),
              "kernel_id": KERNEL, "mode": "kernel-only", "tp": m["tp"], "warmup": args.warmup, "repeat": args.repeat,
              "correctness": {"baseline": base_check["status"], "candidate": cand_check["status"],
                              "same_exact_golden": base_check["status"] == "pass" and cand_check["status"] == "pass"},
              "timing": {"baseline_device_median_ms": bm, "candidate_device_median_ms": cm,
                         "candidate_speedup": (bm / cm if timing_comparable else None),
                         "candidate_change_percent": ((bm - cm) / bm * 100.0 if timing_comparable else None),
                         "hardware_runtime_match": hw_match, "workload_isolation_attested": isolation_attested,
                         "workload_isolation_reason": timing_reason, "timing_comparable": timing_comparable},
              "baseline_workload_isolation": base_iso, "candidate_workload_isolation": cand_iso,
              "baseline_hardware": base.get("hardware", {}), "candidate_hardware": cand.get("hardware", {}),
              "status": "pass" if base_check["status"] == "pass" and cand_check["status"] == "pass" else "fail"}
    (out / "ab.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    change = "not comparable" if not timing_comparable else f"{report['timing']['candidate_change_percent']:+.2f}%"
    (out / "ab.md").write_text("# XRV same-fixture A/B\n\n" +
        f"- Fixture: {m['fixture_id']} ({report['fixture_manifest_sha256']})\n- Kernel: {KERNEL}\n- Warmup/repeat: {args.warmup}/{args.repeat}\n" +
        f"- Exact correctness: baseline {base_check['status']}, candidate {cand_check['status']}\n" +
        f"- Device median: baseline {bm} ms; candidate {cm} ms; change {change}\n" +
        f"- Hardware/runtime fingerprint match: {hw_match}\n- Mode: kernel-only; no end-to-end tok/s claim\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1

def sanitized_command(command: list[str]) -> list[str]:
    # Preserve executable and option names only; values and positional arguments
    # can contain prompts, paths, tokens, or private identifiers.
    out: list[str] = []
    redact_next: str | None = None
    model_options = {"--model", "--model-path"}
    sensitive_options = {"--token", "--api-key", "--password", "--secret", "--auth-token", "--prompt",
                         "--prompt-text", "--prompt-file", "--query", "--input", "--input-file", "--text",
                         "--messages", "--messages-file", "--model", "--model-path", "--dataset", "--data-file", "--ppl"}
    for index, arg in enumerate(command):
        if index == 0:
            out.append(Path(arg).name or "<EXECUTABLE>")
            continue
        if redact_next:
            out.append("<MODEL_DIR>" if redact_next == "model" else "<REDACTED>")
            redact_next = None
            continue
        if arg.startswith("-"):
            key, has_value, _value = arg.partition("=")
            out.append(key + ("=<MODEL_DIR>" if has_value and key.lower() in model_options
                              else ("=<REDACTED>" if has_value else "")))
            if not has_value and key.lower() in sensitive_options:
                redact_next = "model" if key.lower() in model_options else "secret"
            elif not has_value and index + 1 < len(command) and not command[index + 1].startswith("-"):
                redact_next = "model" if key.lower() in model_options else "secret"
            continue
        out.append("<REDACTED_ARG>")
    return out

def _capture_input_origin(command: list[str]) -> dict[str, Any]:
    for index, argument in enumerate(command[:-1]):
        if Path(argument).name != "xenithium-probe":
            continue
        subcommand = command[index + 1]
        if subcommand == "model":
            return {"kind":"full-model-probe","description":"caller-supplied token corpus through xenithium-probe model",
                    "prompt_derived_state":True}
        if subcommand == "qsa":
            return {"kind":"qsa-diagnostic-probe","description":"random K/V history and synthetic chain activations from xenithium-probe qsa",
                    "prompt_derived_state":False}
    return {"kind":"unclassified-probe","description":"explicit capture command was not recognized as a full-model or QSA diagnostic probe",
            "prompt_derived_state":None}

def _capture_attention_fixture(capdir: Path, meta: dict[str, Any], output: Path,
                               args: argparse.Namespace, rev: str, command: list[str],
                               child_returncode: int) -> int:
    import qsa_attention as attention
    if meta.get("kernel") != ATTENTION_KERNEL:
        raise XrvError(f"capture hook returned kernel {meta.get('kernel')!r}, expected {ATTENTION_KERNEL}")
    dimensions = {}
    for field in ("rows", "q_heads", "kv_heads", "cards", "tp_rank", "tp_size", "capacity", "pool_pages", "table_slots", "slot_pages", "tiles_per_group"):
        dimensions[field] = _num(meta.get(field), f"capture.{field}", 1 if field != "tp_rank" else 0)
    if dimensions["rows"] > 8 or dimensions["tp_rank"] != args.tp_rank or dimensions["tp_size"] != args.tp_size:
        raise XrvError("captured attention rows or TP identity do not match the requested capture")
    if dimensions["cards"] != dimensions["tp_size"] or dimensions["q_heads"] // dimensions["cards"] != 12:
        raise XrvError("captured attention topology is incompatible with the XRV sparse-attention adapter")
    if dimensions["kv_heads"] // dimensions["cards"] != 1:
        raise XrvError("captured attention fixture requires one local KV head")
    source_projection_stride = _num(meta.get("source_projection_stride"),"capture.source_projection_stride",12*2*256)
    if dimensions["capacity"] % 64 or dimensions["slot_pages"] < dimensions["capacity"] // 64:
        raise XrvError("captured attention cache dimensions are inconsistent")
    rows, pool, slots, page_count = dimensions["rows"], dimensions["pool_pages"], dimensions["table_slots"], dimensions["slot_pages"]
    input_origin = _capture_input_origin(command)
    tensors = {
        "query": ("query.u16le", "uint16", [rows,12,256], [12*256,256,1], "projected_query"),
        "gate": ("gate.u16le", "uint16", [rows,12,256], [12*256,256,1], "bf16_rounded_gate"),
        "positions": ("positions.i32le", "int32", [rows], [1], "logical_positions"),
        "slots": ("slots.i32le", "int32", [rows], [1], "compact_slot_ids"),
        "page_table": ("page_table.i32le", "int32", [slots,page_count], [page_count,1], "compact_page_table"),
        "keys": ("keys.u8", "uint8", [pool,1,64,256], [64*256,64*256,256,1], "packed_k8_cache"),
        "values": ("values.u8", "uint8", [pool,1,64,128], [64*128,64*128,128,1], "packed_v4_cache"),
        "key_scales": ("key_scales.f32le", "float32", [pool,1,64], [64,64,1], "k8_per_token_scale"),
        "value_scales": ("value_scales.u16le", "uint16", [pool,1,64,8], [64*8,64*8,8,1], "v4_bf16_group_scales"),
        "selected_ids": ("selected_ids.i32le", "int32", [rows,512], [512,1], "selected_logical_block_ids"),
        "selected_counts": ("selected_counts.i32le", "int32", [rows], [1], "selected_block_counts"),
    }
    if int(meta.get("pool_pages", 0)) < 1 or int(meta.get("table_slots", 0)) < 1:
        raise XrvError("captured attention cache has no pages or slots")
    output.mkdir()
    manifest = {"schema": SCHEMA, "schema_version": 2,
        "fixture_id": args.fixture_id or f"qsa-attention-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "kernel": {"id": ATTENTION_KERNEL, "version": 1, "implementation": "xenithium::kernels::xrv::sparse_attention"},
        "model_fingerprint": {"sha256": args.model_fingerprint or None, "weights_included": False},
        "runtime_fingerprint": meta.get("runtime_fingerprint", {}),
        "tp": {"rank": dimensions["tp_rank"], "size": dimensions["tp_size"]},
        "operation": {"rows": rows, "q_heads": dimensions["q_heads"], "kv_heads": dimensions["kv_heads"],
            "cards": dimensions["cards"], "capacity": dimensions["capacity"], "pool_pages": pool,
            "table_slots": slots, "slot_pages": page_count, "tiles_per_group": dimensions["tiles_per_group"],
            "selection_contract": "captured selected blocks plus causal open-block tail",
            "source_projection_stride": source_projection_stride},
        "quantization": {"kv": "K8V4", "key_scale": "float32-per-token", "value_scale": "bf16-per-32-dims",
            "k_code": "signed-symmetric-int8", "v_code": "signed-symmetric-int4"},
        "position": {"kind": "per-row logical token positions", "causal": True, "tokens_per_block": 4,
            "page_size": 64, "page_table": "compact slot x logical page -> compact physical page"},
        "masks": [{"name": "causal", "kind": "selected complete blocks plus current partial block"}],
        "mutable_state": [{"name": "KV page pool", "captured": True, "mutated_by_replay": False}],
        "reference": {"kind": "independent-float64-reference", "independent": True,
            "source": "tools/xrv/qsa_attention.py reference_output over packed K8/V4 inputs and explicit production BF16 gate-rounding points; captured candidate output is not the golden",
            "candidate_output_is_not_golden": True, "tolerance": {"scaled_error_max": 1/64, "bf16_mismatch_fraction_max": 0.005}},
        "provenance": {"kind": "kernel-boundary-capture", "input_source": input_origin["description"],
            "input_origin_kind": input_origin["kind"],
            "captured_utc": meta.get("captured_utc"), "engine_revision": meta.get("engine_revision", rev),
            "capture_command_template": sanitized_command(command),
            "capture_command_sha256": hashlib.sha256(json.dumps(sanitized_command(command),separators=(",",":"),ensure_ascii=False).encode()).hexdigest(),
            "capture_device": meta.get("runtime_fingerprint", {}), "model_fingerprint": args.model_fingerprint or None,
            "prompt_text_included": False, "prompt_derived_state": input_origin["prompt_derived_state"], "weights_included": False,
            "sensitive_capture": True, "sensitive_capture_reason": "tensor values may contain request-derived activations or cache pages; keep outside git and public artifacts"},
        "tensors": {}}
    for name,(filename,dtype,shape,strides,role) in tensors.items():
        src = capdir / filename
        if not src.is_file(): raise XrvError(f"attention capture is missing tensor {filename}")
        dst = output / filename
        shutil.copyfile(src,dst)
        manifest["tensors"][name] = tensor_info(dst,dtype,shape,strides,role)
    candidate_src = capdir / "candidate_output.u16le"
    if not candidate_src.is_file(): raise XrvError("attention capture is missing candidate_output.u16le")
    candidate_dst = output / "captured_candidate_output.u16le"
    shutil.copyfile(candidate_src,candidate_dst)
    manifest["tensors"]["captured_candidate_output"] = tensor_info(candidate_dst,"uint16",[rows,12,256],[12*256,256,1],"captured_candidate_observation")
    expected = attention.reference_output(output,manifest)
    expected_path = output / "expected_output.u16le"
    expected_path.write_bytes(attention.pack_values("uint16",expected))
    manifest["tensors"]["expected_output"] = tensor_info(expected_path,"uint16",[rows,12,256],[12*256,256,1],"independent_reference")
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    root = output
    try:
        attention.validate_fixture(root,manifest,check_data=True)
        attention.verify_golden(root,manifest)
    except attention.FixtureError as e:
        raise XrvError(str(e)) from e
    observation = output / "capture_observation"
    observation.mkdir()
    observed = observation / "output.u16le"
    shutil.copyfile(candidate_dst,observed)
    result = {"xrv_result_version":1,"fixture_id":manifest["fixture_id"],
        "fixture_manifest_sha256":attention._digest(root),"kernel_id":ATTENTION_KERNEL,"backend":"xpu-capture-observation",
        "mode":"kernel-only","status":"completed","tp":manifest["tp"],"output_files":{"output":"output.u16le"},
        "output_sha256":{"output":attention.sha256(observed)}}
    (observation / "result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    comparison = attention.compare_output(root,manifest,observation,write=True)
    capture_report = {"candidate_vs_independent_reference":comparison["status"],"mismatch_count":comparison["bf16_bitwise_mismatch_count"],
        "max_scaled_error":comparison["max_scaled_error"],"comparison":"capture_observation/comparison.json",
        "fixture_manifest_sha256":attention._digest(root),"child_exit_code":child_returncode}
    (output / "capture_validation.json").write_text(json.dumps(capture_report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"Captured {ATTENTION_KERNEL} fixture {manifest['fixture_id']} at {output}; independent reference comparison: {comparison['status'].upper()}.")
    if comparison["status"] != "pass" or child_returncode:
        return 1
    return 0

def cmd_capture(args: argparse.Namespace) -> int:
    command=args.command
    if command and command[0]=="--": command=command[1:]
    if not command:
        raise XrvError("capture requires an explicit GPU/model command after --; XRV will not start a model implicitly")
    if args.tp_rank >= args.tp_size:
        raise XrvError("tp-rank must be less than tp-size")
    if args.model_fingerprint and (len(args.model_fingerprint) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.model_fingerprint)):
        raise XrvError("model-fingerprint must be a 64-character SHA-256 hex digest; do not pass a path or model name")
    if args.fixture_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.fixture_id):
        raise XrvError("fixture-id must be a safe identifier, not a path or private text")
    output=Path(args.output).expanduser().resolve()
    repo=Path(__file__).resolve().parents[2]
    if output == repo or repo in output.parents:
        raise XrvError("real captures contain prompt-derived tensors; choose an output directory outside the repository")
    if output.exists(): raise XrvError(f"refusing to overwrite existing fixture path: {output}")
    output.parent.mkdir(parents=True,exist_ok=True)
    rev=subprocess.run(["git","rev-parse","HEAD"],cwd=repo,text=True,capture_output=True).stdout.strip() or "unknown"
    with tempfile.TemporaryDirectory(prefix="xrv-capture-") as temp_dir:
        raw=Path(temp_dir)
        env=os.environ.copy(); env.update({"XEN_XRV_CAPTURE_DIR":str(raw),"XEN_XRV_CAPTURE_MIN_BOUND":str(args.min_bound),
            "XEN_XRV_CAPTURE_MAX_BYTES":str(args.max_bytes),"XEN_XRV_CAPTURE_TP_RANK":str(args.tp_rank),
            "XEN_XRV_CAPTURE_TP_SIZE":str(args.tp_size),"XEN_XRV_ENGINE_REVISION":rev,
            "XEN_XRV_CAPTURE_KERNEL":args.kernel})
        if args.model_fingerprint: env["XEN_XRV_MODEL_FINGERPRINT"]=args.model_fingerprint
        print("Starting explicitly supplied capture command (this may load weights and run a model):", sanitized_command(command), file=sys.stderr)
        proc=subprocess.run(command,env=env,cwd=repo)
        captures=sorted(raw.glob("capture-*/capture.json"))
        if not captures:
            raise XrvError(f"no matching {args.kernel} capture was produced (child exit={proc.returncode}); check probe build, TP rank/size, kernel mode and min-bound")
        capdir=captures[0].parent
        meta=json.loads(captures[0].read_text(encoding="utf-8"))
        if not isinstance(meta,dict): raise XrvError("capture metadata must be a JSON object")
        if args.kernel == ATTENTION_KERNEL:
            return _capture_attention_fixture(capdir,meta,output,args,rev,command,proc.returncode)
        rows=_num(meta.get("rows"),"capture.rows",1); stride=_num(meta.get("stride"),"capture.stride",1); bound=_num(meta.get("bound"),"capture.bound",1)
        if rows>64 or stride!=bound or bound>MAX_BLOCKS or meta.get("tp_rank")!=args.tp_rank or meta.get("tp_size")!=args.tp_size:
            raise XrvError("captured selector dimensions or TP identity do not match the requested XRV capture")
        score_src=capdir/"scores.u32le"; nblk_src=capdir/"nblk.i32le"; cand_ids=capdir/"candidate_ids.i32le"; cand_counts=capdir/"candidate_counts.i32le"
        scores=unpack_values("uint32",score_src.read_bytes()); nblk=unpack_values("int32",nblk_src.read_bytes())
        observed_ids=unpack_values("int32",cand_ids.read_bytes()); observed_counts=unpack_values("int32",cand_counts.read_bytes())
        if len(scores)!=rows*stride or len(nblk)!=rows or len(observed_ids)!=rows*TOP_K or len(observed_counts)!=rows:
            raise XrvError("captured tensor sizes do not match capture metadata")
        for r in range(rows):
            if observed_counts[r] < 0 or observed_counts[r] > TOP_K:
                raise XrvError(f"captured selector output count is invalid at row {r}")
            start=r*TOP_K
            for j in range(observed_counts[r],TOP_K):
                observed_ids[start+j]=-1
        exp_ids,exp_counts=reference_select(scores,nblk,rows,stride,TOP_K)
        output.mkdir()
        shutil.copyfile(score_src,output/"scores.u32le"); shutil.copyfile(nblk_src,output/"nblk.i32le")
        (output/"expected_ids.i32le").write_bytes(pack_values("int32",exp_ids)); (output/"expected_counts.i32le").write_bytes(pack_values("int32",exp_counts))
        (output/"captured_candidate_ids.i32le").write_bytes(pack_values("int32",observed_ids)); shutil.copyfile(cand_counts,output/"captured_candidate_counts.i32le")
        capture_mismatches=[]
        capture_mismatch_count=0
        for r in range(rows):
            if observed_counts[r] != exp_counts[r]:
                capture_mismatch_count += 1
                if len(capture_mismatches)<16: capture_mismatches.append({"tensor":"selected_counts","row":r,"expected":exp_counts[r],"actual":observed_counts[r]})
            for j in range(exp_counts[r]):
                if observed_ids[r*TOP_K+j] != exp_ids[r*TOP_K+j]:
                    capture_mismatch_count += 1
                    if len(capture_mismatches)<16: capture_mismatches.append({"tensor":"selected_ids","row":r,"index":j,"expected":exp_ids[r*TOP_K+j],"actual":observed_ids[r*TOP_K+j]})
        input_origin = _capture_input_origin(command)
        reference={"kind":"independent-host-reference","independent":True,"source":"tools/xrv/xrv.py reference_select (descending uint32 score, tie by lowest ID; output IDs ascending)",
           "candidate_capture_is_not_golden":True}
        manifest={"schema":SCHEMA,"schema_version":SCHEMA_VERSION,"fixture_id":args.fixture_id or f"qsa-select-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
          "kernel":{"id":KERNEL,"version":1,"implementation":"xenithium::kernels::qsa_select"},"model_fingerprint":{"sha256":args.model_fingerprint or None,"weights_included":False},
          "runtime_fingerprint":meta.get("runtime_fingerprint",{}),"tp":{"rank":meta.get("tp_rank",args.tp_rank),"size":meta.get("tp_size",args.tp_size)},
          "operation":{"rows":rows,"stride":stride,"source_stride":meta.get("source_stride",stride),"bound":bound,"top_k":TOP_K,"chunk":meta.get("chunk",0),"score_encoding":"nonnegative_float32_bits_as_uint32"},
          "quantization":{"score_encoding":"positive-float32-order-preserving-bits","kv":"not-present-at-selector-boundary"},
          "position":{"kind":"already-applied-to-score-and-nblk","start":meta.get("position"),"causal_mask":"already-applied-before-selector"},
          "masks":[],"mutable_state":[],"tensors":{},"reference":reference,
          "capture_validation":{"candidate_vs_reference":"pass" if not capture_mismatches else "fail","mismatch_count":capture_mismatch_count,"sample_mismatches":capture_mismatches},
          "provenance":{"kind":"kernel-boundary-capture","input_source":input_origin["description"],"input_origin_kind":input_origin["kind"],
            "captured_utc":meta.get("captured_utc"),"capture_command_template":sanitized_command(command),
            "capture_command_sha256":hashlib.sha256(json.dumps(sanitized_command(command),separators=(",",":"),ensure_ascii=False).encode()).hexdigest(),"capture_device":meta.get("runtime_fingerprint",{}),
            "engine_revision":meta.get("engine_revision",rev),"prompt_tokens_included":False,"weights_included":False,
            "prompt_derived_state":input_origin["prompt_derived_state"],
            "derived_from_prompt_or_model_state":input_origin["prompt_derived_state"]}}
        for name,file,dtype,shape,strides,role in [
          ("scores","scores.u32le","uint32",[rows,stride],[stride,1],"input"),("nblk","nblk.i32le","int32",[rows],[1],"input"),
          ("expected_ids","expected_ids.i32le","int32",[rows,TOP_K],[TOP_K,1],"independent_reference"),("expected_counts","expected_counts.i32le","int32",[rows],[1],"independent_reference"),
          ("captured_candidate_ids","captured_candidate_ids.i32le","int32",[rows,TOP_K],[TOP_K,1],"candidate_observation"),("captured_candidate_counts","captured_candidate_counts.i32le","int32",[rows],[1],"candidate_observation")]:
            manifest["tensors"][name]=tensor_info(output/file,dtype,shape,strides,role)
        (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    root,manifest=load_fixture(output)
    verify_golden(root,manifest)
    print(f"Captured fixture {manifest['fixture_id']} at {output}; host reference stored separately from candidate output.")
    if proc.returncode:
        raise XrvError(f"capture command exited {proc.returncode}; fixture was saved, but inspect the child run before use")
    if capture_mismatches:
        print("WARNING: captured kernel output disagrees with independent reference; fixture is retained as a failure case.",file=sys.stderr)
    return 0

def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="xrv",description="Xenithium Replay Validation: fixtures, correctness replay and kernel A/B")
    sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("list",help="list checked-in and XRV_FIXTURE_DIR fixtures").set_defaults(func=cmd_list)
    q=sub.add_parser("inspect",help="show fixture metadata and verify checksums"); q.add_argument("fixture"); q.add_argument("--json",action="store_true"); q.add_argument("--no-hash",action="store_true"); q.set_defaults(func=cmd_inspect)
    q=sub.add_parser("validate",help="validate schema, bounds, paths and SHA-256 checksums"); q.add_argument("fixture"); q.set_defaults(func=cmd_validate)
    q=sub.add_parser("capture",help="explicitly capture one QSA kernel call from a supplied probe/model command"); q.add_argument("--kernel",choices=[KERNEL,ATTENTION_KERNEL],default=KERNEL); q.add_argument("--output",required=True); q.add_argument("--fixture-id"); q.add_argument("--min-bound",type=int,default=1); q.add_argument("--max-bytes",type=int,default=268435456); q.add_argument("--tp-rank",type=int,default=0); q.add_argument("--tp-size",type=int,default=1); q.add_argument("--model-fingerprint"); q.add_argument("command",nargs=argparse.REMAINDER); q.set_defaults(func=cmd_capture)
    q=sub.add_parser("replay",help="run CPU reference candidate or production XPU selector"); q.add_argument("--fixture",required=True); q.add_argument("--backend",choices=["cpu","xpu"],default="cpu"); q.add_argument("--output",required=True); q.add_argument("--warmup",type=int,default=0); q.add_argument("--repeat",type=int,default=1); q.add_argument("--device",type=int,default=0); q.set_defaults(func=cmd_replay)
    q=sub.add_parser("compare",help="exactly compare selected IDs and counts to independent reference"); q.add_argument("--fixture",required=True); q.add_argument("--result",required=True); q.set_defaults(func=cmd_compare)
    q=sub.add_parser("bench",help="repeat a fixed fixture and report CPU wall or XPU event timings"); q.add_argument("--fixture",required=True); q.add_argument("--backend",choices=["cpu","xpu"],default="xpu"); q.add_argument("--output",required=True); q.add_argument("--warmup",type=int,default=10); q.add_argument("--repeat",type=int,default=100); q.add_argument("--device",type=int,default=0); q.set_defaults(func=cmd_bench)
    q=sub.add_parser("ab",help="benchmark two XPU adapter builds on the same fixture and exact golden"); q.add_argument("--fixture",required=True); q.add_argument("--baseline-adapter",required=True); q.add_argument("--candidate-adapter",required=True); q.add_argument("--output",required=True); q.add_argument("--warmup",type=int,default=10); q.add_argument("--repeat",type=int,default=100); q.add_argument("--device",type=int,default=0); q.set_defaults(func=cmd_ab)
    return p

def main(argv: list[str] | None = None) -> int:
    try:
        args=parser().parse_args(argv)
        if hasattr(args,"warmup") and args.warmup<0: raise XrvError("warmup must be >= 0")
        if hasattr(args,"device") and args.device<0: raise XrvError("device must be >= 0")
        if hasattr(args,"repeat") and args.repeat<1: raise XrvError("repeat must be >= 1")
        if hasattr(args,"min_bound") and (args.min_bound<0 or args.min_bound>MAX_BLOCKS): raise XrvError("min-bound must be in [0,262144]")
        if hasattr(args,"tp_rank") and (args.tp_rank<0 or args.tp_rank>=args.tp_size): raise XrvError("tp-rank must be less than tp-size")
        if hasattr(args,"max_bytes") and args.max_bytes<1: raise XrvError("max-bytes must be > 0")
        return args.func(args)
    except XrvError as e:
        print(f"xrv: error: {e}",file=sys.stderr)
        return 2
    except (OSError, ValueError, OverflowError) as e:
        print(f"xrv: error: {e}",file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("xrv: interrupted",file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
