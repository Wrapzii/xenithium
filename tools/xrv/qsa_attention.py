"""K8V4 sparse-attention fixture contract and portable high-precision reference."""
from __future__ import annotations
import array, hashlib, json, math, os, platform, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

KERNEL = "qsa.sparse_attention.k8v4"
SCHEMA = "org.xenithium.xrv.fixture"
SCHEMA_VERSION = 2
D, PAGE, COMPRESS, TOP_BLOCKS, V_GROUP = 256, 64, 4, 512, 32
V_SCALES = D // V_GROUP
DTYPES = {"uint16": ("H", 2), "int32": ("i", 4), "uint8": ("B", 1), "int8": ("b", 1), "float32": ("f", 4)}
MAX_BYTES = 1 << 30

class FixtureError(ValueError): pass

def pack_values(dtype: str, values: Any) -> bytes:
    result = array.array(DTYPES[dtype][0], values)
    if sys.byteorder != "little": result.byteswap()
    return result.tobytes()

def unpack_values(dtype: str, data: bytes) -> array.array:
    code, size = DTYPES[dtype]
    if len(data) % size: raise FixtureError(f"{dtype} tensor byte count is not divisible by {size}")
    result = array.array(code); result.frombytes(data)
    if sys.byteorder != "little": result.byteswap()
    return result

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def tensor_path(root: Path, desc: dict[str, Any]) -> Path:
    name = desc.get("file")
    if not isinstance(name, str): raise FixtureError("tensor file must be a relative path")
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts: raise FixtureError(f"unsafe tensor path: {name}")
    path = (root / rel).resolve(strict=True)
    if root.resolve() not in path.parents: raise FixtureError(f"tensor escapes fixture directory: {name}")
    return path

def read_tensor(root: Path, manifest: dict[str, Any], name: str) -> array.array:
    desc = manifest["tensors"][name]
    return unpack_values(desc["dtype"], tensor_path(root, desc).read_bytes())

def bf16_to_float(bits: int) -> float:
    import struct
    return struct.unpack("<f", struct.pack("<I", int(bits) << 16))[0]

def float_to_bf16(value: float) -> int:
    import struct
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF

def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)

def validate_fixture(root: Path, m: dict[str, Any], check_data: bool = True) -> None:
    if m.get("schema") != SCHEMA or m.get("schema_version") != SCHEMA_VERSION: raise FixtureError("expected XRV fixture schema v2")
    fid = m.get("fixture_id")
    if not isinstance(fid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", fid): raise FixtureError("fixture_id is not a safe identifier")
    if m.get("kernel") != {"id": KERNEL, "version": 1, "implementation": "xenithium::kernels::xrv::sparse_attention"}: raise FixtureError(f"expected kernel {KERNEL} v1")
    op = m.get("operation")
    if not isinstance(op, dict): raise FixtureError("operation must be an object")
    fields = ("rows", "q_heads", "kv_heads", "cards", "capacity", "pool_pages", "table_slots", "slot_pages")
    if any(type(op.get(k)) is not int or op[k] < 1 for k in fields): raise FixtureError("operation dimensions must be positive integers")
    rows, qh, kvh, cards = op["rows"], op["q_heads"], op["kv_heads"], op["cards"]
    if rows > 8 or qh % kvh or qh % cards or qh // cards != 12 or kvh // cards != 1 or qh // kvh != 12: raise FixtureError("requires <=8 rows, 12 local Q heads and 1 local KV head")
    if op["capacity"] % PAGE or op["slot_pages"] < op["capacity"] // PAGE or op["pool_pages"] > 65536: raise FixtureError("invalid cache/page-table dimensions")
    tp = m.get("tp")
    if not isinstance(tp, dict) or type(tp.get("rank")) is not int or type(tp.get("size")) is not int or tp["size"] != cards or not 0 <= tp["rank"] < tp["size"]:
        raise FixtureError("TP rank/size must match the QSA head partition")
    required = {
        "query": ("uint16",[rows,12,D]), "gate": ("uint16",[rows,12,D]), "positions": ("int32",[rows]), "slots": ("int32",[rows]),
        "page_table": ("int32",[op["table_slots"],op["slot_pages"]]), "keys": ("uint8",[op["pool_pages"],1,PAGE,D]),
        "values": ("uint8",[op["pool_pages"],1,PAGE,D//2]), "key_scales": ("float32",[op["pool_pages"],1,PAGE]),
        "value_scales": ("uint16",[op["pool_pages"],1,PAGE,V_SCALES]), "selected_ids": ("int32",[rows,TOP_BLOCKS]),
        "selected_counts": ("int32",[rows]), "expected_output": ("uint16",[rows,12,D])}
    tensors = m.get("tensors")
    if not isinstance(tensors,dict): raise FixtureError("tensors must be an object")
    for name,(dtype,shape) in required.items():
        d=tensors.get(name)
        if not isinstance(d,dict) or d.get("dtype")!=dtype or d.get("shape")!=shape: raise FixtureError(f"tensor {name} must be {dtype} {shape}")
        expected_strides=[1]*len(shape)
        for i in range(len(shape)-2,-1,-1): expected_strides[i]=expected_strides[i+1]*shape[i+1]
        if d.get("strides")!=expected_strides:
            raise FixtureError(f"tensor {name} uses unsupported non-contiguous strides; this production kernel boundary requires {expected_strides}")
    total=0
    for name,d in tensors.items():
        if not isinstance(name,str) or not isinstance(d,dict): raise FixtureError("tensor entries must be named objects")
        dtype,shape,strides=d.get("dtype"),d.get("shape"),d.get("strides")
        if dtype not in DTYPES or d.get("byte_order")!="little": raise FixtureError(f"tensor {name} has unsupported dtype/byte order")
        if not isinstance(shape,list) or not shape or any(type(x)is not int or x<1 for x in shape): raise FixtureError(f"tensor {name} has invalid shape")
        if not isinstance(strides,list) or len(strides)!=len(shape) or any(type(x)is not int or x<1 for x in strides): raise FixtureError(f"tensor {name} has invalid strides")
        size=math.prod(shape)*DTYPES[dtype][1]
        if type(d.get("size_bytes")) is not int or d["size_bytes"]!=size: raise FixtureError(f"tensor {name} byte-size declaration is inconsistent")
        total+=size
        if total>MAX_BYTES: raise FixtureError("fixture exceeds the 1 GiB safety limit")
        if check_data:
            path=tensor_path(root,d)
            if path.stat().st_size!=size: raise FixtureError(f"tensor {name} file size mismatch")
            if d.get("sha256")!=sha256(path): raise FixtureError(f"tensor {name} SHA-256 mismatch")
    candidate_desc = tensors.get("captured_candidate_output")
    if candidate_desc is not None and (candidate_desc.get("dtype") != "uint16" or candidate_desc.get("shape") != [rows,12,D]):
        raise FixtureError("captured_candidate_output must be uint16 [rows,12,256]")
    for field in ("model_fingerprint","runtime_fingerprint","quantization","position","masks","mutable_state","reference","provenance"):
        if field not in m: raise FixtureError(f"manifest missing {field}")
    if not all(isinstance(m.get(x),dict) for x in ("model_fingerprint","runtime_fingerprint","quantization","position","reference","provenance")): raise FixtureError("metadata fingerprints/reference must be objects")
    if not isinstance(m["masks"],list) or not isinstance(m["mutable_state"],list): raise FixtureError("masks and mutable_state must be arrays")
    ref=m["reference"]; tol=ref.get("tolerance")
    if ref.get("independent") is not True or not isinstance(ref.get("source"),str) or not ref["source"].strip(): raise FixtureError("reference must identify an independent source")
    if not isinstance(tol,dict) or tol.get("scaled_error_max")!=1/64 or tol.get("bf16_mismatch_fraction_max")!=0.005:
        raise FixtureError("explicit QSA tolerance must be scaled<=1/64 and BF16 mismatches<=0.5%")
    q=m["quantization"]
    if q.get("kv")!="K8V4" or q.get("key_scale")!="float32-per-token" or q.get("value_scale")!="bf16-per-32-dims": raise FixtureError("unsupported packed KV quantization metadata")
    pos=m["position"]
    if pos.get("causal") is not True or pos.get("tokens_per_block")!=COMPRESS or pos.get("page_size")!=PAGE: raise FixtureError("position/causal metadata does not match QSA layout")
    if not check_data: return
    for name in ("query","gate","expected_output","captured_candidate_output"):
        if name in tensors:
            for i,bits in enumerate(read_tensor(root,m,name)):
                if bits & 0x7f80 == 0x7f80: raise FixtureError(f"{name} contains NaN/Inf at flat element {i}")
    positions=read_tensor(root,m,"positions"); slots=read_tensor(root,m,"slots"); pages=read_tensor(root,m,"page_table")
    selected=read_tensor(root,m,"selected_ids"); counts=read_tensor(root,m,"selected_counts")
    keys=read_tensor(root,m,"keys"); values=read_tensor(root,m,"values"); ks=read_tensor(root,m,"key_scales"); vs=read_tensor(root,m,"value_scales")
    if any(p<0 or p>=op["pool_pages"] for p in pages): raise FixtureError("page table references an out-of-range physical page")
    for r,p in enumerate(positions):
        if p<0 or p>=op["capacity"] or slots[r]<0 or slots[r]>=op["table_slots"]: raise FixtureError(f"row {r} has invalid position/slot")
        blocks=(p+1)//COMPRESS; count=counts[r]
        if count!=min(blocks,TOP_BLOCKS): raise FixtureError(f"selected_counts[{r}] must equal min(complete_blocks,512)")
        prev=-1
        for j,b in enumerate(selected[r*TOP_BLOCKS:(r+1)*TOP_BLOCKS]):
            if j<count:
                if b<=prev or b<0 or b>=blocks: raise FixtureError(f"selected_ids[{r},{j}] is not unique ascending/in range")
                prev=b
            elif b!=-1: raise FixtureError(f"selected_ids[{r}] padding must be -1")
    # Validate native packed codes/scales without converting fixture storage to floats.
    for token,scale in enumerate(ks):
        if not math.isfinite(scale) or scale<0: raise FixtureError("K scale must be finite and nonnegative")
        for d in range(D):
            code=keys[token*D+d]; code=code if code<128 else code-256
            if code==-128: raise FixtureError("K8 code -128 is outside symmetric quantizer range")
            if scale==0 and code: raise FixtureError("zero K scale requires all-zero K token")
    for token in range(op["pool_pages"]*PAGE):
        for g in range(V_SCALES):
            scale=bf16_to_float(vs[token*V_SCALES+g])
            if not math.isfinite(scale) or scale<0: raise FixtureError("V4 scale must be finite and nonnegative")
            for d in range(g*V_GROUP,(g+1)*V_GROUP):
                x=values[token*(D//2)+d//2]; n=x&15 if d%2==0 else x>>4; code=n-16 if n>=8 else n
                if code==-8: raise FixtureError("V4 code -8 is outside symmetric quantizer range")
                if scale==0 and code: raise FixtureError("zero V4 scale requires all-zero V4 group")

def _signed_byte(v: int) -> int: return v if v<128 else v-256

def selected_tokens(pos: int, ids: list[int], count: int) -> list[int]:
    blocks=(pos+1)//COMPRESS
    chosen=list(range(blocks)) if blocks<=TOP_BLOCKS else ids[:count]
    out=[]
    for b in chosen: out.extend(range(b*COMPRESS,b*COMPRESS+COMPRESS))
    out.extend(range(blocks*COMPRESS,pos+1))
    return out

def reference_output(root: Path,m: dict[str,Any]) -> array.array:
    """Independent float64 softmax attention over packed K8/V4 cache, with production BF16 gate rounding."""
    op=m["operation"]; rows=op["rows"]; pages_per_slot=op["slot_pages"]
    q=read_tensor(root,m,"query"); gate=read_tensor(root,m,"gate"); positions=read_tensor(root,m,"positions"); slots=read_tensor(root,m,"slots")
    table=read_tensor(root,m,"page_table"); kb=read_tensor(root,m,"keys"); vb=read_tensor(root,m,"values")
    ks=read_tensor(root,m,"key_scales"); vs=read_tensor(root,m,"value_scales"); ids=read_tensor(root,m,"selected_ids"); counts=read_tensor(root,m,"selected_counts")
    output=array.array("H",[0])*(rows*12*D)
    for r in range(rows):
        toks=selected_tokens(positions[r],list(ids[r*TOP_BLOCKS:(r+1)*TOP_BLOCKS]),counts[r])
        dk=array.array("d",[0.0])*(len(toks)*D); dv=array.array("d",[0.0])*(len(toks)*D)
        for ti,tok in enumerate(toks):
            lp,ip=divmod(tok,PAGE); pp=table[slots[r]*pages_per_slot+lp]; pt=pp*PAGE+ip
            kbase=pt*D; vbase=pt*(D//2); scale=ks[pt]; sbase=pt*V_SCALES
            for d in range(D):
                dk[ti*D+d]=_signed_byte(kb[kbase+d])*scale
                packed=vb[vbase+d//2]; nibble=packed&15 if d%2==0 else packed>>4; code=nibble-16 if nibble>=8 else nibble
                dv[ti*D+d]=code*bf16_to_float(vs[sbase+d//V_GROUP])
        for h in range(12):
            qb=(r*12+h)*D; qrow=[bf16_to_float(q[qb+d]) for d in range(D)]; scores=[]
            for ti in range(len(toks)):
                base=ti*D; dot=0.0
                for d in range(D): dot+=qrow[d]*dk[base+d]
                scores.append(dot/16.0)
            if not scores: raise FixtureError("reference received an empty causal token set")
            high=max(scores); probs=[math.exp(x-high) for x in scores]; denom=sum(probs); attended=[0.0]*D
            for ti,p in enumerate(probs):
                weight=p/denom; base=ti*D
                for d in range(D): attended[d]+=weight*dv[base+d]
            for d in range(D):
                a=bf16_to_float(float_to_bf16(attended[d])); g=bf16_to_float(gate[qb+d])
                sigmoid=bf16_to_float(float_to_bf16(_stable_sigmoid(g)))
                output[qb+d]=float_to_bf16(a*sigmoid)
    return output

def _digest(root: Path) -> str: return sha256(root/"manifest.json")

def verify_golden(root: Path,m: dict[str,Any]) -> None:
    expected=list(read_tensor(root,m,"expected_output")); actual=list(reference_output(root,m))
    if expected!=actual:
        first=next(i for i,(a,b) in enumerate(zip(expected,actual)) if a!=b)
        raise FixtureError(f"independent BF16 golden mismatch at flat output {first}; intentional refresh needs review")

def compare_output(root: Path,m: dict[str,Any],out: Path,write: bool=True) -> dict[str,Any]:
    result=json.loads((out/"result.json").read_text(encoding="utf-8"))
    envelope={"xrv_result_version":1,"fixture_id":m["fixture_id"],"fixture_manifest_sha256":_digest(root),"kernel_id":KERNEL,
              "mode":"kernel-only","status":"completed","tp":m["tp"]}
    for key,value in envelope.items():
        if result.get(key)!=value: raise FixtureError(f"result envelope {key} mismatch")
    if result.get("output_files")!={"output":"output.u16le"}: raise FixtureError("result output_files mismatch")
    path=out/"output.u16le"
    if not path.is_file() or result.get("output_sha256",{}).get("output")!=sha256(path): raise FixtureError("result output missing or SHA-256 mismatch")
    got=unpack_values("uint16",path.read_bytes()); want=read_tensor(root,m,"expected_output")
    if len(got)!=len(want): raise FixtureError("attention output element count mismatch")
    rows=m["operation"]["rows"]; total=len(got); mismatches=0; violations=0; max_abs=0.; max_rel=0.; max_scaled=0.
    first=None; first_bad=None; samples=[]; tol=m["reference"]["tolerance"]
    for r in range(rows):
      for h in range(12):
        base=(r*12+h)*D; ref=[bf16_to_float(want[i]) for i in range(base,base+D)]
        rms=math.sqrt(sum(x*x for x in ref)/D)
        for d in range(D):
          i=base+d; a=bf16_to_float(got[i]); e=ref[d]
          if not math.isfinite(a) or not math.isfinite(e): raise FixtureError(f"NaN/Inf at output [{r},{h},{d}]")
          ae=abs(a-e); re=ae/max(abs(e),1e-30); se=ae/max(abs(e),rms,1e-30)
          max_abs=max(max_abs,ae); max_rel=max(max_rel,re); max_scaled=max(max_scaled,se)
          if got[i]!=want[i]:
            mismatches+=1; entry={"row":r,"head":h,"dim":d,"expected":e,"actual":a,"abs_error":ae,"relative_error":re}
            if first is None:first=entry
            if len(samples)<16:samples.append(entry)
          if se>tol["scaled_error_max"]:
            violations+=1
            if first_bad is None:first_bad={"row":r,"head":h,"dim":d,"scaled_error":se}
    frac=mismatches/max(total,1); status="pass" if max_scaled<=tol["scaled_error_max"] and frac<=tol["bf16_mismatch_fraction_max"] else "fail"
    report={"fixture_id":m["fixture_id"],"kernel_id":KERNEL,"status":status,
      "comparison":"BF16 output vs independent float64 K8V4 reference; per-head scaled error and exact-bit mismatch fraction",
      "reference_source":m["reference"]["source"],"tolerance":tol,"output_elements":total,
      "bf16_bitwise_mismatch_count":mismatches,"bf16_bitwise_mismatch_fraction":frac,"tolerance_violation_count":violations,
      "max_absolute_error":max_abs,"max_relative_error":max_rel,"max_scaled_error":max_scaled,
      "first_mismatch":first,"first_tolerance_violation":first_bad,"sample_mismatches":samples,"candidate_backend":result.get("backend")}
    if write:
        (out/"comparison.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        (out/"comparison.md").write_text("# XRV K8V4 attention comparison\n\n"+
          f"- Fixture: {m['fixture_id']}\n- Status: **{status.upper()}**\n- Output elements: {total}\n"+
          f"- BF16 bitwise mismatches: {mismatches} ({frac:.4%})\n- Max abs/relative/scaled error: {max_abs:.6g} / {max_rel:.6g} / {max_scaled:.6g}\n"+
          f"- Tolerance: scaled <= {tol['scaled_error_max']}; BF16 mismatch fraction <= {tol['bf16_mismatch_fraction_max']}\n"+
          (f"- First mismatch: {first}\n" if first else ""),encoding="utf-8")
    return report

def adapter_path() -> Path:
    return Path(os.environ.get("XRV_QSA_ATTENTION_ADAPTER",Path(__file__).resolve().parents[2]/"build"/"xenithium-xrv-qsa-sparse-attention")).expanduser().resolve()

def _adapter_cmd(root:Path,m:dict[str,Any],out:Path,warmup:int,repeat:int,device:int,adapter:str|None=None)->list[str]:
    op=m["operation"]; t=m["tensors"]; exe=Path(adapter).expanduser().resolve() if adapter else adapter_path()
    cmd=[str(exe)]
    files=(("query","query"),("gate","gate"),("positions","positions"),("slots","slots"),("page_table","page-table"),("keys","keys"),
      ("values","values"),("key_scales","key-scales"),("value_scales","value-scales"),("selected_ids","selected-ids"),("selected_counts","selected-counts"))
    for name,flag in files: cmd += ["--"+flag,str(tensor_path(root,t[name]))]
    fields=(("rows","rows"),("q_heads","q-heads"),("kv_heads","kv-heads"),("cards","cards"),("capacity","capacity"),
      ("pool_pages","pool-pages"),("table_slots","table-slots"),("slot_pages","slot-pages"))
    for name,flag in fields: cmd += ["--"+flag,str(op[name])]
    cmd += ["--card",str(m["tp"]["rank"]),"--tp-rank",str(m["tp"]["rank"]),"--tp-size",str(m["tp"]["size"]),
      "--device",str(device),"--warmup",str(warmup),"--repeat",str(repeat),"--output",str(out)]
    return cmd

def run_xpu(root:Path,m:dict[str,Any],out:Path,warmup:int,repeat:int,device:int,adapter:str|None=None)->dict[str,Any]:
    cmd=_adapter_cmd(root,m,out,warmup,repeat,device,adapter)
    if not Path(cmd[0]).is_file(): raise FixtureError(f"attention XPU adapter not found: {cmd[0]}; build xenithium-xrv-qsa-sparse-attention")
    proc=subprocess.run(cmd,capture_output=True,text=True)
    if proc.returncode: raise FixtureError(f"XPU adapter exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    try: meta=json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError,IndexError) as e: raise FixtureError(f"XPU adapter did not return JSON: {e}") from e
    path=out/"output.u16le"
    if not path.is_file(): raise FixtureError("XPU adapter completed without output.u16le")
    result={"xrv_result_version":1,"fixture_id":m["fixture_id"],"fixture_manifest_sha256":_digest(root),"kernel_id":KERNEL,"backend":"xpu",
      "mode":"kernel-only","status":"completed","tp":m["tp"],"output_files":{"output":"output.u16le"},"output_sha256":{"output":sha256(path)},
      "timing":meta.get("timing",{}),"hardware":meta.get("hardware",{}),"runtime":meta.get("runtime",{}),
      "workload_isolation":meta.get("workload_isolation",{"status":"unknown","evidence":"not-reported"}),
      "execution":{"warmup":warmup,"repeat":repeat,"device":device,"measurement_scope":"sparse attention + merge; H2D/D2H excluded",
        "cache_state":"warm-after-warmup; fixture inputs remain device-resident","cold_cache_measurement":False},
      "created_utc":datetime.now(timezone.utc).isoformat()}
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result

def cpu_reference_result(root:Path,m:dict[str,Any],out:Path)->dict[str,Any]:
    start=time.perf_counter(); bits=reference_output(root,m); elapsed=(time.perf_counter()-start)*1000
    path=out/"output.u16le"; path.write_bytes(pack_values("uint16",bits))
    result={"xrv_result_version":1,"fixture_id":m["fixture_id"],"fixture_manifest_sha256":_digest(root),"kernel_id":KERNEL,
      "backend":"cpu-independent-reference","mode":"kernel-only","status":"completed","tp":m["tp"],
      "output_files":{"output":"output.u16le"},"output_sha256":{"output":sha256(path)},
      "timing":{"measurement_kind":"portable_float64_reference_wall_only","device_event_ms":None,"host_wall_ms":{"median":elapsed,"samples":[elapsed]}},
      "hardware":{"device":"CPU reference"},"runtime":{"backend":"python-float64"},
      "workload_isolation":{"status":"not-applicable-reference-only","evidence":"not a GPU timing"},
      "execution":{"warmup":0,"repeat":1,"measurement_scope":"independent reference; no candidate kernel executed"},
      "created_utc":datetime.now(timezone.utc).isoformat()}
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result
