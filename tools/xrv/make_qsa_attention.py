#!/usr/bin/env python3
"""Generate a deterministic packed K8V4 QSA attention fixture with a float64 CPU golden."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import qsa_attention as xrv

def tensor_payloads():
    rows,pool_pages,table_slots,slot_pages=2,66,2,33
    query=[]; gate=[]
    for r in range(rows):
      for h in range(12):
        for d in range(256):
          query.append(xrv.float_to_bf16((((r*13+h*7+d*3)%31)-15)/160.0))
          gate.append(xrv.float_to_bf16((((r*5+h*11+d)%23)-11)/16.0))
    positions=[2053,30]; slots=[0,1]
    page_table=[]
    for slot in range(table_slots):
      base=slot*slot_pages
      page_table.extend(base+((p*7+3)%slot_pages) for p in range(slot_pages))
    keys=[]; values=[]; key_scales=[]; value_scales=[]
    for physical in range(pool_pages):
      for token in range(64):
        pt=physical*64+token
        kzero=(pt%97==0)
        key_scales.append(0.0 if kzero else 0.004+(pt%7)*0.00025)
        for d in range(256):
          code=0 if kzero else ((physical*17+token*7+d*3)%31)-15
          keys.append(code&255)
        scales=[]
        for group in range(8):
          zero=((pt+group*13)%83==0)
          scale=0.0 if zero else 0.012+(pt%5)*0.001+(group%3)*0.0005
          scales.append(xrv.float_to_bf16(scale))
        value_scales.extend(scales)
        for pair in range(128):
          d0=pair*2; d1=d0+1
          g0=d0//32; g1=d1//32
          s0=xrv.bf16_to_float(scales[g0]); s1=xrv.bf16_to_float(scales[g1])
          c0=0 if s0==0 else ((physical*11+token*3+d0*5)%15)-7
          c1=0 if s1==0 else ((physical*11+token*3+d1*5)%15)-7
          values.append((c0&15)|((c1&15)<<4))
    ids=[b for b in range(513) if b!=257]
    selected=ids+[-1]*(512-len(ids))
    ids1=list(range(7))+[-1]*505
    return {
      "query":("uint16",[rows,12,256],[12*256,256,1],query),
      "gate":("uint16",[rows,12,256],[12*256,256,1],gate),
      "positions":("int32",[rows],[1],positions),"slots":("int32",[rows],[1],slots),
      "page_table":("int32",[table_slots,slot_pages],[slot_pages,1],page_table),
      "keys":("uint8",[pool_pages,1,64,256],[64*256,64*256,256,1],keys),
      "values":("uint8",[pool_pages,1,64,128],[64*128,64*128,128,1],values),
      "key_scales":("float32",[pool_pages,1,64],[64,64,1],key_scales),
      "value_scales":("uint16",[pool_pages,1,64,8],[64*8,64*8,8,1],value_scales),
      "selected_ids":("int32",[rows,512],[512,1],selected+ids1),
      "selected_counts":("int32",[rows],[1],[512,7]),
    }

def create(out:Path)->None:
    if out.exists(): raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    op={"rows":2,"q_heads":24,"kv_heads":2,"cards":2,"capacity":2112,"pool_pages":66,"table_slots":2,"slot_pages":33,
        "tiles_per_group":1,"selection_contract":"512 ascending logical blocks; causal tail appended"}
    m={"schema":xrv.SCHEMA,"schema_version":2,"fixture_id":"qsa-k8v4-sparse-synthetic",
       "kernel":{"id":xrv.KERNEL,"version":1,"implementation":"xenithium::kernels::xrv::sparse_attention"},
       "model_fingerprint":{"kind":"synthetic-Qwen-QSA-topology","sha256":None,"weights_included":False},
       "runtime_fingerprint":{"kind":"synthetic-fixture","driver":None,"compiler":None,"device":None},
       "tp":{"rank":0,"size":2},"operation":op,
       "quantization":{"kv":"K8V4","key_scale":"float32-per-token","value_scale":"bf16-per-32-dims","k_code":"signed-symmetric-int8","v_code":"signed-symmetric-int4"},
       "position":{"kind":"per-row logical token positions","causal":True,"tokens_per_block":4,"page_size":64,"page_table":"slot x logical page -> physical page"},
       "masks":[{"name":"causal","kind":"position-bounded selected blocks plus open-block tail"}],"mutable_state":[],
       "reference":{"kind":"independent-float64-reference","independent":True,
          "source":"tools/xrv/qsa_attention.py reference_output: float64 softmax over independently dequantized packed K8/V4 bytes, page mappings, and explicit production BF16 gate-rounding points; candidate output is excluded",
          "candidate_output_is_not_golden":True,"tolerance":{"scaled_error_max":1/64,"bf16_mismatch_fraction_max":0.005}},
       "provenance":{"kind":"deterministic-synthetic","generator":"tools/xrv/make_qsa_attention.py","seed":"integer formulas; no random state",
          "contains_model_weights":False,"contains_prompt_data":False,"purpose":"long sparse selection, short every-block path, permuted page table, packed K8/V4, zero scales and BF16 gate"},
       "tensors":{}}
    for name,(dtype,shape,strides,values) in tensor_payloads().items():
      data=xrv.pack_values(dtype,values); filename=name+(".u16le" if dtype=="uint16" else ".i32le" if dtype=="int32" else ".u8" if dtype=="uint8" else ".f32le")
      path=out/filename; path.write_bytes(data)
      m["tensors"][name]={"file":filename,"dtype":dtype,"shape":shape,"strides":strides,"byte_order":"little",
          "role":"input","size_bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    # Add expected output from the independent reference.
    expected=xrv.reference_output(out,m); data=xrv.pack_values("uint16",expected); path=out/"expected_output.u16le";path.write_bytes(data)
    m["tensors"]["expected_output"]={"file":path.name,"dtype":"uint16","shape":[2,12,256],"strides":[12*256,256,1],
       "byte_order":"little","role":"independent_reference","size_bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    (out/"manifest.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    xrv.validate_fixture(out,m); xrv.verify_golden(out,m)
    print(out)

if __name__=="__main__":
    create(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"tests/data/xrv/qsa-k8v4-sparse-synthetic")
