from __future__ import annotations
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import jsonschema
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/xrv"))
import xrv
CLI = ROOT / "tools/xrv/xrv.py"
WRAPPER = ROOT / "scripts/xrv"
FIXTURE = ROOT / "tests/data/xrv/qsa-select-synthetic"

class XrvCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([str(WRAPPER), *map(str,args)], cwd=ROOT, text=True, capture_output=True)

    def test_checked_in_manifests_match_versioned_json_schemas(self):
        pairs=[
            ("tools/xrv/fixture-schema-v1.json","tests/data/xrv/qsa-select-synthetic/manifest.json"),
            ("tools/xrv/fixture-schema-v1.json","tests/data/xrv/qsa-select-max-bound/manifest.json"),
            ("tools/xrv/fixture-schema-v2.json","tests/data/xrv/qsa-k8v4-sparse-synthetic/manifest.json"),
        ]
        for schema_path,manifest_path in pairs:
            with self.subTest(manifest=manifest_path):
                schema=json.loads((ROOT/schema_path).read_text())
                manifest=json.loads((ROOT/manifest_path).read_text())
                jsonschema.validate(instance=manifest,schema=schema)

    def test_fixture_validates_and_inspects(self):
        p=self.run_cli("validate",str(FIXTURE))
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)["status"],"pass")
        p=self.run_cli("inspect",str(FIXTURE))
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn("qsa-select-synthetic",p.stdout)

    def test_cpu_replay_matches_independent_golden_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.run_cli("replay","--fixture",str(FIXTURE),"--backend","cpu","--output",d)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            report=json.loads((Path(d)/"comparison.json").read_text())
            self.assertEqual(report["status"],"pass")
            self.assertEqual(report["mismatch_count"],0)
            self.assertEqual(report["selected_ids_compared"],4*512)

    def test_max_bound_fixture_covers_empty_row_and_replay(self):
        fixture=ROOT/"tests/data/xrv/qsa-select-max-bound"
        p=self.run_cli("validate",str(fixture))
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertIn("verified-exact",p.stdout)
        with tempfile.TemporaryDirectory() as d:
            p=self.run_cli("replay","--fixture",str(fixture),"--backend","cpu","--output",d)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            report=json.loads((Path(d)/"comparison.json").read_text())
            self.assertEqual(report["status"],"pass")
            self.assertEqual(report["selected_ids_compared"],2*512)
            self.assertEqual(report["sample_mismatches"],[])

    def test_intentional_output_failure_is_reported_and_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.run_cli("replay","--fixture",str(FIXTURE),"--backend","cpu","--output",d)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            path=Path(d)/"selected_ids.i32le"
            data=bytearray(path.read_bytes()); data[0:4]=(123456).to_bytes(4,"little",signed=True); path.write_bytes(data)
            result_path=Path(d)/"result.json"; result=json.loads(result_path.read_text())
            import hashlib
            result["output_sha256"]["selected_ids"]=hashlib.sha256(path.read_bytes()).hexdigest()
            result_path.write_text(json.dumps(result))
            p=self.run_cli("compare","--fixture",str(FIXTURE),"--result",d)
            self.assertEqual(p.returncode,1,p.stdout+p.stderr)
            report=json.loads(p.stdout)
            self.assertEqual(report["status"],"fail")
            self.assertEqual(report["first_mismatch"]["tensor"],"selected_ids")
            self.assertGreater(report["mismatch_count"],0)

    def test_compare_rejects_result_from_another_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.run_cli("replay","--fixture",str(FIXTURE),"--backend","cpu","--output",d)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            other=ROOT/"tests/data/xrv/qsa-select-max-bound"
            p=self.run_cli("compare","--fixture",str(other),"--result",d)
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)
            self.assertIn("result envelope fixture_id mismatch",p.stderr)

    def test_compare_rejects_tampered_result_tensor_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.run_cli("replay","--fixture",str(FIXTURE),"--backend","cpu","--output",d)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            path=Path(d)/"selected_ids.i32le"
            data=bytearray(path.read_bytes()); data[0]^=1; path.write_bytes(data)
            p=self.run_cli("compare","--fixture",str(FIXTURE),"--result",d)
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)
            self.assertIn("result output SHA-256 mismatch",p.stderr)

    def test_checksum_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/"fixture"; shutil.copytree(FIXTURE,copy)
            path=copy/"scores.u32le"; data=bytearray(path.read_bytes()); data[0]^=1; path.write_bytes(data)
            p=self.run_cli("validate",str(copy))
            self.assertEqual(p.returncode,2)
            self.assertIn("SHA-256 mismatch",p.stderr)

    def test_replay_refuses_a_rehashed_but_wrong_golden(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/"fixture"; shutil.copytree(FIXTURE,copy)
            path=copy/"expected_ids.i32le"; data=bytearray(path.read_bytes()); data[0:4]=(777).to_bytes(4,"little",signed=True); path.write_bytes(data)
            m=json.loads((copy/"manifest.json").read_text())
            import hashlib
            m["tensors"]["expected_ids"]["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            (copy/"manifest.json").write_text(json.dumps(m))
            p=self.run_cli("replay","--fixture",str(copy),"--backend","cpu","--output",str(Path(d)/"result"))
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)
            self.assertIn("independent golden does not match host reference",p.stderr)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/"fixture"; shutil.copytree(FIXTURE,copy)
            m=json.loads((copy/"manifest.json").read_text()); m["tensors"]["scores"]["file"]="../escape.bin"
            (copy/"manifest.json").write_text(json.dumps(m))
            p=self.run_cli("validate",str(copy))
            self.assertEqual(p.returncode,2)
            self.assertIn("unsafe tensor path",p.stderr)

    def test_non_finite_scores_are_rejected_after_hash_update(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/"fixture"; shutil.copytree(FIXTURE,copy)
            path=copy/"scores.u32le"; data=bytearray(path.read_bytes()); data[0:4]=(0x7fc00000).to_bytes(4,"little"); path.write_bytes(data)
            m=json.loads((copy/"manifest.json").read_text())
            import hashlib
            m["tensors"]["scores"]["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            (copy/"manifest.json").write_text(json.dumps(m))
            p=self.run_cli("validate",str(copy))
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)
            self.assertIn("NaN or infinity",p.stderr)

    def test_validate_recomputes_golden_from_input(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/"fixture"; shutil.copytree(FIXTURE,copy)
            path=copy/"expected_ids.i32le"; data=bytearray(path.read_bytes()); data[0:4]=(777).to_bytes(4,"little",signed=True); path.write_bytes(data)
            m=json.loads((copy/"manifest.json").read_text())
            import hashlib
            m["tensors"]["expected_ids"]["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            (copy/"manifest.json").write_text(json.dumps(m))
            p=self.run_cli("validate",str(copy))
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)
            self.assertIn("independent golden does not match host reference",p.stderr)

    def test_ab_runs_two_adapters_on_identical_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            adapter=Path(d)/"fake-xpu-adapter.py"
            adapter.write_text("""#!/usr/bin/env python3
import json, shutil, sys
from pathlib import Path
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
out=Path(args['--output'])
fixture=Path(%r)
shutil.copyfile(fixture/'expected_ids.i32le',out/'selected_ids.i32le')
shutil.copyfile(fixture/'expected_counts.i32le',out/'selected_counts.i32le')
print(json.dumps({'hardware':{'device':'test-double','driver':'none'},'runtime':{'backend':'mock'},'timing':{'device_event_ms':{'median':1.0,'p90':1.1,'p99':1.2}}}))
""" % str(FIXTURE))
            adapter.chmod(0o755)
            out=Path(d)/"ab"
            p=self.run_cli("ab","--fixture",str(FIXTURE),"--baseline-adapter",str(adapter),"--candidate-adapter",str(adapter),
                           "--output",str(out),"--warmup","0","--repeat","3")
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            report=json.loads((out/"ab.json").read_text())
            self.assertEqual(report["correctness"]["baseline"],"pass")
            self.assertEqual(report["correctness"]["candidate"],"pass")
            self.assertFalse(report["timing"]["timing_comparable"])
            self.assertIsNone(report["timing"]["candidate_speedup"])
            self.assertIn("workload isolation was not attested",report["timing"]["workload_isolation_reason"])

    def test_ab_requires_caller_attested_idle_workload(self):
        with tempfile.TemporaryDirectory() as d:
            adapter=Path(d)/"isolated-adapter.py"
            adapter.write_text("""#!/usr/bin/env python3
import json, shutil, sys
from pathlib import Path
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
out=Path(args['--output'])
fixture=Path(%r)
shutil.copyfile(fixture/'expected_ids.i32le',out/'selected_ids.i32le')
shutil.copyfile(fixture/'expected_counts.i32le',out/'selected_counts.i32le')
print(json.dumps({'hardware':{'device':'test-double','driver':'none'},'runtime':{'backend':'mock'},
'workload_isolation':{'status':'caller-attested-idle','evidence':'process-check+exclusive-lock-attested'},
'timing':{'device_event_ms':{'median':1.0}}}))
""" % str(FIXTURE))
            adapter.chmod(0o755)
            out=Path(d)/"ab"
            p=self.run_cli("ab","--fixture",str(FIXTURE),"--baseline-adapter",str(adapter),"--candidate-adapter",str(adapter),
                           "--output",str(out),"--warmup","0","--repeat","2")
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            report=json.loads((out/"ab.json").read_text())
            self.assertTrue(report["timing"]["timing_comparable"])
            self.assertEqual(report["timing"]["candidate_speedup"],1.0)

    def test_capture_command_template_redacts_local_paths_and_prompt_text(self):
        command=["./build/xenithium-probe","model","--model",r"C:\Users\FixtureAccount\private-model","--ppl",r"C:\Users\FixtureAccount\private-corpus.bin","--prompt","private prompt contents","--context","131072","--prompt-file",r"C:\Users\FixtureAccount\secret-prompt.txt","private positional data"]
        safe=xrv.sanitized_command(command)
        rendered=" ".join(safe)
        self.assertIn("<MODEL_DIR>",rendered)
        self.assertIn("<REDACTED>",rendered)
        self.assertNotIn("FixtureAccount",rendered)
        self.assertNotIn("private prompt contents",rendered)
        self.assertNotIn("secret-prompt.txt",rendered)
        self.assertNotIn("private-corpus.bin",rendered)
        self.assertNotIn("private positional data",rendered)

    def test_capture_provenance_distinguishes_diagnostic_from_prompt_derived_state(self):
        diagnostic=xrv._capture_input_origin(["./build/xenithium-probe","qsa","--model","/private/model"])
        model=xrv._capture_input_origin(["./build/xenithium-probe","model","--ppl","/private/corpus"])
        self.assertFalse(diagnostic["prompt_derived_state"])
        self.assertEqual(diagnostic["kind"],"qsa-diagnostic-probe")
        self.assertTrue(model["prompt_derived_state"])
        self.assertEqual(model["kind"],"full-model-probe")

    def test_capture_wrapper_writes_self_consistent_fixture_and_redacts_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); child=root/"xenithium-probe"; output=root/"captured"
            child.write_text("""import json, os, struct, sys
from pathlib import Path
sys.path.insert(0, %r)
import xrv
out=Path(os.environ['XEN_XRV_CAPTURE_DIR'])/'capture-000001'
out.mkdir(parents=True)
scores=[struct.unpack('<I',struct.pack('<f',float((i*7)%%13)))[0] for i in range(512)]
nblk=[5]
ids,counts=xrv.reference_select(scores,nblk,1,512,512)
ids[5:]=[987654]*507
(out/'scores.u32le').write_bytes(xrv.pack_values('uint32',scores))
(out/'nblk.i32le').write_bytes(xrv.pack_values('int32',nblk))
(out/'candidate_ids.i32le').write_bytes(xrv.pack_values('int32',ids))
(out/'candidate_counts.i32le').write_bytes(xrv.pack_values('int32',counts))
(out/'capture.json').write_text(json.dumps({'rows':1,'stride':512,'source_stride':512,'bound':512,'chunk':0,'tp_rank':0,'tp_size':1,'runtime_fingerprint':{'device':'mock','driver':'none'},'engine_revision':'test'}))
""" % str(ROOT/"tools/xrv"))
            p=self.run_cli("capture","--kernel","qsa.select","--output",str(output),"--fixture-id","capture-test","--tp-rank","0","--tp-size","1",
                           "--model-fingerprint","a"*64,"--",sys.executable,str(child),"model","--model","/private/local/model","--ppl",str(root/"private-corpus.bin"),"--prompt","private prompt contents")
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            manifest=json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["capture_validation"]["candidate_vs_reference"],"pass")
            self.assertFalse(manifest["provenance"]["prompt_tokens_included"])
            self.assertFalse(manifest["provenance"]["weights_included"])
            self.assertTrue(manifest["provenance"]["derived_from_prompt_or_model_state"])
            self.assertEqual(manifest["provenance"]["kind"],"kernel-boundary-capture")
            self.assertEqual(manifest["provenance"]["input_origin_kind"],"full-model-probe")
            self.assertTrue(manifest["provenance"]["prompt_derived_state"])
            self.assertNotIn("private-corpus.bin"," ".join(manifest["provenance"]["capture_command_template"]))
            captured=xrv.unpack_values("int32",(output/"captured_candidate_ids.i32le").read_bytes())
            self.assertEqual(list(captured[5:]),[-1]*507)
            self.assertEqual(len(manifest["provenance"]["capture_command_sha256"]),64)
            self.assertNotIn("prompt_or_weight_data_included",manifest["provenance"])
            template=" ".join(manifest["provenance"]["capture_command_template"])
            self.assertNotIn("/private/local/model",template)
            self.assertNotIn("private prompt contents",template)
            self.assertIn("<MODEL_DIR>",template)
            self.assertIn("<REDACTED>",template)
            p=self.run_cli("validate",str(output))
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)

    def test_capture_requires_explicit_child_command(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.run_cli("capture","--output",str(Path(d)/"fixture"))
            self.assertEqual(p.returncode,2)
            self.assertIn("explicit GPU/model command",p.stderr)

    def test_capture_refuses_in_repository_output_path(self):
        output=ROOT/"tests/data/xrv/private-capture-must-not-land"
        p=self.run_cli("capture","--kernel","qsa.sparse_attention.k8v4","--output",str(output),"--",sys.executable,"-c","pass")
        self.assertEqual(p.returncode,2,p.stdout+p.stderr)
        self.assertIn("outside the repository",p.stderr)
        self.assertFalse(output.exists())

    def test_cpu_bench_is_labeled_as_host_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.run_cli("bench","--fixture",str(FIXTURE),"--backend","cpu","--output",d,"--warmup","1","--repeat","3")
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            result=json.loads((Path(d)/"result.json").read_text())
            self.assertEqual(result["timing"]["measurement_kind"],"portable_cpu_algorithm")
            self.assertIsNone(result["timing"]["device_event_ms"])
            self.assertIn("p90",result["timing"]["host_wall_ms"])
            self.assertTrue((Path(d)/"bench.md").is_file())
            self.assertIn("host-only",(Path(d)/"bench.md").read_text())


    def test_attention_reference_sigmoid_handles_extreme_finite_gates(self):
        import qsa_attention
        self.assertEqual(qsa_attention._stable_sigmoid(-1000.0),0.0)
        self.assertEqual(qsa_attention._stable_sigmoid(1000.0),1.0)
        self.assertAlmostEqual(qsa_attention._stable_sigmoid(0.0),0.5)

    def test_k8v4_attention_fixture_preserves_packed_inputs_and_validates_reference(self):
        import qsa_attention
        fixture=ROOT/"tests/data/xrv/qsa-k8v4-sparse-synthetic"
        p=self.run_cli("validate",str(fixture))
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        manifest=json.loads((fixture/"manifest.json").read_text())
        self.assertEqual(manifest["schema_version"],2)
        self.assertEqual(manifest["quantization"]["kv"],"K8V4")
        self.assertEqual(manifest["tensors"]["keys"]["dtype"],"uint8")
        self.assertEqual(manifest["tensors"]["values"]["dtype"],"uint8")
        self.assertEqual(manifest["operation"]["rows"],2)
        self.assertEqual(list(qsa_attention.read_tensor(fixture,manifest,"positions")),[2053,30])
        self.assertEqual(manifest["mutable_state"],[])
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"reference"
            p=self.run_cli("replay","--fixture",str(fixture),"--backend","cpu","--output",str(out))
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            result=json.loads((out/"result.json").read_text())
            self.assertEqual(result["backend"],"cpu-independent-reference")
            self.assertIsNone(result["timing"]["device_event_ms"])
            self.assertIn("no candidate kernel executed",result["execution"]["measurement_scope"])
            self.assertEqual(json.loads((out/"comparison.json").read_text())["status"],"pass")

    def test_capture_k8v4_attention_creates_private_fixture_and_checks_candidate(self):
        fixture=ROOT/"tests/data/xrv/qsa-k8v4-sparse-synthetic"
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); child=root/"xenithium-probe"; output=root/"captured-attention"
            child.write_text(f"""import json, os, shutil
from pathlib import Path
fixture=Path({str(fixture)!r})
out=Path(os.environ["XEN_XRV_CAPTURE_DIR"])/"capture-000001"
out.mkdir(parents=True)
manifest=json.loads((fixture/"manifest.json").read_text())
files={{"query":"query.u16le","gate":"gate.u16le","positions":"positions.i32le","slots":"slots.i32le",
"page_table":"page_table.i32le","keys":"keys.u8","values":"values.u8","key_scales":"key_scales.f32le",
"value_scales":"value_scales.u16le","selected_ids":"selected_ids.i32le","selected_counts":"selected_counts.i32le"}}
for name,filename in files.items(): shutil.copyfile(fixture/manifest["tensors"][name]["file"],out/filename)
shutil.copyfile(fixture/manifest["tensors"]["expected_output"]["file"],out/"candidate_output.u16le")
op=manifest["operation"]
meta={{"kernel":"qsa.sparse_attention.k8v4","rows":op["rows"],"q_heads":op["q_heads"],"kv_heads":op["kv_heads"],
"cards":op["cards"],"tp_rank":0,"tp_size":2,"capacity":op["capacity"],"pool_pages":op["pool_pages"],
"table_slots":op["table_slots"],"slot_pages":op["slot_pages"],"tiles_per_group":1,"source_projection_stride":op["q_heads"]//op["cards"]*2*256+2*(op["kv_heads"]//op["cards"])*256,
"runtime_fingerprint":{{"backend":"SYCL","device":"test-double","driver":"none"}},"engine_revision":"test","captured_utc":"2026-11-23T00:00:00Z"}}
(out/"capture.json").write_text(json.dumps(meta))
""")
            p=self.run_cli("capture","--kernel","qsa.sparse_attention.k8v4","--output",str(output),"--fixture-id","attention-capture-test",
                           "--tp-rank","0","--tp-size","2","--",sys.executable,str(child),"model","--ppl",str(root/"private-corpus.bin"))
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            manifest=json.loads((output/"manifest.json").read_text())
            report=json.loads((output/"capture_validation.json").read_text())
            self.assertEqual(manifest["schema_version"],2)
            self.assertEqual(manifest["reference"]["independent"],True)
            self.assertTrue(manifest["provenance"]["sensitive_capture"])
            self.assertTrue(manifest["provenance"]["prompt_derived_state"])
            self.assertEqual(manifest["provenance"]["input_origin_kind"],"full-model-probe")
            self.assertNotIn("private-corpus.bin"," ".join(manifest["provenance"]["capture_command_template"]))
            self.assertEqual(report["candidate_vs_independent_reference"],"pass")
            self.assertTrue((output/"capture_observation/comparison.json").is_file())
            p=self.run_cli("validate",str(output))
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)

    def test_k8v4_attention_reports_intentional_numeric_failure(self):
        import hashlib
        fixture=ROOT/"tests/data/xrv/qsa-k8v4-sparse-synthetic"
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"candidate"
            p=self.run_cli("replay","--fixture",str(fixture),"--backend","cpu","--output",str(out))
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            path=out/"output.u16le"
            data=bytearray(path.read_bytes()); data[0:2]=(0x7f7f).to_bytes(2,"little"); path.write_bytes(data)
            result=json.loads((out/"result.json").read_text())
            result["output_sha256"]["output"]=hashlib.sha256(data).hexdigest()
            (out/"result.json").write_text(json.dumps(result))
            p=self.run_cli("compare","--fixture",str(fixture),"--result",str(out))
            self.assertEqual(p.returncode,1,p.stdout+p.stderr)
            report=json.loads(p.stdout)
            self.assertEqual(report["status"],"fail")
            self.assertGreater(report["bf16_bitwise_mismatch_count"],0)
            self.assertGreater(report["tolerance_violation_count"],0)
            self.assertIsNotNone(report["first_tolerance_violation"])
            self.assertIn("max_absolute_error",report)

    def test_k8v4_attention_rejects_non_contiguous_stride_contract(self):
        fixture=ROOT/"tests/data/xrv/qsa-k8v4-sparse-synthetic"
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/"fixture"; shutil.copytree(fixture,copy)
            m=json.loads((copy/"manifest.json").read_text())
            m["tensors"]["query"]["strides"]=[1,256,3072]
            (copy/"manifest.json").write_text(json.dumps(m))
            p=self.run_cli("validate",str(copy))
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)
            self.assertIn("unsupported non-contiguous strides",p.stderr)

    def test_k8v4_attention_rejects_corrupt_packed_scale_data(self):
        import hashlib
        fixture=ROOT/"tests/data/xrv/qsa-k8v4-sparse-synthetic"
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/"fixture"; shutil.copytree(fixture,copy)
            path=copy/"keys.u8"; data=bytearray(path.read_bytes()); data[0]=1; path.write_bytes(data)
            m=json.loads((copy/"manifest.json").read_text())
            m["tensors"]["keys"]["sha256"]=hashlib.sha256(data).hexdigest()
            (copy/"manifest.json").write_text(json.dumps(m))
            p=self.run_cli("validate",str(copy))
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)
            self.assertIn("zero K scale requires",p.stderr)

    def test_k8v4_ab_binds_fixture_fingerprint_and_keeps_timing_samples(self):
        fixture=ROOT/"tests/data/xrv/qsa-k8v4-sparse-synthetic"
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            expected=(fixture/"expected_output.u16le")
            # Use the fixture manifest to select the independent golden without invoking any GPU code.
            manifest=json.loads((fixture/"manifest.json").read_text())
            expected=fixture/manifest["tensors"]["expected_output"]["file"]
            adapters=[]
            for label,median_ms in (("baseline",2.0),("candidate",1.0)):
                adapter=root/("adapter-"+label)
                adapter.write_text(f"""#!/usr/bin/env python3
import json,shutil,sys
from pathlib import Path
fixture=Path({str(fixture)!r})
out=Path(sys.argv[sys.argv.index("--output")+1])
out.mkdir(parents=True,exist_ok=True)
shutil.copyfile(Path({str(expected)!r}),out/"output.u16le")
print(json.dumps({{"timing":{{"device_event_ms":{{"median":{median_ms},"p90":{median_ms},"p99":{median_ms},"jitter_max_minus_min":0.0,"samples":[{median_ms},{median_ms}]}},
"host_wall_ms":{{"median":{median_ms+0.2},"p90":{median_ms+0.2},"p99":{median_ms+0.2},"jitter_max_minus_min":0.0,"samples":[{median_ms+0.2},{median_ms+0.2}]}}}},
"hardware":{{"device":"test-device","driver":"test-driver"}},
"runtime":{{"backend":"SYCL","compiler":"IntelLLVM"}},
"workload_isolation":{{"status":"caller-attested-idle","evidence":"process-check+exclusive-lock-attested"}}}}))
""")
                adapter.chmod(0o755)
                adapters.append(adapter)
            output=root/"ab"
            p=self.run_cli("ab","--fixture",str(fixture),"--baseline-adapter",str(adapters[0]),
                           "--candidate-adapter",str(adapters[1]),"--warmup","2","--repeat","2","--output",str(output))
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            report=json.loads((output/"ab.json").read_text())
            self.assertEqual(report["status"],"pass")
            self.assertTrue(report["timing"]["timing_comparable"])
            self.assertEqual(report["fixture_manifest_sha256"],xrv.sha256(fixture/"manifest.json"))
            self.assertEqual((report["warmup"],report["repeat"]),(2,2))
            self.assertEqual(report["correctness"]["acceptance"],"declared K8V4 numeric tolerance")
            self.assertEqual(report["correctness"]["baseline_bf16_bitwise_mismatch_count"],0)
            self.assertEqual(report["baseline_timing"]["device_event_ms"]["samples"],[2.0,2.0])
            self.assertEqual(report["timing"]["candidate_speedup"],2.0)

    def test_k8v4_attention_cpu_bench_does_not_claim_reference_as_kernel(self):
        fixture=ROOT/"tests/data/xrv/qsa-k8v4-sparse-synthetic"
        with tempfile.TemporaryDirectory() as d:
            p=self.run_cli("bench","--fixture",str(fixture),"--backend","cpu","--output",d)
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)
            self.assertIn("not a candidate attention kernel",p.stderr)

if __name__ == "__main__":
    unittest.main()
