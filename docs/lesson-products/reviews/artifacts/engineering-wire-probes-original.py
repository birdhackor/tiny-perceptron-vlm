"""真TorchVersion及Modal1.6 wire序列化；實際execute_remote控制流的明示離線替身。"""
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from scripts.lesson_products import checkpoints, modal_runner as runner
import torch

SDK_CHILD = r'''
import hashlib, importlib.metadata, importlib.util, json, pickle, sys
from pathlib import Path
import modal._serialization as wire
mode, directory = sys.argv[1], Path(sys.argv[2])
assert importlib.util.find_spec("torch") is None
result = {"modal_version": importlib.metadata.version("modal"), "initial_torch_available": False, "serialization_sha256": hashlib.sha256(Path(wire.__file__).read_bytes()).hexdigest(), "mode": mode}
if mode == "encode":
    sys.path.append(sys.argv[3])  # producer alone loads the actual installed torch class
    for name in ("original", "normalized"):
        value = pickle.loads((directory / (name + ".pickle")).read_bytes())
        (directory / (name + ".modal")).write_bytes(wire.serialize(value))
    result["producer_torch_version_type"] = type(pickle.loads((directory / "original.pickle").read_bytes())["receipt"]["training"]["environment"]["torch"]).__module__ + ".TorchVersion"
else:
    try:
        wire.deserialize((directory / "original.modal").read_bytes(), None)
    except Exception as error:
        result["original_error"] = {"type": type(error).__name__, "message": str(error), "cause_type": type(error.__cause__).__name__, "missing_module": getattr(error.__cause__, "name", None)}
    else: raise AssertionError("original unexpectedly loads without torch")
    assert result["original_error"]["type"] == "DeserializationError"
    assert result["original_error"]["missing_module"] == "torch"
    normalized = wire.deserialize((directory / "normalized.modal").read_bytes(), None)
    assert type(normalized["receipt"]["training"]["environment"]["torch"]) is str
    for name, raw in normalized["artifacts"].items():
        expected = normalized["receipt"]["client_artifacts"][name]
        assert type(raw) is bytes and len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["sha256"]
    result["normalized_status"] = normalized["receipt"]["status"]
    result["normalized_artifact_count"] = len(normalized["artifacts"])
    result["receipt_canonical_sha256"] = hashlib.sha256(json.dumps(normalized["receipt"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    result["torch_loaded"] = "torch" in sys.modules
    assert result["torch_loaded"] is False and importlib.util.find_spec("torch") is None
print(json.dumps(result))
'''


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def builtin_json(value):
    if type(value) is dict:
        return all(type(key) is str and builtin_json(item) for key, item in value.items())
    if type(value) is list: return all(builtin_json(item) for item in value)
    return type(value) in (str, int, float, bool, type(None))


def main():
    started = datetime.now(UTC).isoformat()
    checked_names = ["scripts/lesson_products/modal_runner.py", "scripts/lesson_products/checkpoints.py", "scripts/selftrained/modal_runner.py", "scripts/selftrained/finance.py", "tiny_perceptron/lesson_products/train.py", "tiny_perceptron/lesson_products/training.py", "tiny_perceptron/lesson_products/data.py", "tiny_perceptron/lesson_products/model.py", "docs/lesson-products/jobs/train-b-v1.json", runner.POLICY_PATH]
    before = {name: sha(ROOT/name) for name in checked_names}
    old = ART / "engineering-snapshots/f5201ddfc1d4f8157f623c0fe977dfc66f5ddfbfc61010c4e922fbea7f69ca25.py"
    assert sha(old) == "f5201ddfc1d4f8157f623c0fe977dfc66f5ddfbfc61010c4e922fbea7f69ca25"
    delta = '        # TorchVersion等str子類也可能要求接收端import torch；跨程序只回傳JSON基本型別。\n        receipt = json.loads(json.dumps(receipt, ensure_ascii=False, allow_nan=False))\n'
    assert (ROOT/checked_names[0]).read_text().replace(delta, "") == old.read_text()
    classes_path = ART / "engineering-runner-probes-original.py"
    spec = importlib.util.spec_from_file_location("wire_fake_classes", classes_path)
    classes = importlib.util.module_from_spec(spec); spec.loader.exec_module(classes)
    version = torch.__version__
    assert type(version).__module__ == "torch.torch_version" and type(version) is not str
    results = []
    with tempfile.TemporaryDirectory(dir=ROOT/"work", prefix="wire-audit-") as folder:
        work = Path(folder)
        sdk_script = work/"sdk_child.py"; sdk_script.write_text(SDK_CHILD)
        envelopes = []
        for kind in ("completed", "failed_after_periodic_backup"):
            case = work/kind; case.mkdir()
            app = classes.FakeApp(); events=[]; uploads=[]
            volume = SimpleNamespace(reload=lambda:events.append("reload"), commit=lambda:events.append("commit"))
            sdk = SimpleNamespace(App=lambda _:app, Volume=SimpleNamespace(from_name=lambda *a,**k:volume), Secret=SimpleNamespace(from_name=lambda *a,**k:"synthetic"), Image=SimpleNamespace(debian_slim=lambda **k:classes.FakeImage()), current_function_call_id=lambda:"fc-synthetic-wire")
            with patch.dict(sys.modules,{"modal":sdk}): _, _, execute, _ = runner.register("run",case/"data")
            job=json.loads((ROOT/"docs/lesson-products/jobs/train-b-v1.json").read_bytes())
            hashes={name:sha(ROOT/name) for name in checkpoints.TRAIN_SOURCES}; hashes[runner.DATA_MANIFEST]=sha(ROOT/runner.DATA_MANIFEST)
            entry={"run_id":"synthetic-wire", "status":"reserved", "experiment_id":"selftrained_lesson_products", "job_sha256":runner.canonical_sha(job), "source_sha256":hashes, "reserved_usd":".10", "gross_quota_guard":{"synthetic":True}}
            book={"reserved_total_usd":"10.10", "reservations":[{"run_id":"history", "status":"completed", "reserved_usd":"10.00"},entry]}
            old_entry=copy.deepcopy(book["reservations"][0]); budget=case/"course/budget.json"; runner.write_json(budget,book)
            data=case/"input-data/manifest.json";data.parent.mkdir();data.write_bytes((ROOT/runner.DATA_MANIFEST).read_bytes())
            out=case/"course/lesson-products/synthetic-wire";out.mkdir(parents=True)
            runner.write_json(out/"reservation.json",{"synthetic_preflight_completed":True})
            training={"last_step":1000,"selected_step":1000,"environment":{"torch":version,"device":"synthetic CUDA signal; no actual GPU"},"scope":"Explicit synthetic stub, no optimizer/model operation"}
            def train(data, directory, *args, **kwargs):
                assert directory==out
                for name in checkpoints.FILES:
                    (directory/name).write_bytes(("explicit synthetic "+name).encode())
                (directory/"model.pt").write_bytes(b"explicit synthetic model; no weights")
                runner.write_json(directory/"progress.json",{"step":1000})
                runner.write_json(directory/"training-receipt.json",training)
                kwargs["on_checkpoint"](directory/"last.pt",1000)
                if kind=="failed_after_periodic_backup":raise RuntimeError("synthetic interruption after complete periodic backup")
                return training
            class API:
                def __init__(self,token):assert token=="SYNTHETIC_WIRE_NOT_REAL_TOKEN"
                def upload_folder(self,**kwargs):
                    allowed=kwargs["allow_patterns"]
                    assert all(name in allowed for name in (*checkpoints.FILES,"checkpoint-bundle.json"))
                    uploads.append({"allow_patterns":allowed,"path_in_repo":kwargs["path_in_repo"],"files":{name:sha(out/name) for name in checkpoints.FILES}})
                    return SimpleNamespace(oid="c"*40)
            def mapped(value):
                value=str(value)
                return case/value.lstrip("/") if value.startswith(("/course","/input-data")) else Path(value)
            fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda _:"synthetic L4"))
            with patch.dict(sys.modules,{"modal":sdk,"torch":fake_torch,"huggingface_hub":SimpleNamespace(HfApi=API),"tiny_perceptron.lesson_products.train":SimpleNamespace(train=train)}), patch.dict(os.environ,{"HF_TOKEN":"SYNTHETIC_WIRE_NOT_REAL_TOKEN"}), patch.object(runner,"Path",side_effect=mapped), patch.object(runner,"LEDGER",budget), patch.object(runner,"check_source_files",return_value=None):
                envelope=execute("synthetic-wire","b"*40,job,hashes,"synthetic/private")
            receipt=envelope["receipt"]; terminal="completed" if kind=="completed" else "failed"
            assert receipt["status"]==terminal and builtin_json(receipt)
            assert all(type(raw) is bytes for raw in envelope["artifacts"].values())
            after=json.loads(budget.read_bytes())
            assert after["reservations"][-1]["status"]==terminal
            assert after["reservations"][-1]["receipt_sha256"]==runner.canonical_sha(receipt)
            assert after["reservations"][0]==old_entry and after["reserved_total_usd"]=="10.10"
            assert json.loads((out/"execution.json").read_bytes())==receipt
            assert all((out/name).read_bytes()==raw for name,raw in envelope["artifacts"].items())
            checkpoints.save_transferred(envelope,case/"client")
            assert len(uploads)==(2 if kind=="completed" else 1)
            assert events.count("commit")>=3
            if kind=="completed":
                assert type(receipt["training"]["environment"]["torch"]) is str
                original=copy.deepcopy(envelope);original["receipt"]["training"]=training
                assert type(original["receipt"]["training"]["environment"]["torch"]) is type(version)
                assert runner.canonical_sha(original["receipt"])==runner.canonical_sha(receipt)
                assert original["artifacts"]==envelope["artifacts"]
                envelopes=[original,envelope]
            results.append({"id":"execute_remote_"+kind,"passed":True,"status":receipt["status"],"artifact_count":len(envelope["artifacts"]),"artifacts_unchanged":True,"receipt_builtin_json_only":True,"ledger_receipt_sha256":after["reservations"][-1]["receipt_sha256"],"prior_entry_and_total_retained":True,"commits":events.count("commit"),"hf_paired_backup_calls":uploads,"scope":"Real current nested execute_remote function; synthetic CUDA availability and stub trainer/HF, zero GPU/model/paid operations."})
        for name,envelope in zip(("original","normalized"),envelopes): (work/(name+".pickle")).write_bytes(pickle.dumps(envelope,protocol=5))
        sdk_results=[]
        for mode in ("encode","decode"):
            command=["uv","run","--isolated","--no-project","--python","/usr/bin/python3","--with","modal==1.6.0","python",str(sdk_script),mode,str(work)]
            if mode=="encode": command.append(str(Path(torch.__file__).parent.parent))
            run=subprocess.run(command,capture_output=True,text=True)
            assert run.returncode==0, run.stdout+run.stderr
            observation=json.loads(run.stdout.strip());sdk_results.append(observation)
        assert sdk_results[-1]["receipt_canonical_sha256"]==runner.canonical_sha(envelopes[-1]["receipt"])
        results.append({"id":"real_modal16_torchfree_deserialization","passed":True,"torch_version_value":str(version),"torch_version_original_type":type(version).__module__+"."+type(version).__name__,"sdk_results":sdk_results,"canonical_sha_unchanged":True,"binary_artifacts_unchanged":True,"production_transport_claim":False})
    assert before=={name:sha(ROOT/name) for name in checked_names}
    report={"reviewer_task":"/root/tech_20261016_review_tools","context":"informed engineering wire original","started_at":started,"completed_at":datetime.now(UTC).isoformat(),"environment":{"executable":sys.executable,"python":sys.version,"platform":platform.platform(),"torch_version":str(version)},"checked_files":[{"path":name,"sha256":digest} for name,digest in before.items()],"results":results,"all_expected":True,"exact_delta":"Only one JSON normalization statement plus its explanatory comment inserted after transfer_payload, before write_json/finish_entry/RPC.","limits":["Actual installed CPU TorchVersion and Modal1.6 SDK serializer used; SDK consumer isolated without torch. No cloud calls/containers/GPU/model optimizer/training/HF writes.","Synthetic execution function trainer uses a fake CUDA signal only to enter real orchestration branches, never claims genuine GPU or model results.","Tested finite JSON-compatible receipt values plus TorchVersion str subclass; allow_nan=False deliberately rejects nonfinite/non-JSON receipts, not arbitrary custom objects."]}
    path=ART/"engineering-wire-probes-results-original.json"
    with path.open("x") as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write("\n")
    print(json.dumps({"cases":len(results),"all_expected":True,"receipt_sha256":sha(path),"torchfree_original_error":sdk_results[-1]["original_error"],"normalized_status":sdk_results[-1]["normalized_status"]}))


if __name__=="__main__":main()
