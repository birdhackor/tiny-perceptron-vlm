"""19.16 版本邊界核對；不載入成品權重，不訓練，不評估語義品質。"""

import difflib
import hashlib
import json
import platform
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.course_product import dispatch_request
from tiny_perceptron.decision_product.runtime import create_model
from tiny_perceptron.decision_product.service import DeviceSession, Prediction
from tiny_perceptron.lesson_products.data import CANDIDATES
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, Tokenizer, candidate_batch
from tiny_perceptron.omni_products.choice import Candidate, ChoiceModel, Question, collate_questions

ROOT = Path(__file__).resolve().parents[4]
COMMAND = (
    "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv "
    "PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 "
    "uv run --extra cpu --no-sync python "
    "docs/technical-reviews/artifacts/c-v3-v21-canonical-19-16-fresh/interface_probe.py"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lesson(text, identifier):
    return re.search(r"^## " + re.escape(identifier) + r" .*?(?=^## |\Z)", text, re.M | re.S)[0]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    freeze21_path = ROOT / "docs/course-revision-20261101-c-v3/source-freeze-v21.json"
    freeze18_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json"
    assert digest(freeze21_path) == "d9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea"
    assert digest(freeze18_path) == "d74eb6e0fc958fbd410c7588dbe4d96a7ced72ff512c7ea8f8863c87f68c9108"
    freeze21 = json.loads(freeze21_path.read_text())
    freeze18 = json.loads(freeze18_path.read_text())
    p21 = next(p for p in freeze21["pages"] if p["page_id"] == "19.16")
    p18 = next(p for p in freeze18["pages"] if p["page_id"] == "19.16")
    new = (ROOT / p21["snapshot"]).read_text()
    old = (ROOT / p18["snapshot"]).read_text()
    assert digest(ROOT / p21["snapshot"]) == p21["source_sha256"]
    assert digest(ROOT / p18["snapshot"]) == p18["source_sha256"]
    insertion = "本節教原版文字C的三型介面；[19.23](19.md#19.23)再教C-v2共享材料，[19.25](19.md#19.25)才是目前推薦的typed裝置C-v3。三者介面與權重不可互換。\n\n"
    assert new.replace(insertion, "", 1) == old
    live = (ROOT / "course/chapters/19.md").read_text()
    assert lesson(live, "19.16") == new
    navigation = {}
    for identifier in ("19.23", "19.25"):
        row = next(p for p in freeze21["pages"] if p["page_id"] == identifier)
        raw = (ROOT / row["snapshot"]).read_text()
        assert hashlib.sha256(raw.encode()).hexdigest() == row["source_sha256"]
        assert lesson(live, identifier) == raw
        navigation[identifier] = {"snapshot": row["snapshot"], "sha256": row["source_sha256"], "heading": raw.splitlines()[0]}

    tokenizer = Tokenizer()
    legacy = CandidateScorer(Config(vocab_size=len(tokenizer), width=64, ff=128)).eval()
    old_interfaces = {}
    for kind, candidates in CANDIDATES.items():
        row = {"state": "卡片進度;申請編號=?", "question": kind, "candidates": list(candidates)}
        tokens, slices, maximum, labels = candidate_batch([row], tokenizer, legacy.config)
        assert slices == [(0, len(candidates))] and labels.numel() == 0
        old_interfaces[kind] = {"candidate_ids": list(candidates), "tokens_shape": list(tokens.shape), "slices": slices, "max_candidates": maximum}

    c2 = ChoiceModel().eval()
    question = Question("共同材料", "choice", "請選服務", (Candidate("x", "服務甲"), Candidate("y", "服務乙")))
    batch = collate_questions([question])
    state_calls = []
    handle = c2.state_encoder.register_forward_hook(lambda module, args, output: state_calls.append(list(args[0].shape)))
    with torch.inference_mode():
        result = c2(batch)
    handle.remove()
    assert state_calls == [list(batch["state"].shape)] and len(state_calls) == 1
    assert list(result["logits"].shape) == [1, 3]

    manifest_path = ROOT / "docs/decision-product/v3/metadata/qualified-bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    c3 = create_model(manifest["config"]).eval()
    models = {"text-C-v1": legacy, "byte-C-v2": c2, "typed-C-v3": c3}
    incompatibility = {}
    for target_name, target in models.items():
        target_shapes = {k: list(v.shape) for k, v in target.state_dict().items()}
        for source_name, source in models.items():
            if target_name == source_name:
                continue
            source_shapes = {k: list(v.shape) for k, v in source.state_dict().items()}
            missing = sorted(set(target_shapes) - set(source_shapes))
            unexpected = sorted(set(source_shapes) - set(target_shapes))
            mismatched = sorted(k for k in target_shapes.keys() & source_shapes.keys() if target_shapes[k] != source_shapes[k])
            assert missing or unexpected or mismatched
            incompatibility[source_name + " -> " + target_name] = {
                "missing_keys": len(missing), "unexpected_keys": len(unexpected), "mismatched_shapes": len(mismatched),
                "compatible_with_strict_state_dict": False,
            }

    class RecordingEngine:
        def __init__(self):
            self.examples = []

        def session(self, devices, language):
            return DeviceSession(self, tuple(devices), language)

        def _predict(self, example, use_cache=True):
            self.examples.append({"kind": example.kind, "state": example.state, "question": example.question})
            return Prediction(example.kind, "probe-only", None, "untrained-interface-probe", 0.0, "No neural inference performed")

    engine = RecordingEngine()
    request = json.loads((ROOT / "examples/decision_product/record-ready.json").read_text())
    dispatch = dispatch_request(engine, request)
    assert dispatch["product"] == "C-v3" and dispatch["dispatch"] == "typed-devices"
    assert {x["kind"] for x in engine.examples} == {"choice", "info", "rubric"}
    try:
        dispatch_request(engine, {"state": "卡片進度;申請編號=?", "question": "choice"})
    except ValueError as error:
        rejection = str(error)
    else:
        raise AssertionError("C-v3 dispatch accepted a legacy text request")

    acquisition_path = ROOT / "docs/decision-product/v3/metadata/acquisition.json"
    acquisition = json.loads(acquisition_path.read_text())
    archive = Path("/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip")
    assert archive.stat().st_size == acquisition["archive"]["bytes"]
    assert digest(archive) == acquisition["archive"]["sha256"]
    metadata_bindings = []
    with zipfile.ZipFile(archive) as source:
        for item in acquisition["files"]:
            raw = (ROOT / item["path"]).read_bytes()
            assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
            assert source.namelist().count(item["archive_member"]) == 1
            assert source.read(item["archive_member"]) == raw
            metadata_bindings.append({"path": item["path"], "sha256": item["sha256"], "archive_member": item["archive_member"]})

    print(json.dumps({
        "created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/c_v3_technical_coord/canonical_19_16_fresh",
        "command": COMMAND, "result": "All stated identity and interface assertions passed.",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": str(torch.get_num_threads()), "training": "none"},
        "identity": {"source21": p21, "source18": p18, "delta": list(difflib.unified_diff(old.splitlines(), new.splitlines(), fromfile="source18", tofile="source21")), "source_identical_remainder": True},
        "navigation": navigation,
        "legacy_interface": {"kinds": old_interfaces, "vocabulary_size": len(tokenizer), "parameters": sum(p.numel() for p in legacy.parameters())},
        "v2_shared_state": {"state_encoder_calls": state_calls, "branches_shape": list(batch["branches"].shape), "logits_shape": list(result["logits"].shape)},
        "v3_manifest_config": manifest["config"],
        "strict_state_dict_compatibility": incompatibility,
        "typed_dispatch": {"dispatch": dispatch["dispatch"], "kinds": [x["kind"] for x in engine.examples], "legacy_request_error": rejection, "neural_inference": False, "scope": "RecordingEngine checks format dispatch only; its fabricated probe results are not model predictions."},
        "local_archive": {"path": str(archive), "bytes": archive.stat().st_size, "sha256": digest(archive), "metadata_bindings": metadata_bindings, "public_download_performed": False},
        "scope": "Only source identity, candidate/interface contracts, v2 single shared-state call and incompatible architecture state-dict schemas. All constructed neural models are random and untrained. No original training, final evaluation, product acceptance, GPU/MPS or public transport was rerun or established."
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
