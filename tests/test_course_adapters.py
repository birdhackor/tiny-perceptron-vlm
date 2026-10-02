"""正式 LoRA payload 能獨立推論；錯誤基底與尺寸在植入之前失敗。"""

import copy
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from scripts import infer
from scripts.course_experiments.behavior import _adapter_state, _add_lora, _base_state, _merge_lora, _state_digest
from tiny_perceptron.adapters import base_state_sha256, load_lora_adapter
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def adapter_files(tmp_path):
    base = TinyLM(ModelConfig(width=8, layers=1, heads=2, max_length=32)).eval()
    base_path = tmp_path / "base.pt"
    save_checkpoint(base_path, base)
    adapted = copy.deepcopy(base)
    _add_lora(adapted, rank=2, alpha=4)
    with torch.no_grad():
        for layer in adapted.modules():
            if isinstance(layer, LoRALinear):
                layer.a.normal_(0, 0.2)
                layer.b.normal_(0, 0.2)
    payload = {
        "format_version": "lora-v1",
        "adapter": _adapter_state(adapted),
        "config": asdict(base.config),
        "base_sha256": _state_digest(base.state_dict()),
        "scaling": "alpha/rank",
    }
    adapter_path = tmp_path / "adapter.pt"
    # 公開部署版不需要 optimizer 或 RNG；保留正式實驗的 adapter schema。
    torch.save(payload, adapter_path)
    return base, adapted, base_path, adapter_path, payload


def test_nonzero_official_adapter_matches_trained_and_merged_logits(adapter_files):
    base, trained, base_path, adapter_path, payload = adapter_files
    loaded, checkpoint = load_checkpoint(base_path)
    originals = {name: layer.weight for name, layer in loaded.named_modules() if isinstance(layer, torch.nn.Linear)}
    rng = torch.get_rng_state().clone()
    report = load_lora_adapter(loaded, adapter_path, base_state=checkpoint["model"])
    assert torch.equal(torch.get_rng_state(), rng)
    ids = torch.tensor([[1, 3, 55, 57, 2, 4]])
    with torch.no_grad():
        expected = trained(ids)["logits"]
        actual = loaded(ids)["logits"]
        merged = _merge_lora(trained)(ids)["logits"]
        original = base(ids)["logits"]
    assert (actual - original).abs().max() > 0.001
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(actual, merged, atol=2e-6)
    assert base_state_sha256(_base_state(loaded)) == payload["base_sha256"]
    for entry in report["modules"]:
        layer = loaded.get_submodule(entry["path"])
        assert layer.base.weight is originals[entry["path"]]
        assert entry["scale"] == 2 and entry["rank"] == 2 and entry["alpha"] == 4
    assert report["base_sha256_verified"] and report["scaling"] == "alpha/rank"
    json.dumps(report, allow_nan=False)


def test_wrong_base_is_rejected_before_any_module_is_wrapped(adapter_files):
    base, _, _, adapter_path, _ = adapter_files
    wrong = copy.deepcopy(base)
    with torch.no_grad():
        wrong.embedding.weight[0, 0] += 1
    before = base_state_sha256(wrong.state_dict())
    with pytest.raises(ValueError, match="base_sha256"):
        load_lora_adapter(wrong, adapter_path)
    assert not any(isinstance(layer, LoRALinear) for layer in wrong.modules())
    assert base_state_sha256(wrong.state_dict()) == before


@pytest.mark.parametrize(
    "problem",
    [
        "format",
        "config",
        "config_type",
        "scaling",
        "rank",
        "rank_type",
        "alpha",
        "a_shape",
        "b_shape",
        "a_dtype",
        "nonfinite",
        "unknown_path",
        "nonlinear",
        "top_rank",
    ],
)
def test_invalid_payload_is_transactional(adapter_files, tmp_path, problem):
    base, _, _, _, original = adapter_files
    payload = copy.deepcopy(original)
    name = "output"  # 最後一筆失敗也不能留下前面已植入的 attention／FFN。
    entry = payload["adapter"][name]
    expected = {
        "format": "lora-v1",
        "config": "config",
        "config_type": "config",
        "scaling": "scaling",
        "rank": "rank",
        "rank_type": "rank",
        "alpha": "alpha",
        "a_shape": "shape",
        "b_shape": "shape",
        "a_dtype": "浮點",
        "nonfinite": "非有限",
        "unknown_path": "未知",
        "nonlinear": "Linear",
        "top_rank": "頂層",
    }[problem]
    if problem == "format":
        payload["format_version"] = 1
    elif problem == "config":
        payload["config"]["heads"] = 1
    elif problem == "config_type":
        payload["config"]["layers"] = True
    elif problem == "scaling":
        payload["scaling"] = "alpha"
    elif problem == "rank":
        entry["rank"] = 0
    elif problem == "rank_type":
        entry["rank"] = 2.0
    elif problem == "alpha":
        entry["alpha"] = float("nan")
    elif problem == "a_shape":
        entry["a"] = torch.zeros(2, base.config.width + 1)
    elif problem == "b_shape":
        entry["b"] = torch.zeros(base.config.vocab_size, 3)
    elif problem == "a_dtype":
        entry["a"] = entry["a"].long()
    elif problem == "nonfinite":
        entry["b"][0, 0] = float("inf")
    elif problem == "unknown_path":
        payload["adapter"]["missing.layer"] = entry
    elif problem == "nonlinear":
        payload["adapter"]["embedding"] = entry
    elif problem == "top_rank":
        payload["rank"] = 3
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    before = _state_digest(base.state_dict())
    with pytest.raises(ValueError, match=expected):
        load_lora_adapter(base, path)
    assert not any(isinstance(layer, LoRALinear) for layer in base.modules())
    assert _state_digest(base.state_dict()) == before


def test_current_dtype_uses_verified_original_state(adapter_files):
    base, trained, _, adapter_path, _ = adapter_files
    original = copy.deepcopy(base.state_dict())
    converted = copy.deepcopy(base).double()
    load_lora_adapter(converted, adapter_path, base_state=original)
    for layer in converted.modules():
        if isinstance(layer, LoRALinear):
            assert layer.a.dtype == layer.b.dtype == layer.base.weight.dtype == torch.float64
            assert layer.a.device == layer.b.device == layer.base.weight.device
    ids = torch.tensor([[1, 55, 2]])
    assert torch.allclose(converted(ids)["logits"], trained.double()(ids)["logits"], atol=1e-9)
    wrong = copy.deepcopy(base).double()
    with torch.no_grad():
        wrong.output.weight[0, 0] += 0.1
    with pytest.raises(ValueError, match="實際 LoRA 基底"):
        load_lora_adapter(wrong, adapter_path, base_state=original)


def test_cli_adapter_changes_actual_raw_generation_and_preserves_eos(tmp_path, monkeypatch, capsys):
    tok = ByteTokenizer()
    base = TinyLM(ModelConfig(width=8, max_length=32))
    token = tok.encode("x")[0]
    with torch.no_grad():
        base.final_norm.weight.zero_()
        base.final_norm.bias.zero_()
        base.final_norm.bias[0] = 1
        base.output.weight.zero_()
        base.output.weight[token, 0] = 1
    base_path, adapter_path = tmp_path / "base.pt", tmp_path / "eos-adapter.pt"
    save_checkpoint(base_path, base)
    a, b = torch.zeros(1, 8), torch.zeros(264, 1)
    a[0, 0], b[tok.eos_id, 0] = 1, 2
    torch.save(
        {
            "format_version": "lora-v1",
            "config": asdict(base.config),
            "base_sha256": _state_digest(base.state_dict()),
            "scaling": "alpha/rank",
            "adapter": {"output": {"a": a, "b": b, "rank": 1, "alpha": 1}},
        },
        adapter_path,
    )
    arguments = [str(base_path), "--prompt", "a", "--device", "cpu", "--json", "--tokens", "2"]
    monkeypatch.setattr(sys, "argv", [infer.__file__, *arguments])
    infer.main()
    plain = json.loads(capsys.readouterr().out)
    assert plain["generated_ids"] == [token, token] and not plain["eos"]
    # 子程序真的經過 CLI 的 --adapter path，沒有 monkeypatch 生成器。
    process = subprocess.run(
        [sys.executable, "scripts/infer.py", *arguments, "--adapter", str(adapter_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(process.stdout)
    assert result["generated_ids"] == [tok.eos_id] and result["eos"] and result["answer"] == ""
    assert result["invalid_special_tokens"] == [] and result["valid_answer_tokens"]
    assert result["adapter"]["path"] == str(adapter_path)
    assert result["adapter"]["base_sha256"] == _state_digest(base.state_dict())
    assert result["adapter"]["modules"] == [{"path": "output", "rank": 1, "alpha": 1, "scale": 1}]
