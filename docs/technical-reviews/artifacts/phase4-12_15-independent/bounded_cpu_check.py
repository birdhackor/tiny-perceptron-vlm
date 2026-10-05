"""Independent, bounded CPU checks for 12.15 design contracts; no training or real audio."""
import ast
import hashlib
import json
import platform
import sys
from pathlib import Path
import importlib.metadata

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from tiny_perceptron.multimodal import log_mel, tone, AudioEncoder, MultiModalLM, generate_modal
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from tiny_perceptron.data import ByteTokenizer

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1)
    assert torch.version.cuda is None and not torch.cuda.is_available()
    env = {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_build": str(torch.version.cuda), "device": "cpu", "torch_threads": str(torch.get_num_threads()),
        "PyYAML": yaml.__version__, "numpy": importlib.metadata.version("numpy"),
        "code_sha256": sha(__file__), "runtime_policy": "CPU, synthetic tones only; no optimizer, pretrained checkpoint, or downloaded dataset/model",
    }
    (BASE / "cpu.environment.json").write_text(json.dumps(env, indent=2) + "\n")

    # Parse publisher's exact pinned metadata, rather than assume 502 from the chapter.
    readme = BASE / "sources/minds14-readme.md"
    frontmatter = readme.read_text().split("---", 2)[1]
    card = yaml.safe_load(frontmatter)
    matching = [(i, d) for i, d in enumerate(card["dataset_info"]) if d["config_name"] == "zh-CN"]
    assert len(matching) == 1
    index, chinese = matching[0]
    fields = [x["name"] for x in chinese["features"]]
    intents = next(x["dtype"]["class_label"]["names"] for x in chinese["features"] if x["name"] == "intent_class")
    observed = chinese["splits"][0]["num_examples"]
    assert card["license"] == ["cc-by-4.0"]
    assert observed == 502 and len(intents) == 14
    selected = {k: intents[k] for k in ("1", "2", "6")}
    assert selected == {"1": "address", "2": "app_error", "6": "card_issues"}
    assert "speaker_id" not in fields and "response" not in fields and "answer" not in fields
    metadata = {
        "origin": "Publisher's README YAML frontmatter at revision 40ce77cb32a384e4d50a568e1ec39ac804019d33",
        "source_sha256": sha(readme),
        "inspected_pointers": ["/license", f"/dataset_info/{index}/config_name", f"/dataset_info/{index}/features", f"/dataset_info/{index}/splits/0/num_examples"],
        "chinese_examples_metadata": observed, "published_fields": fields,
        "selected_intent_ids": selected, "total_intent_classes": len(intents),
        "scope": "Read metadata only. No rows/recordings downloaded; usable three-intent short-utterance counts and speakers remain unmeasured.",
    }

    # Fixed front end has no learned state; encoder output changes with fresh parameters.
    wave = tone(440)[None]
    spec1 = log_mel(wave)
    torch.manual_seed(101)
    first = AudioEncoder(bands=16, width=8)
    features1 = first(wave)
    torch.manual_seed(102)
    second = AudioEncoder(bands=16, width=8)
    features2 = second(wave)
    spec2 = log_mel(wave)
    assert torch.equal(spec1, spec2)
    assert not torch.allclose(features1, features2)
    assert tuple(spec1.shape) == (1, 16, 11) and tuple(features1.shape) == (1, 11, 8)
    assert torch.isfinite(spec1).all() and torch.isfinite(features1).all()

    # Training requires target answers, while generation accepts only a prompt and waveform.
    torch.manual_seed(103)
    tok = ByteTokenizer()
    model = MultiModalLM(TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=128)), audio_width=8)
    prefix = [tok.bos_id, tok.user_id, tok.audio_id, tok.eos_id, tok.assistant_id]
    answer = tok.encode("先檢查網路。") + [tok.eos_id]
    ids = torch.tensor(prefix + answer)
    labels = torch.tensor([-100] * len(prefix) + answer)
    output = model(ids, labels, waveform=wave[0])
    assert (output["labels"] != -100).sum().item() == len(answer)
    loss = masked_loss(output["logits"], output["labels"])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    loss.backward()
    grads = {
        "audio_encoder": model.audio.projection.weight.grad.norm().item(),
        "audio_adapter": model.audio_projector.weight.grad.norm().item(),
        "language_core": model.language.embedding.weight.grad.norm().item(),
    }
    assert all(x > 0 for x in grads.values())
    assert all(torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
    generation_input = torch.tensor(prefix)
    with torch.no_grad():
        generated = generate_modal(model, generation_input, waveform=wave[0], max_new_tokens=2)
    assert generated[:len(prefix)].tolist() == prefix
    assert len(generated) <= len(prefix) + 2

    # A transformed descendant retains the original recording identity for split purposes.
    train_variants = [{"file": "A.original", "family": "A"}, {"file": "B.crop", "family": "B"}]
    naive_test = [{"file": "A.noise", "family": "A"}, {"file": "C.original", "family": "C"}]
    assert not ({x["file"] for x in train_variants} & {x["file"] for x in naive_test})
    overlap = sorted({x["family"] for x in train_variants} & {x["family"] for x in naive_test})
    assert overlap == ["A"]
    grouped_test = [x for x in naive_test if x["family"] not in {x["family"] for x in train_variants}]
    assert not ({x["family"] for x in train_variants} & {x["family"] for x in grouped_test})

    result = {
        "metadata_check": metadata,
        "synthetic_contract": {
            "wave_samples": wave.shape[-1], "sample_rate_hz": 16000, "wave_seconds": wave.shape[-1] / 16000,
            "fixed_log_mel_shape": list(spec1.shape), "time_feature_shape": list(features1.shape),
            "log_mel_identical_across_encoder_reinitialization": True,
            "learned_feature_changed_across_seeds": True,
            "all_values_finite": True,
            "input_prefix_positions": len(prefix), "answer_utf8_bytes_plus_eos": len(answer),
            "training_logits_shape": list(output["logits"].shape), "valid_target_count": len(answer),
            "masked_loss": loss.item(), "gradient_norms": grads,
            "parameter_updates": 0,
            "inference_input_ids": generation_input.tolist(), "inference_has_gold_answer": False,
            "generated_ids": generated.tolist(), "generated_new_tokens": len(generated) - len(prefix),
            "scope": "Finite tensor integration and backward connection only; random output is not a Chinese/intent/ASR capability measurement.",
        },
        "split_counterexample": {
            "naive_file_overlap": [], "naive_recording_family_overlap": overlap,
            "grouped_family_overlap": [], "scope": "Hypothetical sample identities; no MInDS-14 split constructed or evaluated.",
        },
        "assertions": "all passed",
    }
    (BASE / "cpu.result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
