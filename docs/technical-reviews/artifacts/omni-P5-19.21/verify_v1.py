"""Portable independent 19.21 checks: only embedded artificial CPU examples and saved-row arithmetic."""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
PROOFS = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v3-final/sources/19.21.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node_map(text):
    return {node.name: node for node in ast.parse(text).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}


def main():
    torch.set_num_threads(1)
    assert sha(SOURCE) == "00d4bc4a39663f31c5d0b531c7f9dfb8aef1cd3f614d66e3c5b388b84274ce71"
    body = SOURCE.read_text()
    code = re.findall(r"```python\n(.*?)```", body, re.S)
    assert len(code) == 3
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torch_git_version": torch.version.git_version, "device": "cpu", "threads": "1"},
        "source_sha256": sha(SOURCE),
        "embedded_code_sha256": [hashlib.sha256(part.encode()).hexdigest() for part in code],
    }
    namespace = {"__name__": "embedded_19_21"}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for index, part in enumerate(code):
            exec(compile(part, f"19.21-embedded-{index + 1}", "exec"), namespace)
    result["embedded_stdout"] = log.getvalue()
    n = namespace
    assert n["response_demo"][0, 0, 0].tolist() == [5, 8, 11, 4]
    assert n["local_demo"][0, 0].tolist() == [[3.5, 5.5], [11.5, 13.5]]
    assert n["global_demo"].item() == 8.5
    modified = n["pixel_demo"].clone()
    modified[0, 0, 3, 3] = 32
    result["pool_exercise"] = {"local": torch.nn.functional.avg_pool2d(modified, 2)[0, 0].tolist(),
                               "global": torch.nn.functional.adaptive_avg_pool2d(modified, 1).item()}
    assert result["pool_exercise"] == {"local": [[3.5, 5.5], [11.5, 17.5]], "global": 9.5}
    assert tuple(n["logits"].shape) == (3, 3) and tuple(n["weights"].shape) == (3, 2, 25, 25)
    assert torch.equal(n["index_logits"][0], n["index_logits"][1])
    assert torch.equal(n["logits"][2], n["changed_logits"][2])
    assert n["support"].mean(-1)[:2, 1].tolist() == [1300, 2300]
    parameters = [n["model"].vision[0].weight, n["model"].audio.weight,
                  n["model"].layers[0].q.weight, n["model"].layers[1].q.weight, n["model"].head.weight]
    result["mechanism_gradient_norms"] = [float(p.grad.norm()) for p in parameters]
    assert all(x > 0 for x in result["mechanism_gradient_norms"])
    before = [p.detach().clone() for p in n["model"].parameters()]
    no_images = n["frame_valid"].clone()
    no_images[2] = False
    with torch.no_grad():
        exercise_logits, exercise_attention = n["model"](
            n["images"], n["frame_ms"], n["audio_features"], n["support"], no_images,
            n["audio_valid"], n["duration"],
        )
    result["missing_both_exercise"] = {
        "shape": list(exercise_logits.shape),
        "masked_key_attention_sum": float(exercise_attention[2, :, :, 1:17].sum() + exercise_attention[2, :, :, 19:23].sum()),
        "parameters_unchanged": all(torch.equal(old, new) for old, new in zip(before, n["model"].parameters())),
    }
    assert result["missing_both_exercise"]["masked_key_attention_sum"] == 0
    assert result["missing_both_exercise"]["parameters_unchanged"]
    assert tuple(n["features"].shape) == (4, 1, 24, 37)
    assert n["anchors"].tolist() == [100, 300, 500, 650]
    assert tuple(n["formal_logits"].shape) == (2, 3)
    assert sum(p.numel() for p in n["formal_model"].parameters()) == 119693
    result["formal_parameter_count"] = 119693
    wave = torch.linspace(-0.8, 0.8, 9600)
    changed = wave.clone()
    changed[3200:6400] *= -0.3
    fa = n["local_audio"](wave)[0]
    fb = n["local_audio"](changed)[0]
    result["artificial_waveform_locality"] = {
        "unmodified_blocks_identical": torch.equal(fa[[0, 2]], fb[[0, 2]]),
        "modified_block_differs": not torch.equal(fa[1], fb[1]),
        "half_amplitude_changes_features": not torch.equal(fa, n["local_audio"](wave / 2)[0]),
    }
    assert all(result["artificial_waveform_locality"].values())
    repository = node_map((ROOT / "tiny_perceptron/omni_products/av.py").read_text())
    embedded = node_map(code[2])
    result["formal_ast_equality"] = {}
    for name in ("AVConfig", "mel_bank", "local_audio", "TimeAttention", "AVModel"):
        node = repository[name]
        if name == "AVModel":
            node.body = [part for part in node.body if not isinstance(part, ast.FunctionDef) or part.name != "config_sha256"]
        equal = ast.dump(node, include_attributes=False) == ast.dump(embedded[name], include_attributes=False)
        result["formal_ast_equality"][name] = equal
        assert equal, name
    result["audio_entry_inspection"] = {
        "tinyav_audio_type": type(n["model"].audio).__name__,
        "tinyav_audio_in_out": [n["model"].audio.in_features, n["model"].audio.out_features],
        "formal_audio_layers": [type(layer).__name__ for layer in n["formal_model"].audio],
        "contradicted_quote": "兩者的聲音 CNN 都只能讀本塊",
        "reason": "TinyAV audio is Linear(17,12), and the 12.17 summary preprocessor has no CNN. Formal AVModel alone has an audio CNN.",
    }

    # Every result below is recomputed from the compact exact row projection, not model inference.
    primary = json.loads((PROOFS / "primary-extract-v1.json").read_bytes())
    summary = {}
    correct = {}
    for job, rows in primary["final_rows"].items():
        assert len(rows) == 200 and len({row[0] for row in rows}) == 200
        correct[job] = {row[0]: int(row[2:4] == row[4:6]) for row in rows}
        summary[job] = {
            "primary": sum(correct[job].values()),
            "panels": [sum(row[2:4] == row[4:6] for row in rows if row[1] == panel) for panel in range(4)],
            "panel_n": [sum(row[1] == panel for row in rows) for panel in range(4)],
            "remove_image_unknown": sum(row[6] == 2 for row in rows),
            "remove_audio_unknown": sum(row[7] == 2 for row in rows),
            "waveform_swap_pairs": sum(row[8:10] == row[10:12] for row in rows),
            "waveform_swap_observations": sum(sum(a == b for a, b in zip(row[8:10], row[10:12])) for row in rows),
        }
        assert summary[job]["panel_n"] == [50] * 4
    result["recomputed_final"] = summary
    family_ids = sorted(correct["av-additive-s11"])
    assert all(set(rows) == set(family_ids) for rows in correct.values())
    differences = torch.tensor([
        [correct[f"av-time-rope-s{seed}"][family] - correct[f"av-additive-s{seed}"][family] for family in family_ids]
        for seed in (11, 29, 47)
    ], dtype=torch.float64)
    generator = torch.Generator().manual_seed(20261029)
    index = torch.randint(200, (1000, 200), generator=generator)
    averages = differences.mean(0)
    result["independent_bootstrap"] = {
        "families": 200, "fixed_training_seeds": [11, 29, 47], "replicates": 1000, "resampling_seed": 20261029,
        "per_seed_differences_pp": (differences.mean(1) * 100).tolist(),
        "three_seed_additive_percent": sum(summary[f"av-additive-s{s}"]["primary"] for s in (11, 29, 47)) / 6,
        "three_seed_rope_percent": sum(summary[f"av-time-rope-s{s}"]["primary"] for s in (11, 29, 47)) / 6,
        "mean_difference_pp": float(averages.mean() * 100),
        "bootstrap_95_pp": (torch.quantile(averages[index].mean(1), torch.tensor([0.025, 0.975], dtype=torch.float64)) * 100).tolist(),
        "sample_index_sha256": hashlib.sha256(index.numpy().tobytes()).hexdigest(),
    }
    assert result["independent_bootstrap"]["bootstrap_95_pp"] == [24.166666666666668, 31.666666666666664]
    chosen = {}
    for job, training in primary["training"].items():
        best = max(training["logs"], key=lambda row: (row["dev_score"], -row["update"]))
        assert best == training["best"]
        assert training["complete"] and training["updates"] == 6000 and training["optimizer_updates_observed"]
        assert training["trainable_parameters"] == 119693
        assert training["initial_parameter_sha256"] != training["latest_parameter_sha256"]
        chosen[job] = {"update": best["update"], "dev_pairs_correct": round(best["dev_metrics"]["paired_accuracy"] * 64),
                       "dev_missing_cue_accuracies": [best["dev_metrics"]["remove_image_accuracy"], best["dev_metrics"]["remove_audio_accuracy"]]}
    selected = max(primary["job_order"], key=lambda job: (
        primary["training"][job]["best"]["dev_score"], -primary["training"][job]["best"]["update"],
        -primary["job_order"].index(job),
    ))
    assert selected == primary["chronology"]["selected_av"] == "av-time-rope-s29"
    result["recomputed_development_selection"] = {"selected": selected, "jobs": chosen}
    result["matched_initializations"] = {
        str(seed): primary["training"][f"av-additive-s{seed}"]["initial_parameter_sha256"] == primary["training"][f"av-time-rope-s{seed}"]["initial_parameter_sha256"]
        for seed in (11, 29, 47)
    }
    assert all(result["matched_initializations"].values())
    result["scope"] = "All three exact embedded CPU blocks and stated exercises; artificial local waveform perturbation; AST equality; row-only final arithmetic and log-only development selection. No fixed-family NN inference, trained weights loaded, optimizer steps, or policy fitting."
    output = PROOFS / "verification-result-v1.json"
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "bytes": output.stat().st_size, "sha256": sha(output), "selected": selected, "bootstrap": result["independent_bootstrap"]}))


if __name__ == "__main__":
    main()
