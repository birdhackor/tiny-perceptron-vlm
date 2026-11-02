"""Full current 19.21 recheck: exact current artificial CPU cells and portable preserved-row statistics."""

import ast
import contextlib
import difflib
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
REVIEW = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-19.21"
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v6.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def named(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


def main():
    assert sha(FREEZE) == "57f17dde51dac7f9b190e0b02dac68ad8f4f8214f368af1c8c8f8f2aa6ba18df"
    original = REVIEW / "technical-original-v1.json"
    assert sha(original) == "8cdbb218137b2c4e5d82e7d07a2f016faaac7200f428b15fd1012f8d92dadb9d"
    report = json.loads(original.read_bytes())
    page = next(p for p in json.loads(FREEZE.read_bytes())["pages"] if p["page_id"] == "19.21")
    snapshot = ROOT / page["snapshot"]
    assert sha(snapshot) == page["source_sha256"] == "be13d1c7acb49c2f3a750b2aa6250110ba9d46c15f0ded9149c73c40a872c995"
    current = snapshot.read_text()
    chapter = (ROOT / page["source"]).read_text()
    match = re.search(r"^## 19\.21 .+?(?=^## |\Z)", chapter, re.M | re.S)
    assert match and match.group() == current
    old = (ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v3-final/sources/19.21.md").read_text()
    before, after = "兩者的聲音 CNN 都只能讀本塊", "兩者的聲音入口都只能讀本塊"
    assert old.count(before) == 1 and old.replace(before, after) == current
    code = re.findall(r"```python\n(.*?)```", current, re.S)
    old_code = re.findall(r"```python\n(.*?)```", old, re.S)
    assert len(code) == 3 and code == old_code
    assert sha(ROOT / page["notebook"]) == page["notebook_sha256"]
    cells = json.loads((ROOT / page["notebook"]).read_bytes())["cells"]
    notebook_code = ["".join(c["source"]).rstrip() for c in cells if c["cell_type"] == "code"]
    assert all(c.rstrip() in notebook_code for c in code)
    checked_sources = []
    for source in report["sources"]:
        if source["kind"] == "repository_code":
            observed = sha(ROOT / source["path"])
            assert observed == source["sha256"], source["path"]
            checked_sources.append({"path": source["path"], "sha256": observed, "unchanged": True})
    for artifact in report["artifacts"]:
        assert sha(ROOT / artifact["path"]) == artifact["sha256"]
    for figure, expected in page["figures_sha256"].items():
        assert sha(ROOT / figure) == expected == report["figure_sha256"][figure]
    torch.set_num_threads(1)
    namespace = {"__name__": "current_19_21_artificial"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        for i, cell in enumerate(code):
            exec(compile(cell, f"current19.21-cell{i + 1}", "exec"), namespace)
    previous = json.loads((PROOFS / "verification-result-v1.json").read_bytes())
    assert stdout.getvalue() == previous["embedded_stdout"]
    n = namespace
    assert isinstance(n["model"].audio, torch.nn.Linear)
    assert (n["model"].audio.in_features, n["model"].audio.out_features) == (17, 12)
    assert sum(isinstance(layer, torch.nn.Conv2d) for layer in n["formal_model"].audio) == 2
    changed_pixels = n["pixel_demo"].clone()
    changed_pixels[0, 0, 3, 3] = 32
    assert torch.nn.functional.avg_pool2d(changed_pixels, 2)[0, 0].tolist() == [[3.5, 5.5], [11.5, 17.5]]
    assert torch.nn.functional.adaptive_avg_pool2d(changed_pixels, 1).item() == 9.5
    no_images = n["frame_valid"].clone()
    no_images[2] = False
    with torch.no_grad():
        logits, attention = n["model"](n["images"], n["frame_ms"], n["audio_features"], n["support"],
                                         no_images, n["audio_valid"], n["duration"])
    assert tuple(logits.shape) == (3, 3)
    assert float(attention[2, :, :, 1:17].sum() + attention[2, :, :, 19:23].sum()) == 0
    wave = torch.linspace(-0.8, 0.8, 9600)
    changed_wave = wave.clone()
    changed_wave[3200:6400] *= -0.3
    fa, fb = n["local_audio"](wave)[0], n["local_audio"](changed_wave)[0]
    assert torch.equal(fa[[0, 2]], fb[[0, 2]]) and not torch.equal(fa[1], fb[1])
    assert not torch.equal(fa, n["local_audio"](wave / 2)[0])
    repository, embedded = named((ROOT / "tiny_perceptron/omni_products/av.py").read_text()), named(code[2])
    for name in ("AVConfig", "mel_bank", "local_audio", "TimeAttention", "AVModel"):
        node = repository[name]
        if name == "AVModel":
            node.body = [part for part in node.body if not isinstance(part, ast.FunctionDef) or part.name != "config_sha256"]
        assert ast.dump(node, include_attributes=False) == ast.dump(embedded[name], include_attributes=False)
    assert sum(p.numel() for p in n["formal_model"].parameters()) == 119693
    primary = json.loads((PROOFS / "primary-extract-v1.json").read_bytes())
    summary, successes = {}, {}
    for job, rows in primary["final_rows"].items():
        successes[job] = {r[0]: int(r[2:4] == r[4:6]) for r in rows}
        summary[job] = {
            "primary": sum(successes[job].values()),
            "panels": [sum(r[2:4] == r[4:6] for r in rows if r[1] == p) for p in range(4)],
            "panel_n": [sum(r[1] == p for r in rows) for p in range(4)],
            "remove_image_unknown": sum(r[6] == 2 for r in rows),
            "remove_audio_unknown": sum(r[7] == 2 for r in rows),
            "waveform_swap_pairs": sum(r[8:10] == r[10:12] for r in rows),
            "waveform_swap_observations": sum(sum(a == b for a, b in zip(r[8:10], r[10:12])) for r in rows),
        }
    assert summary == previous["recomputed_final"]
    families = sorted(successes["av-additive-s11"])
    assert len(families) == 200 and all(set(rows) == set(families) for rows in successes.values())
    differences = torch.tensor([[successes[f"av-time-rope-s{s}"][f] - successes[f"av-additive-s{s}"][f]
                                 for f in families] for s in (11, 29, 47)], dtype=torch.float64)
    indices = torch.randint(200, (1000, 200), generator=torch.Generator().manual_seed(20261029))
    interval = (torch.quantile(differences.mean(0)[indices].mean(1), torch.tensor([0.025, 0.975], dtype=torch.float64)) * 100).tolist()
    assert interval == previous["independent_bootstrap"]["bootstrap_95_pp"]
    chosen = {}
    for job, training in primary["training"].items():
        best = max(training["logs"], key=lambda r: (r["dev_score"], -r["update"]))
        assert best == training["best"] and training["complete"] and training["updates"] == 6000
        chosen[job] = [best["update"], round(best["dev_metrics"]["paired_accuracy"] * 64)]
    selected = max(primary["job_order"], key=lambda job: (primary["training"][job]["best"]["dev_score"],
                    -primary["training"][job]["best"]["update"], -primary["job_order"].index(job)))
    assert selected == "av-time-rope-s29"
    result = {
        "created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_p5_technical_19_21",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "CPU"},
        "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE)},
        "page": page, "original_report_sha256": sha(original),
        "diff": list(difflib.unified_diff(old.splitlines(), current.splitlines(), lineterm="")),
        "all_three_current_embedded_cells_executed": True, "embedded_stdout": stdout.getvalue(),
        "embedded_cells_byte_identical_to_original": True, "current_notebook_cells_match": True,
        "all_original_artifacts_unchanged": True, "repository_sources_rechecked": checked_sources,
        "formal_ast_match": True, "artificial_exercises_and_waveform_locality_pass": True,
        "issue_T01": {"original_json_pointer": "/issues/0", "current_quote": after,
                      "tinyav_audio": "Linear(17,12)", "formal_audio_conv2d_layers": 2,
                      "resolution": "The revised word names the shared local audio entry without incorrectly claiming the TinyAV summary projection is a CNN. Existing per-block support and padding qualification are preserved.",
                      "status": "resolved"},
        "recomputed_final": summary, "recomputed_dev_best_update_correct_of64": chosen,
        "dev_selected": selected, "mean_difference_pp": float(differences.mean() * 100),
        "bootstrap95_pp": interval,
        "scope": "Entire current body/details read; exact current artificial CPU cells and exercises executed; current dependencies/figure/Notebook hashes checked; preserved rows/logs recomputed. No final-family NN inference, training, weight changes, T/tau selection, Modal or publication. Original report/proofs preserved.",
    }
    output = PROOFS / "current-recheck-result-v1.json"
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "bytes": output.stat().st_size, "sha256": sha(output),
                      "original_issue": result["issue_T01"]["status"], "bootstrap95_pp": interval}))


if __name__ == "__main__":
    main()
