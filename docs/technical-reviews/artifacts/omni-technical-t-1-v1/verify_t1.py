"""T.1 有界 CPU 核驗；不建立模型、不訓練，只寫本 reviewer 證據與 ignored data。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime

import numpy as np
import soundfile as sf
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "execution.json"
DATA = ROOT / "data/omni-technical-t-1-v1"
DATA.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
report = {
    "started_at": datetime.now(UTC).isoformat(),
    "environment": {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": "cpu", "platform": platform.platform(),
    },
    "scope": "bounded CPU data/CLI checks; no model, optimizer, training, GPU, paid service or publication",
    "commands": [], "checks": {},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flush():
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def run(argv, *, env=None, label=None, timeout=60):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, env=env, timeout=timeout)
    entry = {
        "label": label, "argv": [str(a) for a in argv], "cwd": str(ROOT),
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
        "timeout_seconds": timeout,
    }
    report["commands"].append(entry)
    flush()
    return result


assert not OUT.exists(), "immutable execution receipt already exists"
snapshot = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.1.md"
body = snapshot.read_text()
commands = re.search(r"```bash\n(.*?)```", body, re.S).group(1).strip().splitlines()
assert commands == [
    ".venv/bin/python scripts/fetch_training_assets.py --list",
    ".venv/bin/python scripts/fetch_training_assets.py --asset tinystories",
]
report["snapshot_sha256"] = sha(snapshot)
manifest = json.loads((ROOT / "assets/training/manifest.json").read_text())
asset = next(a for a in manifest["assets"] if a["id"] == "tinystories")
archive = ROOT / asset["archive"]
pointer_bytes = archive.read_bytes()
report["checks"]["archive_before"] = {
    "path": asset["archive"], "bytes": len(pointer_bytes), "sha256": sha(archive),
    "is_lfs_pointer": pointer_bytes.startswith(b"version https://git-lfs.github.com/spec/v1\n"),
}
assert report["checks"]["archive_before"]["is_lfs_pointer"]
assert run(["git", "lfs", "version"], label="installed Git LFS version").returncode == 0

# 執行教材兩條原命令，而非替它們另寫等價 helper。
listed = run(commands[0].split(), label="exact T.1 embedded command 1")
assert listed.returncode == 0
rows = [json.loads(line) for line in listed.stdout.splitlines()]
expected = [{k: a[k] for k in ("id", "training_records", "archive_bytes", "license")} for a in manifest["assets"]]
assert rows == expected
report["checks"]["list_matches_manifest"] = {"assets": len(rows), "exact_match": True}
fetched = run(commands[1].split(), label="exact T.1 embedded command 2", timeout=60)
assert fetched.returncode == 0, fetched.stderr
assert archive.stat().st_size == asset["archive_bytes"] == 357896
assert sha(archive) == asset["archive_sha256"]
files = []
for row in asset["files"]:
    path = ROOT / "data/training" / row["path"]
    assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
    files.append({"path": row["path"], "bytes": path.stat().st_size, "sha256": sha(path)})
stories_path = ROOT / "data/training/text-initial/tinystories-train-512.jsonl"
stories = [json.loads(line) for line in stories_path.read_text().splitlines()]
assert len(stories) == 512 and all(isinstance(r.get("text"), str) and r["text"] for r in stories)
report["checks"]["download_and_unpack"] = {
    "archive_bytes": archive.stat().st_size, "archive_sha256": sha(archive),
    "files_verified_independently": files, "story_rows": len(stories),
    "first_row_keys": sorted(stories[0]), "first_text_prefix": stories[0]["text"][:180],
}

# 隔離 fixture：使用未改動函式，只覆蓋其 ROOT；這不是另一台真實新安裝的電腦。
with tempfile.TemporaryDirectory(prefix="fixture-", dir=DATA) as temp:
    fixture = Path(temp)
    (fixture / "assets/training").mkdir(parents=True)
    (fixture / "assets/training/manifest.json").write_text(json.dumps({"assets": [asset]}))
    (fixture / asset["archive"]).write_bytes(pointer_bytes)
    bin_path = fixture / "bin"
    bin_path.mkdir()
    (bin_path / "git").symlink_to(shutil.which("git"))
    fixture_env = os.environ.copy()
    fixture_env["PATH"] = str(bin_path)
    fixture_env["GIT_EXEC_PATH"] = str(bin_path)
    loader = (
        "import importlib.util,sys;from pathlib import Path;"
        f"s=importlib.util.spec_from_file_location('fetch', {str(ROOT / 'scripts/fetch_training_assets.py')!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"m.ROOT=Path({str(fixture)!r});m.main()"
    )
    no_lfs_list = run([sys.executable, "-c", loader, "--list"], env=fixture_env,
                      label="controlled fixture: list without discoverable git-lfs")
    no_lfs_asset = run([sys.executable, "-c", loader, "--asset", "tinystories"], env=fixture_env,
                       label="controlled fixture: pointer asset without discoverable git-lfs")
    assert no_lfs_list.returncode == 0
    assert no_lfs_asset.returncode != 0 and "CalledProcessError" in no_lfs_asset.stderr
    assert not (fixture / "data/training").exists()
    report["checks"]["missing_lfs_fixture"] = {
        "simulation_scope": "real git executable; isolated PATH and GIT_EXEC_PATH without git-lfs; unmodified main with fixture ROOT",
        "list_succeeds": True, "asset_stops_before_unpack": True,
    }

alternate = DATA / "alternate-unpack"
alt = run([sys.executable, "scripts/fetch_training_assets.py", "--asset", "tinystories", "--output", str(alternate)],
          label="explicit --output option; already materialized local archive")
assert alt.returncode == 0
assert json.loads(alt.stdout)["destination"] == str(alternate.resolve())
report["checks"]["output_override"] = {"destination": str(alternate), "files_verified": len(asset["files"])}

generated = DATA / "generated"
prep = run([sys.executable, "scripts/prepare_data.py", "--kind", "attributes-sft", "--seed", "42", "--output", str(generated)],
           label="bounded CPU generator, not training")
assert prep.returncode == 0
all_rows = []
split_families = {}
for split in ("train", "validation", "test"):
    rs = [json.loads(line) for line in (generated / "attributes-sft" / (split + ".jsonl")).read_text().splitlines()]
    split_families[split] = {r["family"] for r in rs}
    all_rows += rs
for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
    assert not split_families[a] & split_families[b]
expected_examples = [
    ("color=red;shape=square;pitch=low;shape?", "square", "red:square:low"),
    ("color=red;shape=square;pitch=low;describe", "square", "red:square:low"),
    ("color=blue;shape=circle;pitch=high;shape?", "circle", "blue:circle:high"),
    ("color=green;shape=circle;pitch=high;shape?", "circle", "green:circle:high"),
    ("color=green;shape=circle;pitch=high;color?", "green", "green:circle:high"),
]
matches = []
for question, answer, family in expected_examples:
    found = [r for r in all_rows if r["messages"][0]["content"] == question]
    assert len(found) == 1 and found[0]["messages"][1]["content"] == answer and found[0]["family"] == family
    matches.append(found[0])
report["checks"]["attribute_examples_and_split"] = {
    "records": len(all_rows), "families": len({r["family"] for r in all_rows}),
    "split_family_counts": {k: len(v) for k, v in split_families.items()},
    "pairwise_family_overlap": 0, "table_A_B_C_and_exercise_matches": matches,
    "note": "the printed random split is not the proposed hypothetical A/B-versus-C split; only family isolation and mappings are checked",
}

from tiny_perceptron.assets import unpack_asset
from tiny_perceptron.data import ByteTokenizer, shifted
from tiny_perceptron.modal_data import modal_example

ids = ByteTokenizer().encode("cat")
x, y = shifted(ids)
assert x.tolist() == ids[:-1] and y.tolist() == ids[1:]
report["checks"]["next_unit_targets"] = {"text": "cat", "ids": ids, "inputs": x.tolist(), "targets": y.tolist()}
with tempfile.TemporaryDirectory(prefix="compat-", dir=DATA) as temp:
    p = Path(temp)
    Image.new("RGB", (12, 10)).save(p / "image.png")
    sf.write(p / "8k.wav", np.zeros(80, dtype="float32"), 8000)
    sf.write(p / "16k.wav", np.zeros(160, dtype="float32"), 16000)
    wave, rate = sf.read(p / "8k.wav")
    assert len(wave) == 80 and rate == 8000
    try:
        modal_example({"audio": "8k.wav", "question": "pitch?", "answer": "low"}, p, "audio")
    except ValueError as error:
        rejection = str(error)
    else:
        raise AssertionError("8k readable file should be rejected by the 16k example loader")
    result = modal_example({"image": "image.png", "audio": "16k.wav", "question": "shape?", "answer": "circle"}, p, "joint")
    assert list(result[2].shape) == [3, 16, 16] and result[3].numel() == 160
    report["checks"]["decodable_not_task_compatible"] = {
        "readable_audio_rate": rate, "loader_rejection": rejection,
        "valid_joint_image_shape": list(result[2].shape), "valid_joint_wave_samples": result[3].numel(),
        "scope": "loader compatibility only; fabricated question/answer does not establish correct real-world labels",
    }
    bad = p / "bad.tar.gz"
    content = bytearray(archive.read_bytes())
    content[-1] ^= 1
    bad.write_bytes(content)
    try:
        unpack_asset(bad, asset, p / "bad-unpack")
    except ValueError as error:
        report["checks"]["tampered_archive_rejected"] = str(error)
    else:
        raise AssertionError("archive SHA mismatch should be rejected")
report["completed_at"] = datetime.now(UTC).isoformat()
report["result"] = "all assertions passed"
flush()
print(json.dumps({"result": report["result"], "receipt": str(OUT), "sha256": sha(OUT)}, ensure_ascii=False))
