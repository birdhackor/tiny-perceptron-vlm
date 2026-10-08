#!/usr/bin/env python3
"""Append deterministic train-only numeric and calculator exercises to V1.

The original twelve record files define the split boundary. Their original
bytes are preserved; only text/tools train gains rows. Heldout questions are
used solely for exclusion checks, never to author labels or fit vocabulary.
All new targets use bounded arithmetic, string copying, or authored finite
contrast examples. No model teacher or inference answer replacement is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.selftrained.tools import (  # noqa: E402
    OPERATIONS,
    execute_tool_call,
    serialize_tool_call,
    serialize_tool_result,
)

VERSION = "selftrained-text-tools-v2-train-augmentation-v1"
SEED = 20261006
RECORD_NAMES = tuple(
    f"{kind}-{split}.jsonl"
    for kind in ("ocr", "text-tools", "vision", "voice")
    for split in ("train", "validation", "test")
)
TEXT_TRAIN = "text-tools-train.jsonl"
OTHER_AUGMENTABLE = frozenset(("ocr-train.jsonl", "voice-train.jsonl"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def original_helpers():
    path = ROOT / "scripts/selftrained/prepare_text_tools.py"
    spec = importlib.util.spec_from_file_location("selftrained_original_text_authoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def new_record(group: str, task: str, messages: list[dict], *, kind: str, **labels) -> dict:
    record = {
        "group_id": group,
        "split": "train",
        "task": task,
        "messages": messages,
        "supervision": {
            "intent": "calculation" if task.startswith("tool_") else "numeric_copy",
            "augmentation_kind": kind,
            **labels,
        },
        "provenance": {
            "source": "project-original-deterministic-v2-exercise",
            "version": VERSION,
            "author": "project-authored-by-codex",
            "human_review": "pending",
            "external_teacher": None,
            "license": "MIT (repository original exercises)",
        },
    }
    record["id"] = "v2-text-" + sha256(canonical(record))[:24]
    return record


def operand_family(call: dict) -> tuple[int, int]:
    return tuple(sorted((call["a"], call["b"])))


def selected_pairs(existing: set[tuple[int, int]], heldout: set[tuple[int, int]]) -> list[tuple[int, int]]:
    # The V1 majority (0,n) creates unusually many zero multiplication results.
    # Add nonzero pairs, without reusing a single heldout canonical family.
    bins = {digits: [] for digits in (1, 2, 3, 4)}
    for a in range(1, 100):
        for b in range(a, 100):
            if (a, b) not in existing | heldout:
                bins[len(str(a * b))].append((a, b))
    quotas = {1: 10, 2: 20, 3: 40, 4: 180}
    pairs = []
    for digits in (1, 2, 3, 4):
        candidates = sorted(bins[digits], key=lambda pair: sha256(canonical([VERSION, SEED, pair])))
        pairs.extend(candidates[: quotas[digits]])
    if len(pairs) < 250:
        remainder = sorted(
            {pair for bucket in bins.values() for pair in bucket} - set(pairs),
            key=lambda pair: sha256(canonical([VERSION, "fill", SEED, pair])),
        )
        pairs.extend(remainder[: 250 - len(pairs)])
    assert len(pairs) == len(set(pairs)) == 250
    assert not set(pairs) & (existing | heldout)
    return pairs


def result_values() -> list[int]:
    # 900 unique bounded values, explicitly covering negative and 1–4 digits.
    three = [100 + i * 899 // 249 for i in range(250)]
    four = [1000 + i * 8801 // 450 for i in range(451)]
    values = list(range(-99, 0)) + list(range(100)) + three + four
    assert len(values) == len(set(values)) == 900
    return values


def build_additions(base: list[dict]) -> tuple[list[dict], dict]:
    helper = original_helpers()
    train_calls = [row for row in base if row["split"] == "train" and row["task"] == "tool_call"]
    existing = {operand_family(row["supervision"]["expected_call"]) for row in train_calls}
    heldout = {
        operand_family(row["supervision"]["expected_call"])
        for row in base
        if row["split"] != "train" and isinstance(row.get("supervision", {}).get("expected_call"), dict)
    }
    pairs = selected_pairs(existing, heldout)
    rows = []
    for index, (left, right) in enumerate(pairs):
        group = f"tools:operands:{left:02d}:{right:02d}"
        for operation_index, operation in enumerate(OPERATIONS):
            for direction, (a, b) in enumerate(((left, right), (right, left))):
                # Keep twelve descendants even for equal operands by choosing
                # two different known training expression families.
                family_index = (index + operation_index + direction) % 4
                template = helper.TOOL_TEMPLATES["train"][family_index][operation_index]
                question = template.format(a=a, b=b)
                call_text = serialize_tool_call(operation, a, b)
                call = json.loads(call_text)
                result = execute_tool_call(call_text)
                for style in helper.FORMATS:
                    prompt = helper.style_history(style, direction, tool=True) + [message("user", question)]
                    labels = {
                        "format": style,
                        "expected_call": call,
                        "expected_result": result["result"],
                        "expected_tool": True,
                        "operand_family": group,
                        "template_family": f"tools:train:expression-family-{family_index:02d}",
                        "parent_template_source": "scripts/selftrained/prepare_text_tools.py:TOOL_TEMPLATES.train",
                        "semantic_all": [],
                        "semantic_any": [],
                        "depends_on_history": True,
                    }
                    rows.append(
                        new_record(
                            group,
                            "tool_call",
                            prompt + [message("assistant", call_text)],
                            kind="normal_nonzero_calculator_call",
                            **labels,
                        )
                    )
                    reply_labels = {**labels, "expected_tool": False, "replay_kind": "actual_executor"}
                    rows.append(
                        new_record(
                            group,
                            "tool_reply",
                            prompt
                            + [
                                message("assistant", call_text),
                                message("tool", serialize_tool_result(result)),
                                message("assistant", helper.result_answer(result["result"], style)),
                            ],
                            kind="normal_nonzero_calculator_reply",
                            **reply_labels,
                        )
                    )
    values = result_values()
    for index, value in enumerate(values):
        # This is a structured-return reading exercise, without an arithmetic
        # request or a purported calculator execution. No false tool trace is
        # presented; normal roundtrip rows above use the real executor.
        for style in helper.FORMATS:
            question = (
                "請讀取接下來tool訊息中的result數字值，依照指定格式回答。"
                if index % 2 == 0
                else "回覆接下來的tool返回值；答案要使用result欄位的數字。"
            )
            payload = {"tool": "calculator", "ok": True, "result": value}
            messages = helper.style_history(style, index % 2, tool=True) + [
                message("user", question),
                message("tool", serialize_tool_result(payload)),
                message("assistant", helper.result_answer(value, style)),
            ]
            rows.append(
                new_record(
                    f"v2:result-reading:{value}",
                    "tool_reply",
                    messages,
                    kind="isolated_tool_result_reading",
                    format=style,
                    expected_result=value,
                    expected_tool=False,
                    replay_kind="structured_value_reading_train_only",
                    protocol_fixture=True,
                    template_family=f"v2:train:tool-value-reading-{index % 2}",
                    semantic_all=[],
                    semantic_any=[],
                    depends_on_history=True,
                )
            )
    rng = random.Random(SEED)
    copy_values = list(values)
    rng.shuffle(copy_values)
    copy_templates = (
        "數字值是{value}，請原樣抄寫這個數字字串。",
        "只複製下面的字串，不要計算：{value}",
        "請照抄這個值並保留負號：{value}",
        "練習讀取數字，直接寫出{value}。",
        "差值字串為{value}，請逐字複製，不作運算。",
    )
    for index in range(1500):
        value = copy_values[index % len(copy_values)]
        template_index = (index // len(copy_values) + index) % len(copy_templates)
        question = copy_templates[template_index].format(value=value)
        messages = [
            message("system", "這是字串抄寫練習；只抄寫使用者提供的數字，保留每個字元，不作計算。"),
            message("user", "這次只回覆數字字串。"),
            message("assistant", "好，請提供字串。"),
            message("user", question),
            message("assistant", str(value)),
        ]
        rows.append(
            new_record(
                f"v2:number-copy:{value}",
                "text_pretrain",
                messages,
                kind="signed_digit_string_copy",
                format=None,
                expected_numeric_string=str(value),
                expected_tool=False,
                template_family=f"v2:train:digit-copy-family-{template_index}",
                semantic_all=[],
                semantic_any=[],
            )
        )
    for index in range(600):
        a, b = pairs[index % len(pairs)]
        if index % 2:
            a, b = b, a
        operation = OPERATIONS[index % 3]
        call_text = serialize_tool_call(operation, a, b)
        call = json.loads(call_text)
        # Slot-copy scaffolding is supplied by the user's request, not hidden
        # gold labels. These 600 exercises complement the 3000 natural-language
        # calls; no computed result enters either first prompt.
        question = (
            f"將這組運算資料整理成計算器JSON呼叫：operation={operation}；a={a}；b={b}。"
            if index % 2 == 0
            else f"請組成計算器JSON呼叫，欄位資料為b={b}、operation={operation}、a={a}。"
        )
        style = helper.FORMATS[index % 2]
        prompt = helper.style_history(style, index % 2, tool=True) + [
            message("user", question),
            message("assistant", call_text),
        ]
        group = f"tools:operands:{min(a, b):02d}:{max(a, b):02d}"
        rows.append(
            new_record(
                group,
                "tool_call",
                prompt,
                kind="explicit_operation_operand_slot_copy",
                expected_call=call,
                expected_result=execute_tool_call(call_text)["result"],
                expected_tool=True,
                format=style,
                operand_family=group,
                template_family=f"v2:train:slot-copy-family-{index % 2}",
                semantic_all=[],
                semantic_any=[],
            )
        )
    # Ninety independent numeric families, each with a paired missing-argument
    # versus conceptual-reading request. Full arithmetic requests remain in
    # the normal call pool; these contrast targets do not invent missing data.
    for index in range(90):
        number = 1 + index
        style = helper.FORMATS[index % 2]
        operation = OPERATIONS[index % 3]
        word = {"add": "加", "subtract": "減", "multiply": "乘"}[operation]
        prompt = helper.style_history(style, index % 2, tool=True)
        missing = f"要作{word}法，目前只給了一個數字{number}；另一個參數尚未提供。"
        missing_target = (
            "請提供完整的兩個整數與運算方式。"
            if style == "one_sentence"
            else "1. 計算參數還不完整。\n2. 請提供兩個整數與運算方式。"
        )
        rows.append(
            new_record(
                f"v2:missing-contrast:{number}",
                "tool_missing",
                prompt + [message("user", missing), message("assistant", missing_target)],
                kind="explicit_missing_second_operand",
                format=style,
                expected_tool=False,
                template_family=f"v2:train:missing-operation-{operation}",
                semantic_all=["整數"],
                semantic_any=["提供"],
            )
        )
        concept = f"字串{number}只是背景資料，請解釋加法概念；這不是計算題。"
        concept_target = (
            "加法是把兩個數合起來，字串讀取不需要計算器。"
            if style == "one_sentence"
            else "1. 加法是把兩個數合起來。\n2. 字串讀取不需要計算器。"
        )
        rows.append(
            new_record(
                f"v2:concept-contrast:{number}",
                "tool_concept",
                prompt + [message("user", concept), message("assistant", concept_target)],
                kind="explicit_noncalculation_with_visible_number",
                intent="scope",
                format=style,
                expected_tool=False,
                template_family="v2:train:concept-with-number",
                semantic_all=["兩個數"],
                semantic_any=["合"],
            )
        )
    return rows, {
        "heldout_operand_families_excluded": [list(pair) for pair in sorted(heldout)],
        "original_train_operand_families": len(existing),
        "added_operand_families": [list(pair) for pair in pairs],
        "result_reading_unique_values": len(values),
        "seed": SEED,
    }


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".v2-text-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_and_copy_base(source: Path, destination: Path, frozen: dict) -> dict:
    expected_records = {item["path"]: item for item in frozen["records"]}
    if set(expected_records) != set(RECORD_NAMES):
        raise ValueError("frozen manifest must list exactly the twelve known record files")
    for name, entry in expected_records.items():
        path = source / name
        if file_sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"original source record differs from frozen V1 manifest: {name}")
    recipe = json.loads((ROOT / "docs/selftrained/package-recipe.json").read_text(encoding="utf-8"))
    packaged_sources = {entry["path"]: ROOT / entry["source"] for entry in recipe["source_files"]}
    identities = {}
    for entry in frozen["assets"]:
        path = source / entry["path"]
        # Four packaging notices are stored in the repository/source cache
        # rather than the producers' default data root. Resolve them through
        # the already pinned V1 packaging recipe; their hashes still must be
        # exactly the frozen V1 assets, never newly authored replacements.
        if not path.exists():
            path = packaged_sources[entry["path"]]
        if file_sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"original source asset differs from frozen V1: {entry['path']}")
        target = destination / entry["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        elif target.is_symlink() or file_sha(target) != entry["sha256"]:
            raise ValueError(f"existing destination asset differs from frozen V1: {entry['path']}")
        identities[entry["path"]] = entry["sha256"]
    # Walk regular files only for byte identity/copy, never interpret all JSONL
    # files as records. In particular voice-audit is protected metadata only.
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("base data cannot contain links")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        target = destination / relative
        original_sha = file_sha(path)
        identities[relative] = original_sha
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        elif relative in OTHER_AUGMENTABLE | {TEXT_TRAIN}:
            if not target.read_bytes().startswith(path.read_bytes()):
                raise ValueError(f"existing augmentation has lost original byte prefix: {relative}")
        elif target.is_symlink() or file_sha(target) != original_sha:
            raise ValueError(f"heldout/base destination changed: {relative}")
    return identities


def augment(source_data: Path, output_data: Path) -> dict:
    source, output = source_data.resolve(), output_data.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("source and output data roots must be separate")
    frozen_path = ROOT / "docs/selftrained/manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    source_identities = verify_and_copy_base(source, output, frozen)
    # Exactly twelve record filenames, deliberately excluding voice-audit.
    base = [
        json.loads(line)
        for name in RECORD_NAMES
        for line in (source / name).read_text(encoding="utf-8").splitlines()
        if line
    ]
    additions, generation = build_additions(base)
    ids = {row["id"] for row in base}
    heldout_groups = {row["group_id"] for row in base if row["split"] != "train"}
    heldout_templates = {row.get("supervision", {}).get("template_family") for row in base if row["split"] != "train"}
    heldout_prompts = {sha256(canonical(row["messages"][:-1])) for row in base if row["split"] != "train"}
    for row in additions:
        if row["id"] in ids or row["group_id"] in heldout_groups:
            raise ValueError("augmentation ID/group conflicts with source or heldout")
        ids.add(row["id"])
        if row["supervision"]["template_family"] in heldout_templates:
            raise ValueError("augmentation uses a heldout template family")
        if sha256(canonical(row["messages"][:-1])) in heldout_prompts:
            raise ValueError("augmentation duplicates a heldout prompt")
        assert row["split"] == "train" and row["messages"][-1]["role"] == "assistant"
        assert all(set(item) == {"role", "content"} for item in row["messages"])
        call = row["supervision"].get("expected_call")
        if call and operand_family(call) in {tuple(pair) for pair in generation["heldout_operand_families_excluded"]}:
            raise ValueError("augmentation operand family crosses heldout")
    original = (source / TEXT_TRAIN).read_bytes()
    if original and not original.endswith(b"\n"):
        raise ValueError("original train bytes must end in newline")
    extra_bytes = b"".join(canonical(row) + b"\n" for row in additions)
    combined = original + extra_bytes
    current = (output / TEXT_TRAIN).read_bytes()
    if current != original and current != combined:
        prior_path = output.parent / "manifests/text-tools-v2-augmentation.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else {}
        # During local authoring, only an exact previous output of this owned
        # producer may be refreshed. An unknown append is never overwritten.
        if (
            prior.get("version") != VERSION
            or prior.get("output", {}).get("sha256") != sha256(current)
            or prior.get("output", {}).get("original_bytes_preserved") != len(original)
            or prior.get("parent_file_sha256", {}).get(TEXT_TRAIN) != sha256(original)
            or prior.get("producer", {}).get("path") != "scripts/selftrained/augment_text_tools_v2.py"
        ):
            raise ValueError("refusing to replace unknown text/tools train augmentation")
    atomic_bytes(output / TEXT_TRAIN, combined)
    # All non-text files are unchanged; preserve other owners' append-only
    # OCR/voice train work when this producer is invoked after their work.
    for relative, expected_sha in source_identities.items():
        path = output / relative
        if relative == TEXT_TRAIN:
            assert path.read_bytes().startswith(original)
        elif relative in OTHER_AUGMENTABLE:
            assert path.read_bytes().startswith((source / relative).read_bytes())
        elif file_sha(path) != expected_sha:
            raise ValueError(f"a heldout/media/base file changed: {relative}")
    original_train = [row for row in base if row["task"] == "tool_call" and row["split"] == "train"]
    added_calls = [
        row for row in additions if row["supervision"]["augmentation_kind"] == "normal_nonzero_calculator_call"
    ]
    packaged_names = {entry["path"] for entry in frozen["records"] + frozen["assets"]}
    stable_parent_sha = {name: value for name, value in source_identities.items() if name in packaged_names}
    assert set(stable_parent_sha) == packaged_names
    manifest = {
        "version": VERSION,
        "seed": SEED,
        "split": "train",
        "external_teacher": None,
        "human_review": "pending",
        "producer": {"path": "scripts/selftrained/augment_text_tools_v2.py", "sha256": file_sha(Path(__file__))},
        "parent_manifest": {"path": "docs/selftrained/manifest.json", "sha256": file_sha(frozen_path)},
        "reused_sources": {
            "scripts/selftrained/prepare_text_tools.py": file_sha(ROOT / "scripts/selftrained/prepare_text_tools.py"),
            "tiny_perceptron/selftrained/tools.py": file_sha(ROOT / "tiny_perceptron/selftrained/tools.py"),
        },
        "record_filenames": sorted(RECORD_NAMES),
        "parent_file_sha256": stable_parent_sha,
        "parent_packaged_files_sha256": sha256(canonical(stable_parent_sha)),
        "parent_package_recipe": {
            "path": "docs/selftrained/package-recipe.json",
            "sha256": file_sha(ROOT / "docs/selftrained/package-recipe.json"),
        },
        "output": {
            "path": TEXT_TRAIN,
            "original_bytes_preserved": len(original),
            "original_rows": len(original.splitlines()),
            "added_rows": len(additions),
            "added_bytes": len(extra_bytes),
            "bytes": len(combined),
            "sha256": sha256(combined),
        },
        "counts_by_task": dict(sorted(Counter(row["task"] for row in additions).items())),
        "counts_by_exercise": dict(
            sorted(Counter(row["supervision"]["augmentation_kind"] for row in additions).items())
        ),
        "normal_call_result_digits_before": dict(
            sorted(Counter(len(str(abs(row["supervision"]["expected_result"]))) for row in original_train).items())
        ),
        "normal_call_result_digits_added": dict(
            sorted(Counter(len(str(abs(row["supervision"]["expected_result"]))) for row in added_calls).items())
        ),
        "generation": generation,
        "audit": {
            "original_train_byte_prefix": True,
            "all_validation_test_bytes_unchanged": True,
            "all_media_and_nonrecord_source_bytes_unchanged": True,
            "heldout_operand_overlap": 0,
            "heldout_template_overlap": 0,
            "heldout_exact_prompt_overlap": 0,
            "new_ids_unique": True,
            "other_train_augmentations": "OCR/voice may retain append-only additions; their bytes are never rewritten",
            "unpackaged_cache": "existing raw/cache source files also byte-checked in this run; volatile cache hashes are not serialized as package inputs",
        },
        "limits": [
            "No trained capability claim; validation/test and acceptance thresholds remain V1.",
            "Structured-value reading rows are copying exercises, not claimed calculator executions.",
            "Targets are deterministic arithmetic/copy labels or authored contrast replies; no model-generated teacher.",
            "Additional rows do not increase tool draw probability under the old equal-bucket sampler.",
            "Rebuild train-only tokenizer for V2; 值/差 occur in general copy exercises, not copied heldout templates.",
        ],
    }
    manifest_path = output.parent / "manifests/text-tools-v2-augmentation.json"
    atomic_bytes(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=ROOT / "outputs/selftrained/data")
    parser.add_argument("--output-data", type=Path, default=ROOT / "outputs/selftrained-v2/data")
    args = parser.parse_args()
    manifest = augment(args.source_data, args.output_data)
    print(
        json.dumps(
            {
                "version": VERSION,
                "output": manifest["output"],
                "counts_by_task": manifest["counts_by_task"],
                "audit": manifest["audit"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
