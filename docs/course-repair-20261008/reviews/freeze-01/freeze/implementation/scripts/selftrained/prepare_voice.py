#!/usr/bin/env python3
"""Prepare genuine MInDS-14 zh-CN audio without pretrained ASR or answer leakage.

Run with the repository environment:
  .venv/bin/python scripts/selftrained/prepare_voice.py

Source labels are teaching supervision, not a promise of correct ASR transcripts.
No speaker/session identifiers exist in this source. The resulting evaluation is
within-source intent demonstration, never speaker-disjoint evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

REVISION = "40ce77cb32a384e4d50a568e1ec39ac804019d33"
PARQUET_SHA256 = "cea0246e1a54afe5a3ab9d9542baa53c7cb45fc8f4e28e7872ebafefb25ba99d"
PARQUET_SIZE = 32357378
SOURCE_ROOT = f"https://huggingface.co/datasets/PolyAI/minds14/resolve/{REVISION}"
INTENTS = (
    "abroad",
    "address",
    "app_error",
    "atm_limit",
    "balance",
    "business_loan",
    "card_issues",
    "cash_deposit",
    "direct_debit",
    "freeze",
    "high_value_payment",
    "joint_account",
    "latest_transactions",
    "pay_bill",
)
SELECTED = {"address": 0, "app_error": 1, "card_issues": 2}
SPLIT_SEED = "phase5-voice-source-groups-20261006-v1"
SYSTEM = "你是教學用的有限客服助手，用繁體中文簡短回答，遵守對話中的格式要求。"
ANSWERS = {
    "address": {
        "one_sentence": "請透過官方管道查詢地址更新方式。",
        "two_points": "1. 請使用官方管道。\n2. 詢問地址更新方式。",
    },
    "app_error": {
        "one_sentence": "先檢查網路並重新啟動App，仍失敗再詢問官方客服。",
        "two_points": "1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。",
    },
    "card_issues": {
        "one_sentence": "請先確認卡片無法使用的情況，再詢問官方客服。",
        "two_points": "1. 請先確認卡片無法使用的情況。\n2. 再詢問官方客服。",
    },
}
RUBRICS = {
    "address": "指向官方管道查詢地址更新方式，不宣稱已更新地址。",
    "app_error": "建議檢查網路、重新啟動App，持續失敗詢問官方客服。",
    "card_issues": "確認卡片無法使用的情況並詢問官方客服，不編造拒付原因。",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def fetch(path: Path, url: str, expected_sha256: str | None = None) -> dict:
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "tiny-perceptron-phase5-data/1"})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.rename(path)
    data = path.read_bytes()
    digest = sha256(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Source checksum mismatch: {path}")
    return {"file": path.name, "url": url, "bytes": len(data), "sha256": digest}


def normalized_transcript(text: str) -> str:
    """Conservative lexical duplicate key; no transcript correction or ASR."""
    text = "".join(c for c in unicodedata.normalize("NFKC", text).lower() if c.isalnum())
    return re.sub(r"^(?:喂你好|嗨你好|您好|你好|请问)|(?:谢谢你|谢谢)$", "", text)


class Groups:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def root(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.root(i)] = self.root(j)


def quality(wave: np.ndarray, sample_rate: int) -> dict:
    size = max(1, round(sample_rate * 0.02))
    padding = (-len(wave)) % size
    frames = np.pad(wave, (0, padding)).reshape(-1, size)
    energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    return {
        "duration_seconds": len(wave) / sample_rate,
        "num_samples": len(wave),
        "peak_amplitude": float(np.abs(wave).max()),
        "rms_amplitude": rms,
        "rms_dbfs": float(20 * np.log10(max(rms, 1e-12))),
        "dc_offset": float(wave.mean()),
        "samples_ge_0_97_fraction": float(np.mean(np.abs(wave) >= 0.97)),
        "frames_below_minus_50dbfs_fraction": float(np.mean(energy < 10 ** (-50 / 20))),
        "finite_samples": bool(np.isfinite(wave).all()),
        "automated_qc_only": True,
        "human_listening_performed": False,
    }


def acoustic_duplicate_edges(indices: list[int], waves: list[np.ndarray]) -> list[dict]:
    """Flag gain/crop duplicates, never identify speakers or classify speech.

    Average each 16-sample block (500 Hz check only), trim low-amplitude edges,
    compare pairs with >=80% duration overlap by sliding normalized correlation.
    The 0.995 threshold detects almost identical waveforms. Different utterances
    from one speaker cannot establish a speaker grouping from this procedure.
    """
    signatures = {}
    for i in indices:
        wave = waves[i]
        usable = wave[: len(wave) // 16 * 16].reshape(-1, 16).mean(axis=1).astype(np.float64)
        active = np.flatnonzero(np.abs(usable) > 0.002)
        if len(active):
            usable = usable[active[0] : active[-1] + 1]
        signatures[i] = usable - usable.mean()
    edges = []
    for i, j in itertools.combinations(indices, 2):
        a, b = signatures[i], signatures[j]
        if len(a) > len(b):
            a, b = b, a
        if len(a) < 250 or len(a) / len(b) < 0.8:
            continue
        width = 1 << (len(a) + len(b) - 1).bit_length()
        convolution = np.fft.irfft(np.fft.rfft(b, width) * np.fft.rfft(a[::-1], width), width)
        dots = convolution[len(a) - 1 : len(b)]
        cumulative = np.concatenate(([0.0], np.cumsum(b * b)))
        local_energy = cumulative[len(a) :] - cumulative[: -len(a)]
        denominator = np.sqrt(np.dot(a, a) * np.maximum(local_energy, 1e-20))
        correlation = float(np.max(dots / np.maximum(denominator, 1e-20)))
        if correlation >= 0.995:
            edges.append({"rows": [i, j], "reason": "near_identical_audio", "correlation": correlation})
    return edges


def split_groups(audit: list[dict], groups: Groups, selected: list[int]) -> dict[int, str]:
    by_class = defaultdict(lambda: defaultdict(list))
    for i in selected:
        by_class[audit[i]["intent"]][groups.root(i)].append(i)
    assignment = {}
    for intent in SELECTED:
        entries = list(by_class[intent].values())
        if len(entries) < 21:
            raise ValueError(f"Not enough independent lexical/recording groups for {intent}: {len(entries)}")
        # Multiple-recording duplicate groups go to train first. Then hash single
        # groups before training/model results exist: ten test, five validation.
        multiple = [g for g in entries if len(g) > 1]
        single = [g for g in entries if len(g) == 1]
        single.sort(key=lambda g: sha256((SPLIT_SEED + audit[g[0]]["source_path"]).encode()))
        if len(single) < 15:
            raise ValueError(f"Need 15 singleton heldout groups for {intent}, got {len(single)}")
        for position, group in enumerate(single):
            split = "test" if position < 10 else "validation" if position < 15 else "train"
            assignment.update({i: split for i in group})
        for group in multiple:
            assignment.update({i: "train" for i in group})
    # A duplicate relation across intent labels would invalidate this splitting.
    checked = defaultdict(set)
    for i, split in assignment.items():
        checked[groups.root(i)].add(split)
    if any(len(splits) > 1 for splits in checked.values()):
        raise ValueError("Cross-intent duplicate group would cross splits")
    return assignment


def history(format_name: str, switched: bool = False) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    first = "two_points" if switched else format_name
    request = "接下來請用兩點回答。" if first == "two_points" else "接下來請用一句回答。"
    confirmation = "好，我會用兩點回答。" if first == "two_points" else "好，我會用一句回答。"
    messages += [{"role": "user", "content": request}, {"role": "assistant", "content": confirmation}]
    if switched:
        messages += [{"role": "user", "content": "改成一句。"}, {"role": "assistant", "content": "好，我會改成一句。"}]
    return messages + [{"role": "user", "content": "請回答這段語音問題。"}]


def research_supplements(source_dir: Path, manifest_dir: Path, offline: bool) -> None:
    """Inspect actual official transcripts; never relabel news as banking QA."""
    revision = "bbe295d530192a4cd41644b711c9aecd087df653"
    research_sources = [
        (
            "AISHELL-transcript-tree.json",
            f"https://huggingface.co/api/datasets/AISHELL/AISHELL-1/tree/{revision}/data_aishell/transcript?expand=true",
            None,
        ),
        (
            "AISHELL-transcript.txt",
            f"https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/{revision}/data_aishell/transcript/aishell_transcript_v0.8.txt",
            "b5f33b9e0b47548e20a5ea4e504297f80df41d559133924c4e5b7b544c15b5c4",
        ),
        ("AISHELL-README.md", f"https://huggingface.co/datasets/AISHELL/AISHELL-1/raw/{revision}/README.md", None),
        ("AISHELL-openslr33.html", "https://www.openslr.org/33/", None),
        ("AISHELL-Apache-2.0.txt", "https://www.apache.org/licenses/LICENSE-2.0.txt", None),
    ]
    queries = ["CATSLU", "chinese intent", "Mandarin banking", "Chinese SLU"]
    for query in queries:
        research_sources.append(
            (
                f"search-{query}.json",
                "https://huggingface.co/api/datasets?search=" + urllib.parse.quote(query) + "&limit=20&full=true",
                None,
            )
        )
    if offline and any(not (source_dir / name).exists() for name, _, _ in research_sources):
        raise FileNotFoundError("Offline supplemental research requires the cached metadata and transcript files")
    receipts = [fetch(source_dir / name, url, digest) for name, url, digest in research_sources]
    transcript = (source_dir / "AISHELL-transcript.txt").read_bytes()
    tree = json.loads((source_dir / "AISHELL-transcript-tree.json").read_text())
    transcript_entry = next(x for x in tree if x["path"].endswith("aishell_transcript_v0.8.txt"))
    git_blob = hashlib.sha1(f"blob {len(transcript)}\0".encode() + transcript).hexdigest()
    if git_blob != transcript_entry["oid"]:
        raise ValueError("Official AISHELL transcript Git blob checksum mismatch")
    patterns = {
        "address": r"(?:更改|修改|更新|变更|换).{0,6}(?:地址|住址)|(?:地址|住址).{0,6}(?:更改|修改|更新|变更|换)",
        "app_error": r"(?:应用|程序|网银|银行App|手机银行).{0,12}(?:无法|不能|失败|错误|加载|不起作用|死机|没法)",
        "card_issues": r"(?:银行|信用|储蓄)卡.{0,10}(?:无效|不能|无法|拒绝|用不了)|(?:付款|支付).{0,6}(?:拒绝|失败)",
    }
    matches = {intent: [] for intent in SELECTED}
    lines = transcript.decode("utf-8").splitlines()
    for line in lines:
        utterance, text = line.split(" ", 1)
        text = text.replace(" ", "")
        for intent, pattern in patterns.items():
            if re.search(pattern, text):
                matches[intent].append({"id": utterance, "speaker_id": utterance[6:11], "text": text})
    write_json(
        manifest_dir / "voice-supplement-research.json",
        {
            "dataset": "AISHELL/AISHELL-1",
            "revision": revision,
            "author": "Beijing Shell Shell Technology Co., Ltd.; Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng (2017)",
            "official_corpus_license": "Apache-2.0 per OpenSLR33; official HF card retains academic-use prose",
            "license_source": "https://www.openslr.org/33/",
            "license_copy": "data/voice-sources/AISHELL-Apache-2.0.txt",
            "documents": receipts,
            "git_blob_verified": git_blob,
            "transcript_lines": len(lines),
            "transcript_modification": "Spaces removed only for research matching; original transcript retained byte-for-byte",
            "patterns": patterns,
            "actual_matches": matches,
            "finding": "Matches are unrelated news/declarative statements, not same-task first-person e-banking intent requests. No AISHELL audio or training rows incorporated.",
            "speaker_metadata": "Speaker resource IDs are available in AISHELL, but source/task mismatch prevents using these rows as target-task heldout speakers.",
            "huggingface_keyword_searches": {
                q: [d["id"] for d in json.loads((source_dir / f"search-{q}.json").read_text())] for q in queries
            },
            "search_limits": "Limited public metadata and lexical transcript research, not an exhaustive claim that no suitable Chinese corpus exists.",
        },
    )
    (manifest_dir / "voice-AISHELL-research-NOTICE.md").write_text(
        "# AISHELL-1 supplemental-source research\n\n"
        "Source: Beijing Shell Shell Technology Co., Ltd.; Hui Bu, Jiayu Du, Xingyu Na, "
        "Bengu Wu, Hao Zheng, AIShell-1: An Open-Source Mandarin Speech Corpus and "
        "A Speech Recognition Baseline (2017). https://www.openslr.org/33/ .\n\n"
        "Official transcript from AISHELL/AISHELL-1, revision "
        "`bbe295d530192a4cd41644b711c9aecd087df653`, is retained unchanged for "
        "source research only. OpenSLR declares Apache License 2.0; the full license "
        "is retained in `data/voice-sources/AISHELL-Apache-2.0.txt`. The official "
        "HF README, including its academic-use wording, is also retained.\n\n"
        "Changes in research results: source text spaces removed for lexical matching; "
        "five matches listed in the research manifest. No AISHELL audio was downloaded "
        "or added to the teaching/evaluation task because the matches are unrelated statements.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("outputs/selftrained/data"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/selftrained/manifests"))
    parser.add_argument("--offline", action="store_true", help="Use only already checksum-verified source files")
    parser.add_argument(
        "--research-supplements",
        action="store_true",
        help="Inspect AISHELL official full transcript and public metadata, without downloading its audio",
    )
    args = parser.parse_args()
    data_root = args.data_root
    manifest_dir = args.manifest_dir
    source_dir = data_root / "voice-sources"
    audio_dir = data_root / "audio"
    source_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    source_documents = [
        ("minds14-api.json", f"https://huggingface.co/api/datasets/PolyAI/minds14/revision/{REVISION}"),
        ("minds14-README.md", f"https://huggingface.co/datasets/PolyAI/minds14/raw/{REVISION}/README.md"),
        (
            "minds14-zh-CN-tree.json",
            f"https://huggingface.co/api/datasets/PolyAI/minds14/tree/{REVISION}/zh-CN?expand=true",
        ),
        ("minds14-info.json", "https://datasets-server.huggingface.co/info?dataset=PolyAI%2Fminds14&config=zh-CN"),
    ]
    if args.offline and any(not (source_dir / name).exists() for name, _ in source_documents):
        raise FileNotFoundError("Offline preparation requires the four source evidence files")
    documents = [fetch(source_dir / name, url) for name, url in source_documents]
    source_api = json.loads((source_dir / "minds14-api.json").read_text())
    if source_api["sha"] != REVISION or "license:cc-by-4.0" not in source_api["tags"]:
        raise ValueError("Unexpected source revision or license metadata")
    tree = json.loads((source_dir / "minds14-zh-CN-tree.json").read_text())
    entry = next(x for x in tree if x["path"] == "zh-CN/train-00000-of-00001.parquet")
    if entry["lfs"]["oid"] != PARQUET_SHA256 or entry["size"] != PARQUET_SIZE:
        raise ValueError("Source LFS metadata changed")
    source_path = source_dir / "minds14-zh-CN.parquet"
    if args.offline and not source_path.exists():
        raise FileNotFoundError(source_path)
    parquet_receipt = fetch(source_path, SOURCE_ROOT + "/zh-CN/train-00000-of-00001.parquet", PARQUET_SHA256)
    if parquet_receipt["bytes"] != PARQUET_SIZE:
        raise ValueError("Unexpected parquet size")
    rows = pq.read_table(source_path).to_pylist()
    if len(rows) != 502:
        raise ValueError(f"Expected the fixed 502-row source, received {len(rows)}")
    audit, waves = [], []
    for i, row in enumerate(rows):
        audio_bytes = row["audio"]["bytes"]
        information = sf.info(io.BytesIO(audio_bytes))
        wave, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if sample_rate != 8000 or wave.ndim != 1 or information.format != "WAV" or row["lang_id"] != 13:
            raise ValueError(f"Unexpected waveform metadata at row {i}")
        digest = sha256(audio_bytes)
        audio_file = audio_dir / f"{digest[:24]}.wav"
        if audio_file.exists() and sha256(audio_file.read_bytes()) != digest:
            raise ValueError(f"Opaque asset filename collision: {audio_file}")
        audio_file.write_bytes(audio_bytes)
        waves.append(wave)
        audit.append(
            {
                "source_row": i,
                "source_path": row["path"],
                "source_utterance_id": Path(row["path"]).stem,
                "source_revision": REVISION,
                "source_split": "train",
                "source_lang_id": row["lang_id"],
                "intent": INTENTS[row["intent_class"]],
                "intent_source_label": row["intent_class"],
                "transcription": row["transcription"],
                "english_transcription": row["english_transcription"],
                "transcription_provenance": "Source Google-ASR transcript; not manually verified listening gold",
                "transcript_contains_latin": bool(re.search("[A-Za-z]", row["transcription"])),
                "audio": str(audio_file.relative_to(data_root)),
                "audio_sha256": digest,
                "decoded_float32_sha256": sha256(wave.astype("<f4").tobytes()),
                "audio_bytes": len(audio_bytes),
                "sample_rate": sample_rate,
                "channels": information.channels,
                "format": information.format,
                "subtype": information.subtype,
                "speaker_id": None,
                "session_id": None,
                "synthetic": False,
                "quality": quality(wave, sample_rate),
            }
        )
    groups = Groups(len(rows))
    duplicate_edges = []
    for field in ["audio_sha256", "decoded_float32_sha256"]:
        known = {}
        for i, record in enumerate(audit):
            value = record[field]
            if value in known:
                groups.union(i, known[value])
                duplicate_edges.append({"rows": [known[value], i], "reason": field})
            known[value] = i
    normalized = [normalized_transcript(r["transcription"]) for r in rows]
    for i, j in itertools.combinations(range(len(rows)), 2):
        a, b = normalized[i], normalized[j]
        exact = a == b
        near = min(len(a), len(b)) >= 6 and min(len(a), len(b)) / max(len(a), len(b)) >= 0.8
        ratio = SequenceMatcher(None, a, b, autojunk=False).ratio() if near else 0
        if exact or ratio >= 0.9:
            groups.union(i, j)
            duplicate_edges.append(
                {
                    "rows": [i, j],
                    "reason": "normalized_transcript" if exact else "near_transcript",
                    "similarity": 1 if exact else ratio,
                }
            )
    selected = [i for i, record in enumerate(audit) if record["intent"] in SELECTED]
    acoustic_edges = acoustic_duplicate_edges(selected, waves)
    for edge in acoustic_edges:
        groups.union(*edge["rows"])
    duplicate_edges += acoustic_edges
    assignment = split_groups(audit, groups, selected)
    members = defaultdict(list)
    for i in range(len(rows)):
        members[groups.root(i)].append(audit[i]["source_utterance_id"])
    group_ids = {root: "voice-source-" + sha256("|".join(sorted(ids)).encode())[:20] for root, ids in members.items()}
    for i, record in enumerate(audit):
        record["group_id"] = group_ids[groups.root(i)]
        record["prepared_split"] = assignment.get(i)
        record["selected"] = i in assignment
    prepared = []
    for i in selected:
        source = audit[i]
        intent = source["intent"]
        opaque_id = "voice-" + sha256((REVISION + source["source_path"]).encode())[:20]
        for variant, format_name, switched in [
            ("one", "one_sentence", False),
            ("two", "two_points", False),
            ("switch", "one_sentence", True),
        ]:
            messages = history(format_name, switched) + [{"role": "assistant", "content": ANSWERS[intent][format_name]}]
            prepared.append(
                {
                    "id": opaque_id + "-" + variant,
                    "group_id": source["group_id"],
                    "split": assignment[i],
                    "task": "voice_qa",
                    "audio": source["audio"],
                    "modality_message_index": len(messages) - 2,
                    "messages": messages,
                    "supervision": {
                        "intent": intent,
                        "intent_id": SELECTED[intent],
                        "format": format_name,
                        "history_variant": variant,
                        "rubric": RUBRICS[intent],
                        "label_author": "Project-authored teaching response; source-provided intent label",
                    },
                }
            )
        for variant, initial_format, final_format, rewrite_request in [
            ("topic_one_to_two", "one_sentence", "two_points", "請將剛才的答覆改成兩點。"),
            ("topic_two_to_one", "two_points", "one_sentence", "請將剛才的答覆改成一句。"),
        ]:
            # These are legitimate supervised training conversations. Heldout
            # inference MUST reconstruct both turns using the evaluation indices:
            # [:4] -> actual first generation -> user[5] -> actual second generation.
            # The demonstrated topic answer at messages[4] is a scoring/training
            # reference, never a heldout inference history message.
            first_prompt = history(initial_format)
            messages = first_prompt + [
                {"role": "assistant", "content": ANSWERS[intent][initial_format]},
                {"role": "user", "content": rewrite_request},
                {"role": "assistant", "content": ANSWERS[intent][final_format]},
            ]
            prepared.append(
                {
                    "id": opaque_id + "-" + variant,
                    "group_id": source["group_id"],
                    "split": assignment[i],
                    "task": "voice_topic_continuation",
                    "audio": source["audio"],
                    "modality_message_index": 3,
                    "messages": messages,
                    "evaluation": {
                        "mode": "voice_topic_continuation",
                        "first_prompt_message_count": len(first_prompt),
                        "first_target_message_index": len(first_prompt),
                        "continuation_user_message_index": len(first_prompt) + 1,
                        "initial_format": initial_format,
                        "final_format": final_format,
                    },
                    "supervision": {
                        "intent": intent,
                        "intent_id": SELECTED[intent],
                        "format": final_format,
                        "initial_format": initial_format,
                        "history_variant": variant,
                        "rubric": RUBRICS[intent],
                        "label_author": "Project-authored teaching conversation; source-provided intent label",
                    },
                }
            )
    for split in ["train", "validation", "test"]:
        write_jsonl(data_root / f"voice-{split}.jsonl", [r for r in prepared if r["split"] == split])
    write_jsonl(data_root / "voice-audit.jsonl", audit)
    summary = {}
    for intent in INTENTS:
        records = [r for r in audit if r["intent"] == intent]
        durations = [r["quality"]["duration_seconds"] for r in records]
        summary[intent] = {
            "recordings": len(records),
            "duration_min_seconds": min(durations),
            "duration_max_seconds": max(durations),
            "duration_mean_seconds": float(np.mean(durations)),
            "duration_total_seconds": sum(durations),
            "recordings_2_to_6_seconds": sum(2 <= d <= 6 for d in durations),
            "split_recordings": dict(Counter(r["prepared_split"] for r in records if r["selected"])),
            "split_source_groups": {
                s: len({r["group_id"] for r in records if r["prepared_split"] == s})
                for s in ["train", "validation", "test"]
            },
            "rms_dbfs_range": [
                min(r["quality"]["rms_dbfs"] for r in records),
                max(r["quality"]["rms_dbfs"] for r in records),
            ],
            "high_peak_recordings": sum(r["quality"]["samples_ge_0_97_fraction"] > 0.001 for r in records),
            "low_rms_recordings": sum(r["quality"]["rms_dbfs"] < -35 for r in records),
            "subtypes": dict(Counter(r["subtype"] for r in records)),
        }
    manifest = {
        "schema_version": 2,
        "dataset": "PolyAI/minds14",
        "config": "zh-CN",
        "revision": REVISION,
        "preparation_script": {
            "path": "scripts/selftrained/prepare_voice.py",
            "sha256": sha256(Path(__file__).read_bytes()),
        },
        "author": "PolyAI; Daniela Gerz, Pei-Hao Su, Razvan Kusztos, Avishek Mondal, Michał Lis, Eshan Singhal, Nikola Mrkšić, Tsung-Hsien Wen, Ivan Vulić (2021)",
        "citation": "Multilingual and Cross-Lingual Intent Detection from Spoken Data, https://arxiv.org/abs/2104.08524",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_original_location": documents[1]["url"],
        "documents": documents,
        "parquet": parquet_receipt,
        "source_recordings": len(audit),
        "selected_recordings": len(selected),
        "intent_mapping": SELECTED,
        "asset_root": str(data_root),
        "source_labels_and_transcripts_file": "voice-audit.jsonl",
        "inference_input_fields": [
            "voice_qa: messages except final assistant target",
            "voice_topic_continuation first turn: messages[:evaluation.first_prompt_message_count]",
            "voice_topic_continuation second turn: first-turn prompt + actual model-generated first assistant + messages[evaluation.continuation_user_message_index]",
            "decoded audio waveform",
            "modality_message_index for structural soft-token placement only",
        ],
        "never_model_inputs": [
            "supervision",
            "source transcript",
            "intent",
            "source_path",
            "source_row",
            "asset filename",
            "group_id",
            "voice_topic_continuation demonstrated first assistant answer (heldout inference)",
        ],
        "topic_continuation_evaluation_contract": {
            "mode": "voice_topic_continuation",
            "first_prompt_message_count": 4,
            "first_target_message_index": 4,
            "continuation_user_message_index": 5,
            "modality_message_index": 3,
            "first_prompt_stops_after": "The first audio user's message",
            "second_turn_history": "Actual first model-generated answer, never the demonstrated reference answer",
            "metrics": [
                "initial_semantic",
                "initial_format",
                "final_semantic",
                "final_format",
                "end_to_end_semantic",
                "end_to_end_format",
                "end_to_end_semantic_and_format",
            ],
            "initial_and_final_references": "Project-authored replies corresponding to the source-provided intent; score-only/training supervision, not runtime answer tables",
        },
        "recording_domain": "Web-recorded human Chinese e-banking intent paraphrases; not real customer telephone calls",
        "modifications": [
            "Extracted original WAV bytes unchanged from checksum-verified official Parquet",
            "Opaque audio filenames",
            "Lexical/recording duplicate groups and deterministic splits",
            "Project-authored short teaching replies and three pre-audio format-history variants per recording",
            "Two post-audio topic-continuation format variants per recording, with actual first-generation history required during heldout evaluation",
        ],
        "audio_processing": "No crop, resampling, normalization, synthesis, ASR, pretrained model or perceptual speaker embeddings in preparation",
        "quality_limits": "Automated waveform QC only; no human listening was performed. Source ASR errors are preserved and never used as inference prompts or manual listening gold.",
        "grouping_limits": "Source machine transcription is a lexical grouping proxy. Unidentified speakers/sessions and mistranscribed re-recordings cannot be conclusively separated without listening and reliable metadata.",
        "speaker_disjoint": False,
        "session_disjoint": False,
        "evaluation_scope": "Within-source intent demonstration; unseen human speaker question answering remains unestablished",
        "split": {
            "seed": SPLIT_SEED,
            "test_groups_per_intent": 10,
            "validation_groups_per_intent": 5,
            "method": "Exact and >=0.90 normalized lexical similarity plus PCM and high waveform correlation grouping. Multi-recording groups train; hashed singleton groups test then validation before any model training.",
            "test_used_for_selection_or_thresholds": False,
        },
        "duplicate_edges": duplicate_edges,
        "class_summary": summary,
        "selected_source_rows": selected,
        "prepared_files": {
            f"voice-{s}.jsonl": {
                "rows": sum(r["split"] == s for r in prepared),
                "task_rows": dict(Counter(r["task"] for r in prepared if r["split"] == s)),
                "independent_recordings": sum(source["prepared_split"] == s for source in audit),
                "sha256": sha256((data_root / f"voice-{s}.jsonl").read_bytes()),
            }
            for s in ["train", "validation", "test"]
        },
    }
    write_json(manifest_dir / "voice.json", manifest)
    write_json(
        manifest_dir / "voice-quality-summary.json",
        {
            "source_recordings": len(audit),
            "automated_qc_only": True,
            "decode_failures": 0,
            "byte_duplicate_edges": sum(e["reason"] == "audio_sha256" for e in duplicate_edges),
            "pcm_duplicate_edges": sum(e["reason"] == "decoded_float32_sha256" for e in duplicate_edges),
            "selected_near_audio_edges": acoustic_edges,
            "classes": summary,
            "selected_source_transcript_latin_rows_requiring_listening": [
                i for i in selected if audit[i]["transcript_contains_latin"]
            ],
            "selected_high_peak_rows": [i for i in selected if audit[i]["quality"]["samples_ge_0_97_fraction"] > 0.001],
            "selected_low_rms_rows": [i for i in selected if audit[i]["quality"]["rms_dbfs"] < -35],
            "warning": "Flags are review aids. No source row was excluded by transcript language, waveform quality, model outcome or test result.",
        },
    )
    (manifest_dir / "voice-NOTICE.md").write_text(
        "# MInDS-14 Chinese teaching subset\n\n"
        "Audio and source transcripts attribution: PolyAI and the MInDS-14 authors (2021), "
        "CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ .\n\n"
        "Gerz et al., Multilingual and Cross-Lingual Intent Detection from Spoken Data: "
        "https://arxiv.org/abs/2104.08524 . Official source: https://huggingface.co/datasets/PolyAI/minds14 "
        f"revision `{REVISION}`. The original license statement is retained in `voice-sources/minds14-README.md`.\n\n"
        "Original WAV bytes are retained unchanged. Changes: opaque filenames, selected intents, "
        "duplicate groups, deterministic splits, original project teaching responses and format-history examples. "
        "Two topic-continuation examples per selected recording add a text format rewrite after the audio response; "
        "heldout evaluation must use the model's actual first response as history, never the demonstrated topic response. "
        "Project-written replies are MIT-licensed teaching examples, not verified banking procedures. "
        "No relationship or endorsement by PolyAI is implied.\n\n"
        "The source has no speaker or session identifiers. These splits cannot prove unseen-speaker capability. "
        "Source machine transcripts are audit labels only and are not listening gold or inference prompts. "
        "Automated decoding/quality checks ran; human listening did not.\n",
        encoding="utf-8",
    )
    if args.research_supplements:
        research_supplements(source_dir, manifest_dir, args.offline)
    print(
        json.dumps(
            {
                "selected_recordings": len(selected),
                "records": len(prepared),
                "splits": Counter(r["split"] for r in prepared),
                "class_summary": {k: summary[k] for k in SELECTED},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
