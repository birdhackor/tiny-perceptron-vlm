"""Fetch preselected human read questions from the official AISHELL-1 source."""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PIN = "bbe295d530192a4cd41644b711c9aecd087df653"
BASE = f"https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/{PIN}"
PROOF = ROOT / "docs/natural-assistant/evidence/v4-research/chat-speech"
DESTINATION = ROOT / "outputs/natural-v4/data/voice"
LIMIT = 48 * 1024 * 1024
QUESTION_POLICY = "用繁體中文，以一至三句直接回應；資訊不足時先澄清，不捏造使用者背景或自己的親身經驗。"
# Explicit selection and semantic expectations are fixed before any ASR/model inference.
SPEC = [
    ("validation", "BAC009S0005W0205", "建議具體溝通方法，例如定期交流、傾聽或確認需求；不假設某家公司已發生衝突。"),
    ("validation", "BAC009S0007W0430", "不能声称实际拥有或使用实体八達通；可以說沒有實體交通卡、但可介紹用途。"),
    (
        "validation",
        "BAC009S0023W0297",
        "回應智慧設備的能力限制，例如依賴資料、情境、感測或人類設定；不编造個別設備事故。",
    ),
    ("validation", "BAC009S0019W0272", "给出平台互通的建議，如確認介面/API、資料格式與协议；不聲稱已经操作任何平台。"),
    ("test", "BAC009S0031W0414", "詢問是什麼考試或澄清無自己的應試成績；不得虚构分数或同学成绩。"),
    ("test", "BAC009S0039W0471", "不能虛構親自看電影的經歷；可以表示沒有觀看經驗、可協助討論電影。"),
    ("test", "BAC009S0042W0480", "『這樣的事情』缺少指代，应询问具体发生什么事；不凭空补情节。"),
    (
        "test",
        "BAC009S0032W0151",
        "直接提出產品定價或差異化建議，如了解客群/价值、比较竞争品、清楚呈现优势；无股价/时效断言。",
    ),
]


def read_bytes(url):
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "tiny-vlm-aishell-questions/1"}), timeout=60
    ) as response:
        return response.read()


class CountedReader:
    def __init__(self, response):
        self.response, self.bytes_read = response, 0

    def read(self, count=-1):
        if self.bytes_read >= LIMIT:
            raise ValueError("Bounded speaker archive read exceeded 48 MiB")
        value = self.response.read(min(count, LIMIT - self.bytes_read) if count >= 0 else LIMIT - self.bytes_read)
        self.bytes_read += len(value)
        return value


def retrieve(spec, annotations, archive_info):
    split, utterance_id, rubric = spec
    speaker = re.search(r"S\d{4}", utterance_id).group()
    archive_path = f"data_aishell/wav/{speaker}.tar.gz"
    url = f"{BASE}/{archive_path}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "tiny-vlm-aishell-questions/1"}), timeout=90
    ) as response:
        stream = CountedReader(response)
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            found = None
            for member in archive:
                if not member.isfile() or Path(member.name).name != utterance_id + ".wav":
                    continue
                raw = archive.extractfile(member).read()
                info = sf.info(io.BytesIO(raw))
                if info.samplerate != 16000 or info.channels != 1 or not 0.5 <= info.duration <= 25:
                    raise ValueError(f"Question audio outside route support: {member.name}")
                found = member.name
                break
        requested_bytes = stream.bytes_read
    if found is None:
        raise ValueError(f"Preselected utterance absent from pinned speaker archive: {utterance_id}")
    relative = Path("audio") / split / ("aishell1-" + utterance_id + ".wav")
    target = DESTINATION / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return {
        "id": "aishell1-v4-" + utterance_id,
        "split": split,
        "task": "voice_question",
        "family": "aishell1-speaker-" + speaker,
        "audio": str(relative),
        "user": annotations[utterance_id],
        "answer": None,
        "system": QUESTION_POLICY,
        "reference_kind": "Predeclared semantic rubric, not a source-authored assistant answer",
        "semantic_rubric": rubric,
        "history": [],
        "source": "aishell1-human-read-questions-v4",
        "source_utterance_id": utterance_id,
        "source_speaker_id": speaker,
        "source_member": found,
        "source_archive": {
            "url": url,
            "bytes_compressed_read": requested_bytes,
            "sha256_from_huggingface_lfs_metadata": archive_info[archive_path]["lfs"]["oid"],
            "full_archive_downloaded_and_verified": False,
        },
        "original_spaced_source_transcription": annotations[utterance_id + "-spaced"],
        "transcription_changes": "Removed source character-separating spaces for chat input; no character, script or punctuation correction",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_seconds": info.duration,
        "num_samples": info.frames,
        "synthetic": False,
        "recording_domain": "A person reading a question-shaped corpus utterance; not spontaneously recorded assistant conversation",
        "audio_inspection": "Metadata verified by SoundFile and official transcript read; audio not listened to, no ASR inference",
        "selection": "Utterance IDs, split/speaker and semantic rubrics predeclared in source script before download and model inference",
    }


def main():
    PROOF.mkdir(parents=True, exist_ok=True)
    text = (PROOF / "AISHELL1-transcript-first-1MiB.raw.txt").read_text(errors="strict").splitlines()[:-1]
    annotations = {}
    for line in text:
        item, spaced = line.split(maxsplit=1)
        annotations[item], annotations[item + "-spaced"] = "".join(spaced.split()), spaced
    tree_url = (
        f"https://huggingface.co/api/datasets/AISHELL/AISHELL-1/tree/{PIN}/data_aishell/wav?recursive=true&expand=true"
    )
    raw = read_bytes(tree_url)
    (PROOF / "AISHELL1-official-speaker-tree.json").write_bytes(raw)
    archive_info = {row["path"]: row for row in json.loads(raw)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda spec: retrieve(spec, annotations, archive_info), SPEC))
    speakers = [{row["source_speaker_id"] for row in rows if row["split"] == split} for split in ("validation", "test")]
    if speakers[0] & speakers[1] or len(set(row["source_speaker_id"] for row in rows)) != len(rows):
        raise ValueError("Question speaker families overlap")
    result = {
        "schema_version": 1,
        "dataset": "AISHELL/AISHELL-1",
        "revision": PIN,
        "official_corpus_license": "Apache-2.0",
        "official_corpus_license_source": "https://www.openslr.org/33",
        "license_scope": "OpenSLR 33 describes the speech corpus itself as Apache License v.2.0; official AISHELL HF card also marks Apache-2.0",
        "attribution": "Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng, AIShell-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline (2017); Beijing Shell Shell Technology Co., Ltd.",
        "modification_notice": "Selected original WAVs from official speaker archives; removed transcript inter-character spaces; project-authored response policy and semantic rubrics added; no synthetic audio or source transcript corrections",
        "audio_root": str(DESTINATION.relative_to(ROOT)),
        "audio_rows": rows,
        "counts": {split: sum(row["split"] == split for row in rows) for split in ("validation", "test")},
        "duration_seconds": sum(row["duration_seconds"] for row in rows),
        "audio_bytes": sum(row["bytes"] for row in rows),
        "speaker_family_disjointness_checked": True,
        "v3_overlap": "Old natural speech source is FLEURS. No AISHELL source utterance/speaker families occur in its manifest. No question audio from this source is used for v4 SFT.",
        "scoring_policy": {
            "instruction_and_content": "Respond directly to question or necessary clarification, no invented user facts or physical experience, Traditional Chinese, one to three sentences. Each requirement is checked separately before any combined pass.",
            "typed_vs_spoken": "Use same system/history with gold character-spaceless source question vs actual ASR prediction; do not insert gold text into spoken route.",
            "ASR_raw_CER_reference": "Original source character string with only annotation-separating whitespace removed; corpus has no punctuation, original Simplified script retained",
            "ASR_normalized_CER": "NFKC and all-whitespace removal, punctuation and original script retained; script-equivalence diagnostic reported separately if used",
            "EOS": "Stopping criterion is independent of response adequacy",
            "scope": "Small sample of human read questions only; not free speech-chat success rate or unseen foundation-pretraining-data claim",
        },
    }
    destination = ROOT / "docs/natural-assistant/v4/data/voice-question-sources.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("counts", "duration_seconds", "audio_bytes")}, indent=2))


if __name__ == "__main__":
    main()
