"""只以Volume SDK下載已結束attempt；不啟動容器、不預約、不重跑訓練。"""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.lesson_products.checkpoints import TRANSFER_FILES, save_transferred
from scripts.lesson_products.modal_runner import POLICY_PATH, POLICY_SHA
from scripts.selftrained.modal_runner import canonical_sha, read_live_ledger, safe_name, write_json

ROOT = Path(__file__).resolve().parents[2]


def read_bounded(volume, name, limit):
    parts, total = [], 0
    for chunk in volume.read_file(name):
        total += len(chunk)
        if total > limit:
            raise ValueError("已提交Volume檔案超過下載上限：" + name)
        parts.append(chunk)
    return b"".join(parts)


def inspect_attempt(volume, live, run_id, expected_revision, destination):
    safe_name(run_id)
    entry = next((x for x in live["ledger"]["reservations"] if x["run_id"] == run_id), None)
    if (
        not entry
        or entry.get("experiment_id") != "selftrained_lesson_products"
        or entry["status"] not in ("completed", "failed", "failed-or-cancelled")
    ):
        raise ValueError("只下載已確認結束的成品attempt")
    prefix = f"lesson-products/{run_id}"
    raw = read_bounded(volume, prefix + "/execution.json", 4 * 1024 * 1024)
    receipt = json.loads(raw)
    if (
        receipt.get("run_id") != run_id
        or receipt.get("code_revision") != expected_revision
        or entry.get("revision") != expected_revision
        or canonical_sha(receipt) != entry.get("receipt_sha256")
    ):
        raise ValueError("遠端完成紀錄與原帳本receipt／source revision不符")
    if receipt.get("status") != entry["status"] or receipt.get("source_hashes") != entry.get("source_sha256"):
        raise ValueError("實際完成狀態或來源不符原預約")
    files = receipt.get("client_artifacts", {})
    if not set(files) <= set(TRANSFER_FILES) or sum(item["bytes"] for item in files.values()) > 20 * 1024 * 1024:
        raise ValueError("attempt下載不在已封存的有限artifact清單內")
    artifacts = {name: read_bounded(volume, prefix + "/" + name, expected["bytes"]) for name, expected in files.items()}
    destination = Path(destination)
    save_transferred({"receipt": receipt, "artifacts": artifacts}, destination / "files")
    write_json(destination / "execution.json", receipt)
    result = {
        "run_id": run_id,
        "status": receipt["status"],
        "code_revision": expected_revision,
        "execution_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_canonical_sha256": canonical_sha(receipt),
        "downloaded_files": files,
        "gpu_started_by_inspection": False,
        "remote_container_started": False,
        "ledger_written": False,
        "billing_snapshot": live["live"]["gross_quota_snapshot"],
        "scope": "Read existing terminal attempt and SHA-verified artifacts only; does not rerun training or release retained bounds.",
    }
    write_json(destination / "inspection.json", result)
    return result


def main():
    import modal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lesson-products-inspect")
    args = parser.parse_args()
    policy = json.loads((ROOT / POLICY_PATH).read_bytes())
    live = read_live_ledger(policy, POLICY_SHA)
    volume = modal.Volume.from_name("tiny-perceptron-course", create_if_missing=False)
    result = inspect_attempt(volume, live, args.run_id, args.expected_revision, args.output)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": result["status"],
                "files": len(result["downloaded_files"]),
                "remote_container_started": False,
            }
        )
    )


if __name__ == "__main__":
    main()
