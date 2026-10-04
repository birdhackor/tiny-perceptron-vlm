"""Verify selection records and save exact original-source excerpts and derivations."""

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tiny_perceptron.capstone import (  # noqa: E402
    TOK,
    build_dataset,
    calculator_runtime,
    expected_final,
    parse_action,
    prompt_ids,
)

PREFIX = ROOT / "docs/technical-reviews/artifacts/fact_v2_19_01"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    splits, manifest = build_dataset()
    validation_rows = {row["id"]: row for row in splits["validation"]}
    validation_checks = {}
    for stage in ("joint", "dpo"):
        path = ROOT / f"docs/course-experiments/capstone-evidence/{stage}/validation.json"
        saved = json.loads(path.read_text())
        checks = []
        for record in saved["records"]:
            row = validation_rows[record["id"]]
            trace = record["action_trace"]
            assert trace["prompt_ids"] == prompt_ids(row)
            assert trace["raw"] == TOK.decode(trace["generated_ids"])
            assert trace["eos"] == (trace["generated_ids"][-1] == TOK.eos_id)
            assert record["expected_action"] == row["answer"]
            assert record["expected_final"] == expected_final(row)
            action = parse_action(trace)
            assert record["parsed_action"] == action
            answer = None
            if action["status"] in ("direct", "ask"):
                answer = action["content"]
                assert record["runtime"] is None and record["final_trace"] is None
            elif action["status"] == "tool":
                runtime = calculator_runtime(action, row["available"])
                assert runtime == record["runtime"]
                if runtime["status"] == "ok":
                    final = record["final_trace"]
                    followup = dict(row, image=None, audio=None)
                    followup["user"] = f"原題：{action['a']}+{action['b']}。計算器回報：{runtime['result']}。請回答。"
                    assert final["prompt_ids"] == prompt_ids(followup)
                    assert final["raw"] == TOK.decode(final["generated_ids"])
                    assert final["eos"] == (final["generated_ids"][-1] == TOK.eos_id)
                    parsed = parse_action(final)
                    if parsed["status"] == "direct":
                        answer = parsed["content"]
            first_ok = bool(trace["eos"] and trace["raw"] == row["answer"])
            final_ok = bool(first_ok and answer == expected_final(row))
            assert record["answer"] == answer
            assert record["action_correct"] == first_ok
            assert record["end_to_end_correct"] == final_ok
            checks.append(
                {
                    "id": row["id"],
                    "task": row["task"],
                    "action_correct_recomputed": first_ok,
                    "end_to_end_correct_recomputed": final_ok,
                }
            )
        assert len(checks) == 84 and len({row["id"] for row in checks}) == 84
        assert sum(row["end_to_end_correct_recomputed"] for row in checks) == saved["end_to_end_correct"]
        validation_checks[stage] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "count": len(checks),
            "end_to_end_correct_recomputed": saved["end_to_end_correct"],
            "rows": checks,
        }
    stage_configs = {}
    for stage in ("pretrain", "sft", "joint", "dpo"):
        path = ROOT / f"docs/course-experiments/capstone-evidence/{stage}/train-report.json"
        report = json.loads(path.read_text())
        assert report["data_manifest"] == manifest and report["schedule_completed"] is True
        stage_configs[stage] = {key: value for key, value in report.items() if key not in ("history", "data_manifest")}
        stage_configs[stage].update({"report_path": str(path.relative_to(ROOT)), "report_sha256": digest(path)})
    run_path = Path(str(PREFIX) + "_deployment_run.json")
    run = json.loads(run_path.read_text())
    selection = json.loads((ROOT / "docs/course-experiments/capstone-selection.json").read_text())
    selected = datetime.fromisoformat(selection["selected_at_utc"])
    started = datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00"))
    assert selected < started
    assert run["head_sha"] == "6ffc653199a71ad83afcee24aa8a2388122771bd"
    excerpt_ranges = {
        "llava_v2.txt": [(185, 196), (243, 259), (438, 460)],
        "toolformer_v1.txt": [(83, 124), (198, 212), (607, 653)],
        "helm_v1.txt": [(350, 385)],
        "torch_dtype.md": [(14, 35)],
        "hf_download.py": [(918, 930), (1008, 1025)],
        "hf_headers.py": [(38, 65), (124, 139)],
        "uv_cli.rs": [(3808, 3848), (3960, 3990)],
        "python_ipaddress.py": [(1400, 1408), (1578, 1584)],
    }
    excerpt_receipts = []
    for filename, ranges in excerpt_ranges.items():
        original_path = Path(str(PREFIX) + "_" + filename)
        lines = original_path.read_text().split("\n")
        text = []
        for first, last in ranges:
            text.append(f"ORIGINAL LINES {first}-{last}\n" + "\n".join(lines[first - 1 : last]))
        path = Path(str(PREFIX) + "_excerpt_" + filename.replace(".py", ".txt").replace(".rs", ".txt"))
        path.write_text("\n\n".join(text) + "\n")
        excerpt_receipts.append(
            {
                "original_file": str(original_path.relative_to(ROOT)),
                "original_sha256": digest(original_path),
                "excerpt_path": str(path.relative_to(ROOT)),
                "excerpt_sha256": digest(path),
                "original_line_ranges": ranges,
            }
        )
    derivation = """19.1 手算與執行對照（只核參數與當次總分，不估GPU速度）
字表 embedding: 264*64 = 16,896。
未綁定輸出表: 64*264 = 16,896。
每層 attention: q/out各64*64，k/v各64*32 => 12,288。
每層兩個RMSNorm: 2*64 = 128。
每位expert兩個Linear(有bias): (64*256+256)+(256*64+64) = 33,088。
每層4位expert: 4*33,088 = 132,352；router無bias: 64*4=256。
每層合計: 12,288+128+132,352+256 = 145,024，兩層290,048。
最後RMSNorm64；RGB接頭48*64+64=3,136；音訊接頭16*64+64=1,088。
全部16,896+16,896+290,048+64+3,136+1,088 = 328,128，與全部state_dict numel精確相同。
正式test各task分母依序12,12,6,6,9,9,18,6,3,3,3,3，總和90。
整題正確依序10,12,5,6,9,0,18,6,3,3,3,3，總和78（78/90=13/15=0.866666...）。
action_correct總80；calculator12/12第一段吻合，但第二次回答僅10/12。
範例1+2=3；錯誤案例0+1=1，runtime為1而模型回0，故整題失敗。
參數是實際全部儲存數；數字不含檔案容器、更新器、梯度或執行記憶體。
"""
    Path(str(PREFIX) + "_derivation.txt").write_text(derivation)
    receipt = {
        "command": " ".join(sys.argv),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "validation_checks": validation_checks,
        "stage_configuration_reports": stage_configs,
        "source_bound_training_configuration": {
            "batch_size": 24,
            "learning_rates": {"pretrain": 0.003, "sft": 0.003, "joint": 0.0015, "dpo": 0.0002},
            "optimizer": "torch.optim.AdamW with version defaults except lr",
            "sampling": "random.Random(seed + stage_index*1000); balanced sample across task groups",
            "auxiliary_weight": 0.01,
            "gradient_clip_norm": 1.0,
            "joint_dtype": "float32; no autocast",
            "evaluation": {
                "max_new_tokens": 64,
                "batch_size": 24,
                "strategy": "greedy argmax; KV cache; group equal prefix lengths; context limit 192",
            },
            "source_sha256": {
                name: digest(ROOT / name)
                for name in ["scripts/course_experiments/capstone.py", "tiny_perceptron/capstone.py"]
            },
            "configuration_basis": "_run_context passes no batch-size override; train_stage default24, stage learning rates and objective lines; formal reports carry requested/completed updates and effective target-token counts.",
        },
        "selection_time": selection["selected_at_utc"],
        "official_deployment_run_started_at": run["run_started_at"],
        "selection_precedes_deployment_run": True,
        "chronology_scope": "Saved selection time precedes official successful GitHub deployment job, which performs first official test. Cannot prove absence of unlogged runs.",
        "excerpt_receipts": excerpt_receipts,
    }
    Path(str(PREFIX) + "_supplement.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "validation_joint": validation_checks["joint"]["end_to_end_correct_recomputed"],
                "validation_dpo": validation_checks["dpo"]["end_to_end_correct_recomputed"],
                "validation_count_each": 84,
                "selection_precedes_deployment": True,
                "excerpt_count": len(excerpt_receipts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
