"""成品入口驗證真檔案SHA、候選平手規則、手動優先與端到端成本。"""

import hashlib
import json

import pytest
import torch

from tiny_perceptron.lesson_products.inference import Product, auto_solve, check_manifest, parse_state, rule_depth
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, Tokenizer


@pytest.mark.parametrize(
    "state",
    [
        "兩盒追蹤;A=0;B=2;事件=A設5,交換;查詢=B",
        "三數加法;x=?;y=0;z=?",
        "卡片進度;申請編號=?",
        "地址填寫規則",
        "App重設密碼;裝置=手機",
        "天氣預測",
    ],
)
def test_public_finite_grammar_accepts_missing_and_zero_without_computing_truth(state):
    assert parse_state(state)["task"] in ("boxes", "sum", "card", "address", "app", "none")


@pytest.mark.parametrize(
    "state",
    [
        "你好",
        "三數加法;x=7;y=1;z=2",
        "兩盒追蹤;A=0;Q=1;事件=交換;查詢=A",
        "兩盒追蹤;A=0;B=1;事件=A設9;查詢=A",
        "兩盒追蹤;A=0;B=1;事件=交換;查詢=P",
    ],
)
def test_outside_scope_or_inconsistent_entity_names_are_rejected(state):
    with pytest.raises(ValueError):
        parse_state(state)


def test_public_weights_sha_and_stable_semantic_tie_break(tmp_path):
    tokenizer = Tokenizer()
    tokenizer.save(tmp_path / "tokenizer.json")
    config = Config(len(tokenizer), width=16, heads=2, ff=32, c_layers=1)
    config.save(tmp_path / "config.json")
    model = CandidateScorer(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "calibration.json").write_text(json.dumps({"temperatures": {"choice": 1, "info": 1, "rubric": 1}}))
    names = ("model.pt", "config.json", "tokenizer.json", "calibration.json")
    manifest = {
        "version": 1,
        "product": "c",
        "files": {
            name: {
                "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
                "bytes": (tmp_path / name).stat().st_size,
            }
            for name in names
        },
    }
    (tmp_path / "artifact-manifest.json").write_text(json.dumps(manifest))
    product = Product(tmp_path, "c")
    for candidates in (("none", "card"), ("card", "none")):
        result = product.judge("卡片進度;申請編號=?", "choice", candidates)
        assert result["selected"] == "card" and result["concentration"] == 0
        assert result["probabilities"] == {x: 0.5 for x in candidates}
    (tmp_path / "calibration.json").write_text('{"temperatures": {"choice": 2}}')
    with pytest.raises(ValueError, match="指紋"):
        check_manifest(tmp_path, "c")


class FakeB:
    auto_policy = {"version": 1, "low_max_events": 1, "medium_max_events": 3, "sum_depth": 3}

    def synchronize(self):
        pass

    def solve(self, state, loops):
        return {"answer": 5, "loops": loops}


class FakeC:
    def synchronize(self):
        pass

    def judge(self, state, question):
        return {"selected": "boxes" if question == "choice" else "yes", "seconds": 0.25}


def test_manual_effort_overrides_auto_and_hard_cap_always_wins():
    state = "兩盒追蹤;A=0;B=2;事件=A設5;查詢=A"
    automatic = auto_solve(FakeB(), FakeC(), state)
    assert automatic["b_loops"] == 1
    manual = auto_solve(FakeB(), FakeC(), state, "high", 2)
    assert manual["requested_loops"] == 5 and manual["b_loops"] == 2
    assert manual["status"] == "budget_limited" and manual["c_seconds"] == 0.5
    assert rule_depth(FakeB.auto_policy, "sum") == 3


def test_auto_does_not_call_solver_when_C_rejects_complete_example():
    class Reject(FakeC):
        def judge(self, state, question):
            return {"selected": "boxes" if question == "choice" else "no", "seconds": 0.25}

    class MustNotSolve(FakeB):
        def solve(self, *args, **kwargs):
            raise AssertionError("不應呼叫B")

    result = auto_solve(MustNotSolve(), Reject(), "兩盒追蹤;A=0;B=2;事件=A設5;查詢=A")
    assert result["status"] == "clarification_needed" and result["b_loops"] == 0
