"""付費工作描述與取消收尾；全部offline，不能啟動Modal。"""

import copy
import json
from pathlib import Path

import pytest

from scripts.lesson_products.modal_runner import confirm_stopped, validate_job

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def job():
    return json.loads((ROOT / "docs/lesson-products/jobs/train-b-v1.json").read_bytes())


def test_committed_job_contracts_preserve_same_quota_and_bounded_resources():
    for path in (ROOT / "docs/lesson-products/jobs").glob("*.json"):
        value = validate_job(json.loads(path.read_bytes()))
        assert value["train_seconds"] <= value["wall_seconds"] - 300
        assert value["stage"] == "pretrain"
    fixed = json.loads((ROOT / "docs/lesson-products/jobs/mechanism-fixed3-v1.json").read_bytes())
    odd = json.loads((ROOT / "docs/lesson-products/jobs/mechanism-odd135-v1.json").read_bytes())
    assert {key: value for key, value in fixed.items() if key != "mode"} == {
        key: value for key, value in odd.items() if key != "mode"
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("wall_seconds", 3600),
        ("train_seconds", 1700),
        ("batch_size", True),
        ("learning_rate", float("nan")),
        ("seed", -1),
        ("depth_weights", [0, 1, 1, 1, 1]),
        ("product", "general_chat"),
    ],
)
def test_invalid_or_unbounded_job_is_rejected_before_remote_work(job, key, value):
    job[key] = value
    with pytest.raises(ValueError):
        validate_job(job)


def test_policy_cannot_be_changed_or_increased(job):
    job["gross_quota_policy"] = {**job["gross_quota_policy"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="額度"):
        validate_job(job)
    extra = copy.deepcopy(job)
    extra["additional_quota_usd"] = 300
    with pytest.raises(ValueError):
        validate_job(extra)


class TerminalOutput(Exception):
    pass


class FakeCall:
    def __init__(self, error=None):
        self.error = error
        self.events = []

    def cancel(self, terminate_containers):
        self.events.append(("cancel", terminate_containers))

    def get(self, timeout):
        self.events.append(("get", timeout))
        if self.error:
            raise self.error
        return {"status": "completed"}


@pytest.mark.parametrize("error", [None, TerminalOutput("server cancellation output")])
def test_terminal_output_is_required_after_container_termination_request(error):
    call = FakeCall(error)
    confirm_stopped(call, (TerminalOutput,))
    assert call.events == [("cancel", True), ("get", 30)]


@pytest.mark.parametrize("error", [TimeoutError("still running"), ConnectionError("status unavailable")])
def test_poll_timeout_or_network_failure_is_not_terminal_confirmation(error):
    call = FakeCall(error)
    with pytest.raises(type(error)):
        confirm_stopped(call, (TerminalOutput,))
