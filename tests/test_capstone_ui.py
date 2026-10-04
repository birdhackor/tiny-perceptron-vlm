"""Exercise real local HTTP generation and the playground's input boundaries."""

import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tiny_perceptron import capstone_ui
from tiny_perceptron.capstone import CapstoneModel, run_assistant
from tiny_perceptron.model import ModelConfig
from tiny_perceptron.multimodal import scene, tone


@pytest.fixture
def playground():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    model = CapstoneModel(ModelConfig(width=8, layers=1, heads=2, kv_heads=1, max_length=192, experts=2, top_k=1))
    server = capstone_ui.create_server(model, port=0, stage="sft")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    torch.set_num_threads(previous)


def request(server, path="/api/chat", data=None, *, method="POST", headers=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    body = json.dumps(data).encode() if raw is None and method == "POST" else raw
    supplied = {"Content-Type": "application/json", **(headers or {})}
    try:
        connection.request(method, path, body=body, headers=supplied)
        response = connection.getresponse()
        content = response.read()
        if response.getheader("Content-Type", "").startswith("application/json"):
            content = json.loads(content)
        return response.status, content
    finally:
        connection.close()


def test_http_generates_the_same_actual_model_trace(playground):
    fields = {"prompt": "1+2等於多少？", "calculator_available": True, "style": "短"}
    expected = run_assistant(playground.model, capstone_ui.request_row(fields))
    status, actual = request(playground, data=fields)
    assert status == 200
    assert actual == expected
    assert actual["action_trace"]["generated_ids"]
    assert "expected_final" not in actual


def test_previews_are_real_inputs_not_text_answers(playground):
    data = {"image": {"color": "green", "shape": "circle"}, "audio": {"frequency": 510}}
    row = capstone_ui.request_row(data, require_prompt=False)
    status, actual = request(playground, "/api/preview", data)
    assert status == 200
    expected_rgb = (scene("green", "circle") * 0.9).permute(1, 2, 0).mul(255).round().int()
    assert actual["image_pixels"] == expected_rgb.tolist()
    assert actual["audio_waveform"] == tone(510, seconds=0.04).tolist()
    assert row["audio"]["delta"] == 70
    assert actual["sample_rate"] == 16000
    assert len(actual["audio_waveform"]) == 640
    assert "answer" not in actual


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"prompt": " "},
        {"prompt": "x", "calculator_available": "false"},
        {"prompt": "x", "style": "詩意"},
        {"prompt": "x", "image": {"color": "red", "shape": "triangle"}},
        {"prompt": "x", "image": {"color": "red", "shape": "square", "path": "/tmp/photo.png"}},
        {"prompt": "x", "audio": {"frequency": True}},
        {"prompt": "x", "audio": {"frequency": 0}},
        {"prompt": "x", "audio": {"frequency": float("nan")}},
        {"prompt": "字" * 100},
        {"prompt": "x", "expected_answer": "3"},
    ],
)
def test_invalid_input_is_rejected_before_model_runs(playground, monkeypatch, data):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid input must not reach inference")

    monkeypatch.setattr(capstone_ui, "run_assistant", forbidden)
    status, error = request(playground, data=data)
    assert status == 400
    assert error["error"]


@pytest.mark.parametrize(
    "headers,raw,status",
    [
        ({"Content-Type": "text/plain"}, b"{}", 415),
        ({"Content-Length": "9000"}, b"{}", 413),
        ({"Origin": "https://elsewhere.example"}, b"{}", 403),
        ({"Host": "rebinding.example:8765"}, b"{}", 403),
        ({}, b"{broken", 400),
    ],
)
def test_request_boundaries(playground, headers, raw, status):
    actual, error = request(playground, headers=headers, raw=raw)
    assert actual == status
    assert error["error"]


def test_page_and_model_information_are_local_and_accessible(playground):
    status, html = request(playground, "/", method="GET")
    assert status == 200
    assert 'lang="zh-Hant"' in html.decode()
    assert "小世界" in html.decode()
    assert "https://" not in html.decode()
    assert "textContent" in html.decode()
    status, info = request(playground, "/api/info", method="GET")
    assert status == 200
    assert info["stage"] == "sft"
    assert info["parameters"] == playground.model.description()["parameters"]
    assert request(playground, "/missing", method="GET")[0] == 404


def test_requests_serialize_generation_and_do_not_replace_failures(playground, monkeypatch):
    active = 0
    maximum = 0
    state_lock = threading.Lock()
    record = {
        "action_trace": {"raw": "TOOL:calculator:1+2", "generated_ids": [50], "eos": True},
        "parsed_action": {"status": "tool"},
        "runtime": {"status": "error", "reason": "calculator_unavailable"},
        "final_trace": None,
        "answer": None,
    }

    def inference(model, row):
        nonlocal active, maximum
        assert row["available"] is False
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with state_lock:
            active -= 1
        return record

    monkeypatch.setattr(capstone_ui, "run_assistant", inference)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(
                lambda _: request(playground, data={"prompt": "1+2等於多少？", "calculator_available": False}), range(3)
            )
        )
    assert maximum == 1
    assert all(status == 200 and actual == record for status, actual in results)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.4", "gpu.example"])
def test_server_cannot_accidentally_publish_the_model(host):
    with pytest.raises(ValueError, match="loopback"):
        capstone_ui.create_server(None, host=host)
