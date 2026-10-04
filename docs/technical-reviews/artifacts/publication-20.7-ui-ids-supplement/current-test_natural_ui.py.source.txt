"""Real HTTP / uploads / history tests, with no model-capability claims."""

import base64
import copy
import http.client
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from tiny_perceptron import natural_ui


def request(server, path="/api/chat", data=None, *, method="POST", headers=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    body = json.dumps(data).encode() if raw is None and method == "POST" else raw
    try:
        connection.request(method, path, body=body, headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        content = response.read()
        if response.getheader("Content-Type", "").startswith("application/json"):
            content = json.loads(content)
        return response.status, content
    finally:
        connection.close()


@pytest.fixture
def playground(tmp_path, monkeypatch):
    calls = []

    def runner(model, processor, row, data_root, options):
        calls.append(copy.deepcopy(row))
        for name in natural_ui.assistant.row_assets(row):
            assert natural_ui.assistant.asset_path(name, data_root).is_file()
        return {"prediction": "模型輸出：" + row["user"]}

    monkeypatch.setattr(natural_ui.assistant, "generate", runner)
    options = SimpleNamespace(adapter=None)
    server = natural_ui.create_server(object(), object(), options, tmp_path, port=0, asr=(object(), object()))
    server.calls = calls
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield server
    uploads = Path(server.upload_directory.name)
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    assert not uploads.exists()


def session(server):
    status, result = request(server, "/api/session", {})
    assert status == 200
    return result["session"]


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 16), (20, 50, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def audio_bytes(seconds=0.1):
    buffer = io.BytesIO()
    waveform = np.sin(2 * np.pi * 440 * np.arange(int(8000 * seconds)) / 8000).astype("float32")
    sf.write(buffer, waveform, 8000, format="WAV")
    return buffer.getvalue()


def upload(server, identifier, *, kind="image", content=None, filename=None):
    return request(
        server,
        "/api/upload",
        {
            "session": identifier,
            "kind": kind,
            "filename": filename or ("照片.png" if kind == "image" else "語音.wav"),
            "base64": base64.b64encode(content or (image_bytes() if kind == "image" else audio_bytes())).decode(),
        },
    )


def test_upload_and_chat_preserve_actual_image_and_every_turn(playground):
    identifier = session(playground)
    status, uploaded = upload(playground, identifier)
    assert status == 200
    image = uploaded["asset"]
    status, first = request(playground, data={"session": identifier, "prompt": "照片裡有什麼？", "image": image})
    assert status == 200
    assert first["prediction"] == "模型輸出：照片裡有什麼？"
    assert first["image"] == image
    assert first["asr"] is None
    assert playground.calls[0]["history"] == []
    status, second = request(playground, data={"session": identifier, "prompt": "它在做什麼？"})
    assert status == 200
    assert second["prediction"] == "模型輸出：它在做什麼？"
    previous = playground.calls[1]["history"]
    assert len(previous) == 2
    assert previous[0]["content"][0] == {"type": "image", "image": playground.calls[0]["image"]}
    assert previous[0]["content"][1] == {"type": "text", "text": "照片裡有什麼？"}
    assert previous[1]["content"] == [{"type": "text", "text": first["prediction"]}]
    assert "answer" not in playground.calls[0]
    query = urlencode({"session": identifier, "asset": image})
    assert request(playground, "/api/asset?" + query, method="GET") == (200, image_bytes())


def test_speech_is_two_explicit_stations_and_correction_reaches_same_core(playground, monkeypatch):
    identifier = session(playground)
    _, uploaded = upload(playground, identifier, kind="audio")
    speech = uploaded["asset"]
    called = []

    def transcribe(model, processor, path):
        called.append(path)
        assert path.read_bytes() == audio_bytes()
        return {"transcript": "你好請幫我看這張圖", "audio_seconds": 0.1, "audio_sha256": "test-fixture-only"}

    monkeypatch.setattr(natural_ui.assistant, "transcribe", transcribe)
    status, transcript = request(playground, "/api/transcribe", {"session": identifier, "audio": speech})
    assert status == 200
    assert transcript["asr"]["transcript"] == "你好請幫我看這張圖"
    assert playground.calls == []
    corrected = "你好，請幫我看這張照片。"
    status, answer = request(playground, data={"session": identifier, "prompt": corrected, "speech": speech})
    assert status == 200
    assert answer["prediction"] == "模型輸出：" + corrected
    assert answer["asr"]["transcript"] == transcript["asr"]["transcript"]
    assert answer["asr"]["submitted_text"] == corrected
    assert answer["asr"]["corrected"] is True
    assert playground.calls[0]["user"] == corrected
    assert len(called) == 1


def test_speech_loader_is_lazy_and_loaded_once(playground, monkeypatch):
    identifier = session(playground)
    _, uploaded = upload(playground, identifier, kind="audio")
    playground.asr = None
    loads = []
    pair = (object(), object())

    def loader(options):
        assert options is playground.options
        loads.append(options)
        return pair

    def transcribe(model, processor, path):
        assert (model, processor) == pair
        return {"transcript": "你好"}

    monkeypatch.setattr(natural_ui.assistant, "load_asr", loader)
    monkeypatch.setattr(natural_ui.assistant, "transcribe", transcribe)
    for _ in range(2):
        assert request(playground, "/api/transcribe", {"session": identifier, "audio": uploaded["asset"]})[0] == 200
    assert len(loads) == 1


def test_session_assets_and_history_are_isolated_and_no_arbitrary_paths(playground):
    first, second = session(playground), session(playground)
    _, uploaded = upload(playground, first)
    for image in (uploaded["asset"], "../../pyproject.toml", "/tmp/photo.png"):
        assert request(playground, data={"session": second, "prompt": "看圖", "image": image})[0] == 400
    assert request(playground, data={"session": first, "prompt": "你好"})[0] == 200
    assert request(playground, data={"session": second, "prompt": "您好"})[0] == 200
    assert playground.calls[1]["history"] == []
    query = urlencode({"session": second, "asset": uploaded["asset"]})
    assert request(playground, "/api/asset?" + query, method="GET")[0] == 400


def test_reset_clears_history_and_owned_uploads_and_shutdown_removes_directory(playground):
    identifier = session(playground)
    _, uploaded = upload(playground, identifier)
    assert request(playground, data={"session": identifier, "prompt": "看圖", "image": uploaded["asset"]})[0] == 200
    path = natural_ui.assistant.asset_path(playground.calls[0]["image"], playground.data_root)
    assert path.is_file()
    assert request(playground, "/api/reset", {"session": identifier}) == (200, {"reset": True})
    assert not path.exists()
    assert request(playground, data={"session": identifier, "prompt": "重新開始"})[0] == 200
    assert playground.calls[-1]["history"] == []
    assert playground.sessions[identifier]["assets"] == {}


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"prompt": " "},
        {"prompt": True},
        {"prompt": "x", "expected_answer": "假答案"},
        {"prompt": "x", "speech": "unknown"},
        {"prompt": "x", "speech": []},
        {"prompt": "x", "image": []},
        {"prompt": "字" * 3000},
    ],
)
def test_invalid_chat_cannot_reach_model(playground, data):
    if isinstance(data, dict):
        data = {"session": session(playground), **data}
    assert request(playground, data=data)[0] == 400
    assert playground.calls == []


@pytest.mark.parametrize(
    "kind,filename,content",
    [
        ("image", "photo.svg", b"<svg></svg>"),
        ("image", "../photo.png", image_bytes()),
        ("image", "photo.png", b"not an image"),
        ("audio", "sound.wav", b"not audio"),
        ("audio", "sound.wav", audio_bytes(30.01)),
        ("audio", "sound.txt", audio_bytes()),
    ],
    ids=["svg", "path-traversal", "invalid-image", "invalid-audio", "overlong-audio", "audio-extension"],
)
def test_invalid_or_long_upload_does_not_write_assets(playground, kind, filename, content):
    identifier = session(playground)
    status, error = upload(playground, identifier, kind=kind, filename=filename, content=content)
    assert status == 400
    assert error["error"]
    assert playground.sessions[identifier]["assets"] == {}
    assert list(Path(playground.upload_directory.name).iterdir()) == []


@pytest.mark.parametrize(
    "headers,raw,status",
    [
        ({"Content-Type": "text/plain"}, b"{}", 415),
        ({"Content-Length": str(natural_ui.MAX_REQUEST_BYTES + 1)}, b"{}", 413),
        ({"Origin": "https://elsewhere.example"}, b"{}", 403),
        ({"Host": "rebinding.example:8766"}, b"{}", 403),
        ({}, b"{broken", 400),
    ],
)
def test_request_boundaries(playground, headers, raw, status):
    assert request(playground, headers=headers, raw=raw)[0] == status


def test_failures_keep_the_previous_conversation_and_do_not_replace_answer(playground, monkeypatch):
    identifier = session(playground)
    assert request(playground, data={"session": identifier, "prompt": "你好"})[0] == 200
    before = copy.deepcopy(playground.sessions[identifier]["history"])

    def broken(*args):
        raise RuntimeError("test runner failed")

    monkeypatch.setattr(natural_ui.assistant, "generate", broken)
    status, result = request(playground, data={"session": identifier, "prompt": "下一句"})
    assert status == 500
    assert "prediction" not in result
    assert playground.sessions[identifier]["history"] == before
    assert request(playground, "/", method="GET")[0] == 200


def test_model_requests_are_serialized_and_append_each_turn_once(playground, monkeypatch):
    identifier = session(playground)
    active, maximum = 0, 0

    def runner(model, processor, row, root, options):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        time.sleep(0.02)
        active -= 1
        return {"prediction": row["user"]}

    monkeypatch.setattr(natural_ui.assistant, "generate", runner)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(lambda i: request(playground, data={"session": identifier, "prompt": str(i)}), range(3))
        )
    assert maximum == 1
    assert all(status == 200 for status, _ in results)
    assert len(playground.sessions[identifier]["history"]) == 6


def test_page_is_local_accessible_and_displays_text_without_html_execution(playground):
    status, page = request(playground, "/", method="GET")
    assert status == 200
    html = page.decode()
    assert 'lang="zh-Hant"' in html
    assert 'name="viewport"' in html
    assert "文字" in html and "照片" in html and "語音" in html
    assert "textContent" in html and "innerHTML" not in html
    assert "https://" not in html
    assert request(playground, "/missing", method="GET")[0] == 404


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.4", "gpu.example"])
def test_server_stays_local(tmp_path, host):
    with pytest.raises(ValueError, match="loopback"):
        natural_ui.create_server(None, None, None, tmp_path, host=host)
