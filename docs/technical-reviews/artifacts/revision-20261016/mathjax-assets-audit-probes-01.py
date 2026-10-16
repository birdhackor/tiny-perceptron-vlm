"""獨立 MathJax 資產審核；第三方 bytes、截圖及暫存網站只留 outputs。"""

import copy
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import stage_mathjax as stage

ART = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/mathjax-assets-independent-audit"
OUTPUT.mkdir(parents=True, exist_ok=True)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def main():
    started = datetime.now(UTC).isoformat()
    manifest_path = ROOT / "course/web/mathjax-assets.json"
    manifest = json.loads(manifest_path.read_bytes())
    upstream = json.loads((ART / "mathjax-assets-independent-source-01.json").read_bytes())
    tree = {x["path"]: x for x in upstream["entries"]}
    verified = []
    for item in manifest["files"]:
        raw = (ROOT / "outputs/mathjax-cache/3.2.2" / item["path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        original = tree[item["upstream_path"]]
        assert len(raw) == item["bytes"] == original["bytes"]
        assert sha(raw) == item["sha256"]
        assert blob == item["git_blob_sha1"] == original["git_blob_sha1"]
        verified.append({"path": item["path"], "bytes": len(raw), "sha256": sha(raw), "git_blob_sha1": blob})
    sample = OUTPUT / "font-sample.woff"
    assert sample.read_bytes() == (ROOT / "outputs/mathjax-cache/3.2.2/output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff").read_bytes()
    print(json.dumps({"phase": "upstream_identity", "files": len(verified), "bytes": sum(x["bytes"] for x in verified)}), flush=True)
    results = []
    commands = []

    def record(name, operation, expected="pass"):
        try:
            detail = operation()
        except Exception as error:
            results.append({"id": name, "expected": expected, "actual": "rejected", "error_type": type(error).__name__, "details": str(error)})
        else:
            results.append({"id": name, "expected": expected, "actual": "accepted", "details": detail})

    with tempfile.TemporaryDirectory(prefix="mathjax-audit-", dir=OUTPUT) as temporary:
        work = Path(temporary)
        fixture_count = 0

        def fixture():
            nonlocal fixture_count
            fixture_count += 1
            root = work / f"fixture-{fixture_count}"
            cache = root / "outputs/mathjax-cache/3.2.2"
            value = {k: copy.deepcopy(v) for k, v in manifest.items() if k != "files"}
            value["files"] = []
            for name in ("LICENSE", "tex-chtml.js", "output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff"):
                raw = ("synthetic-independent-fixture:" + name).encode()
                path = cache / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                value["files"].append({"path": name, "upstream_path": name if name == "LICENSE" else "es5/" + name, "bytes": len(raw), "sha256": sha(raw)})
            source = root / "manifest.json"
            source.write_text(json.dumps(value), encoding="utf-8")
            return root, cache, source, value

        def offline_positive():
            root, cache, source, value = fixture()
            previous = stage.urllib.request.urlopen
            stage.urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network called for complete cache"))
            try:
                prepared = stage.prepare_assets(root, source)
                dest = root / "destination"
                stage.stage_assets(dest, prepared)
                assert all((dest / f["path"]).read_bytes() == (cache / f["path"]).read_bytes() for f in value["files"])
            finally:
                stage.urllib.request.urlopen = previous
            return "all files copied byte-for-byte without network"

        record("complete_cache_and_stage_positive", offline_positive)

        def download_case(kind):
            root, cache, source, value = fixture()
            target = cache / "tex-chtml.js"
            original = target.read_bytes()
            target.unlink()
            urls, reads = [], []

            class Response(io.BytesIO):
                def read(self, size=-1):
                    reads.append(size)
                    return super().read(size)

            def remote(url, timeout):
                urls.append(url)
                assert timeout == 30
                if kind == "http503":
                    raise urllib.error.HTTPError(url, 503, "independent fixture", {}, None)
                raw = original if kind == "valid" else b"" if kind == "empty" else original + b"extra" if kind == "oversized" else b"wrong"
                return Response(raw)

            previous = stage.urllib.request.urlopen
            stage.urllib.request.urlopen = remote
            error = None
            try:
                stage.prepare_assets(root, source)
            except Exception as caught:
                error = caught
            finally:
                stage.urllib.request.urlopen = previous
            expected_url = "https://raw.githubusercontent.com/mathjax/MathJax/" + value["revision"] + "/es5/tex-chtml.js"
            assert urls == [expected_url]
            if kind != "http503":
                assert reads == [len(original) + 1]
            if error:
                assert not target.exists(), "failed download wrote cache"
                raise error
            assert target.read_bytes() == original
            return {"urls": urls, "read_limits": reads, "cache_bytes_match": True}

        for kind in ("valid", "empty", "oversized", "wrong", "http503"):
            record("download_" + kind, lambda k=kind: download_case(k), "pass" if kind == "valid" else "reject")

        for unsafe in ("../outside", "/absolute", "a/../b", "a\\b", "./alias", "a//b", "."):
            def bad_path(name=unsafe):
                _, _, _, value = fixture()
                item = copy.deepcopy(value["files"][0])
                item.update(path=name, upstream_path="es5/" + name)
                value["files"].append(item)
                return stage.validate_manifest(value)
            record("relative_path_" + repr(unsafe), bad_path, "reject")

        for name, mutate in (
            ("single_file_limit", lambda v: v["files"][0].update(bytes=2 * 1024 * 1024 + 1)),
            ("duplicate_path", lambda v: v["files"].append(copy.deepcopy(v["files"][0]))),
            ("wrong_repository", lambda v: v.update(repository="unrelated/project")),
            ("floating_revision", lambda v: v.update(revision="main")),
            ("wrong_version", lambda v: v.update(version="latest")),
            ("wrong_license", lambda v: v.update(license="unknown")),
        ):
            def malformed(change=mutate):
                _, _, _, value = fixture()
                change(value)
                return stage.validate_manifest(value)
            record(name, malformed, "reject")

        def total_limit():
            _, _, _, value = fixture()
            for i in range(9):
                value["files"].append({"path": f"padding/{i}.js", "upstream_path": f"es5/padding/{i}.js", "bytes": 2 * 1024 * 1024, "sha256": "0" * 64})
            return stage.validate_manifest(value)
        record("total_byte_limit", total_limit, "reject")

        def corrupted_cache():
            root, cache, source, _ = fixture()
            (cache / "tex-chtml.js").write_bytes(b"corrupt")
            return stage.prepare_assets(root, source)
        record("corrupt_cache_rejected", corrupted_cache, "reject")

        def cache_symlink(parent=False):
            root, cache, source, _ = fixture()
            outside = root / "outside"
            outside.mkdir()
            if parent:
                shutil.rmtree(cache / "output")
                (cache / "output").symlink_to(outside, target_is_directory=True)
            else:
                (cache / "LICENSE").unlink()
                (outside / "license").write_bytes(b"outside")
                (cache / "LICENSE").symlink_to(outside / "license")
            return stage.prepare_assets(root, source)
        record("cache_file_symlink_rejected", cache_symlink, "reject")
        record("cache_parent_symlink_rejected", lambda: cache_symlink(True), "reject")

        def staged_cache_corruption():
            root, cache, source, _ = fixture()
            prepared = stage.prepare_assets(root, source)
            (cache / "LICENSE").write_bytes(b"changed after prepare")
            return stage.stage_assets(root / "destination", prepared)
        record("stage_rechecks_cache_bytes", staged_cache_corruption, "reject")

        def destination_symlink(parent=False):
            root, cache, source, value = fixture()
            prepared = stage.prepare_assets(root, source)
            destination = root / "destination"
            destination.mkdir()
            outside = root / "outside"
            outside.mkdir()
            victim = outside / "victim"
            victim.write_bytes(b"outside sentinel")
            if parent:
                (destination / "output").symlink_to(outside, target_is_directory=True)
            else:
                (destination / "LICENSE").symlink_to(victim)
            stage.stage_assets(destination, prepared)
            return {"outside_changed": victim.read_bytes() != b"outside sentinel" if not parent else (outside / "chtml/fonts/woff-v2/MathJax_Main-Regular.woff").exists(), "normal_manifest": True, "normal_prepared_tuple": True}
        record("destination_file_symlink_must_not_escape", destination_symlink, "reject")
        record("destination_parent_symlink_must_not_escape", lambda: destination_symlink(True), "reject")

        # 單一真實冷下載；其餘93份用已獨立核對的cache，避免重抓全部資產。
        cold = work / "real-cold-download"
        cold_cache = cold / "outputs/mathjax-cache/3.2.2"
        shutil.copytree(ROOT / "outputs/mathjax-cache/3.2.2", cold_cache)
        (cold_cache / "LICENSE").unlink()
        network_start = datetime.now(UTC).isoformat()
        try:
            stage.prepare_assets(cold, manifest_path)
            cold_result = {"status": "downloaded_and_verified", "license_sha256": sha((cold_cache / "LICENSE").read_bytes())}
        except Exception as error:
            cold_result = {"status": "network_unavailable", "error_type": type(error).__name__, "details": str(error)}
        commands.append({"command": "stage.prepare_assets(temporary_root_with_93_verified_assets, actual_manifest); missing LICENSE downloaded from immutable raw URL", "started_at": network_start, "completed_at": datetime.now(UTC).isoformat(), "result": cold_result})
        print(json.dumps({"phase": "cold_download", "result": cold_result}), flush=True)

        # CLI和module各建置一次；所有生成檔留獨立clone，避免碰共用preview/config。
        clone = work / "repository"
        clone.mkdir()
        for name in ("docs", "course", "notebooks", "scripts", ".agents", "assets", "tiny_perceptron", "tests", ".github"):
            shutil.copytree(ROOT / name, clone / name, copy_function=os.link, symlinks=True)
        for path in ROOT.iterdir():
            if path.is_file() and not path.is_symlink() and path.name != ".zensical-build.toml":
                os.link(path, clone / path.name)
        shutil.copytree(ROOT / "outputs/mathjax-cache/3.2.2", clone / "outputs/mathjax-cache/3.2.2")
        for mode, entry in (("cli", [str(clone / "scripts/export_course.py")]), ("module", ["-m", "scripts.export_course"])):
            destination = clone / "outputs" / (mode + "-site")
            command = [sys.executable, *entry, "--output", str(destination), "--reading-times", str(clone / "outputs/absent-audit-reading-times.json")]
            began = datetime.now(UTC).isoformat()
            result = subprocess.run(command, cwd=clone, capture_output=True, text=True, check=False)
            record_value = {"command": command, "started_at": began, "completed_at": datetime.now(UTC).isoformat(), "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            if result.returncode == 0:
                record_value["staged_hash_mismatches"] = [f["path"] for f in manifest["files"] if not (destination / "javascripts/mathjax" / f["path"]).is_file() or sha((destination / "javascripts/mathjax" / f["path"]).read_bytes()) != f["sha256"]]
                record_value["all_94_staged_hashes_match"] = not record_value["staged_hash_mismatches"]
                html = (destination / "9.15.html").read_text()
                scripts = []
                class ScriptURLs(HTMLParser):
                    def handle_starttag(self, tag, attrs):
                        if tag == "script":
                            scripts.append(dict(attrs).get("src"))
                parser = ScriptURLs()
                parser.feed(html)
                record_value["script_urls"] = scripts
                resolved = [urljoin("https://audit.invalid/9.15.html", url) for url in scripts if url]
                record_value["resolved_script_urls"] = resolved
                record_value["uses_local_mathjax_entry"] = "https://audit.invalid/javascripts/mathjax/tex-chtml.js" in resolved and not any(url.startswith("https://cdn.jsdelivr.net/npm/mathjax") for url in resolved)
            commands.append(record_value)
            print(json.dumps({"phase": mode + "_build", "exit_code": result.returncode}), flush=True)

    # 新browser context封鎖站外，親測CHTML與本地WOFF；截圖在ignored outputs。
    from playwright.sync_api import sync_playwright
    browser_results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
        for mode, viewport in (("desktop", {"width": 1280, "height": 900}), ("mobile", {"width": 390, "height": 844})):
            context = browser.new_context(viewport=viewport)
            blocked, responses = [], []
            def route(request):
                if request.request.url.startswith("http://127.0.0.1:8891/"):
                    request.continue_()
                else:
                    blocked.append(request.request.url)
                    request.abort()
            context.route("**/*", route)
            page = context.new_page()
            page.on("response", lambda response: responses.append(response))
            page.goto("http://127.0.0.1:8891/9.15.html", wait_until="networkidle")
            page.wait_for_function('window.MathJax && MathJax.startup && MathJax.startup.document && document.querySelectorAll("mjx-container").length > 0', timeout=15000)
            screenshot = OUTPUT / ("9.15-" + mode + "-formula.png")
            page.locator(".arithmatex").first.screenshot(path=str(screenshot))
            local_assets = []
            expected = {f["path"]: f for f in manifest["files"]}
            for response in responses:
                if "/javascripts/mathjax/" in response.url:
                    name = response.url.split("/javascripts/mathjax/", 1)[1]
                    raw = response.body()
                    local_assets.append({"path": name, "status": response.status, "sha256": sha(raw), "manifest_match": name in expected and sha(raw) == expected[name]["sha256"]})
            value = {"mode": mode, "viewport": viewport, "url": page.url, "mathjax_version": page.evaluate("MathJax.version"), "math_nodes": page.locator("mjx-container").count(), "arithmatex_nodes": page.locator(".arithmatex").count(), "math_errors": page.locator("mjx-merror").count(), "local_assets": local_assets, "blocked_external_requests": blocked, "formula_screenshot": file_ref(screenshot)}
            value["checks_passed"] = value["mathjax_version"] == "3.2.2" and value["math_nodes"] == value["arithmatex_nodes"] == 1 and value["math_errors"] == 0 and bool(local_assets) and all(x["status"] == 200 and x["manifest_match"] for x in local_assets)
            browser_results.append(value)
            context.close()
        browser.close()

    unexpected = [x["id"] for x in results if (x["expected"] == "reject") == (x["actual"] == "accepted")]
    unexpected += ["export_integrity_" + str(i) for i, command in enumerate(commands) if "exit_code" in command and (command["exit_code"] != 0 or not command.get("all_94_staged_hashes_match") or not command.get("uses_local_mathjax_entry"))]
    unexpected += ["browser_" + value["mode"] for value in browser_results if not value["checks_passed"]]
    output = {"schema_version": 1, "reviewer_task": "/root/tech_20261016_review_tools", "started_at": started, "completed_at": datetime.now(UTC).isoformat(), "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/mathjax-assets-audit-probes-01.py", "environment": {"python": sys.version, "platform": platform.platform(), "torch_distribution": importlib.metadata.version("torch"), "playwright": importlib.metadata.version("playwright"), "browser": "/usr/bin/chromium"}, "inputs": [file_ref(ROOT / name) for name in ["scripts/stage_mathjax.py", "scripts/export_course.py", "course/web/mathjax-assets.json", "course/web/mathjax.js", "zensical.toml", "tests/test_stage_mathjax.py"]], "upstream_identity": {"source_receipt": file_ref(ART / "mathjax-assets-independent-source-01.json"), "verified_files": verified, "font_sample": file_ref(sample), "bytes": sum(x["bytes"] for x in verified)}, "results": results, "unexpected_results": unexpected, "commands": commands, "browser": browser_results}
    target = ART / "mathjax-assets-audit-probes-results-01.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"probes": len(results), "unexpected_results": unexpected, "builds": [(x.get("exit_code"), x.get("all_94_staged_hashes_match")) for x in commands if "exit_code" in x], "browser_modes": len(browser_results)}), flush=True)


if __name__ == "__main__":
    main()
