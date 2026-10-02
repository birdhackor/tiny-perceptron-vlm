"""單節原文／執行事實；不產生審閱結論，不執行 bash，不批次選課節。"""

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
OFFLINE = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def original_section(path, lesson):
    raw = path.read_bytes()
    raw.decode("utf-8")  # 只驗證編碼；不以解碼／重新編碼取代原始 bytes。
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    selected = [i for i, h in enumerate(headers) if re.match(rb"^## " + re.escape(lesson.encode()) + rb" ", h[0])]
    if len(selected) != 1:
        raise ValueError("source 中必須恰有一個指定小節")
    index = selected[0]
    start = headers[index].start()
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[start:end], raw, raw[:start].count(b"\n") + 1


def fences(raw, first_line):
    result, opened = [], None
    for index, line in enumerate(raw.splitlines(keepends=True)):
        marker = re.match(rb"^ {0,3}(`{3,}|~{3,})([^\r\n]*)", line)
        if opened is None:
            if marker:
                info = marker[2].strip().decode("utf-8")
                opened = {
                    "marker": marker[1],
                    "info": info,
                    "language": info.split()[0] if info else "",
                    "code_line": first_line + index + 1,
                    "lines": [],
                    "closed": False,
                }
        elif (
            marker
            and marker[1][:1] == opened["marker"][:1]
            and len(marker[1]) >= len(opened["marker"])
            and not marker[2].strip()
        ):
            opened["raw"] = b"".join(opened.pop("lines"))
            opened["closed"] = True
            result.append(opened)
            opened = None
        else:
            opened["lines"].append(line)
    if opened:
        opened["raw"] = b"".join(opened.pop("lines"))
        result.append(opened)
    return result


def extract(source, output):
    path_text, separator, lesson = source.rpartition("#")
    if not separator or not re.fullmatch(r"[A-Z\d]+\.\d+", lesson):
        raise ValueError("需要 repo-relative Markdown source#ID")
    relative = Path(path_text)
    original = (ROOT / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or original.suffix != ".md"
        or not original.is_relative_to(ROOT)
    ):
        raise ValueError("source 必須是 repo 內 Markdown 原稿；不接受 Notebook")
    output = output.resolve()
    if not (output.is_relative_to(ROOT / "outputs") or output.is_relative_to(Path("/tmp"))):
        raise ValueError("只可在 ignored outputs 或 /tmp 建立工具證據")
    output.mkdir(parents=True, exist_ok=False)
    body, whole, first_line = original_section(original, lesson)
    (output / "section.md").write_bytes(body)
    metadata = {
        "schema_version": 1,
        "kind": "section_extraction_facts",
        "source": source,
        "source_sha256": digest(body),
        "source_file_sha256": digest(whole),
        "section_first_line": first_line,
        "newline_policy": "original UTF-8 bytes; no stripping or newline normalization",
        "crlf_count": body.count(b"\r\n"),
        "python_fences": [],
        "other_fences": [],
        "svg_references": [],
        "figure_sha256": {},
        "helper_sha256": digest(Path(__file__).read_bytes()),
        "scope": "Extraction and execution facts only; reviewer must read original sources and judge claims independently.",
    }
    for number, item in enumerate(fences(body, first_line), 1):
        code = item.pop("raw")
        item.pop("marker")
        item.update(number=number, sha256=digest(code), bytes=len(code))
        if item["language"] in ("python", "python3", "py"):
            item["file"] = f"fence-{number}.py"
            (output / item["file"]).write_bytes(code)
            metadata["python_fences"].append(item)
        else:
            metadata["other_fences"].append(item)
    text = body.decode("utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    refs += re.findall(r'(?:src|href)=["\']([^"\']+)["\']', text)
    for reference in dict.fromkeys(refs):
        url = urlsplit(reference)
        if not url.path.lower().endswith(".svg"):
            continue
        item = {"reference": reference, "external": bool(url.scheme or url.netloc)}
        if item["external"]:
            item["snapshot"] = None
        else:
            target = (original.parent / url.path).resolve()
            item["inside_repo"] = target.is_relative_to(ROOT)
            item["exists"] = target.is_file() if item["inside_repo"] else False
            if item["inside_repo"]:
                name = target.relative_to(ROOT).as_posix()
                item["path"] = name
                if item["exists"]:
                    raw = target.read_bytes()
                    item["sha256"] = digest(raw)
                    metadata["figure_sha256"][name] = digest(raw)
                    snapshot = output / "figures" / name
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    snapshot.write_bytes(raw)
                    item["snapshot"] = snapshot.relative_to(output).as_posix()
        metadata["svg_references"].append(item)
    build = ROOT / "scripts/build_course.py"
    tree = ast.parse(build.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "BOOTSTRAP" for target in node.targets)
    )
    bootstrap = ast.literal_eval(node.value).encode("utf-8")
    (output / "bootstrap.py").write_bytes(bootstrap)
    metadata["bootstrap"] = {
        "repository_code": "scripts/build_course.py",
        "source_file_sha256": digest(build.read_bytes()),
        "bootstrap_sha256": digest(bootstrap),
        "snapshot": "bootstrap.py",
    }
    write_json(output / "extraction.json", metadata)
    return metadata


def worker(output):
    metadata = json.loads((output / "extraction.json").read_text())
    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "device_requested": "cpu",
        "cwd": str(Path.cwd()),
        "repo_import_root": str(ROOT),
        "offline_environment": OFFLINE,
        "guard_policy": "No socket network, child commands, or writes outside this artifact directory. This is a Python audit guard, not an OS sandbox.",
        "guard_events": [],
    }
    attempted = []

    def guard(event, arguments):
        blocked = event in (
            "socket.connect",
            "socket.getaddrinfo",
            "socket.bind",
            "socket.sendto",
            "subprocess.Popen",
            "os.system",
            "os.exec",
            "os.posix_spawn",
        )
        paths = []
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            flags = arguments[2]
            if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                paths = [arguments[0]]
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod"):
            paths = [arguments[0]]
        elif event in ("os.rename", "os.link", "os.symlink"):
            paths = [arguments[0], arguments[1]]
        for value in paths:
            path = Path(os.fsdecode(value)).resolve()
            if path != Path("/dev/null") and not path.is_relative_to(output):
                blocked = True
        if blocked:
            environment["guard_events"].append({"event": event, "reason": "CPU/offline artifact-only execution guard"})
            raise PermissionError(
                f"CPU/offline helper blocked {event}; this is an execution restriction, not a review verdict"
            )

    sys.addaudithook(guard)
    write_json(output / "environment.json", environment)
    namespace = {"__name__": "__main__"}
    try:
        bootstrap = (output / "bootstrap.py").read_bytes()
        if digest(bootstrap) != metadata["bootstrap"]["bootstrap_sha256"]:
            raise ValueError("bootstrap snapshot SHA changed")
        exec(compile(bootstrap, "scripts/build_course.py:BOOTSTRAP", "exec"), namespace)
        torch = namespace["torch"]
        environment.update(
            torch=str(torch.__version__),
            torch_git_version=str(torch.version.git_version),
            cuda_build=str(torch.version.cuda),
            cuda_available=str(torch.cuda.is_available()),
        )
        if torch.cuda.is_available() or torch.version.cuda is not None:
            raise RuntimeError("需要 repo .venv 的 CPU wheel；不以 CUDA wheel 或 GPU 執行此 helper")
        torch.set_default_device("cpu")
        write_json(output / "environment.json", environment)
        for item in metadata["python_fences"]:
            if not item["closed"]:
                raise ValueError("有未關閉的原始 Python fence；不猜測或修補")
            code = (output / item["file"]).read_bytes()
            if digest(code) != item["sha256"]:
                raise ValueError("original fence snapshot SHA changed")
            attempted.append(item["number"])
            environment["attempted_fences"] = attempted
            write_json(output / "environment.json", environment)
            name = f"{metadata['source']}:fence-{item['number']}:source-line-{item['code_line']}"
            exec(compile(code, name, "exec"), namespace)
    finally:
        environment["attempted_fences"] = attempted
        environment["repository_modules"] = {}
        for module_name, module in sorted(sys.modules.items()):
            module_path = getattr(module, "__file__", None)
            if module_path and module_name.startswith(("tiny_perceptron", "scripts.")):
                path = Path(module_path).resolve()
                if path.is_file() and path.is_relative_to(ROOT):
                    environment["repository_modules"][module_name] = {
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": digest(path.read_bytes()),
                    }
        write_json(output / "environment.json", environment)


def execute(output, metadata, timeout):
    if not metadata["python_fences"]:
        write_json(
            output / "execution.json",
            {
                "kind": "execution_facts",
                "executed": False,
                "reason": "No Python fences; nothing executed",
                "exit_code": None,
            },
        )
        return 2
    workspace = output / "workspace"
    workspace.mkdir()
    for name in ("pyproject.toml", "tiny_perceptron", "scripts", "assets", "course"):
        if (ROOT / name).exists():
            (workspace / name).symlink_to(ROOT / name, target_is_directory=(ROOT / name).is_dir())
    # 現有輸入只讀指向實際 repo；本次 outputs 與新檔案留在獨立工作目錄。
    for name in ("data", "checkpoints"):
        target = workspace / name
        target.mkdir()
        if (ROOT / name).is_dir():
            for existing in (ROOT / name).iterdir():
                (target / existing.name).symlink_to(existing, target_is_directory=existing.is_dir())
    for name in ("outputs", "tmp", "mpl-cache", "hf-cache"):
        (workspace / name).mkdir()
    env = {
        key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "LD_LIBRARY_PATH") if key in os.environ
    }
    env.update(
        OFFLINE,
        TMPDIR=str(workspace / "tmp"),
        MPLCONFIGDIR=str(workspace / "mpl-cache"),
        HF_HOME=str(workspace / "hf-cache"),
    )
    command = [str(ROOT / ".venv/bin/python"), "-I", str(Path(__file__).resolve()), "--worker", str(output)]
    started = time.perf_counter()
    result = {
        "kind": "execution_facts",
        "command_argv": command,
        "command": shlex.join(command),
        "helper_sha256": metadata["helper_sha256"],
        "environment_file": "environment.json",
        "cwd": str(workspace),
        "timeout_seconds": timeout,
        "executed": True,
        "source_sha256": metadata["source_sha256"],
    }
    with (output / "stdout.txt").open("wb") as stdout, (output / "stderr.txt").open("wb") as stderr:
        try:
            completed = subprocess.run(
                command, cwd=workspace, env=env, stdout=stdout, stderr=stderr, timeout=timeout, check=False
            )
            result["exit_code"] = completed.returncode
        except subprocess.TimeoutExpired:
            result.update(exit_code=124, timed_out=True)
        except OSError as error:
            result.update(
                exit_code=127, executed=False, process_start_error={"type": type(error).__name__, "message": str(error)}
            )
    result["elapsed_seconds"] = time.perf_counter() - started
    result["artifacts"] = [
        {"path": name, "sha256": digest((output / name).read_bytes()), "bytes": (output / name).stat().st_size}
        for name in ("stdout.txt", "stderr.txt", "environment.json")
        if (output / name).is_file()
    ]
    write_json(output / "execution.json", result)
    return result["exit_code"]


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]).resolve())
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="例如 course/chapters/14.md#14.1；只選一節")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="另啟新 CPU process 執行本節原始 Python fences")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        raise ValueError("CPU timeout 必須介於 1 至 600 秒")
    output = args.output or ROOT / "outputs/reviewer-tools/runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    metadata = extract(args.source, output)
    code = execute(output.resolve(), metadata, args.timeout) if args.execute else 0
    print(
        json.dumps(
            {
                "artifact_directory": str(output.resolve()),
                "source_sha256": metadata["source_sha256"],
                "python_fences": len(metadata["python_fences"]),
                "svg_references": len(metadata["svg_references"]),
                "execution_exit_code": code if args.execute else None,
            },
            ensure_ascii=False,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
