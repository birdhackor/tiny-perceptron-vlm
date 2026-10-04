"""Exercise C.1/W.1 commands in temporary repositories without changing shared Git config."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent


def main():
    commands = []

    def run(command, cwd=ROOT, env=None):
        result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
        commands.append(
            {
                "command": command,
                "cwd": str(cwd),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.returncode:
            raise RuntimeError(commands[-1])
        return result.stdout

    run(
        [
            "bash",
            "-c",
            "source .venv/bin/activate\npython -c 'import sys; print(sys.executable)'\npython scripts/check_env.py",
        ]
    )
    run([str(ROOT / ".venv/bin/python"), "scripts/course_experiments/run.py", "--help"])
    run([str(ROOT / ".venv/bin/python"), "scripts/course_experiments/run.py", "--list-assets", "reasoning"])
    run(["git", "lfs", "version"])
    run(["git", "lfs", "install", "-h"])
    run(["git", "lfs", "pull", "-h"])
    with tempfile.TemporaryDirectory(prefix="fact_finish_c_1_lfs-") as temporary:
        parent = Path(temporary)
        origin, producer, consumer = parent / "origin.git", parent / "producer", parent / "consumer"
        run(["git", "init", "--bare", str(origin)])
        run(["git", "init", str(producer)])
        run(["git", "lfs", "install", "--local"], producer)
        (producer / ".gitattributes").write_text(
            "assets/training/gsm8k-v1.tar.gz filter=lfs diff=lfs merge=lfs -text\n"
        )
        target = producer / "assets/training/gsm8k-v1.tar.gz"
        target.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "assets/training/gsm8k-v1.tar.gz", target)
        run(["git", "add", ".gitattributes", "assets/training/gsm8k-v1.tar.gz"], producer)
        run(
            [
                "git",
                "-c",
                "user.name=Technical Probe",
                "-c",
                "user.email=probe@example.invalid",
                "commit",
                "-m",
                "Isolated LFS command fixture",
            ],
            producer,
        )
        run(["git", "remote", "add", "origin", str(origin)], producer)
        run(["git", "push", "origin", "HEAD"], producer)
        environment = dict(os.environ)
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        run(["git", "clone", str(origin), str(consumer)], env=environment)
        archive = consumer / "assets/training/gsm8k-v1.tar.gz"
        pointer = archive.read_text()
        assert pointer.startswith("version https://git-lfs.github.com/spec/v1")
        run(["git", "lfs", "install", "--local"], consumer)
        run(["git", "lfs", "pull", "--include=assets/training/gsm8k-v1.tar.gz", "--exclude="], consumer)
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        expected = hashlib.sha256((ROOT / "assets/training/gsm8k-v1.tar.gz").read_bytes()).hexdigest()
        assert actual == expected
        lfs = {
            "pre_pull_pointer": pointer,
            "post_pull_archive_sha256": actual,
            "expected_archive_sha256": expected,
            "matches": True,
            "scope": "Exact quoted LFS commands were exercised against a local temporary LFS origin containing this fixed real archive; shared repository Git config was untouched.",
        }
    value = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_c_1_dependency/software_probe.py",
        "environment": {"python": platform.python_version(), "os": platform.platform(), "device": "cpu"},
        "commands": commands,
        "lfs": lfs,
    }
    (HERE / "software-results.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "commands_executed": len(commands), "lfs_archive_matches": actual == expected}, indent=2
        )
    )


if __name__ == "__main__":
    main()
