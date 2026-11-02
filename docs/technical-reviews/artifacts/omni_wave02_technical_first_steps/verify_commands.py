"""Record the exact, non-installing command probes used by this review."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
commands = [
    (['uv', 'sync', '--frozen', '--extra', 'cpu', '--group', 'notebook', '--dry-run', '--offline'], 0),
    (['bash', '--noprofile', '--norc', '-c', 'source .venv/bin/activate\npython -c "import sys; print(sys.executable); print(sys.prefix)"'], 0),
    (['git', 'check-ignore', 'data/example', 'checkpoints/example', 'outputs/example', '.venv/example'], 0),
    ([sys.executable, '-I', '-S', '-c', 'import torch'], 1),
    ([sys.executable, '-c', 'from scripts.export_course import output_html; print(output_html([{ "output_type": "stream", "text": "cat\\n" }]))'], 0),
]
rows = []
for argv, expected in commands:
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == expected, (argv, result.returncode, result.stderr)
    rows.append({'argv': argv, 'returncode': result.returncode, 'stdout': result.stdout,
                 'stderr': result.stderr, 'expected_returncode': expected})
assert "ModuleNotFoundError: No module named 'torch'" in rows[3]['stderr']
assert '實際執行結果 · CPU' in rows[4]['stdout']
print(json.dumps({'created_at': datetime.now(timezone.utc).isoformat(),
                  'environment': {'python': platform.python_version(), 'executable': sys.executable,
                                  'platform': platform.system(), 'device': 'cpu'},
                  'checks': rows,
                  'scope': 'Existing environment and local source probes; no clean installation, network cloning, Windows or Colab runtime, or full-course notebook execution.'},
                 ensure_ascii=False, indent=2))
