"""重用原immutable離線守門反例；恢復改動另以專用callback probe核對。"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent


def main():
    prior = ROOT / "docs/lesson-products/reviews/engineering-runner-original.json"
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == "368139cecf659abc40602b93f2ec66b3230499f2aaba23b72c382e5f843057e0"
    original = json.loads(prior.read_bytes())
    reference = next(x for x in original["artifacts"] if x["id"] == "probe_code")
    source = ROOT / reference["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == reference["sha256"]
    text = source.read_text()
    generated = text.split("        def recovery_after_remote_failure():", 1)[0] + "    assert before ==" + text.split("    assert before ==", 1)[1]
    generated = generated.replace("engineering-runner-probes-results-original.json", "engineering-runner-regression-probes-results-callback-01.json")
    generated = generated.replace("離線一致性/付費守門/失敗收尾；其中恢復不完整是實證缺口，all expectations met不是runner可付費pass。無真正Modal/HF I/O、GPU或optimizer step。", "原image/finance接線/HF/terminal/client finally的18個回歸；不含已另測的新恢復路徑。無真Modal/HF/GPU或optimizer step。")
    namespace = {"__file__": str(__file__), "__name__": "independent_original_regressions"}
    exec(compile(generated, str(__file__) + "::immutable-original-prefix", "exec"), namespace)
    code = namespace["main"]()
    assert code == 0


if __name__ == "__main__":
    main()
