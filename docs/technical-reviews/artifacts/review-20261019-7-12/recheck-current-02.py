"""依固定雜湊載入既有核對程式，將輸入版本與新收據路徑設為指定工作檔。"""

import hashlib
from pathlib import Path
import re

TEMPLATE = Path(__file__).with_name("recheck-current-01.py")
EXPECTED_TEMPLATE_SHA256 = "867bf8cc7baa9cf8aabe9457cfe63595a1b898fa4fd63d3bb1c89506d0d0d069"
raw = TEMPLATE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_TEMPLATE_SHA256
replacements = {
    "technical-7-12-original-01.json": "technical-7-12-current-01.json",
    "146c64714e00c5d170da277bc30ad7daa927a9233b71d988c5d3228c281b585e": "cbbedb4a6ba8556576b18aad96a74001c36c8586e8d59e5e0400b5c2ac2ceab7",
    "5ef56bc756f3": "ecf6dec71280",
    "6a7aa1de5227": "c6b9339d2595",
    "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21": "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94",
    "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94": "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b",
    "freeze-06.json": "freeze-07.json",
    "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9": "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753",
    "recheck-current-01.py": "recheck-current-02.py",
    "recheck-current-01.json": "recheck-current-02.json",
    '"original_report":': '"previous_report":',
    "其他訓練頁面的多模態與量化新增敘述不屬於本節主張": "這次變動的多模態段落不屬於本節主張",
}
text = raw.decode("utf-8")
assert all(key in text for key in replacements)
pattern = re.compile("|".join(re.escape(key) for key in replacements))
configured = pattern.sub(lambda match: replacements[match.group()], text)
namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__"}
exec(compile(configured, str(TEMPLATE), "exec"), namespace)
