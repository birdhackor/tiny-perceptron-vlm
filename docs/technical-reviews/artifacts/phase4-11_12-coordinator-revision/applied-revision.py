import hashlib
import json
import re
from pathlib import Path

source = Path('course/chapters/11.md')
before = source.read_bytes()

old_pairing = '字形與標籤要人工核對，梯度不會發現畫圖器把 1 畫成 0。本例只改位移或粗細而未改字元時，答案標籤仍相同。同一原始字形及其變體稱為一個來源家族；切分資料時，整個家族都放訓練組或都放留出檢查組，避免把練習字形的變體當成全新材料。若用 RGB 入口，要把單通道小圖放到固定畫布、再明確複製通道，不能把 `(5,3)` 直接當 `(3,16,16)`。'
new_pairing = '字形與標籤要人工核對，梯度不會發現畫圖器把 1 畫成 0。本例只改位移或粗細而未改字元時，答案標籤仍相同。每個字目前只有一個原形；若把數字 0 的原形與所有變體整組留出，訓練側就沒有 0 這個類別。要檢查熟悉類別的新原圖，需為每類準備多張獨立原圖，再把同一原圖及其變體當一個來源家族，整組放在訓練側或留出側。若用 RGB 入口，要把單通道小圖放到固定畫布、再明確複製通道，不能把 `(5,3)` 直接當 `(3,16,16)`。'
old_experiment = '既有 0–99 固定字形實驗，訓練代價降低，最後新字串完整讀對只有 2/30。它留下「熟悉練習改善不等於新組合讀對」這個例子；沒有測另一套字體或手寫。下一節再看整串與逐字判尺。'
new_experiment = '既有 0–99 固定字形實驗按完整字串切分：同一字串的位移版本放在同一側。訓練側已包含 0–9 全部字元，留出側測的是熟悉字元的新字串組合。訓練代價降低，最後新字串完整讀對只有 2/30，留下「熟悉練習改善不等於新組合讀對」這個例子。這個實驗沒有測新字元類別、另一套字體或手寫。下一節再看整串與逐字判尺。'

after = before
for old, new in [(old_pairing, new_pairing), (old_experiment, new_experiment)]:
    assert after.count(old.encode('utf-8')) == 1
    after = after.replace(old.encode('utf-8'), new.encode('utf-8'), 1)

def section_hashes(raw):
    headings = list(re.finditer(rb'(?m)^## (11\.\d+) [^\r\n]+', raw))
    return {heading[1].decode(): hashlib.sha256(raw[heading.start(): headings[index + 1].start() if index + 1 < len(headings) else len(raw)]).hexdigest() for index, heading in enumerate(headings)}

old_sections = section_hashes(before)
new_sections = section_hashes(after)
changed_sections = [section for section in old_sections if old_sections[section] != new_sections[section]]
assert changed_sections == ['11.12']
source.write_bytes(after)
print(json.dumps({'scope': 'Actual coordinator prose revision only; not a scientific review verdict.', 'path': str(source), 'changed_sections': changed_sections, 'before_file_sha256': hashlib.sha256(before).hexdigest(), 'after_file_sha256': hashlib.sha256(after).hexdigest(), 'before_section_sha256': old_sections['11.12'], 'after_section_sha256': new_sections['11.12'], 'other_section_bytes_verified_unchanged': 17, 'figure_or_intro_or_fence_changes': False}, ensure_ascii=False))
