from pathlib import Path
import hashlib
import json
import datetime
import shutil

root = Path('/workspace/tiny-perceptron-vlm')
report = root / 'docs/technical-reviews/10.5.json'
report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
assert report_sha == '80d70aab8a4e5bf7ae0609b6787158c5f8550cc9aa586cc1668d2c3af71b26bf'
history = root / f'docs/technical-reviews/history/phase4-10_5-own-initial-revise-{report_sha}.json'
assert not history.exists()
shutil.copyfile(report, history)
assert history.read_bytes() == report.read_bytes()

source = root / 'course/chapters/10.md'
raw = source.read_bytes()
old = '原實驗的資料切分、更新設定與逐題結果見[既有實報](../../docs/course-experiments/results/encoders.json)；重做入口見[實驗說明](../../docs/course-experiments/README.md)。這是上述有限任務的歷史紀錄。'
new = '相關的六類「顏色＋形狀」分類實驗，其資料切分、更新設定與逐題結果見[既有實報](../../docs/course-experiments/results/encoders.json)；重做入口見[實驗說明](../../docs/course-experiments/README.md)。它用紅、綠、藍搭配方塊、圓形組成六類答案，入口特徵寬度為 16，並實際更新參數。本節則用寬度 8 示範兩類形狀標籤如何產生梯度。'
assert raw.count(old.encode()) == 1
source.write_bytes(raw.replace(old.encode(), new.encode()))

fig = root / 'course/figures/rewrite-10-shape-gradient.svg'
svg = fig.read_text()
old_fig_sha = hashlib.sha256(fig.read_bytes()).hexdigest()
assert old_fig_sha == '5e068757bdfa526fc3486b76dbb96778c37ab2ff91cbca35aa0568488a50b523'
cut = svg.index('<path d="M159,423 L159,462"')
prefix = svg[:cut]
defs_end = '</defs>'
markers = '<marker id="target-arr" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#a16f10"/></marker><marker id="grad-arr" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#b83b3b"/></marker>'
prefix = prefix.replace(defs_end, markers + defs_end, 1)
font = 'font-family="Noto Sans CJK TC, Noto Sans CJK SC, sans-serif"'
tail = f'''<path d="M263,191 L315,191 L315,480" fill="none" stroke="#486885" stroke-width="4" marker-end="url(#arr)"/>
<path d="M373,220 L335,220 L335,480" fill="none" stroke="#486885" stroke-width="4" marker-end="url(#arr)"/>
<path d="M159,423 L159,450 L32,450 L32,662 L55,662" fill="none" stroke="#a16f10" stroke-width="4" marker-end="url(#target-arr)"/>
<path d="M477,423 L477,450 L608,450 L608,662 L585,662" fill="none" stroke="#a16f10" stroke-width="4" marker-end="url(#target-arr)"/>
<rect x="55" y="480" width="530" height="78" rx="10" fill="#edf3f8" stroke="#54718b" stroke-width="2"/>
<text x="320" y="530" {font} font-size="28" fill="#17293c" text-anchor="middle">入口 → 平均 → 分類頭 → 分數</text>
<path d="M320,558 L320,632" fill="none" stroke="#486885" stroke-width="4" marker-end="url(#arr)"/>
<rect x="55" y="632" width="530" height="78" rx="10" fill="#fff3d5" stroke="#54718b" stroke-width="2"/>
<text x="320" y="682" {font} font-size="28" fill="#17293c" text-anchor="middle">交叉熵：分數對正確標籤</text>
<path d="M470,632 L470,558" fill="none" stroke="#b83b3b" stroke-width="4" marker-end="url(#grad-arr)"/>
<text x="490" y="607" {font} font-size="28" fill="#b83b3b" text-anchor="start">梯度</text>
<text x="45" y="761" {font} font-size="30" fill="#17293c" text-anchor="start">此處未做 optimizer.step()</text></svg>'''
fig.write_text(prefix + tail)

progress = root / 'docs/course-revision-20261005/factual-review-progress.json'
j = json.loads(progress.read_bytes())
r = j['records']['10.5']
r['events'].append({'at': datetime.datetime.now(datetime.UTC).isoformat(), 'event': 'coordinator_localized_revision', 'old_source_sha256': r['current_source_sha256'], 'old_figure_sha256': old_fig_sha, 'new_figure_sha256': hashlib.sha256(fig.read_bytes()).hexdigest(), 'prior_report_path': history.relative_to(root).as_posix(), 'prior_report_sha256': report_sha, 'description': 'Clarifies related historical six-class color+shape width16 updated experiment; preserves width8 two-class gradient illustration. Keeps all512 original SVG pixels/labels, routes image data to entry and targets to loss separately, shows reverse gradient arrow and unchanged no-step note. Original fence/implementation/intro/other sections unchanged.'})
progress.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'history_path': history.relative_to(root).as_posix(), 'history_sha256': report_sha, 'new_figure_sha256': hashlib.sha256(fig.read_bytes()).hexdigest()}))
