"""獨立核對凍結 12.14 的實際程式、練習與文字歷史接線；不載入模型。"""

import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import jinja2
import torch

ROOT = Path(__file__).resolve().parents[4]
RAW = Path('/workspace/work/reference/omni-technical-technical_12_14-v1')
FREEZE = 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json'
PACKET = 'docs/course-revision-20261029-omni/reviews/technical-stage/dispatch-briefs-v9/technical_12_14.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(path, lesson_id):
    return re.search(r'^## ' + re.escape(lesson_id) + r' .*?(?=^## |\Z)', path.read_text(), re.M | re.S).group()


def run_exact(code):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, '<frozen-12.14>', 'exec'), namespace)
    return namespace['report'], output.getvalue()


def main():
    started = datetime.now(UTC).isoformat()
    torch.set_num_threads(1)
    freeze = json.loads((ROOT / FREEZE).read_bytes())
    page = next(page for page in freeze['pages'] if page['page_id'] == '12.14')
    packet = json.loads((ROOT / PACKET).read_bytes())
    assert packet['reader_task'] == '/root/rv1019_reader_ch12'
    snapshot = ROOT / page['snapshot']
    raw = snapshot.read_bytes()
    assert digest(snapshot) == page['source_sha256']
    assert section(ROOT / page['source'], '12.14').encode() == raw
    assert all(digest(ROOT / p) == sha for p, sha in page['figures_sha256'].items())
    assert digest(ROOT / page['notebook']) == page['notebook_sha256']
    blocks = re.findall(r'```python\n(.*?)```', raw.decode(), re.S)
    assert len(blocks) == 1
    exact = blocks[0]
    observed, stdout = run_exact(exact)
    assert observed['transcription'] == {
        'reference': '我不吃辣', 'prediction': '我不吃拉', 'exact': False,
        'edits': 1, 'reference_characters': 4, 'cer': 0.25,
    }
    assert observed['answers_identical'] is True and observed['answers_correct'] is None
    assert stdout == '轉寫的字元錯誤率 0.25\n兩路回答完全相同 True\n相同是否足以證明正確 None\n'
    exercise, exercise_stdout = run_exact(exact.replace('"我不吃拉"', '"我不吃辣"'))
    assert exercise['transcription']['edits'] == 0 and exercise['transcription']['cer'] == 0
    assert exercise['answers_identical'] is True and exercise['answers_correct'] is None
    notebook = json.loads((ROOT / page['notebook']).read_bytes())
    notebook_blocks = [''.join(cell['source']).strip() for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert exact.strip() in notebook_blocks

    from tiny_perceptron.natural_concepts import speech_stages
    from tiny_perceptron.natural_assistant import messages_for

    wrong = speech_stages('我不吃辣', '我不吃辣', '選特辣的湯麵。', '選特辣的湯麵。')
    assert wrong['answers_identical'] is True and wrong['answers_correct'] is None
    different = speech_stages('我不吃辣', '我不吃拉', '選清淡的湯麵。', '選清淡的米飯。')
    assert different['answers_identical'] is False and different['answers_correct'] is None
    assert different['transcription'] == observed['transcription']

    # 只檢查序列化與資料是否保留，沒有生成模型回答。
    history = [{'role': 'user', 'content': '我不吃辣'}, {'role': 'assistant', 'content': '選清淡的湯麵。'}]
    typed = messages_for({'history': history, 'user': '還有別的嗎'}, ROOT)
    spoken = messages_for({'history': history, 'user': '還有別的嗎'}, ROOT)
    bad = messages_for({'history': [], 'user': '要加辣'}, ROOT)
    good = messages_for({'history': [], 'user': '不要加辣'}, ROOT)
    template_raw = RAW / 'qwen3-vl-tokenizer_config.json'
    template = jinja2.Environment().from_string(json.loads(template_raw.read_text())['chat_template'])
    render = lambda messages: template.render(messages=messages, tools=None, add_generation_prompt=True)
    rendered = render(typed)
    assert typed == spoken and rendered == render(spoken)
    assert rendered.index('我不吃辣') < rendered.index('選清淡的湯麵。') < rendered.index('還有別的嗎')
    assert render(bad) != render(good)
    assert len(typed) == 3 and [m['role'] for m in typed] == ['user', 'assistant', 'user']

    release_path = ROOT / 'docs/natural-assistant/v4/public-release.json'
    release = json.loads(release_path.read_bytes())
    spec = importlib.util.spec_from_file_location('review_fetch_natural_release', ROOT / 'scripts/fetch_natural_release.py')
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    fetch.validate_manifest(release)
    options = fetch.student_options(release, RAW / 'no-model-download', device='cpu', local_files_only=True)
    assert release['selected_variant'] == 'base' and release['adapter_parameters'] == 0
    assert options.model == 'Qwen/Qwen3-VL-2B-Instruct'
    assert options.asr_model == 'openai/whisper-large-v3-turbo'
    assert options.adapter is None

    premises = [('07', '7.1'), ('11', '11.16'), ('12', '12.11'), ('12', '12.13'), ('12', '12.15'), ('12', '12.16'), ('20', '20.1')]
    code_paths = ['tiny_perceptron/natural_concepts.py', 'tiny_perceptron/natural_assistant.py',
                  'tiny_perceptron/natural_ui.py', 'scripts/fetch_natural_release.py',
                  'docs/natural-assistant/v4/public-release.json']
    result = {
        'reviewer_task': '/root/omni_technical_coordinator/technical_12_14',
        'started_at': started, 'completed_at': datetime.now(UTC).isoformat(),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'jinja2': jinja2.__version__, 'device': 'cpu', 'platform': platform.platform()},
        'freeze': {'path': FREEZE, 'sha256': digest(ROOT / FREEZE)},
        'packet': {'path': PACKET, 'sha256': digest(ROOT / PACKET), 'reader_task': packet['reader_task']},
        'snapshot': {'path': page['snapshot'], 'sha256': digest(snapshot), 'matches_current_section': True},
        'figures_sha256': page['figures_sha256'],
        'notebook': {'path': page['notebook'], 'sha256': digest(ROOT / page['notebook']), 'lesson_code_matches_exact_block': True},
        'exact_block_sha256': hashlib.sha256(exact.encode()).hexdigest(),
        'exact_block_stdout': stdout, 'exact_block_return': observed,
        'exercise_stdout': exercise_stdout, 'exercise_transcription': exercise['transcription'],
        'equal_but_wrong_answer': {'typed_answer': wrong['typed_answer'], 'answers_identical': wrong['answers_identical'], 'answers_correct': wrong['answers_correct']},
        'different_answers': {'answers_identical': different['answers_identical'], 'answers_correct': different['answers_correct'], 'transcription_unchanged': True},
        'history_wiring': {'messages': typed, 'official_template_source_sha256': digest(template_raw),
                           'rendered_prompt': rendered, 'same_text_same_prompt': True,
                           'negation_changes_prompt': True, 'scope': '只驗文字歷史序列化，不驗模型理解、遵從或 ASR 能力'},
        'chapter20_selector': {'selected_variant': release['selected_variant'], 'adapter_parameters': release['adapter_parameters'],
                               'model': options.model, 'model_revision': options.model_revision,
                               'asr_model': options.asr_model, 'asr_revision': options.asr_revision,
                               'adapter': options.adapter, 'scope': '只執行本機 manifest 驗證與 student_options，沒有下載、載入或評測模型'},
        'necessary_current_premises': [{'source': f'course/chapters/{chapter}.md#{lesson}',
                                        'section_sha256': hashlib.sha256(section(ROOT / f'course/chapters/{chapter}.md', lesson).encode()).hexdigest()}
                                       for chapter, lesson in premises],
        'repository_source_sha256': {path: digest(ROOT / path) for path in code_paths},
        'result': 'PASS: exact frozen code, exercise, diagnostic limits, history serialization, and published base selector assertions passed; no training or model inference',
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
