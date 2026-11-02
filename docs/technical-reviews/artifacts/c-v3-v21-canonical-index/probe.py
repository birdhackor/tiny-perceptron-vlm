"""Canonical index technical probe: static source and entry generation only."""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import sys
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/tmp/tpb').resolve()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nav_paths(items):
    for entry in items:
        if isinstance(entry, str):
            yield entry
        else:
            for value in entry.values():
                if isinstance(value, list):
                    yield from nav_paths(value)
                else:
                    yield value


started_at = datetime.now(timezone.utc).isoformat()
setup = load_module('index_native_setup', 'scripts/omni_revision/review_setup.py')
exporter = load_module('scripts.index_native_exporter', 'scripts/export_course.py')
freeze_path = 'docs/course-revision-20261101-c-v3/source-freeze-v21.json'
freeze_sha = 'd9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea'
freeze = setup.read_ref({'path': freeze_path, 'sha256': freeze_sha})
page = next(row for row in freeze['pages'] if row['page_id'] == 'index')
frozen_body = setup.frozen_body(page)
index = json.loads((ROOT / 'course/lesson-index.json').read_bytes())
generated = exporter.home_introduction(index, True)
assert generated == frozen_body
assert page['selector'] == 'home-introduction-without-chapter-navigation'
assert page['figures_sha256'] == {}

sources = sorted({row['source'] for row in index})
headings = {}
for source in sources:
    raw = (ROOT / source).read_text(encoding='utf-8')
    for section_id, title in re.findall(r'^## ([A-Z\d]+\.\d+) (.+)$', raw, re.M):
        assert section_id not in headings
        headings[section_id] = {'title': title, 'source': source}
ids = [row['id'] for row in index]
assert len(ids) == len(set(ids)) == 332
assert set(ids) == set(headings)
assert all(headings[row['id']] == {'title': row['title'], 'source': row['source']} for row in index)

configuration = tomllib.loads((ROOT / 'zensical.toml').read_text(encoding='utf-8'))['project']
nav = list(nav_paths(configuration['nav']))
counts = Counter(nav)
assert all(counts[f'{section_id}.md'] == 1 for section_id in ids)
assert all(counts[target] == 1 for target in ['index.md', 'first-steps.md', 'course.md'])

freeze_pages = {row['page_id']: row for row in freeze['pages']}
footers = 0
source_code_cells = 0
source_cells_with_output = 0
for item in index:
    notebook_path = ROOT / item['notebook']
    notebook = json.loads(notebook_path.read_bytes())
    frozen_notebook = freeze_pages[item['id']]
    assert frozen_notebook['notebook'] == item['notebook']
    assert digest(notebook_path) == frozen_notebook['notebook_sha256']
    for direct in [True, False]:
        footer = exporter.section_footer(item, notebook, 'main', True, direct_colab=direct)
        assert f'href="{item["notebook"]}" download' in footer
        expected_colab = (
            f'https://colab.research.google.com/github/{exporter.REPOSITORY}/blob/main/{item["notebook"]}'
            if direct else 'https://colab.research.google.com/'
        )
        assert f'href="{expected_colab}"' in footer
    footers += 1
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and not cell.get('metadata', {}).get('course_setup') and 'course_figure' not in cell.get('metadata', {}):
            source_code_cells += 1
            source_cells_with_output += bool(cell.get('outputs'))

home_links = re.findall(r'\]\(([^)]+\.md)\)', frozen_body)
assert set(home_links) == {'1.1.md', 'first-steps.md', 'course.md'}
assert (ROOT / exporter.DOCUMENTS['first-steps']).is_file()
assert (ROOT / exporter.DOCUMENTS['course']).is_file()
assert '1.1' in ids
guide = (ROOT / exporter.DOCUMENTS['course']).read_text(encoding='utf-8')
guide_entries = {
    'personality_and_safety': ['如何教文風、誠實、安全與偏好？', 'chapters/08.md', 'chapters/09.md'],
    'images': ['圖片與聲音怎麼成為輸入？', 'chapters/10.md#10.1'],
    'quantization': ['Dense、MoE、加速與壓縮各改善什麼？', 'chapters/17.md'],
}
for fragments in guide_entries.values():
    assert all(fragment in guide for fragment in fragments)

scope_sections = {
    'context': ['2.1', '2.2', '2.5'],
    'questions_and_stopping': ['7.1', '7.9', '7.11'],
    'information_sufficiency': ['9.13'],
    'images_audio_time': ['10.1', '10.12', '12.1', '12.17'],
    'long_material_compute_memory': ['16.3', '16.4', '16.9', '16.15'],
}
scope_rows = {key: {section_id: headings[section_id] for section_id in wanted} for key, wanted in scope_sections.items()}

phase_records = []
for relative, expected_sha, time_key in [
    ('docs/course-revision-20261101-c-v3/reader-phase-proof-v21-v2.json', '1c8a6002b9816a9986c201c613ddf91c214127a945a45df01468f14696d90259', 'completed_at'),
    ('docs/course-revision-20261101-c-v3/technical-stage-open-v21-v1.json', 'd9b760e2261709607e0f5bad417e44275f00cef3266bc61efc46e67fc72bff9d', 'started_at'),
]:
    path = ROOT / relative
    assert digest(path) == expected_sha
    record = json.loads(path.read_bytes())
    phase_records.append({'path': relative, 'sha256': expected_sha, 'status': record['status'], time_key: record[time_key]})
assert datetime.fromisoformat(phase_records[0]['completed_at']) < datetime.fromisoformat(phase_records[1]['started_at'])

result = {
    'started_at': started_at,
    'completed_at': datetime.now(timezone.utc).isoformat(),
    'scope': 'No site build, browser UI run, training, model execution, scientific final opening, package installation, publication or Colab session was performed.',
    'environment': {'python': sys.version.split()[0], 'platform': platform.platform(), 'zensical': importlib.metadata.version('zensical'), 'device': 'CPU; static Python source/entry generation'},
    'phase_metadata_only': phase_records,
    'source_freeze': {'path': freeze_path, 'sha256': freeze_sha},
    'page': page,
    'native_frozen_body_equals_live_selector': generated == frozen_body,
    'index_count': len(index),
    'numbered_source_headings': len(headings),
    'unique_ids': len(set(ids)),
    'all_headings_titles_sources_match_index': True,
    'all_332_lesson_nav_entries_exactly_once': True,
    'all_332_frozen_source_notebooks_match': True,
    'all_332_footer_functions_have_download_and_colab': footers,
    'home_links': home_links,
    'guide_entries_checked': guide_entries,
    'curriculum_scope_locators': scope_rows,
    'theme_features': configuration['theme']['features'],
    'source_notebook_code_cells': source_code_cells,
    'source_notebook_code_cells_with_output': source_cells_with_output,
    'output_provenance_limit': 'Source Notebook output presence is not proof of an executed bundle or a currently deployed site; that claim awaits immutable execution/render evidence.',
    'ui_limit': 'Configuration and generated navigation do not independently establish search indexing or mobile visual position; no browser search or mobile render is claimed.',
    'full_file_sha256': {relative: digest(ROOT / relative) for relative in ['scripts/export_course.py', 'course/lesson-index.json', 'zensical.toml', 'course/README.md', 'course/chapters/01.md', 'course/chapters/02.md', 'course/chapters/07.md', 'course/chapters/09.md', 'course/chapters/12.md', 'course/chapters/16.md']},
}
print(json.dumps(result, ensure_ascii=False, indent=2))
