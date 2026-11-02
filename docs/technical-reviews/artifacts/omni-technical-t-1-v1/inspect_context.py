"""保存本人讀過的原始來源定位與 T.1 必要前文／資料清單核對；不保存外部全文。"""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-t-1-v1')
OUT = Path(__file__).with_name('inspection.json')
assert not OUT.exists()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


originals = [
    ('gpt2', 'gpt2-paper.pdf', 'gpt2-paper-raw.txt',
     'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
     'OpenAI technical report, 2019; PDF retrieved 2026-10-29; fingerprint below',
     'PDF p.2 section 2, equation (1) and paragraph p(output|input, task)',
     'input, it should condition not only on the input but also\non the task to be performed. That is, it should model\np(output|input, task).'),
    ('grouped', 'cross_validation.rst', 'cross_validation.rst',
     'https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst',
     'scikit-learn 1.7.2 tag', 'lines 628-658, grouped data and GroupKFold',
     'ensure that all the samples in the validation fold come from groups that are\nnot represented at all in the paired training fold.'),
    ('lfs-spec', 'git-lfs-spec.md', 'git-lfs-spec.md',
     'https://raw.githubusercontent.com/git-lfs/git-lfs/v3.6.1/docs/spec.md',
     'git-lfs v3.6.1 tag', 'The Pointer; required keys and example v1 pointer',
     'The core Git LFS idea is that instead of writing large blobs to a Git repository,\nonly a pointer file is written.'),
    ('lfs-pull', 'git-lfs-pull.adoc', 'git-lfs-pull.adoc',
     'https://raw.githubusercontent.com/git-lfs/git-lfs/v3.6.1/docs/man/git-lfs-pull.adoc',
     'git-lfs v3.6.1 tag', 'DESCRIPTION; OPTIONS; INCLUDE AND EXCLUDE',
     'Download Git LFS objects for the currently checked out ref, and update\nthe working copy with the downloaded content if required.'),
    ('lfs-config', 'git-lfs-config.adoc', 'git-lfs-config.adoc',
     'https://raw.githubusercontent.com/git-lfs/git-lfs/v3.6.1/docs/man/git-lfs-config.adoc',
     'git-lfs v3.6.1 tag', 'lines 388-395, GIT_LFS_SKIP_SMUDGE',
     "working copy. If 'true', '1', 'on', or similar, Git LFS will skip the\nsmudge process in both `git lfs smudge` and `git lfs filter-process`."),
    ('lfs-install', 'git-lfs-install.adoc', 'git-lfs-install.adoc',
     'https://raw.githubusercontent.com/git-lfs/git-lfs/v3.6.1/docs/man/git-lfs-install.adoc',
     'git-lfs v3.6.1 tag', 'OPTIONS --local and --skip-smudge',
     '  Sets the "lfs" smudge and clean filters in the local repository\'s git config,\n  instead of the global git config (~/.gitconfig).'),
    ('tinystories', 'tinystories-upstream-readme.md', 'tinystories-upstream-readme.md',
     'https://huggingface.co/datasets/roneneldan/TinyStories/raw/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md',
     'roneneldan/TinyStories revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64',
     'YAML license and paragraph naming TinyStories-train.txt',
     'license: cdla-sharing-1.0'),
    ('pillow', 'pillow-image.rst', 'pillow-image.rst',
     'https://raw.githubusercontent.com/python-pillow/Pillow/11.3.0/docs/reference/Image.rst',
     'Pillow 11.3.0 tag; dimension definition only, not a claim about installed Pillow version',
     'lines 210-220 resize; lines 290-293 Image.size',
     '    Image size, in pixels. The size is given as a 2-tuple (width, height).'),
    ('audio-augmentation', 'audio_data_tutorial.py', 'audio_data_tutorial.py',
     'https://raw.githubusercontent.com/pytorch/audio/v2.8.0/examples/tutorials/audio_data_augmentation_tutorial.py',
     'torchaudio v2.8.0 tag; read only, not imported or claimed to be installed',
     'lines 205-227 Adding background noise',
     '# To introduce background noise to audio data, we can add a noise Tensor to\n# the Tensor representing the audio data according to some desired'),
    ('audio-resampling', 'audio_resampling_tutorial.py', 'audio_resampling_tutorial.py',
     'https://raw.githubusercontent.com/pytorch/audio/v2.8.0/examples/tutorials/audio_resampling_tutorial.py',
     'torchaudio v2.8.0 tag; read only, not imported or claimed to be installed',
     'lines 110-126 Resampling Overview',
     '# To resample an audio waveform from one freqeuncy to another, you can use\n# :py:class:`torchaudio.transforms.Resample` or'),
]
receipts = []
for identifier, raw, readable, url, version, locator, quote in originals:
    text = (CACHE / readable).read_text()
    assert quote in text, identifier
    receipts.append({
        'id': identifier, 'url': url, 'version': version, 'accessed_on': '2026-10-29',
        'original_cache_path': str(CACHE / raw), 'original_sha256': sha(CACHE / raw),
        'inspection_text_path': str(CACHE / readable), 'inspection_text_sha256': sha(CACHE / readable),
        'locator': locator, 'quote': quote, 'checked_original': True,
    })

manifest = json.loads((ROOT / 'assets/training/manifest.json').read_text())
starter = []
for a in manifest['assets']:
    source_path = ROOT / a['source_metadata']
    assert source_path.is_file()
    provenance = json.loads(source_path.read_text())
    provenance_fields = {k: v for k, v in provenance.items() if k in (
        'source_revision', 'source_commit', 'revision', 'source_url',
        'source_repository_revision', 'hf_revision', 'source_sha', 'source',
    )}
    assert provenance.get('license') and provenance_fields
    starter.append({
        'id': a['id'], 'archive': a['archive'], 'archive_sha256': a['archive_sha256'],
        'source_metadata': a['source_metadata'], 'source_metadata_sha256': sha(source_path),
        'license': a['license'], 'metadata_has_provenance_and_license': True,
        'provenance_field_names': sorted(provenance_fields),
    })
m19 = json.loads((ROOT / 'docs/selftrained/v2-manifest.json').read_text())
m20 = json.loads((ROOT / 'docs/natural-assistant/v4/manifest.json').read_text())
starter_archives = {a['archive'] for a in manifest['assets']}
later_archives = {m19['package']['path']} | {a['path'] for a in m20['archives']}
assert not starter_archives & later_archives
assert {p.split('-')[0] for p in [r['path'] for r in m19['records']]} >= {'ocr', 'vision', 'voice'}

freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json'
freeze = json.loads(freeze_path.read_text())
page = next(p for p in freeze['pages'] if p['page_id'] == 'T.1')
assert sha(ROOT / page['snapshot']) == page['source_sha256']
assert page['figures_sha256'] == {}
current = (ROOT / page['source']).read_text()
start = current.index('## T.1 ')
end = current.index('\n## ', start + 1)
assert current[start:end + 1] == (ROOT / page['snapshot']).read_text()
prerequisites = [
    'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/W.1.md',
    'docs/course-review-20261019/freeze-01/sources/5.10.md',
    'assets/training/README.md', 'docs/natural-assistant/v4/DATA.md',
]
output = {
    'created_at': datetime.now(UTC).isoformat(),
    'environment': {'python': platform.python_version(), 'device': 'cpu; JSON/text/hash inspection only'},
    'result': 'all source quote, freeze, provenance-metadata presence and separate-manifest assertions passed',
    'reviewer_script_note': 'An initial metadata assertion assumed common top-level revision keys; before this saved run the inspector was corrected to recognize the existing per-source schema keys. No textbook or implementation changed.',
    'source_freeze': {'path': str(freeze_path.relative_to(ROOT)), 'sha256': sha(freeze_path)},
    'snapshot': {'path': page['snapshot'], 'sha256': page['source_sha256'], 'current_source_equal': True, 'figures': {}},
    'read_prerequisites': [{'path': p, 'sha256': sha(ROOT / p)} for p in prerequisites],
    'original_source_receipts': receipts,
    'starter_metadata_presence': starter,
    'later_package_registration': {
        'chapter19_manifest': 'docs/selftrained/v2-manifest.json',
        'chapter19_manifest_sha256': sha(ROOT / 'docs/selftrained/v2-manifest.json'),
        'chapter19_archive': m19['package']['path'],
        'chapter20_manifest': 'docs/natural-assistant/v4/manifest.json',
        'chapter20_manifest_sha256': sha(ROOT / 'docs/natural-assistant/v4/manifest.json'),
        'chapter20_archives': [a['path'] for a in m20['archives']],
        'starter_archive_overlap': [],
        'scope': 'only existence, identities and separation inspected; no later package acquired or model results verified',
    },
    'limits': [
        'Git LFS may satisfy pull from local object cache; no claim of an independently fresh remote download.',
        'The missing-LFS check is an isolated PATH/ROOT fixture, not a second operating-system installation.',
        'Other starter source metadata presence was inspected; upstream datasets and their full records were not independently downloaded.',
        'No GPU, training, model quality, speed, memory or end-to-end chapter19/20 result verified.',
    ],
}
OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'result': output['result'], 'receipt': str(OUT), 'sha256': sha(OUT)}, ensure_ascii=False))
