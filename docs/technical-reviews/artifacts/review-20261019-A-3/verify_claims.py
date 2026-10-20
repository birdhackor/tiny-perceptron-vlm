import ast
import contextlib
import hashlib
import inspect
import io
import json
import platform
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.retrieval import lexical_terms, retrieve
from scripts.course_experiments.applications import _rag_splits
from scripts.course_experiments.common import records_sha256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def score(query, text):
    query_counts = Counter(lexical_terms(query))
    counts = Counter(lexical_terms(text))
    return sum(min(counts[term], amount) for term, amount in query_counts.items())


snapshot = ROOT / 'docs/course-review-20261019/freeze-01/sources/A.3.md'
source = snapshot.read_text()
live = (ROOT / 'course/chapters/0A.md').read_text()
live_section = re.search(r'(?ms)^## A\.3 .*?(?=^## |\Z)', live)[0]
require(source == live_section, '凍結原文與目前小節不相同')
fences = re.findall(r'```python\n(.*?)```', source, re.S)
require(len(fences) == 3, 'Python 區塊數不符')
namespace = {'__name__': '__main__'}
captured = io.StringIO()
identities = []
with contextlib.redirect_stdout(captured):
    for code in fences:
        exec(compile(code, str(snapshot), 'exec'), namespace)
        identities.append(namespace['lexical_terms'] is lexical_terms)
require(identities == [True, False, False], '節錄定義的名稱繫結不符')
for name in ('lexical_terms', 'retrieve'):
    excerpt = next(node for node in ast.parse(fences[1]).body if isinstance(node, ast.FunctionDef) and node.name == name)
    implementation = ast.parse(inspect.getsource(globals()[name])).body[0]
    require(ast.dump(excerpt, include_attributes=False) == ast.dump(implementation, include_attributes=False), '節錄函式與函式庫不符')

query = '書店地址'
docs = [{'id': 'd1', 'text': '貓喜歡曬太陽'}, {'id': 'd2', 'text': '書店地址是青街8號'}]
cases = []
for text, expected in [('書店地址是青街8號', 4), ('書書書書書店', 2), ('K017 address=C2', 0), ('書店地址尚未公布', 4)]:
    actual = score(query, text)
    require(actual == expected, '片段計分不符')
    cases.append({'text': text, 'terms': lexical_terms(text), 'score': actual})
require(lexical_terms(query) == list(query), '查詢片段不符')
require(dict(Counter(lexical_terms(query))) == dict.fromkeys(query, 1), '查詢計數不符')
require(lexical_terms('K017 address=C2') == ['k017', 'address', 'c2'], '英數切分不符')
require(lexical_terms('8') == ['8'] and lexical_terms('80') == ['80'], '數字切分不符')
require(lexical_terms('K017') == lexical_terms('k017'), '大小寫處理不符')
require(lexical_terms(' !，\t\n') == [], '標點與空白未被略過')
require(lexical_terms('\u4e00\u9fff\u3400\U00020000') == ['\u4e00', '\u9fff'], '漢字範圍不符')
require(len(r'\u4e00') == 6 and re.findall(r'\u4e00', '一') == ['一'], 'raw string 與 regex Unicode 解讀不符')
require(Counter('書')['店'] == 0, 'Counter 缺項不為零')
require(list(Counter('書書店').items()) == [('書', 2), ('店', 1)], 'Counter.items 不符')
require(not bool(0) and bool(1), '數字真值不符')
require(score('書', '書' * 5) == 1, '重複片段上限不符')
require(score('書書', '書' * 5) == 2, '查詢重複次數上限不符')
require(retrieve(query, docs, k=1) == [docs[1]], '取回文件不符')
require(retrieve('營業處在哪', docs, k=1) == [], '無共同片段時仍有文件')
require(retrieve(query, docs, k=2) == [docs[1]], '零分文件不應因 k 增加而取回')
ties = [{'id': 'd2', 'text': '書'}, {'id': 'd1', 'text': '書'}, {'id': 'd3', 'text': '書店'}]
require([d['id'] for d in retrieve('書店', ties, k=3)] == ['d3', 'd1', 'd2'], '排序優先序不符')
require(sorted([(1, 'd2'), (1, 'd1'), (0, 'd3')]) == [(0, 'd3'), (1, 'd1'), (1, 'd2')], 'tuple 字典順序不符')
changed = [docs[0], {'id': 'd2', 'text': '書店地址尚未公布'}]
require(len(lexical_terms(changed[1]['text'])) == 8, '練習字數不符')
require(retrieve(query, changed, k=1) == [changed[1]], '資料不足範例未取回 d2')
require(retrieve(query, changed, k=2) == [changed[1]], '練習中零分文件不應取回')

record_path = ROOT / 'docs/course-experiments/results/rag.json'
record = json.loads(record_path.read_bytes())
result = record['results']
rows = [row for row in result['samples'] if row['mode'] == 'retrieved_context']
seed = result['seed']
splits = _rag_splits(seed)
keys = sorted({row['key'] for row in splits['test']})
rng = random.Random(seed + 100)
facts = []
for key in keys:
    source_id = f'D{rng.randrange(10)}'
    address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
    facts.append({'family': key, 'key': key, 'address': address, 'source': source_id})
corpus = [{'id': f"doc-{fact['key']}", 'text': f"{fact['key']} address={fact['address']}"} for fact in facts]
corpus += [{'id': f'noise-{index}', 'text': f"Z{index:03d} address={rng.choice('ABCD')}{rng.randrange(10)}"} for index in range(60)]
require(records_sha256(facts) == result['test_facts_sha256'], '重建測試事實的雜湊不符')
require(records_sha256(corpus) == result['corpus_sha256'], '重建檢索母體的雜湊不符')
require(len(facts) == 12 and len(corpus) == 72 and len(rows) == 12, '測試或母體大小不符')
samples = []
for index, (fact, row) in enumerate(zip(facts, rows, strict=True)):
    query_text = fact['key'] if index % 4 else fact['key'].replace('K', 'shop')
    hits = retrieve(query_text, corpus, k=1)
    ids = [hit['id'] for hit in hits]
    hit = any(value == f"doc-{fact['key']}" for value in ids)
    mapping = {value: fact['source'] for value in ids}
    require(row['fact'] == fact, '逐題事實不符')
    require(row['query'] == query_text and row['retrieved_ids'] == ids and row['retrieval_hit'] == hit, '檢索重放與逐題紀錄不符')
    require(row['citation_id_map'] == mapping, '文件與引用編號對應不符')
    require(row['documents'] == [{'id': fact['source'], 'text': item['text']} for item in hits], '提示內文件不符')
    samples.append({'key': fact['key'], 'query': query_text, 'terms': lexical_terms(query_text), 'retrieved_ids': ids, 'retrieval_hit': hit, 'citation_id_map': mapping})
metric = {'numerator': sum(row['retrieval_hit'] for row in samples), 'denominator': len(samples)}
metric['rate'] = metric['numerator'] / metric['denominator']
require(metric == result['retrieval_recall_at_1'] == {'numerator': 9, 'denominator': 12, 'rate': 0.75}, '檢索命中統計不符')
require(sum(row['query'].startswith('shop') for row in samples) == 3, '別名題數不符')
code_hashes = {}
for name in ['tiny_perceptron/retrieval.py', 'scripts/course_experiments/applications.py', 'scripts/course_experiments/common.py']:
    current = sha(ROOT / name)
    code_hashes[name] = {'current_sha256': current, 'recorded_sha256': record['code_sha256'][name]}
    if name == 'tiny_perceptron/retrieval.py':
        require(current == record['code_sha256'][name], '檢索程式與實驗紀錄綁定版本不符')
historical_command = ['git', 'show', record['revision'] + ':scripts/course_experiments/applications.py']
historical = subprocess.run(historical_command, capture_output=True, text=True)

output = {
    'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-3/verify_claims.py',
    'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'cpu', 'seed': str(seed)},
    'result': '所有斷言通過；只重放資料準備與檢索，未訓練或執行模型生成。',
    'source_sha256': sha(snapshot),
    'section_stdout': captured.getvalue(),
    'excerpt_matches_library_ast': True,
    'lexical_function_is_library_after_each_fence': identities,
    'query_counts': dict(Counter(lexical_terms(query))),
    'numeric_cases': cases,
    'tie_order': [d['id'] for d in retrieve('書店', ties, k=3)],
    'exercise': {'terms': lexical_terms(changed[1]['text']), 'score': score(query, changed[1]['text']), 'k1': retrieve(query, changed, k=1), 'k2': retrieve(query, changed, k=2)},
    'code_hashes': code_hashes,
    'historical_code_lookup': {'command_argv': historical_command, 'exit_code': historical.returncode, 'stdout': historical.stdout if historical.returncode else '歷史程式可取得；全文不複製。', 'stderr': historical.stderr},
    'preliminary_attempt': {'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-3/verify_claims.py', 'exit_code': 1, 'error': 'AssertionError: 目前程式與實驗紀錄綁定版本不符', 'interpretation': '整份 applications.py 的雜湊不同，不能宣稱整份實驗程式未變；目前證據只使用相同的 retrieval.py 及能吻合已保存事實與母體雜湊的檢索重放。'},
    'rag_record': {'path': str(record_path.relative_to(ROOT)), 'sha256': sha(record_path), 'recorded_revision': record['revision'], 'seed': seed, 'device': record['device'], 'python_version': record['python_version'], 'torch_version': record['torch_version']},
    'rag_replay': {'test_fact_count': len(facts), 'noise_count': 60, 'corpus_count': len(corpus), 'facts_sha256': records_sha256(facts), 'corpus_sha256': records_sha256(corpus), 'metric': metric, 'samples': samples},
    'limitations': ['沒有重新訓練模型；9/12 是所記固定母體與別名規則的檢索結果。', '常用漢字範圍不包括全部漢字；k 的示例與排序推導限定為非負整數、具有可比較字串 id 的文件。', '排序分數只數字面重疊，未校準為機率；4 分也可能對應沒有地址的文件。', '目前 applications.py 與結果綁定的整檔雜湊不同；歷史程式查詢結果另列，不由資料重放推論整份模型實驗程式相同。'],
}
print(json.dumps(output, ensure_ascii=False, indent=2))
