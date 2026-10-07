"""Rebuild historical corpus and run only lexical retrieval on CPU."""
import json
import random
from pathlib import Path
from scripts.course_experiments.applications import _rag_splits
from scripts.course_experiments.common import records_sha256
from tiny_perceptron.retrieval import retrieve, lexical_terms
from collections import Counter

raw = json.loads(Path('docs/course-experiments/results/rag.json').read_text())['results']
splits = _rag_splits(raw['seed'])
rng = random.Random(raw['seed'] + 100)
facts = []
for key in sorted({row['key'] for row in splits['test']}):
    source = f'D{rng.randrange(10)}'
    address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
    facts.append({'family': key, 'key': key, 'address': address, 'source': source})
corpus = [{'id': f"doc-{row['key']}", 'text': f"{row['key']} address={row['address']}"} for row in facts]
corpus += [{'id': f'noise-{index}', 'text': f"Z{index:03d} address={rng.choice('ABCD')}{rng.randrange(10)}"} for index in range(60)]
assert records_sha256(corpus) == raw['corpus_sha256']
count = 0
for row in [r for r in raw['samples'] if r['mode'] == 'retrieved_context']:
    hits = retrieve(row['query'], corpus, k=1)
    ids = [hit['id'] for hit in hits]
    assert ids == row['retrieved_ids']
    hit = f"doc-{row['family']}" in ids
    count += hit
    assert row['citation_id_map'] == {i: row['fact']['source'] for i in ids}
    print(row['query'], ids, 'expected', f"doc-{row['family']}", 'hit', hit, 'citation mapping', row['citation_id_map'])
print('corpus', len(facts), '+', len(corpus)-len(facts), '=', len(corpus), 'hash matches', True)
print('source hit', count, '/', len(facts))
docs = [{'id': 'd3', 'text': '書店地址尚未公佈'}, {'id': 'd0', 'text': '書店地址不在此處'}]
print('tie id order', [r['id'] for r in retrieve('書店地址', docs, k=2)])
print('unpublished score', sum(min(Counter(lexical_terms(docs[0]['text']))[t], n) for t,n in Counter(lexical_terms('書店地址')).items()))
