"""Independent, bounded CPU verification of A.3 and its original retrieval rows."""
import hashlib
import json
import random
import re
from pathlib import Path

from tiny_perceptron.retrieval import lexical_terms, retrieve

OUT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def scalar_overlap(query, document):
    # Deliberately independent count formulation for the supported token alphabet.
    query_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", query.lower())
    doc_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", document.lower())
    return sum(min(query_tokens.count(t), doc_tokens.count(t)) for t in set(query_tokens))


query = "書店地址"
docs = [{"id": "d1", "text": "貓喜歡曬太陽"}, {"id": "d2", "text": "書店地址是青街8號"}]
assert lexical_terms(query) == ["書", "店", "地", "址"]
assert [scalar_overlap(query, d["text"]) for d in docs] == [0, 4]
assert retrieve(query, docs, k=1) == [docs[1]]
assert retrieve("營業處在哪", docs, k=1) == []
assert lexical_terms("ABC123 中文 456 a-B") == ["abc123", "中", "文", "456", "a", "b"]
assert scalar_overlap("書", "書" * 10) == 1
assert scalar_overlap("書書", "書" * 10) == 2
changed = {"id": "d2", "text": "書店地址尚未公佈"}
assert scalar_overlap(query, changed["text"]) == 4
assert retrieve(query, [docs[0], changed], k=1) == [changed]
# Positive scores sort descending; tied scores use ascending IDs, independent of input order.
tie_docs = [{"id": "z", "text": "書店"}, {"id": "b", "text": "书"},
            {"id": "a", "text": "書店"}, {"id": "best", "text": "書店地址"}]
assert [d["id"] for d in retrieve(query, tie_docs, k=10)] == ["best", "a", "z"]
assert retrieve(query, list(reversed(tie_docs)), k=10) == retrieve(query, tie_docs, k=10)
assert retrieve(query, tie_docs, k=0) == []

report_raw = (OUT / "raw/rag.json").read_bytes()
report = json.loads(report_raw)
corpus_raw = (OUT / "raw/retrieval-corpus.json").read_bytes()
corpus = json.loads(corpus_raw)
artifact = next(row for row in report["artifacts"] if row["path"] == "retrieval-corpus.json")
assert digest(corpus_raw) == artifact["sha256"] and len(corpus_raw) == artifact["bytes"]
assert digest(json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode()) == report["results"]["corpus_sha256"]
assert len(corpus) == 72
assert len({d["id"] for d in corpus}) == 72
new = [d for d in corpus if d["id"].startswith("doc-K")]
noise = [d for d in corpus if d["id"].startswith("noise-")]
assert len(new) == 12 and len(noise) == 60
assert all(re.fullmatch(r"K\d{3} address=[ABCD]\d", d["text"]) for d in new)
assert all(re.fullmatch(r"Z\d{3} address=[ABCD]\d", d["text"]) for d in noise)
assert report["results"]["split"]["test"]["families"] == 12

samples = report["results"]["samples"]
selected = [s for s in samples if s["mode"] == "retrieved_context"]
assert len(selected) == 12
assert len({s["family"] for s in selected}) == 12
assert len(samples) == 84  # Seven modes repeat each underlying retrieval decision.
facts = [s["fact"] for s in selected]
assert digest(json.dumps(facts, sort_keys=True, ensure_ascii=False).encode()) == report["results"]["test_facts_sha256"]
assert [f["key"] for f in facts] == sorted(f["key"] for f in facts)
# Check the original seed+100 fact/noise construction against the retained original bytes.
rng = random.Random(report["seed"] + 100)
for fact in facts:
    source = f"D{rng.randrange(10)}"
    address = f"{rng.choice('ABCD')}{rng.randrange(10)}"
    assert (fact["source"], fact["address"]) == (source, address)
reconstructed = [{"id": f"doc-{f['key']}", "text": f"{f['key']} address={f['address']}"} for f in facts]
reconstructed += [{"id": f"noise-{i}", "text": f"Z{i:03d} address={rng.choice('ABCD')}{rng.randrange(10)}"} for i in range(60)]
assert reconstructed == corpus

recomputed = []
for i, sample in enumerate(selected):
    key = sample["fact"]["key"]
    expected_query = key if i % 4 else key.replace("K", "shop")
    assert sample["query"] == expected_query
    hits = retrieve(sample["query"], corpus, k=1)
    ids = [hit["id"] for hit in hits]
    hit = any(d["id"] == f"doc-{key}" for d in hits)
    mapping = {d["id"]: sample["fact"]["source"] for d in hits}
    documents = [{"id": sample["fact"]["source"], "text": d["text"]} for d in hits]
    assert ids == sample["retrieved_ids"]
    assert hit == sample["retrieval_hit"]
    assert mapping == sample["citation_id_map"]
    assert documents == sample["documents"]
    # Confirm the same recorded retrieval is shared by all seven generation conditions.
    siblings = [s for s in samples if s["family"] == sample["family"]]
    assert len(siblings) == 7
    for sibling in siblings:
        for field in ["query", "retrieved_ids", "retrieval_hit", "citation_id_map"]:
            assert sibling[field] == sample[field]
    recomputed.append({"key": key, "query": sample["query"], "ids": ids,
                       "hit": hit, "citation_id_map": mapping})
numerator = sum(s["hit"] for s in recomputed)
denominator = len(recomputed)
exact_name_queries = sum(s["query"] == s["key"] for s in recomputed)
aliases = [s["query"] for s in recomputed if s["query"] != s["key"]]
assert (numerator, denominator, exact_name_queries, len(aliases)) == (9, 12, 9, 3)
assert report["results"]["retrieval_recall_at_1"] == {
    "numerator": numerator, "denominator": denominator, "rate": numerator / denominator}
assert next(s for s in recomputed if s["key"] == "K017")["citation_id_map"] == {"doc-K017": "D1"}

result = {
    "device": "cpu; stdlib-only retrieval, no embedding model or generation loaded",
    "example": {"terms": lexical_terms(query), "document_scores": [0, 4], "hit_ids": ["d2"],
                "synonym_query_hits": [], "document_repeat_score": 1, "query_repeat_score": 2,
                "unpublished_address_score": 4, "tie_ids": ["best", "a", "z"]},
    "original_measurement_check": {"report_sha256": digest(report_raw),
        "corpus_sha256": digest(corpus_raw), "corpus_bytes": len(corpus_raw),
        "corpus_documents": len(corpus), "new_documents": len(new), "noise_documents": len(noise),
        "original_seed": report["seed"], "original_run_revision": report["revision"],
        "original_device": report["device"], "original_python": report["python_version"],
        "original_torch": report["torch_version"], "saved_sample_rows": len(samples),
        "unique_retrieval_trials": denominator, "exact_name_queries": exact_name_queries,
        "alias_queries": aliases, "recall_at_1": {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator}, "recomputed": recomputed},
    "inspected_result_pointers": ["/seed", "/revision", "/device", "/python_version", "/torch_version",
        "/artifacts/*/path", "/artifacts/*/bytes", "/artifacts/*/sha256", "/code_sha256",
        "/results/corpus_sha256", "/results/test_facts_sha256", "/results/split/test/families",
        "/results/retrieval_recall_at_1", "/results/samples/*/mode", "/results/samples/*/family",
        "/results/samples/*/query", "/results/samples/*/fact", "/results/samples/*/retrieved_ids",
        "/results/samples/*/retrieval_hit", "/results/samples/*/citation_id_map",
        "/results/samples/*/documents"],
    "excluded_scope": "No model scores, generated answers, training execution or claim about general retrieval accuracy."
}
(OUT / "execution/verification-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
