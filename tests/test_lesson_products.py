"""有限世界的標籤、觀測隔離與模型共享性不能靠案例ID或候選索引成立。"""

import copy
import hashlib
import math

import pytest
import torch
from torch.nn import functional as F

from tiny_perceptron.lesson_products.data import (
    CANDIDATES,
    answer,
    audit_rows,
    build_rows,
    c_rows,
    judgment,
    observable_key,
    render,
)
from tiny_perceptron.lesson_products.model import (
    CandidateScorer,
    Config,
    LoopedSolver,
    Tokenizer,
    audit_tokens,
    candidate_batch,
    grouped_scores,
    padded,
    validate_candidates,
)
from tiny_perceptron.lesson_products.training import (
    checkpoint,
    encoded_data,
    logits_for,
    metrics,
    state_digest,
    step,
    verify_resume,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_state_truth_depends_on_initial_value_query_and_event_order():
    world = {"task": "boxes", "a": 2, "b": 6, "events": [("A", 5), ("swap", 0)], "query": "B"}
    assert answer(world) == 5
    assert answer({**world, "query": "A"}) == 6
    assert answer({**world, "events": list(reversed(world["events"]))}) == 2
    assert answer({**world, "events": [("swap", 0), ("swap", 0)]}) == 6


def test_missing_card_id_does_not_change_task_and_zero_is_present():
    card = {"task": "card", "application": "123456", "family_id": "fixture-card"}
    rows = list(c_rows(card, "train"))
    missing = [r for r in rows if r["omitted"] == ["application"]]
    assert {r["question"]: r["gold"] for r in missing} == {"choice": "card", "info": "no", "rubric": "grade0"}
    world = {"task": "boxes", "a": 0, "b": 0, "events": [("A", 0)], "query": "A"}
    assert judgment(world, (), "info", CANDIDATES["info"]) == "yes"
    assert judgment(world, (), "rubric", CANDIDATES["rubric"]) == "grade2"
    assert judgment(world, ("a",), "rubric", CANDIDATES["rubric"]) == "grade1"
    assert judgment({"task": "address"}, (), "rubric", CANDIDATES["rubric"]) == "grade2"


@pytest.mark.parametrize("task,needed", [("boxes", ("a", "b", "events", "query")), ("sum", ("x", "y", "z"))])
def test_v2_every_missing_subset_has_consistent_type_info_and_rubric(task, needed):
    import itertools

    world = {
        "task": task,
        "family_id": "fixture",
        "a": 0,
        "b": 2,
        "events": [("A", 0)],
        "query": "B",
        "x": 0,
        "y": 1,
        "z": 2,
    }
    rows = [r for r in c_rows(world, "train") if not r.get("missing_best")]
    assert {tuple(r["omitted"]) for r in rows} == {
        s for n in range(len(needed) + 1) for s in itertools.combinations(needed, n)
    }
    for row in rows:
        count = len(row["omitted"])
        expected = (
            task
            if row["question"] == "choice"
            else ("yes" if count == 0 else "no")
            if row["question"] == "info"
            else "grade2"
            if count == 0
            else "grade0"
            if count == len(needed)
            else "grade1"
        )
        assert row["gold"] == expected


@pytest.fixture(scope="module")
def dataset():
    return build_rows(per_length=20, stress_per_length=4, services=8)


def test_family_lineage_and_actual_observation_audit(dataset):
    b, c, config = dataset
    audit = audit_rows(b, c)
    assert audit["joint_family_overlap"] == 0
    assert audit["b"]["cross_split_shared_observations"] == 0
    assert set(map(int, audit["arithmetic_labels"]["train"])) == set(range(19))
    assert "Metadata only" in config["case_id_policy"]
    assert all(row["case_id"] not in row["state"] for row in b + c)
    assert audit["c"]["cross_split_shared_observations"] > 0
    assert audit["c"]["conflicting_labels"] == 0
    assert {r["split"] for r in b} >= {"long_shift", "wording_shift", "entity_shift"}


def test_different_ids_cannot_hide_cross_split_solver_duplicates(dataset):
    b, _, _ = dataset
    row = next(r for r in b if r["split"] == "train")
    duplicate = {
        **copy.deepcopy(row),
        "id": "another-id",
        "case_id": "another-case",
        "family_id": "another-family",
        "split": "test",
    }
    assert observable_key(row, "b") == observable_key(duplicate, "b")
    with pytest.raises(ValueError, match="跨切分重複"):
        audit_rows([row, duplicate], [])


def test_identical_visible_judgment_with_conflicting_labels_is_rejected():
    row = {
        "family_id": "f1",
        "split": "train",
        "task": "card",
        "state": "卡片進度;申請編號=?",
        "question": "info",
        "candidates": ["yes", "no"],
        "gold": "no",
    }
    bad = {**row, "family_id": "f2", "split": "test", "gold": "yes"}
    with pytest.raises(ValueError, match="衝突答案"):
        audit_rows([], [row, bad])


def test_token_audit_rejects_real_encoding_collision_even_if_visible_strings_differ():
    tokenizer = Tokenizer()
    tokenizer.ids["?"] = tokenizer.ids["0"]
    row = {
        "split": "train",
        "state": "卡片進度;申請編號=?",
        "question": "info",
        "candidates": ["yes", "no"],
        "gold": "no",
    }
    other = {**row, "split": "test", "state": "卡片進度;申請編號=0", "gold": "yes"}
    with pytest.raises(ValueError, match="實際token"):
        audit_tokens([], [row, other], tokenizer)


def small_config(tokenizer):
    return Config(len(tokenizer), width=32, heads=4, ff=64, c_layers=1)


def test_one_shared_block_receives_multi_depth_gradients():
    torch.manual_seed(17)
    tokenizer = Tokenizer()
    model = LoopedSolver(small_config(tokenizer))
    world = {"task": "boxes", "a": 2, "b": 6, "events": [("A", 5), ("swap", 0)], "query": "B"}
    tokens = padded([tokenizer.encode(render(world))], 128)
    parameters = {name: id(p) for name, p in model.named_parameters()}
    calls = []
    hook = model.shared.register_forward_hook(lambda *_: calls.append(True))
    logits, norms = model(tokens, loops=5, all_depths=True, diagnostics=True)
    hook.remove()
    assert len(calls) == 5 and logits.shape == (1, 5, 19)
    assert all(0 < n < 100 for n in norms)
    F.cross_entropy(logits.reshape(-1, 19), torch.full((5,), 5)).backward()
    assert model.shared.qkv.weight.grad.norm() > 0
    assert model.embedding.weight.grad.norm() > 0
    assert parameters == {name: id(p) for name, p in model.named_parameters()}
    assert model(tokens, loops=6).shape == (1, 19)


def test_candidate_reordering_preserves_meaning_scores_and_has_no_list_context():
    torch.manual_seed(28)
    tokenizer = Tokenizer()
    model = CandidateScorer(small_config(tokenizer)).eval()
    row = {"state": "卡片進度;申請編號=?", "question": "choice", "candidates": ["card", "address", "none"]}
    reverse = {**row, "candidates": list(reversed(row["candidates"]))}
    with torch.no_grad():
        encoded, slices, n, _ = candidate_batch([row, reverse], tokenizer, model.config)
        scores = grouped_scores(model(encoded), slices, n)
        assert torch.allclose(scores[0], scores[1].flip(0), atol=1e-6)
        short = {**row, "candidates": ["card", "none"]}
        encoded, slices, n, _ = candidate_batch([short], tokenizer, model.config)
        fewer = grouped_scores(model(encoded), slices, n)[0]
        assert torch.allclose(scores[0][[0, 2]], fewer, atol=1e-6)
    assert tokenizer.encode(row["state"], "info", "no") != tokenizer.encode(row["state"], "rubric", "grade0")


@pytest.mark.parametrize("candidates", [["card", "card", "none"], ["card"], ["card", "address"], ["yes", "none"]])
def test_ambiguous_or_untyped_candidate_contract_is_rejected(candidates):
    with pytest.raises(ValueError):
        validate_candidates("choice", candidates)


def test_pretokenized_candidate_batch_matches_separate_candidate_scores():
    torch.manual_seed(42)
    tokenizer = Tokenizer()
    model = CandidateScorer(small_config(tokenizer)).eval()
    rows = [
        {
            "state": "卡片進度;申請編號=?",
            "question": "choice",
            "candidates": ["card", "address", "none"],
            "gold": "card",
        },
        {"state": "地址填寫規則", "question": "info", "candidates": ["yes", "no"], "gold": "yes"},
    ]
    data = encoded_data(rows, "c", tokenizer, model.config)
    tokens, slices, n, labels = candidate_batch(rows, tokenizer, model.config)
    with torch.no_grad():
        expected = grouped_scores(model(tokens), slices, n)
        actual = logits_for(model, data, torch.tensor([0, 1]))
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.isneginf(actual[1, 2])
    assert torch.equal(labels, data["labels"])


def test_probability_metrics_use_classwise_sum_brier_then_mean():
    result = metrics(torch.zeros(2, 2), [0, 1])
    assert result["cross_entropy"] == pytest.approx(math.log(2))
    assert result["brier"] == pytest.approx(0.5)
    assert result["accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize("product", ["b", "c"])
def test_checkpoint_exact_next_update_includes_optimizer_and_sampler_without_advancing_main(tmp_path, product):
    torch.manual_seed(53)
    tokenizer = Tokenizer()
    config = small_config(tokenizer)
    model = (LoopedSolver if product == "b" else CandidateScorer)(config)
    if product == "b":
        rows = [
            {"state": "三數加法;x=0;y=1;z=2", "gold": 3, "task": "sum"},
            {"state": "三數加法;x=1;y=2;z=3", "gold": 6, "task": "sum"},
        ]
    else:
        rows = list(c_rows({"task": "card", "application": "123456", "family_id": "f"}, "train"))
    data = encoded_data(rows, product, tokenizer, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    generator = torch.Generator().manual_seed(17)
    options = {"batch_size": 4, "mode": "multi", "depth_weights": [1, 1, 1, 1, 1]}
    step(model, optimizer, data, generator, **options)
    path = tmp_path / "checkpoint.pt"
    checkpoint(path, model, optimizer, generator, {"step": 1, "scope": "fictional CPU fixture"})
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    before = state_digest(model.state_dict())
    receipt = verify_resume(path, product, torch.device("cpu"), data, options)
    assert receipt["exact_next_step"] and receipt["before_sha256"] == before
    assert receipt["after_sha256"] != before
    assert state_digest(model.state_dict()) == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == file_hash


@pytest.mark.parametrize("mismatch", [None, "run_identity", "mode", "seed", "training_recipe", "config", "model"])
def test_training_rejects_foreign_best_and_preserves_matching_best(tmp_path, monkeypatch, mismatch):
    from tiny_perceptron.lesson_products import train as trainer

    tokenizer = Tokenizer()
    tokenizer.save(tmp_path / "tokenizer.json")
    (tmp_path / "manifest.json").write_text('{"scope":"fictional CPU fixture"}\n')
    torch.manual_seed(301)
    model = LoopedSolver(small_config(tokenizer))
    initial = state_digest(model.state_dict())
    rows = [{"task": "sum", "state": "三數加法;x=0;y=1;z=2", "gold": 3}]
    data = encoded_data(rows, "b", tokenizer, model.config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    generator = torch.Generator().manual_seed(302)
    step(model, optimizer, data, generator, 4, "multi", [1] * 5)
    metadata = {
        "run_identity": "fixture-run-A",
        "product": "b",
        "mode": "multi",
        "seed": 301,
        "initial_weights_sha256": initial,
        "data_manifest_sha256": hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest(),
        "depth_weights": [1] * 5,
        "reference_depth": 5,
        "step": 20,
        "best_step": 20,
        "best_key": [1.0, -0.01],
        "best_weights_sha256": state_digest(model.state_dict()),
        "training_recipe": {"batch_size": 4, "learning_rate": 0.001},
    }
    last = tmp_path / "last.pt"
    checkpoint(last, model, optimizer, generator, metadata)
    best = torch.load(last, weights_only=True)
    if mismatch == "config":
        best["config"]["input_injection"] = 0.75
    elif mismatch == "model":
        best["model"]["answer.bias"][0] += 1
    elif mismatch:
        best["metadata"][mismatch] = "foreign"
    torch.save(best, tmp_path / "best.pt")
    monkeypatch.setattr(trainer, "read_dataset", lambda *args: rows)
    monkeypatch.setattr(trainer, "policy_report", lambda *args: ({"fixture": True}, (0.5, -1.0)))
    options = dict(max_steps=1, max_seconds=30, batch_size=4, resume=last, resume_best=tmp_path / "best.pt")
    if mismatch:
        with pytest.raises(ValueError, match="同一份原policy"):
            trainer.train(tmp_path, tmp_path / "result", "b", "multi", **options)
    else:
        result = trainer.train(tmp_path, tmp_path / "result", "b", "multi", **options)
        assert result["run_identity"] == "fixture-run-A"
        assert result["selected_weights_sha256"] == metadata["best_weights_sha256"]
        assert result["steps_this_attempt"] == 1 and result["resume_verification"]["exact_next_step"]
