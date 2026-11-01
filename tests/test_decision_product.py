"""C 成品的標籤獨立核對、防洩漏與分支不變性；不依賴網路或真實權重。"""

import random
from dataclasses import replace

import pytest
import torch

from tiny_perceptron.decision_product.data import (
    Example,
    action_gold,
    family_key,
    info_gold,
    make_example,
    make_families,
    partition,
    rubric_gold,
)
from tiny_perceptron.decision_product.indexed_model import IndexedConfig, IndexedDecisionModel, encode_indexed_example
from tiny_perceptron.decision_product.model import Config, DecisionModel, Tokenizer, collate, encode_example
from tiny_perceptron.decision_product.qualification import (
    register_exposure,
    reserve_final_families,
    selective_metrics,
    upper95,
)
from tiny_perceptron.decision_product.record_model import RecordConfig, RecordDecisionModel, encode_record_example
from tiny_perceptron.decision_product.runtime import candidate_eligibility, collate_for_model
from tiny_perceptron.decision_product.service import Action, DecisionEngine, Device


def test_visible_family_partition_ignores_names_order_and_target_role():
    facts = ((0, 1, 1), (1, -1, 0), (2, 0, 1), (0, 1, 0))
    assert family_key(facts) == family_key(tuple(reversed(facts)))
    assert family_key(facts) != family_key((*facts[:-1], (0, -1, 0)))
    groups = {
        name: make_families(name, 80, seed) for seed, name in enumerate(("train", "dev", "calibration", "policy"))
    }
    keys = set()
    for name, families in groups.items():
        current = {family["key"] for family in families}
        assert len(current) == len(families)
        assert all(partition(key) == name for key in current)
        assert not current & keys
        keys.update(current)


@pytest.mark.parametrize(
    "task,permission,ready,action,info,grade",
    [
        (0, 0, 0, "stop", "yes", "0"),
        (0, 0, 1, "stop", "yes", "1"),
        (0, 1, 0, "wait", "yes", "1"),
        (0, 1, 1, "record", "yes", "2"),
        (1, 0, 0, "stop", "yes", "0"),
        (1, 0, 1, "stop", "yes", "1"),
        (1, 1, 0, "wait", "yes", "2"),
        (1, 1, 1, "play", "yes", "2"),
        (2, 0, 0, "stop", "yes", "0"),
        (2, 0, 1, "stop", "yes", "2"),
        (2, 1, 0, "wait", "yes", "1"),
        (2, 1, 1, "save", "yes", "2"),
        (0, -1, 1, "none", "no", "none"),
        (1, 1, -1, "none", "yes", "none"),
        (2, -1, 1, "none", "yes", "none"),
        (2, 0, -1, "none", "no", "none"),
        (1, 0, -1, "none", "yes", "none"),
    ],
)
def test_label_truth_table(task, permission, ready, action, info, grade):
    fact = (task, permission, ready)
    assert action_gold(fact) == action
    assert info_gold(fact) == info
    assert rubric_gold(fact) == grade


def test_candidate_count_does_not_reveal_presence_of_answer():
    rng = random.Random(411)
    family = {"key": "test", "facts": [(0, 1, 1), (1, 0, 1), (2, 1, 0), (0, 0, 0)]}
    observed = {"complete": set(), "removed": set()}
    for _ in range(400):
        for panel in observed:
            example = make_example(family, rng, kind="choice", panel=panel)
            observed[panel].add(len(example.candidates))
            assert (example.gold == "none") == (panel == "removed")
    assert observed["complete"] == observed["removed"] == set(range(1, 8))


def test_encoder_never_receives_gold_or_family_metadata():
    example = Example(
        "A: ready=yes",
        "Choose A",
        (("first", "record"), ("second", "wait")),
        "first",
        "choice",
        "family-a",
        "source-a",
        "complete",
    )
    other = replace(example, gold="second", family="unrelated", source="unrelated", panel="removed")
    tokenizer = Tokenizer.fit((example.state, example.question, "record", "wait"))
    first, second = encode_example(example, tokenizer), encode_example(other, tokenizer)
    assert first.pop("label") != second.pop("label")
    assert first == second


def test_branch_permutation_padding_and_shared_state_memory():
    torch.manual_seed(4)
    torch.set_num_threads(1)
    example = Example(
        "A: task=record; permission=yes; ready=no. B: task=play; permission=no; ready=yes.",
        "Choose A",
        (("x", "Device A: wait"), ("y", "Device B: stop")),
        "x",
        "choice",
        "f",
        "compact",
        "complete",
    )
    tokenizer = Tokenizer.fit(
        (
            example.state,
            example.question,
            *(description for _, description in example.candidates),
            "Extra long description for a padding candidate",
        )
    )
    model = DecisionModel(Config(tokenizer.size, width=32, state_layers=1, readers=1)).eval()
    row = encode_example(example, tokenizer)
    reverse = encode_example(replace(example, candidates=tuple(reversed(example.candidates))), tokenizer)
    padded = encode_example(
        replace(example, candidates=(*example.candidates, ("z", "Extra long description for a padding candidate"))),
        tokenizer,
    )
    with torch.inference_mode():
        batch = collate([row])
        plain = model(batch)[0]
        shuffled = model(collate([reverse]))[0]
        mixed = model(collate([row, padded]))[0, :3]
        memory = model.state_encoder(batch["state"])
        cached = model(batch, memory=memory)[0]
    torch.testing.assert_close(plain, shuffled[[1, 0, 2]], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(plain, mixed, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(plain, cached, atol=2e-6, rtol=2e-6)


def test_unknown_unicode_uses_lossless_utf8_fallback():
    tokenizer = Tokenizer.fit(("Known",))
    encoded = tokenizer.encode("新🙂")
    assert len(encoded) > 2
    assert tokenizer.encode("新🙂") != tokenizer.encode("新🙃")
    with pytest.raises(UnicodeEncodeError):
        tokenizer.encode("\ud800")


def test_selective_policy_counts_confident_wrong_and_does_not_accept_none():
    rows = [
        {"logits": [8.0, 0.0, -1.0], "ids": ["x", "y", "none"], "label": 1},
        {"logits": [0.0, -1.0, 8.0], "ids": ["x", "y", "none"], "label": 2},
        {"logits": [8.0, -1.0, 0.0], "ids": ["x", "y", "none"], "label": 0},
    ]
    for row in rows:
        row["eligible"] = [True, True, False]
    metrics = selective_metrics(rows, 1.0, 0.95)
    assert metrics["accepted"] == 2
    assert metrics["errors"] == 1
    assert metrics["risk"] == 0.5
    assert upper95(0, 600) < 0.005
    assert upper95(0, 0) == 1.0


@pytest.fixture
def engine():
    from tiny_perceptron.decision_product.data import evaluation_examples

    torch.set_num_threads(1)
    examples = evaluation_examples(make_families("train", 30, 42), 44)
    tokenizer = Tokenizer.fit(
        text for ex in examples for text in (ex.state, ex.question, *(d for _, d in ex.candidates))
    )
    model = DecisionModel(Config(tokenizer.size, width=32, state_layers=1, readers=1))
    return DecisionEngine._for_development(model, tokenizer, {kind: 1.0 for kind in ("choice", "info", "rubric")}, 0.9)


def devices():
    return (
        Device("錄音室1", "record", True, False),
        Device("player-2", "play", None, True),
        Device("archive-3", "save", False, True),
    )


def test_service_inference_does_not_call_label_oracles(engine, monkeypatch):
    import tiny_perceptron.decision_product.data as data

    def forbidden(*args, **kwargs):
        raise AssertionError("推論不可呼叫標籤規則")

    for name in ("action_gold", "info_gold", "rubric_gold", "make_example"):
        monkeypatch.setattr(data, name, forbidden)
    session = engine.session(devices())
    assert session.choose("錄音室1").kind == "choice"
    assert session.information("player-2").kind == "info"
    assert session.grade("archive-3").kind == "rubric"


def test_service_rejects_ambiguous_or_invalid_inputs(engine):
    with pytest.raises(ValueError):
        engine.session(devices()[:2])
    with pytest.raises(ValueError):
        engine.session((devices()[0], devices()[0], devices()[2]))
    for bad in (1, 0.0, "yes"):
        with pytest.raises(ValueError):
            engine.session((replace(devices()[0], permission=bad), *devices()[1:]))
    with pytest.raises(ValueError):
        engine.session((replace(devices()[0], task=[]), *devices()[1:]))
    session = engine.session(devices())
    with pytest.raises(ValueError):
        session.choose([])
    with pytest.raises(ValueError):
        session.choose("錄音室1", (Action("x", "錄音室1", "wait"), Action("y", "錄音室1", "wait")))
    with pytest.raises(ValueError):
        session.choose("錄音室1", (Action("none", "錄音室1", "wait"),))
    with pytest.raises(ValueError):
        session.choose("錄音室1", (Action("x", [], "wait"),))


def test_state_cache_reuses_state_across_questions_and_invalidates_weights(engine):
    session = engine.session(devices())
    first = session.choose("錄音室1")
    second = session.choose("錄音室1", use_cache=False)
    assert first.model_choice == second.model_choice
    assert first.confidence == pytest.approx(second.confidence, abs=2e-6)
    session.information("player-2")
    assert len(engine._cache) == 1
    with torch.no_grad():
        engine._model.state_encoder.positions.weight.add_(0.03)
    session.choose("錄音室1")
    assert len(engine._cache) == 2


def test_nonfinite_model_output_is_never_accepted(engine):
    with torch.no_grad():
        engine._model.score[-1].bias.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="模型輸出異常"):
        engine.session(devices()).choose("錄音室1")


def test_record_encoder_is_invariant_to_row_order_and_has_no_all_padding_nan():
    torch.manual_seed(87)
    torch.set_num_threads(1)
    rows = [
        "A: task=record; permission=yes; ready=no",
        "B: task=play; permission=no; ready=yes",
        "C: task=save; permission=yes; ready=yes",
        "D: task=play; permission=yes; ready=no",
    ]
    example = Example(
        ". ".join(rows) + ".",
        "Choose A",
        (("wait", "Device A: wait"), ("stop", "Device A: stop")),
        "wait",
        "choice",
        "f",
        "compact",
        "complete",
    )
    tokenizer = Tokenizer.fit((example.state, example.question, *(desc for _, desc in example.candidates)))
    model = RecordDecisionModel(RecordConfig(tokenizer.size, width=32, state_layers=1, readers=1)).eval()
    first = encode_record_example(example, tokenizer)
    second = encode_record_example(replace(example, state=". ".join(reversed(rows)) + "."), tokenizer)
    short = encode_record_example(replace(example, state=rows[0] + "."), tokenizer)
    with torch.inference_mode():
        a = model(collate([first]))
        b = model(collate([second]))
        mixed = model(collate([first, short]))
    assert bool(torch.isfinite(mixed).all())
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(a[0], mixed[0], atol=3e-6, rtol=3e-6)


def test_round_final_is_shared_across_runs_and_never_reuses_exposed_families(tmp_path, monkeypatch):
    import tiny_perceptron.decision_product.qualification as qualification

    # 用人工 key 測鎖定與排除機制；不建立任何真實 final 世界。
    def fake_families(split, count, seed):
        assert split == "final"
        return [{"key": f"fixture-{i}", "facts": []} for i in range(count)]

    monkeypatch.setattr(qualification, "make_families", fake_families)
    round_a = tmp_path / "outputs/round-a"
    register_exposure(round_a, ["fixture-0"], "unit-test-fake-keys", "fake-digest")
    run_a = round_a / "run-a"
    (run_a / "qualification").mkdir(parents=True)
    result = reserve_final_families(run_a, {"protocol": {"families": 3, "family_seed": 123}})
    assert [family["key"] for family in result] == ["fixture-1", "fixture-2", "fixture-3"]
    run_b = round_a / "run-b"
    (run_b / "qualification").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        reserve_final_families(run_b, {"protocol": {"families": 3, "family_seed": 123}})
    round_b = tmp_path / "outputs/round-b/run-c"
    (round_b / "qualification").mkdir(parents=True)
    result = reserve_final_families(round_b, {"protocol": {"families": 3, "family_seed": 456}})
    assert [family["key"] for family in result] == ["fixture-4", "fixture-5", "fixture-6"]


def test_cli_rejects_missing_boolean_fields_and_unknown_keys(engine):
    from tiny_perceptron.decision_product.cli import predict_request

    request = {"devices": [{"id": "a", "task": "record", "permission": True}], "target": "a"}
    with pytest.raises(ValueError, match="明列"):
        predict_request(engine, request)
    with pytest.raises(ValueError, match="未知欄位"):
        predict_request(engine, {**request, "gold": "record"})


def test_indexed_memory_uses_only_id_lookup_and_reuses_all_state_memory():
    torch.manual_seed(89)
    torch.set_num_threads(1)
    state = "A: task=record; permission=yes; ready=no. B: task=play; permission=no; ready=yes. C: task=save; permission=yes; ready=yes."
    example = Example(
        state, "Choose A", (("x", "Device A: wait"), ("y", "Device B: stop")), "x", "choice", "f", "compact", "complete"
    )
    tokenizer = Tokenizer.fit((state, "Choose A", "Choose B", *(desc for _, desc in example.candidates)))
    first = encode_indexed_example(example, tokenizer)
    other_gold = encode_indexed_example(replace(example, gold="y", family="other"), tokenizer)
    assert first["target_row"] == other_gold["target_row"] == 0
    assert first["state"] == other_gold["state"]
    second = encode_indexed_example(replace(example, question="Choose B"), tokenizer)
    assert second["target_row"] == 1
    assert first["state"] == second["state"]
    model = IndexedDecisionModel(IndexedConfig(tokenizer.size, width=32, state_layers=1, readers=1)).eval()
    with torch.inference_mode():
        batch = collate_for_model([first, second], model)
        memory = model.state_encoder(batch["state"])
        shared = model(batch, memory=memory)
        plain = model(batch)
        model.fact_head.weight.fill_(float("nan"))
        unused_auxiliary = model(batch, memory=memory)
    torch.testing.assert_close(shared, plain, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(shared, unused_auxiliary, atol=3e-6, rtol=3e-6)
    assert bool(torch.isfinite(shared).all())
    for bad in ("Choose D", "Choose A and B", "Choose target"):
        with pytest.raises(ValueError):
            encode_indexed_example(replace(example, question=bad), tokenizer)


def test_indexed_prediction_is_invariant_to_whole_record_permutation():
    torch.manual_seed(90)
    rows = [
        "A: task=record; permission=yes; ready=no",
        "B: task=play; permission=no; ready=yes",
        "C: task=save; permission=yes; ready=yes",
    ]
    example = Example(
        ". ".join(rows) + ".",
        "Choose A",
        (("x", "Device A: wait"), ("y", "Device A: stop")),
        "x",
        "choice",
        "f",
        "compact",
        "complete",
    )
    tokenizer = Tokenizer.fit((example.state, example.question, *(desc for _, desc in example.candidates)))
    model = IndexedDecisionModel(IndexedConfig(tokenizer.size, width=32, state_layers=1, readers=1)).eval()
    first = encode_indexed_example(example, tokenizer)
    second = encode_indexed_example(replace(example, state=". ".join(reversed(rows)) + "."), tokenizer)
    assert first["target_row"] == 0 and second["target_row"] == 2
    with torch.inference_mode():
        a = model(collate_for_model([first], model))
        b = model(collate_for_model([second], model))
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)


def test_candidate_target_guard_never_computes_correct_action():
    example = Example(
        "A: task=record; permission=no; ready=no. B: task=play; permission=yes; ready=yes.",
        "Choose A",
        (("a", "Device A: record"), ("b", "Device B: record"), ("c", "Device A: stop")),
        "c",
        "choice",
        "f",
        "compact",
        "complete",
    )
    assert candidate_eligibility(example) == [True, False, True, False]
    # 把真值／動作條件改掉仍只做對象核對，不能因此改選正解。
    assert candidate_eligibility(replace(example, state="A: task=record; permission=yes; ready=yes.", gold="a")) == [
        True,
        False,
        True,
        False,
    ]
    row = {
        "logits": [0.0, 12.0, -1.0, -2.0],
        "ids": ["a", "b", "c", "none"],
        "label": 2,
        "eligible": [True, False, True, False],
    }
    result = selective_metrics([row], 1.0, 0.9)
    assert result["accepted"] == 0
    assert result["coverage"] == 0


def test_service_rejects_high_confidence_foreign_device_prediction(engine, monkeypatch):
    def foreign_logits(batch, memory=None):
        return torch.tensor([[14.0, -3.0, -4.0]])

    monkeypatch.setattr(engine._model, "forward", foreign_logits)
    session = engine.session(devices())
    result = session.choose("錄音室1", (Action("foreign", "player-2", "record"), Action("local", "錄音室1", "wait")))
    assert result.status == "abstained"
    assert result.selected is None
    assert result.model_choice == "foreign"
    assert result.confidence > 0.99


def test_retargeting_covers_both_unknown_without_changing_or_crossing_worlds():
    from collections import Counter
    from itertools import product

    from tiny_perceptron.decision_product.hardening_experiment import target_views

    expected = set(product(range(3), (-1, 0, 1), (-1, 0, 1)))
    all_keys = set()
    for split, count, seed in (("train", 8000, 826019), ("dev", 600, 826021)):
        families = make_families(split, count, seed)
        originals = {family["key"]: Counter(map(tuple, family["facts"])) for family in families}
        groups = target_views(families)
        assert set(groups) == expected
        current = set()
        for fact, views in groups.items():
            for view in views:
                assert tuple(view["facts"][0]) == fact
                assert Counter(map(tuple, view["facts"])) == originals[view["key"]]
                assert family_key(view["facts"]) == view["key"]
                assert partition(view["key"]) == split
                current.add(view["key"])
        assert not current & all_keys
        all_keys.update(current)


def test_competing_options_do_not_use_count_or_foreign_action_as_the_gold():
    from tiny_perceptron.decision_product.hardening_experiment import competing_example
    from tiny_perceptron.decision_product.indexed_model import ENTITY

    family = {"key": "synthetic", "facts": [(0, 1, 1), (1, 0, 0), (2, -1, -1), (0, 1, 0)]}
    rng = random.Random(838)
    counts = {False: set(), True: set()}
    for removed in (False, True):
        for _ in range(100):
            example = competing_example(family, rng, source="chinese", removed=removed)
            target = ENTITY.findall(example.question)[0]
            options = dict(example.candidates)
            counts[removed].add(len(options))
            assert any(ENTITY.findall(desc)[0] != target and desc.endswith("record") for desc in options.values())
            if removed:
                assert example.gold == "none"
                assert not any(
                    ENTITY.findall(desc) == [target] and desc.endswith("record") for desc in options.values()
                )
            else:
                assert ENTITY.findall(options[example.gold]) == [target]
                assert options[example.gold].endswith("record")
    assert counts[False] == counts[True] == {5, 6, 7}
