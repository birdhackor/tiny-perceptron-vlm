"""Bounded CPU checks of the chapter's position relation, without a model."""
import hashlib
import json
import platform
import sys
from pathlib import Path


def visible(position, window):
    return set(range(max(0, position - window + 1), position + 1))


def ancestors(position, window, layers):
    positions = {position}
    for _ in range(layers):
        positions = set().union(*(visible(p, window) for p in positions))
    return positions


cases = []
for window in (3, 2):
    direct = sorted(visible(7, window))
    two = sorted(ancestors(7, window, 2))
    full_pairs = sum(p + 1 for p in range(8))
    local_pairs = sum(len(visible(p, window)) for p in range(8))
    expected = ([5, 6, 7], [3, 4, 5, 6, 7], 21) if window == 3 else ([6, 7], [5, 6, 7], 15)
    assert (direct, two, local_pairs) == expected
    assert full_pairs == 36
    cases.append({"length": 8, "window_including_self": window, "direct_at_7": direct, "two_layers_at_7": two, "full_causal_pairs": full_pairs, "sliding_pairs": local_pairs})

checked_pair_formulas = 0
checked_receptive_fields = 0
for length in range(1, 33):
    for window in range(1, 41):
        width = min(length, window)
        count = sum(len(visible(p, window)) for p in range(length))
        assert count == length * width - width * (width - 1) // 2
        checked_pair_formulas += 1
        for position in range(length):
            for layers in range(1, 5):
                actual = ancestors(position, window, layers)
                expected = set(range(max(0, position - layers * (window - 1)), position + 1))
                assert actual == expected
                checked_receptive_fields += 1

result = {
    "kind": "bounded_cpu_position_relation_verification",
    "cases": cases,
    "pair_formula": "q=min(n,w); n*q-q*(q-1)/2",
    "full_causal_pair_formula": "n*(n+1)/2",
    "reachable_positions_formula": "[max(0,p-L*(w-1)),p] inclusive",
    "denominator_unit": "ordered (query position, key position) pairs per layer; includes self",
    "pair_formula_cases": checked_pair_formulas,
    "receptive_field_cases": checked_receptive_fields,
    "scope": "Position graph and counts only; no timing, attention values, training, token recall, or model performance was measured.",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu", "third_party_dependencies": "none"},
    "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
print(json.dumps(result, ensure_ascii=False, indent=2))
