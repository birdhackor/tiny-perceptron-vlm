import json
from tiny_perceptron.data import ByteTokenizer

# Exact raw-token comparison is taken from common.evaluate_lm, not model inference.
tok = ByteTokenizer()
expected = "red"
observed = []
for generated in ("red", "green", "", "無法回答", "pink"):
    raw = tok.encode(generated)
    exact = raw == tok.encode(expected)
    lacks_injected_color = "pink" not in generated
    assert exact == (generated == expected)
    observed.append({"generated": generated, "expected": expected, "exact_raw_token_match": exact, "lacks_injected_color": lacks_injected_color})
assert sum(x["exact_raw_token_match"] for x in observed) == 1
assert any(x["lacks_injected_color"] and not x["exact_raw_token_match"] for x in observed)
print(json.dumps({"synthetic_comparator_inputs": observed, "scope": "ByteTokenizer plus the inspected original evaluate_lm equality expression only; no generated model samples or inference"}, ensure_ascii=False, indent=2))
