"""Independent numeric checks and exact-section-code exercises; CPU only."""
import contextlib
from decimal import Decimal, localcontext
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent
original = (ROOT / "fence-1.py").read_bytes()
checks = {}
with localcontext() as ctx:
    ctx.prec = 40
    table = []
    for value in ("1", "1.1", "1.01"):
        w = Decimal(value)
        loss = (w - 3) ** 2
        delta_w = w - 1
        delta_loss = loss - 4
        quotient = delta_loss / delta_w if delta_w else None
        table.append({"w": value, "loss": str(loss), "delta_w": str(delta_w),
                      "delta_loss": str(delta_loss), "quotient": str(quotient)})
    assert [row["loss"] for row in table] == ["4", "3.61", "3.9601"]
    assert [row["quotient"] for row in table[1:]] == ["-3.9", "-3.99"]
    checks["exact_table"] = table
    h = Decimal("0.001")
    checks["exact_centered_difference"] = []
    for w in (Decimal(1), Decimal(5), Decimal(3)):
        left, right = (w-h-3)**2, (w+h-3)**2
        gradient = (right-left)/(2*h)
        assert gradient == 2*(w-3)
        checks["exact_centered_difference"].append({
            "w":str(w), "h":str(h), "left":str(left), "right":str(right),
            "delta_loss":str(right-left), "denominator":str(2*h), "gradient":str(gradient)})
    delta = (Decimal("1.001")-3)**2 - 4
    linear = Decimal(-4) * h
    assert delta == Decimal("-0.003999")
    assert delta-linear == h*h
    checks["local_linear_approximation"] = {
        "delta_w":"0.001", "exact_delta_loss":str(delta),
        "derivative_times_delta_w":str(linear), "absolute_error":str(abs(delta-linear)),
        "relative_error_to_exact_change":str(abs((delta-linear)/delta))}
    # A distant positive move from 1 to 6 reverses the local improvement.
    assert (Decimal(6)-3)**2 > (Decimal(1)-3)**2
    checks["distant_step"] = {"start_w":1,"end_w":6,"start_loss":4,"end_loss":9}

checks["actual_code_runs"] = []
for value, expected in (("1.0", -4.0), ("5.0", 4.0), ("3.0", 0.0)):
    code = original if value == "1.0" else original.replace(b"w = 1.0", f"w = {value}".encode())
    name = f"exercise-w-{value}.py"
    (ROOT / name).write_bytes(code)
    namespace = {"__name__":"__main__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, str(ROOT / name), "exec"), namespace)
    assert abs(namespace["gradient"] - expected) <= 2e-12
    assert namespace["w"] == float(value)
    checks["actual_code_runs"].append({
        "w":namespace["w"], "left":namespace["left"], "right":namespace["right"],
        "gradient":namespace["gradient"], "analytic":expected, "w_unchanged":True,
        "code_path":name, "code_sha256":hashlib.sha256(code).hexdigest(), "stdout":stdout.getvalue()})

def centered(func, w, h):
    return (func(w+h)-func(w-h))/(2*h)

# These are evidence of numerical scope, not a replacement for official analysis.
quadratic = lambda value: (value-3)**2
checks["finite_difference_scope"] = {
    "quadratic_h_1":centered(quadratic,1.0,1.0),
    "quadratic_h_1e_minus_16":centered(quadratic,1.0,1e-16),
    "sin_at_zero_h_1":centered(math.sin,0.0,1.0),
    "sin_at_zero_h_0_001":centered(math.sin,0.0,0.001),
    "sin_derivative_at_zero":1.0,
}
assert checks["finite_difference_scope"]["quadratic_h_1"] == -4.0
assert abs(checks["finite_difference_scope"]["quadratic_h_1e_minus_16"] + 4.0) > 0.1
assert abs(checks["finite_difference_scope"]["sin_at_zero_h_0_001"] - 1.0) < 2e-7

result = {
    "environment":{"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
                   "device":"CPU","numeric_type":"Python binary64 float and Decimal(precision=40)",
                   "float_mantissa_bits":sys.float_info.mant_dig},
    "axis_and_units":"w is the single parameter axis; L is arbitrary loss units. Derivative units are loss units per w unit. Centered denominator is 2h, not h; one synthetic scalar problem, no dataset/token mean.",
    "checks":checks,
    "tolerance":"Exact Decimal identities; binary64 gradients absolute error <=2e-12. No tolerance is used to hide error in the tiny-h example.",
    "passed":True,
}
(ROOT / "bounded-results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))
