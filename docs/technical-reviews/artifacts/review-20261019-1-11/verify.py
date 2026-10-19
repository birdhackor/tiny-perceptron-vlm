"""保存獨立算例與官方來源的小段摘錄。"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime
from fractions import Fraction
from html.parser import HTMLParser
from zoneinfo import ZoneInfo


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def between(text, start, end):
    position = text.index(start)
    return text[position:text.index(end, position)].strip()


def sources():
    commit = "5c4886908584029761b579af026dcfb627c84070"
    base = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/"
    specs = [
        ("partial", "https://openstax.org/books/calculus-volume-3/pages/4-3-partial-derivatives",
         "Calculus Volume 3，當次線上版，Derivatives of a Function of Two Variables",
         [("計算偏導時固定其他變數", "The idea to keep in mind", "To see why this is true")]),
        ("gradient", "https://openstax.org/books/calculus-volume-3/pages/4-6-directional-derivatives-and-the-gradient",
         "Calculus Volume 3，當次線上版，The Gradient / Properties of the Gradient",
         [("梯度的定義", "Definition Let z = f", "Checkpoint 4.28"),
          ("反梯度方向", "is minimized when u u points", "Figure 4.41 The gradient")]),
        ("mechanics", "https://docs.pytorch.org/docs/2.14/notes/autograd.html",
         "PyTorch 2.14 官方版本文件，How autograd encodes the history / Setting requires_grad",
         [("計算圖及 chain rule", "Autograd is a reverse", "Internally, autograd"),
          ("累積梯度的預設對象", "During the forward pass, an operation is only recorded", "It is important to note")]),
        ("backward", base + "_tensor.py", f"PyTorch 安裝版本 2.14.1 的 git commit {commit}",
         [("Tensor.backward", "        r\"\"\"Computes the gradient of current tensor", "        .. note::")]),
        ("leaf", base + "_tensor_docs.py", f"PyTorch 安裝版本 2.14.1 的 git commit {commit}",
         [("Tensor.is_leaf", "All Tensors that have :attr:`requires_grad`", "Example::")]),
        ("allclose", base + "_torch_docs.py", f"PyTorch 安裝版本 2.14.1 的 git commit {commit}",
         [("torch.allclose", "allclose(input: Tensor", "Args:")]),
        ("assert", "https://docs.python.org/3.13/reference/simple_stmts.html",
         "Python 3.13 Language Reference，The assert statement，當次線上版",
         [("assert 的檢查行為", "Assert statements are a convenient", "The extended form"),
          ("最佳化模式限制", "In the current implementation, the built-in variable", "Note that it is unnecessary")]),
    ]
    receipts = []
    for identifier, url, version, selections in specs:
        command = ["curl", "-fLsS", "--max-time", "45", "--write-out",
                   "\nRECEIPT\t%{http_code}\t%{url_effective}", url]
        completed = subprocess.run(command, capture_output=True, check=True)
        raw, trailer = completed.stdout.rsplit(b"\nRECEIPT\t", 1)
        status, effective = trailer.decode().split("\t", 1)
        body = raw.decode()
        if not url.endswith(".py"):
            parser = PlainText()
            parser.feed(body)
            body = " ".join(" ".join(parser.parts).split())
        excerpts = [{"locator": label, "text": between(body, start, end)}
                    for label, start, end in selections]
        receipts.append({"id": identifier, "url": url, "effective_url": effective,
                         "version": version, "http_status": status,
                         "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
                         "command_argv": command, "exit_code": completed.returncode,
                         "response_bytes": len(raw),
                         "response_sha256": hashlib.sha256(raw).hexdigest(),
                         "excerpts": excerpts})
    return {"policy": "只保存支持主張的小段摘錄；原始回應僅在記憶體中讀取及計算指紋。",
            "sources": receipts}


def execute():
    import torch

    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    observations = []
    for coefficient in (0.5, 2.0):
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        before = w.detach().clone()
        first = (w[0] - 3).square()
        second = coefficient * w[1].square()
        loss = first + second
        loss.backward()
        expected = torch.tensor([-4.0, 4.0 * coefficient])
        assert torch.equal(w.grad, expected)
        assert torch.equal(w.detach(), before)
        assert w.grad.shape == w.shape
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            intermediate_grad = first.grad
        assert intermediate_grad is None
        leaf = {"w_is_leaf": w.is_leaf, "w_grad_fn_is_none": w.grad_fn is None,
                "first_is_leaf": first.is_leaf, "first_grad_is_none": intermediate_grad is None,
                "nonleaf_access_warning_count": len(caught)}
        assert leaf["w_is_leaf"] and leaf["w_grad_fn_is_none"]
        assert not leaf["first_is_leaf"]
        branch_grads = []
        for branch in ("first", "second"):
            v = torch.tensor([1.0, 2.0], requires_grad=True)
            term = (v[0] - 3).square() if branch == "first" else coefficient * v[1].square()
            term.backward()
            branch_grads.append(v.grad.tolist())
        observations.append({"coefficient": coefficient, "dtype": str(w.dtype),
                             "device": str(w.device), "costs": [first.item(), second.item()],
                             "loss": loss.item(), "before": before.tolist(),
                             "after_backward": w.detach().tolist(), "gradient": w.grad.tolist(),
                             "shape": list(w.shape), "gradient_shape": list(w.grad.shape),
                             "leaf": leaf, "branch_gradients": branch_grads})
    exact = []
    d = Fraction(-2)
    for h in (Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000)):
        lhs = ((d + h) ** 2 - d ** 2) / h
        rhs = 2 * d + h
        assert lhs == rhs
        exact.append({"d": str(d), "h": str(h), "quotient": str(lhs), "two_d_plus_h": str(rhs)})
    def cost(values):
        return (values[0] - 3) ** 2 + Fraction(1, 2) * values[1] ** 2
    base = [Fraction(1), Fraction(2)]
    step = Fraction(1, 10)
    direction_checks = {"initial": str(cost(base)),
                        "increase_first": str(cost([base[0] + step, base[1]])),
                        "decrease_second": str(cost([base[0], base[1] - step])),
                        "negative_gradient_step": str(cost([Fraction(7, 5), Fraction(9, 5)])),
                        "assign_gradient_as_parameters": str(cost([Fraction(-4), Fraction(2)])),
                        "minimum_at_target": str(cost([Fraction(3), Fraction(0)]))}
    assert direction_checks == {"initial": "6", "increase_first": "561/100",
                                "decrease_second": "1161/200", "negative_gradient_step": "209/50",
                                "assign_gradient_as_parameters": "51", "minimum_at_target": "0"}
    near = torch.tensor([-4.0, 2.000001])
    far = torch.tensor([-4.0, 2.001])
    reference = torch.tensor([-4.0, 2.0])
    comparison = {"exact": torch.allclose(reference, reference.clone()),
                  "near": torch.allclose(near, reference), "far": torch.allclose(far, reference)}
    assert comparison == {"exact": True, "near": True, "far": False}
    snapshot = reference.clone()
    assert torch.allclose(reference, torch.tensor([-4.0, 2.0]))
    assert torch.equal(reference, snapshot)
    retained = torch.tensor([1.0, 2.0], requires_grad=True)
    intermediate = (retained[0] - 3).square()
    intermediate.retain_grad()
    intermediate.backward()
    assert intermediate.grad.item() == 1.0
    return {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + sys.argv[0] + " --execute",
            "environment": {"python": sys.version, "torch": str(torch.__version__),
                            "torch_git_version": str(torch.version.git_version),
                            "platform": platform.platform(), "device": "cpu"},
            "observations": observations, "exact_difference_quotients": exact,
            "direction_checks": direction_checks, "allclose": comparison,
            "assert_preserved_tensor": torch.equal(reference, snapshot),
            "retained_nonleaf_grad": intermediate.grad.item(), "result": "所有檢查通過"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sources", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = sources() if args.sources else execute()
    print(json.dumps(result, ensure_ascii=False, indent=2))
