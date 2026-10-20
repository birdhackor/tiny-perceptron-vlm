"""用 curl 讀官方原文，只保存定位附近的短摘錄與版本指紋。"""

from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import html
from html.parser import HTMLParser
import json
import platform
import re
import subprocess


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


SOURCES = [
    ("python_expressions", "https://docs.python.org/3.13/reference/expressions.html", ["floor division of integers", "Unicode code points", "generator object", "bitwise", "dictionary object"]),
    ("python_data", "https://docs.python.org/3.13/tutorial/datastructures.html", ["keys are unique", "set is an unordered", "enumerate()", "zip()", "list comprehensions"]),
    ("python_functions", "https://docs.python.org/3.13/library/functions.html", ["range ( stop", "next ( iterator", "key specifies", "sorted ( iterable", "ndigits", "class", "Return the length", "ord ("]),
    ("python_control", "https://docs.python.org/3.13/tutorial/controlflow.html", ["def introduces", "return", "for statement", "lambda"]),
    ("python_syntax", "https://docs.python.org/3.13/tutorial/introduction.html", ["Indexing returns", "while", "indentation", "equal sign"]),
    ("python_assert", "https://docs.python.org/3.13/reference/simple_stmts.html", ["assert statements", "AssertionError", "import statement"]),
    ("python_errors", "https://docs.python.org/3.13/library/exceptions.html", ["ModuleNotFoundError", "FileNotFoundError", "IndexError", "KeyError", "SyntaxError"]),
    ("python_path", "https://docs.python.org/3.13/library/pathlib.html", ["Path.cwd", "Path.exists", "current working directory"]),
    ("python_math", "https://docs.python.org/3.13/library/math.html", ["natural logarithm", "log(0.0)", "constant e"]),
    ("python_venv", "https://docs.python.org/3.13/library/venv.html", ["Activate.ps1", "activate a virtual environment", "full path"]),
    ("torch_tensor", "https://docs.pytorch.org/docs/2.14/tensors.html", ["multi-dimensional matrix", "tolist", "item", "shape"]),
    ("torch_mean", "https://docs.pytorch.org/docs/2.14/generated/torch.mean.html", ["keepdim", "reduced", "dim"]),
    ("torch_reshape", "https://docs.pytorch.org/docs/2.14/generated/torch.reshape.html", ["same data", "compatible", "shape"]),
    ("torch_linear", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html", ["linear transformation", "out_features", "learnable weights", "bias"]),
    ("torch_nograd", "https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html", ["Disabling gradient", "requires_grad", "thread local"]),
    ("torch_backward", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.backward.html", ["accumulates gradients", "chain rule", "zero"]),
    ("torch_ce", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html", ["unnormalized logits", "reduction", "mean"]),
    ("torch_softmax", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Softmax.html", ["sum to 1", "rescaled", "dim"]),
    ("uv_sync", "https://docs.astral.sh/uv/concepts/projects/sync/", ["--frozen", "--group", "--extra"]),
    ("uv_python", "https://docs.astral.sh/uv/concepts/python-versions/", ["only download", "automatically", "Python"]),
    ("git_clone", "https://git-scm.com/docs/git-clone", ["new directory", "clone", "working tree"]),
    ("git_windows", "https://gitforwindows.org/faq.html", ["260", "core.longpaths"]),
    ("lfs_config", "https://raw.githubusercontent.com/git-lfs/git-lfs/main/docs/man/git-lfs-config.adoc", ["GIT_LFS_SKIP_SMUDGE", "smudge", "download"]),
    ("lfs_install", "https://raw.githubusercontent.com/git-lfs/git-lfs/main/docs/man/git-lfs-install.adoc", ["--local", "filter", "repository"]),
    ("jupyter_notebook", "https://jupyterlab.readthedocs.io/en/stable/user/notebook.html", ["Shift", "kernel", "code"]),
    ("jupyter_commands", "https://jupyterlab.readthedocs.io/en/stable/user/commands.html", ["Restart Kernel and Run All Cells", "Shift Enter", "Insert Cell Below"]),
    ("jupyter_start", "https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html", ["jupyter lab", "notebook-dir", "directory"]),
    ("ipython_kernel", "https://ipython.readthedocs.io/en/stable/install/kernel_install.html", ["--prefix", "--display-name", "--name"]),
    ("ipykernel_install", "https://raw.githubusercontent.com/ipython/ipykernel/v6.30.1/ipykernel/kernelspec.py", ["sys-prefix", "sys.prefix", "display_name"]),
    ("powershell_policy", "https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-5.1", ["cannot be loaded", "running scripts", "Restricted"]),
    ("colab_faq", "https://research.google.com/colaboratory/faq.html", ["Google Account", "notebooks", "Jupyter"]),
    ("calculus_derivative", "https://openstax.org/books/calculus-volume-1/pages/3-1-defining-the-derivative", ["The derivative of a function", "difference quotient", "rate of change"]),
    ("calculus_gradient", "https://openstax.org/books/calculus-volume-3/pages/4-6-directional-derivatives-and-the-gradient", ["partial derivatives", "The gradient", "greatest rate"]),
    ("probability", "https://openstax.org/books/introductory-statistics-2e/pages/3-1-terminology", ["between zero and one", "sum", "sample space"]),
    ("independence", "https://openstax.org/books/introductory-statistics-2e/pages/3-2-independent-and-mutually-exclusive-events", ["independent", "mutually exclusive", "P(A"]),
    ("published_build", "https://birdhackor.github.io/tiny-perceptron-vlm/build-info.json", ["revision", "executed_cpu_outputs"]),
    ("published_example", "https://birdhackor.github.io/tiny-perceptron-vlm/1.1.html", ["實際執行結果", "下載本節", "Colab"]),
]


def fetch(entry):
    identifier, url, needles = entry
    command = ["curl", "-fLsS", "--max-time", "30", url]
    p = subprocess.run(command, capture_output=True, timeout=35)
    record = {"id": identifier, "url": url, "command": command, "returncode": p.returncode,
              "accessed_on": datetime.date.today().isoformat(), "stderr": p.stderr.decode(errors="replace")}
    if p.returncode:
        return record
    raw = p.stdout.decode(errors="replace")
    record["original_sha256"] = hashlib.sha256(p.stdout).hexdigest()
    if "<html" in raw.lower() or "<!doctype" in raw.lower():
        parser = Text()
        main = re.search(r"<article[^>]*>(.*?)</article>", raw, re.S|re.I)
        parser.feed(main.group(1) if main else raw)
        text = re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts)))
        title = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S|re.I)
        record["title"] = html.unescape(title.group(1)) if title else identifier
    else:
        text = re.sub(r"\s+", " ", raw)
        record["title"] = identifier
    record["excerpts"] = []
    for needle in needles:
        position = text.casefold().find(needle.casefold())
        record["excerpts"].append({"locator": needle, "found": position >= 0,
            "text": text[max(0,position-100):position+650] if position >= 0 else ""})
    return record


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, SOURCES))
    print(json.dumps({"environment":{"python":platform.python_version(),"reader":"curl + HTMLParser"},
        "command":"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-first-steps/inspect_sources.py",
        "sources":records},ensure_ascii=False,indent=2))
