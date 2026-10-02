# Tiny Perceptron 小小感知機

A from-scratch tutorial that grows from a next-character table into small text, image and audio models. Each lesson explains one idea through an analogy, a diagram and a short numerical experiment.

[繁體中文](README.md) | English

**[Online course](https://birdhackor.github.io/tiny-perceptron-vlm/) · [Reading routes](course/README.md) · [First steps](course/first-steps.md) · [222 lessons](course/lessons.md) · [Training recipes](course/training.md)** — the main course is in Traditional Chinese.

This preview provides lesson text, SVG diagrams and actual CPU outputs. Each lesson has an Open in Colab button for the same complete notebook, with repository and package setup in its first cell. Small exercises need no GPU. Notebook downloads are also available for local practice. Execution in a real Colab runtime has not been verified.

The course has 18 chapters, three optional branches, 222 independently runnable notebooks and 40 original SVG diagrams with progressive highlights. Begin with small matrices, bigrams, MLPs and manual attention; introduce modern architecture, Dense/MoE, cache/SDPA, quantization and distillation later. Readers can switch models and revisit earlier examples.

Core modules implement data handling, Transformers, multimodal inputs, DPO, packed quantization and training utilities directly in PyTorch. CLI tools cover data preparation, training, inference and evaluation. LoRA, QAT, video sequences and small RL exercises demonstrate individual mechanisms; this release does not include a trained general-purpose audiovisual assistant.

University calculus or linear algebra is sufficient background; the first-steps guide introduces Python, tensors and notebook operation. Validation uses CPU numerical and gradient checks. No actual model training was performed; readers will train and measure capabilities on their own hardware. See the [validation report](docs/validation.md).

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. For CPU exercises:

```bash
git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
uv sync --frozen --extra cpu --group notebook
source .venv/bin/activate
python scripts/check_env.py
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
jupyter lab notebooks
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Select **Tiny Perceptron** and open `notebooks/01/1.1.ipynb`. Use `uv sync --frozen --group notebook` on Apple Silicon, or the matching `cu126`/`cu130` extra on compatible NVIDIA hardware. Driver requirements are in [environment notes](docs/environment.md).

Activate `.venv` and use `python` directly, or pass the same extra to each `uv run`. CUDA/MPS execution was not verified in the current CPU environment.

Use the online course, or run `python scripts/export_course.py` and open `outputs/site/index.html` for an offline copy. Search lessons there; edit and execute code in notebooks. SVGs retain static labels and respect reduced-motion preferences. MathJax typesets formulas when online; offline copies retain TeX text. See [publishing instructions](docs/publishing.md) to include verified outputs or deploy GitHub Pages.

## Minimal workflow

```bash
python scripts/prepare_data.py --kind toy-text
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl
```

The default runs one forward/backward check with no optimizer update or checkpoint write. Add `--train` explicitly for training. [Training recipes](course/training.md) describe prerequisites, formats, evaluation and limitations for text, style, safety, multimodal, preferences and distillation.

[Fixed training snapshots](assets/training/README.md) are published as separate Git LFS archives. Use `python scripts/fetch_training_assets.py --list` to choose and unpack them. Raw caches, checkpoints and outputs stay in ignored `data/`, `checkpoints/` and `outputs/`. External datasets retain their own licenses; see [asset storage](docs/asset-storage.md).

## Maintenance

Edit `course/chapters/`, then regenerate notebooks. Components live in `tiny_perceptron/`; SVG sources are in `scripts/build_visuals.py`.

```bash
python scripts/build_visuals.py
python scripts/build_course.py
python scripts/build_course.py --check
python scripts/check_notebooks.py --mode python
python scripts/check_notebooks.py --mode kernel --lesson 3.6
pytest -ra
ruff check .
ruff format --check .
```

Kernel mode starts a fresh kernel per lesson and saves executed copies under `outputs/`. Existing GitHub Actions run core checks on Linux, macOS and Windows; complete notebook validation is a separate workflow.

## References and license

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat), [MiniMind](https://github.com/jingyaogong/minimind), [MiniMind-V](https://github.com/jingyaogong/minimind-v) and [nanoVLM](https://github.com/huggingface/nanoVLM). Research sources and limitations are in [curriculum notes](docs/curriculum.md).

Code, course text and original diagrams: [MIT](LICENSE). External data has separate licensing.
