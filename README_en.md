# Tiny Perceptron 小小感知機

How does a computer choose the next character after a short phrase? This from-scratch course starts with character IDs and a next-character table, using worked examples, diagrams and short programs to explain text, image and audio models.

[繁體中文](README.md) | English

**[Online course](https://birdhackor.github.io/tiny-perceptron-vlm/) · [Reading routes](course/README.md) · [First steps](course/first-steps.md) · [222 lessons](course/lessons.md) · [Training recipes](course/training.md)** — the main course is in Traditional Chinese.

This preview provides lesson text, SVG diagrams and actual CPU outputs. Each lesson has an Open in Colab button for the same complete notebook, with repository and package setup in its first cell. Small exercises need no GPU. Notebook downloads are also available for local practice. Execution in a real Colab runtime has not been verified.

The course has 18 chapters, three optional branches, 222 independently runnable notebooks and original SVG diagrams with progressive highlights. Begin with small matrices, bigrams, MLPs and manual attention; introduce modern architecture, Dense/MoE, cache/SDPA, quantization and distillation later. Readers can switch models and revisit earlier examples.

Core modules implement data handling, Transformers, multimodal inputs, DPO, packed quantization and training utilities directly in PyTorch. CLI tools cover data preparation, training, inference and evaluation. LoRA, QAT and small RL exercises demonstrate individual mechanisms. Video currently has frame-slicing helpers; a complete video course is planned for a later extension. This release does not include a trained general-purpose audiovisual assistant.

You can begin without prior knowledge of this project. Linked warmups introduce Python, tensors, probability, matrices, gradients and notebook operation as they become necessary. Lessons explain the problem and a worked example before introducing terminology. Validation uses CPU numerical and gradient checks. No actual model training was performed; readers will train and measure capabilities on their own hardware. See the [validation report](docs/validation.md).

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

Read the online course built with [Zensical](https://zensical.org/): full-text search, chapter navigation, page contents, light/dark modes and code copying. Every lesson retains its notebook download and Colab entry. SVGs can be enlarged, retain static labels and respect reduced-motion preferences. Website builds require the separate `--group site`. See [publishing instructions](docs/publishing.md) for local reading, verified outputs and GitHub Pages deployment. MathJax typesets formulas when online.

## Minimal workflow

```bash
python scripts/prepare_data.py --kind toy-text
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl
```

The default runs one forward/backward check with no optimizer update or checkpoint write. Add `--train` explicitly for training. [Training recipes](course/training.md) describe prerequisites, formats, evaluation and limitations for text, style, safety, multimodal, preferences and distillation.

[Fixed training snapshots](assets/training/README.md) are published as separate Git LFS archives. Use `python scripts/fetch_training_assets.py --list` to choose and unpack them. Raw caches, checkpoints and outputs stay in ignored `data/`, `checkpoints/` and `outputs/`. External datasets retain their own licenses; see [asset storage](docs/asset-storage.md).

## Maintenance

Edit `course/chapters/`, format that file’s Python examples, then regenerate notebooks. Each section needs an independent reader review of its current text and SVGs; the [editorial protocol](docs/editorial-guide.md) documents the reports. Publication rejects outdated reviews. Components live in `tiny_perceptron/`; SVG sources are in `scripts/build_visuals.py`.

```bash
python scripts/build_visuals.py
python scripts/format_course_code.py course/chapters/01.md
python scripts/build_course.py
python scripts/build_course.py --check
python scripts/check_course_reviews.py
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
