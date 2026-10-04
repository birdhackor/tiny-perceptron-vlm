# Tiny Perceptron 小小感知機

How does a computer choose the next character after a short phrase? This from-scratch course starts with character IDs and a next-character table, using worked examples, diagrams and short programs to explain text, image and audio models.

[繁體中文](README.md) | English

**[Online course](https://birdhackor.github.io/tiny-perceptron-vlm/) · [Reading routes](course/README.md) · [First steps](course/first-steps.md) · [248 lessons](course/lessons.md) · [Training recipes](course/training.md)** — the main course is in Traditional Chinese.

This preview provides lesson text, SVG diagrams and actual CPU outputs. Each lesson has an Open in Colab button for the same complete notebook, with repository and package setup in its first cell. Small exercises need no GPU. Notebook downloads are also available for local practice. Execution in a real Colab runtime has not been verified.

The course has 19 chapters, three optional branches and 248 lesson notebooks, with original SVG diagrams and progressive highlights. Including 26 numbered reading, warmup, training and glossary sections, there are 274 numbered sections. Begin with small matrices, bigrams, MLPs and manual attention; introduce modern architecture, Dense/MoE, cache/SDPA, quantization and distillation later. Readers can switch models and revisit earlier examples.

Core modules implement data handling, Transformers, multimodal inputs, DPO, packed quantization and training utilities directly in PyTorch. CLI tools cover data preparation, training, inference and evaluation. LoRA, QAT and small RL exercises demonstrate individual mechanisms. Video currently has frame-slicing helpers; a complete video course is planned for a later extension. This release does not include a trained general-purpose audiovisual assistant.

After the concrete continuation example in [7.11](course/chapters/07.md#7.11), [7.17–7.18](course/chapters/07.md#7.17) explain why pre-training and post-training often use different data and learning signals. Start with the two-answer comparison in [13.1](course/chapters/13.md#13.1), then follow [13.10–13.17](course/chapters/13.md#13.10) for reward models, the roles in a typical RLHF pipeline, PPO and DPO. The PPO experiment selects finite, prewritten candidate responses; it does not train an autoregressive language model.

[Chapter 19](course/chapters/19.md#19.1) has trained text, conversation and joint tasks sequentially on the same small MoE core, preserving a DPO comparison, quantized versions and [deployment evaluations](docs/course-experiments/results/capstone_deployment.json). Tasks are limited to a synthetic small world with individually checkable answers. The recommended joint-task model passes 78/90 final test records; its remaining failures are listed in [19.12](course/chapters/19.md#19.12). Before trying the interface in 19.1, follow [19.11](course/chapters/19.md#19.11) to obtain the specified inference file. The 11 capstone and student inference files are now public in a separate [Chapter 19 download manifest](docs/course-experiments/capstone-public.json). They share the same Hugging Face repository as the original 30 experiment groups and 120 checkpoint files below; use the appropriate manifest for each batch.

The [smaller Dense students](course/chapters/19.md#19.10) have also completed the demonstration-only and distillation comparison. On the same 90 records, the demonstration-only student passes 62, the student trained with teacher distributions passes 61, and its 4-bit export still passes 61. Distillation did not outperform demonstration-only training in this fixed recipe. The [student report](docs/course-experiments/results/capstone_student.json) preserves individual results; these models do not establish general assistant capability.

You can begin without prior knowledge of this project. Linked warmups introduce Python, tensors, probability, matrices, gradients and notebook operation as they become necessary. Lessons explain the problem and a worked example before introducing terminology. Small CPU programs verify individual mechanisms. Separate formal experiments update weights, evaluate fixed held-out records and preserve code versions, data provenance and per-record results. Controlled synthetic tasks and short natural-data pilots do not establish general assistant capability. See the [experiment plan and progress](docs/course-experiments/README.md) and [validation report](docs/validation.md).

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

Activate `.venv` and use `python` directly, or pass the same extra to each `uv run`. Local lesson checks use CPU; separate formal training runs use NVIDIA L4 on Modal with PyTorch 2.14.1+cu126. Versions and results are recorded in the [experiment reports](docs/course-experiments/README.md). The macOS CI verified basic matrix forward/backward operations on Apple MPS and a CPU comparison. The full course and formal model training have not been validated on MPS; see the [CI evidence](docs/validation-artifacts/release-compatibility-ci.json).

Read the online course built with [Zensical](https://zensical.org/): full-text search, chapter navigation, page contents, light/dark modes and code copying. Every lesson retains its notebook download and Colab entry. SVGs can be enlarged, retain static labels and respect reduced-motion preferences. Website builds require the separate `--group site`. See [publishing instructions](docs/publishing.md) for local reading, verified outputs and GitHub Pages deployment. MathJax typesets formulas when online.

## Minimal workflow

```bash
python scripts/prepare_data.py --kind toy-text
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl
```

The default runs one forward/backward check with no optimizer update or checkpoint write. Add `--train` explicitly for training. [Training recipes](course/training.md) describe prerequisites, formats, evaluation and limitations for text, style, safety, multimodal, preferences and distillation.

[Fixed training snapshots](assets/training/README.md) are published as separate Git LFS archives. Use `python scripts/fetch_training_assets.py --list` to choose and unpack them. Raw caches, checkpoints and outputs stay in ignored `data/`, `checkpoints/` and `outputs/`. External datasets retain their own licenses; see [asset storage](docs/asset-storage.md).

With Modal and Hugging Face configured, manually run the **GPU training smoke test** GitHub Actions workflow to verify GPU parameter updates, checkpoint uploads and resuming from an HF download. See the [GPU operations guide](docs/gpu-training.md) for account settings, time limits and result checks.

To inspect trained models without retraining, use the [public student weights](https://huggingface.co/birdhackor/tiny-perceptron-course-models): 30 experiment groups with 120 checkpoint files, including different sizes and teacher/student comparisons. The [download manifest](docs/course-experiments/public-models.json) contains only published, pinned revisions; downloads verify each file hash. Model cards document scope, data licenses and measured results. From the installed CPU environment above:

```bash
python scripts/fetch_course_models.py --list
python scripts/check_course_models.py --model text_foundation
```

The second command anonymously downloads the foundation models, verifies files and runs CPU inference. It checks loading and execution; task accuracy comes from the separate held-out evaluation. Public inference exports omit optimizer and RNG state and cannot reproduce an exact resumed training trajectory. Full training backups are stored separately.

## Maintenance

Edit `course/chapters/`, format that file’s Python examples, then regenerate notebooks. Each section needs an independent reader review of its current text and SVGs, followed by a separate reviewer checking facts, original sources and measured evidence. See the [editorial protocol](docs/editorial-guide.md) and [technical review protocol](docs/technical-review-guide.md). Publication rejects outdated reviews. Components live in `tiny_perceptron/`; SVGs live in `course/figures/`, with generated examples in `scripts/build_visuals.py`.

```bash
python scripts/build_visuals.py
python scripts/format_course_code.py course/chapters/01.md
python scripts/build_course.py
python scripts/build_course.py --check
python scripts/check_course_reviews.py
python scripts/check_technical_reviews.py
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
