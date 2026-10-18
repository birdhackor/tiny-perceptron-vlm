# Tiny Perceptron 小小感知機

How does a computer choose the next character after a short phrase? This from-scratch course starts with character IDs and a next-character table, using worked examples, diagrams and short programs to explain text, image and audio models.

[繁體中文](README.md) | English

**[Online course](https://birdhackor.github.io/tiny-perceptron-vlm/) · [Reading routes](course/README.md) · [First steps](course/first-steps.md) · [312 lessons](course/lessons.md) · [Glossary](course/glossary.md) · [Training recipes](course/training.md)** — the main course is in Traditional Chinese.

This preview provides lesson text, SVG diagrams and the actual CPU output of every lesson's code, so you can read it like a book without installing anything; start with [1.1](course/chapters/01.md#1.1). To change the code and rerun it, download the project to your computer as described in Setup below and open the same lesson's notebook, or use the Open in Colab button in each lesson's footer on the website for the same notebook; in notebooks with code, the first code cell downloads the project and installs the packages in Colab. Small exercises need no GPU. Colab requires a Google account, and execution in a real Colab runtime has not been verified.

The course has 20 chapters, the A/B extensions, a required reasoning-compute unit C and 312 lesson notebooks. Including 26 numbered reading, warmup, training and glossary sections, there are 338 numbered sections. Begin with small matrices, bigrams, MLPs and manual attention; introduce multi-token prediction (MTP), modern architecture, Dense/MoE, longer context, cache/SDPA, quantization and distillation later. Each lesson addresses one question, and readers can revisit examples without training one model throughout.

[7.17](course/chapters/07.md#7.17) explains why pre-training and post-training commonly use different data and objectives. [Chapter 13](course/chapters/13.md) develops preference feedback, reward models, PPO and DPO step by step. Instruction checks distinguish content, requested scope, formatting and stopping; context lessons distinguish capacity, positions, computation and information use.

The main [Chapter 19 capstone](course/chapters/19.md#19.1) combines text, vision, audio and tool use into the MoE assistant. Its current v2 version comes in two architectures, MoE and a simpler Dense one; the language core, vision/audio encoders and connectors were trained from scratch in this project. Training and public inference exports are complete, and each architecture was evaluated on a fixed test set of 3,734 records. The MoE model has 5,447,107 parameters and the Dense model has 2,288,067; neither passed every original capability criterion. Tasks are limited to three clothing classes and two-slot positions, caller-provided contiguous regions of 1–4 characters from 12 known Chinese glyphs, three banking-service speech intents, finite text conversations and a calculator. Results and limitations are in [19.12](course/chapters/19.md#19.12); begin with the [public CPU instructions](docs/selftrained/v2-public-cpu-commands.md). A separate synthetic-world integration experiment keeps its weights, recipes and per-record reports in the [experiment archive](docs/course-experiments/README.md); those results describe other models.

[Chapter 20](course/chapters/20.md#20.1) is a separate application extension using Qwen3-VL for vision and language and Whisper for transcription. It has its own [student](docs/natural-assistant/v4/STUDENT.md), [data](docs/natural-assistant/v4/DATA.md) and [training](docs/natural-assistant/v4/TRAINING.md) guides. Upstream model capabilities are evaluated separately from the self-trained main course.

Python, tensor, probability, matrix and gradient [warmups](course/first-steps.md) are there when needed. Operational commands live on the [training page](course/training.md); [validation notes](docs/validation.md) record evidence and limitations.

Thinking effort, Looped Transformers, and Jev-style typed probabilistic judgments are now required topics: [7.23](course/chapters/07.md#7.23), [14.11](course/chapters/14.md#14.11), and [9.11](course/chapters/09.md#9.11). C.1–C.5 follows Chapter 7. [Chapter 19](course/chapters/19.md#19.13) has three products, the MoE assistant, a bounded text Looped solver and an independent probability judge, and users pick one by task. The probability judge can also be connected to the Looped solver: it first judges whether the task is supported and the information complete, then a rule picks how many loops the solver runs. This is Auto effort ([19.17](course/chapters/19.md#19.17)). The Looped solver and the probability judge completed NVIDIA L4 training and were evaluated on test data only after their settings were frozen: the Looped solver solved 2928/2956 two-box tracking records and 2/42 three-number additions; the probability judge answered every candidate judgment in the fixed test set correctly but made errors when the wording changed. See [product instructions](docs/lesson-products/README.md) for pinned downloads and commands, and [19.20](course/chapters/19.md#19.20) for denominators, shifts and limitations.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Git](https://git-scm.com/downloads) to obtain the project. Before cloning, run `git config --get filter.lfs.process`. If it prints `git-lfs filter-process`, your Git has Git LFS enabled for all repositories, and the `git clone` below also downloads every training data archive (currently about 328 MB in total). To get only the code, replace the first line with `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/tiny-perceptron-vlm.git`; the Windows PowerShell form is in [W.1](course/first-steps.md#W.1). If it prints nothing, run the commands below as they are. For CPU exercises:

```bash
git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
uv sync --frozen --extra cpu --group notebook
source .venv/bin/activate
python scripts/check_env.py
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
jupyter lab notebooks
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`; the commands without activation are in [W.1](course/first-steps.md#W.1). JupyterLab lists files starting from the `notebooks` folder; select **Tiny Perceptron** and open `01/1.1.ipynb` there. The commands above install the CPU build of PyTorch, which also works on Apple Silicon Macs. For another option, replace the `uv sync` line: use `uv sync --frozen --group notebook` on Apple Silicon, or the matching `cu126`/`cu130` extra on compatible NVIDIA hardware. Driver requirements are in [environment notes](docs/environment.md). Reading the course and running lesson code do not need Git LFS; it is needed only to download the training data archives. Installation and the `git lfs version` check are in [W.1](course/first-steps.md#W.1).

Activate `.venv` and use `python` directly, or pass the same extra to each `uv run`. Local lesson checks use CPU; formal GPU training for chapters 1–19 uses NVIDIA L4 on Modal with PyTorch 2.14.1+cu126. Versions and results are recorded in the [experiment records](docs/course-experiments/README.md). Chapter 20 uses a separate Python 3.12 and PyTorch 2.8.0 environment, with CUDA 12.8 packages for GPU training; see the [student instructions](docs/natural-assistant/v4/STUDENT.md). The macOS CI verified basic matrix forward/backward operations on Apple MPS and a CPU comparison. The full course and formal model training have not been validated on MPS; see the [CI evidence](docs/validation-artifacts/release-compatibility-ci.json).

Read the online course built with [Zensical](https://zensical.org/): full-text search, chapter navigation, page contents, light/dark modes and code copying. Each lesson footer offers a notebook download and an Open in Colab button. SVGs can be enlarged, retain static labels and respect reduced-motion preferences. Website builds require the separate `--group site`. See [publishing instructions](docs/publishing.md) for local reading, verified outputs and GitHub Pages deployment. MathJax typesets formulas when online.

## Minimal workflow

First check the computation path: prepare a small text set, predict a batch of targets, calculate the loss, then use backpropagation to calculate how the model parameters could change. This check leaves the parameters unchanged and saves no weight file.

```bash
python scripts/prepare_data.py --kind toy-text
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl
```

Expect `mode` to be `dry-run-no-weight-update`, a finite `loss` and finite, positive `grad_norm` in the first `history` entry. This checks the computation path; it does not establish learned answers or held-out ability. Add `--train` explicitly for training. [Training recipes](course/training.md) describe prerequisites, formats, evaluation and limitations for text, style, safety, multimodal, preferences and distillation.

[Fixed training snapshots](assets/training/README.md) are published as separate Git LFS archives, so install Git LFS first ([W.1](course/first-steps.md#W.1)). List available snapshots with `python scripts/fetch_training_assets.py --list` (this step does not need Git LFS), then download, verify and unpack the chosen archive with `python scripts/fetch_training_assets.py --asset NAME`, replacing `NAME` with a listed asset name. Raw caches, checkpoints and outputs stay in ignored `data/`, `checkpoints/` and `outputs/`. External datasets retain their own licenses; see [asset storage](docs/asset-storage.md).

With Modal and Hugging Face configured, manually run the **GPU training smoke test** GitHub Actions workflow to verify GPU parameter updates, checkpoint uploads and resuming from an HF download. See the [GPU operations guide](docs/gpu-training.md) for account settings, time limits and result checks.

To try the Chapter 19 v2 model, follow the [public CPU instructions](docs/selftrained/v2-public-cpu-commands.md), install the `cpu` and `selftrained` extras and anonymously download the [pinned HF revision](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2). It contains four inference exports—MoE pretrain, MoE SFT, MoE joint and Dense joint—with 16 paired files under the MIT license. The MoE joint weights were selected at step 1,000 of a completed 4,000-step joint run; Dense joint weights were selected at step 1,000 of a completed 10,000-step run. Selection used validation results; successful examples do not mean every test criterion passed.

To reproduce v2 from random initialization, use the [local training guide](docs/selftrained/TRAINING.md) and [fixed v2 data manifest](docs/selftrained/v2-manifest.json). The Git LFS archive is 67,862,431 bytes with 8,962 files; train/validation/test contain 28,876/2,435/3,734 records. Each stage continues from your own full checkpoint, without the author's private Volume. Public safetensors inference weights omit optimizer and other training state and cannot reproduce an exact resumed trajectory.

For local chapter comparisons, use the [public student weights](https://huggingface.co/birdhackor/tiny-perceptron-course-models): the 30 experiment groups contain 120 weight files, including different sizes and teacher/student comparisons. The [download manifest](docs/course-experiments/public-models.json) preserves pinned revisions and file hashes; the [synthetic-world capstone manifest](docs/course-experiments/capstone-public.json) retains another 11 stage, quantization and Dense-student files. Model cards document scope, data licenses and measured results. From the installed CPU environment above:

```bash
python scripts/fetch_course_models.py --list
python scripts/check_course_models.py --model text_foundation
```

The second command anonymously downloads the `text_foundation` text model, verifies files and runs CPU inference. It checks loading and execution; task accuracy comes from the experiment's validation and test evaluation. Public inference exports omit optimizer and RNG state and cannot reproduce an exact resumed training trajectory. Full training backups are stored separately.

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
