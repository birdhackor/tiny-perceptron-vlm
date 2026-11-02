# Tiny Perceptron 小小感知機

How does a computer choose the next character after a short phrase? This from-scratch course starts with character IDs and a next-character table, using worked examples, diagrams and short programs to explain text, image and audio models.

[繁體中文](README.md) | English

**[Online course](https://birdhackor.github.io/tiny-perceptron-vlm/) · [Reading routes](course/README.md) · [First steps](course/first-steps.md) · [All lessons](course/lessons.md) · [Glossary](course/glossary.md) · [Training recipes](course/training.md)** — the main course is in Traditional Chinese.

Essential code, key outputs and explanations are included directly in the lesson text. The website also shows CPU outputs, so you can read the course without installation; start with [1.1](course/chapters/01.md#1.1). To change numbers and rerun an example, open the same lesson notebook locally or use its Colab entry. Colab is an optional way to rerun experiments, requires a Google account, and has not been verified in an actual Colab runtime. Small examples need no GPU.

The course has 20 chapters, two extension units (A and B), a required reasoning-compute unit (C), and 332 lesson sections with notebooks generated from the same sources. Including 26 numbered reading, warmup, training and glossary sections, there are 358 numbered sections. Each lesson addresses one question.

A table that uses only the previous character assigns the same next-character probabilities when that character matches, even if earlier text calls for a different continuation. The course therefore moves to models that read context and learn from examples, then uses dialogue examples to practice answering requests and stopping. When an answer will choose a service or trigger a next step, the system also needs checks for supported tasks and missing information; evidence in pictures and sound adds content and timing to preserve. Longer material, repeated questions and audio that should start playing promptly bring repeated work, waiting time and memory limits. These needs lead to shared computation, chunked output, and ways to store and run models with fewer resources, before assembling products whose behavior can be checked task by task.

[7.17](course/chapters/07.md#7.17) explains why pre-training and post-training commonly use different data and objectives. [Chapter 13](course/chapters/13.md) develops preference feedback, reward models, PPO and DPO step by step. Instruction checks distinguish content, requested scope, formatting and stopping; context lessons distinguish capacity, positions, computation and information use.

The main [Chapter 19 capstone](course/chapters/19.md#19.1) combines text, vision, audio and tool use into the MoE assistant (product A). Its published v2 baseline comes in two architectures, MoE and a simpler Dense one; the language core, vision/audio encoders and connectors were trained from scratch in this project. Training and public inference exports are complete, and each architecture was evaluated on a fixed test set of 3,734 records. The MoE model has 5,447,107 parameters and the Dense model has 2,288,067; neither passed every original capability criterion. Tasks are limited to three clothing classes and two-slot positions, caller-provided contiguous regions of 1–4 characters from 12 known Chinese glyphs, three banking-service speech intents, finite text conversations and a calculator. Results and limitations are in [19.12](course/chapters/19.md#19.12); begin with the [public CPU instructions](docs/selftrained/v2-public-cpu-commands.md). A separate synthetic-world integration experiment keeps its weights, recipes and per-record reports in the [experiment archive](docs/course-experiments/README.md); those results describe other models.

This revision adds 19 lessons throughout the main course: [7.26](course/chapters/07.md#7.26) and [9.17–9.18](course/chapters/09.md#9.17) develop candidate data and reliable decisions; [10.12](course/chapters/10.md#10.12) and [12.17–12.18](course/chapters/12.md#12.17) introduce frame timing, audiovisual evidence and reconstructable sound codes; [14.19](course/chapters/14.md#14.19), [16.15–16.16](course/chapters/16.md#16.15) and [17.16–17.17](course/chapters/17.md#17.16) add time positions, shared computation, chunked output and codebook learning. [19.21–19.24](course/chapters/19.md#19.21) connect these methods to an audiovisual extension of the MoE assistant (product A), finite-tone generation, the next probability judge (product C-v2), and Auto integration. These product letters refer to Chapter 19 models; the A/B extensions and required C refer to teaching units. All ten new-model jobs completed real training and fixed final measurements on standard Linux CPUs. The audiovisual task asks which shape is on screen when the tone sounds: each scenario forms two items by shifting only the sound, and both must be correct for a successful pair. The preselected model passed 200/200 pairs. Finite sound generation passed 200/200 programs under checks of frequency, order, duration and pauses; a separate waveform-content swap still exposed audiovisual failures. That batch’s C-v2 selects among candidate actions (Choice), answering 39/60 correctly. Its frozen acceptance rule accepted 53 of those cases, including 19 wrong answers, so that version remains a research candidate; the historical default was retained at that stage. See the [new-model instructions](docs/omni-products/README.md) for results, limits and weight downloads. Historical results remain attached to their original models.

The recommended device-decision product is now [C-v3 in 19.25](course/chapters/19.md#19.25), with [pinned CPU downloads and commands](docs/decision-product/README.md). It accepts typed JSON for 3–6 devices. On 1,500 unseen 4–6-device fact families, action choice, information sufficiency and the fixed rubric each score 1,500/1,500. Its frozen policy accepts all 1,025 answerable actions with zero errors and correctly refuses the remaining 475. The three question types share families; three-device support has separate manual checks and historical semantic regression. The Looped solver B keeps its existing automatic loop budget (Auto effort), and C-v2’s neural-routing measurements retain their original scope. The mechanisms and essential code are included directly in the textbook.

[Chapter 20](course/chapters/20.md#20.1) retains the Qwen3-VL/Whisper application and adds [20.14–20.16](course/chapters/20.md#20.14), which compare Kev, an open-source system that returns probabilities for predefined answer candidates, and the two Omni generations using public implementations, reports and explicit evaluation methods. The existing application has its own [student](docs/natural-assistant/v4/STUDENT.md), [data](docs/natural-assistant/v4/DATA.md) and [training](docs/natural-assistant/v4/TRAINING.md) guides. Upstream capabilities are evaluated separately from the self-trained course.

Python, tensor, probability, matrix and gradient [warmups](course/first-steps.md) are there when needed. Operational commands live on the [training page](course/training.md); [validation notes](docs/validation.md) record evidence and limitations.

Thinking effort, Looped Transformers, and Jev-style typed probabilistic judgments are now required topics: [7.23](course/chapters/07.md#7.23), [14.11](course/chapters/14.md#14.11), and [9.11](course/chapters/09.md#9.11). C.1–C.5 follows Chapter 7. [Chapter 19](course/chapters/19.md#19.13) has three products: A, the MoE assistant; B, a bounded text Looped solver; and C, an independent probability judge. Users pick one by task. The probability judge can also be connected to the Looped solver: it first judges whether the task is supported and the information complete, then a rule picks how many loops the solver runs. This is Auto effort ([19.17](course/chapters/19.md#19.17)). The existing Looped solver and first-version probability judge completed NVIDIA L4 training and were evaluated on test data only after their settings were frozen: the Looped solver solved 2928/2956 two-box tracking records and 2/42 three-number additions; the probability judge answered every candidate judgment in the fixed test set correctly but made errors when the wording changed. See [product instructions](docs/lesson-products/README.md) for pinned downloads and commands, and [19.20](course/chapters/19.md#19.20) for denominators, shifts and limitations.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Git](https://git-scm.com/downloads) to obtain the project. The training data archives are stored with Git LFS (a tool for storing large files with Git), and the `GIT_LFS_SKIP_SMUDGE=1` at the start of the first line below keeps the clone from downloading them. You can type it even without Git LFS installed. In Windows PowerShell, leave that prefix out: run `git config --global core.longpaths true` first, then `$env:GIT_LFS_SKIP_SMUDGE="1"` and `git clone --depth 1 --no-tags https://github.com/birdhackor/tiny-perceptron-vlm.git`. Some files in the project have deep paths, and Git for Windows cannot create paths longer than 260 characters by default; the line that sets `core.longpaths` lets Git on this computer use long paths, including in later clones and when switching versions. For CPU exercises, run these in order. Pause at `check_env.py` to inspect the result described below before registering the kernel and starting Jupyter. This script lists Python and package versions, selects an available device, and performs one small matrix forward/backward calculation on it:

`--depth 1 --no-tags` downloads the current version only. See [historical versions](docs/repository-archives.md) when older history is needed.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --no-tags https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
uv sync --frozen --extra cpu --group notebook
source .venv/bin/activate
python scripts/check_env.py
python -m ipykernel install --sys-prefix --name tiny-perceptron --display-name "Tiny Perceptron"
jupyter lab notebooks
```

The environment check should finish without an error, list versions for the packages with no `未安裝` ("not installed") entries, and show `OK` for the selected device under `compute`. This is an excerpt from one actual CPU run; versions and thread counts depend on the environment:

```text
torch           2.14.1+cpu
device          CPU（5 執行緒）
compute         OK（cpu）
```

This confirms basic matrix computation on CPU in that run; it does not validate the full course, other backends or model capabilities.

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`; the commands without activation are in [W.1](course/first-steps.md#W.1). JupyterLab lists files starting from the `notebooks` folder; select **Tiny Perceptron** and open `01/1.1.ipynb` there. The commands above install the CPU build of PyTorch, which also works on Apple Silicon Macs. For another option, replace the `uv sync` line: use `uv sync --frozen --group notebook` on Apple Silicon, or the matching `cu126`/`cu130` extra on compatible NVIDIA hardware. Driver requirements are in [environment notes](docs/environment.md). Reading the course and running lesson code do not need Git LFS; it is needed only to download the training data archives. Installation and the `git lfs version` check are in [W.1](course/first-steps.md#W.1).

Activate `.venv` and use `python` directly, or pass the same extra to each `uv run` so uv keeps selecting the same PyTorch package build. Local lesson checks use CPU; the existing fixed training recipes for chapters 1–19 were tested on NVIDIA L4 on Modal with PyTorch 2.14.1+cu126. Versions and results are recorded in the [experiment records](docs/course-experiments/README.md). Chapter 20 uses a separate Python 3.12 and PyTorch 2.8.0 environment, with CUDA 12.8 packages for GPU training; see the [student instructions](docs/natural-assistant/v4/STUDENT.md). The macOS CI verified basic matrix forward/backward operations on Apple MPS and a CPU comparison. The full course and formal model training have not been validated on MPS; see the [CI evidence](https://github.com/birdhackor/tiny-perceptron-vlm/blob/83630ea1cea23d47c4b77162a31cbb7566686d11/docs/validation-artifacts/release-compatibility-ci.json).

Read the online course built with [Zensical](https://zensical.org/): full-text search, chapter navigation, page contents, light/dark modes and code copying. Each lesson footer offers a notebook download and an Open in Colab button. SVGs can be enlarged, retain static labels and respect reduced-motion preferences. Website builds require the separate `--group site`. See [publishing instructions](docs/publishing.md) for local reading, verified outputs and GitHub Pages deployment. Formulas are typeset in the browser by the MathJax copy shipped with the site.

## Minimal workflow

First check the computation path: prepare a small text set, predict probabilities for what comes next, and compare them with the known answers in the data. The `loss` summarizes the mismatch as a penalty: assigning a correct answer a lower probability gives a larger penalty. Lower loss means a better match on this same batch. Backpropagation then calculates gradients: how small changes to each adjustable number (parameter) would affect the loss. `grad_norm` is the combined size of those gradients, showing whether the backward calculation produced a finite, nonzero numerical signal. This check calculates that signal, leaves the parameters unchanged and saves no weight file.

```bash
python scripts/prepare_data.py --kind toy-text
python scripts/train.py --task text --data data/generated/toy-text/train.jsonl
```

These are the key fields from one actual CPU report; other fields are omitted:

```json
{
  "mode": "dry-run-no-weight-update",
  "device": "cpu",
  "history": [
    {
      "step": 1,
      "loss": 5.770333290100098,
      "grad_norm": 0.9595053195953369
    }
  ]
}
```

Look for `mode` to be `dry-run-no-weight-update`. In the first `history` entry, `loss` and `grad_norm` should both be finite (neither `NaN` nor infinity), and `grad_norm` should be positive. Here about 5.77 comes from the prediction/answer comparison, and about 0.96 means that backpropagation produced a nonzero gradient; gradient size is not accuracy. Last digits may vary with software versions, so check the meanings and conditions rather than matching the exact numbers. This run checks only the CPU computation path. It updates no parameters and establishes neither learned answers, ability on new questions, nor capabilities of the newly added models.

Add `--train` explicitly for training. [Training recipes](course/training.md) describe prerequisites, formats, evaluation and limitations for text, style, safety, multimodal, preferences and distillation.

[Fixed training snapshots](assets/training/README.md) are published as separate Git LFS archives, so install Git LFS first ([W.1](course/first-steps.md#W.1)). List available snapshots with `python scripts/fetch_training_assets.py --list` (this step does not need Git LFS), then download, verify and unpack the chosen archive with `python scripts/fetch_training_assets.py --asset NAME`, replacing `NAME` with a listed asset name. Raw caches, checkpoints and outputs stay in ignored `data/`, `checkpoints/` and `outputs/`. External datasets retain their own licenses; see [asset storage](docs/asset-storage.md).

With Modal and Hugging Face configured, manually run the **GPU training smoke test** GitHub Actions workflow to verify GPU parameter updates, checkpoint uploads and resuming from an HF download. See the [GPU operations guide](docs/gpu-training.md) for account settings, time limits and result checks.

To try the Chapter 19 v2 model, follow the [public CPU instructions](docs/selftrained/v2-public-cpu-commands.md), install the `cpu` and `selftrained` extras and anonymously download the [pinned HF revision](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2). It contains four inference exports—MoE pretrain, MoE SFT, MoE joint and Dense joint—with 16 paired files under the MIT license. The MoE joint weights were selected at step 1,000 of a completed 4,000-step joint run; Dense joint weights were selected at step 1,000 of a completed 10,000-step run. Selection used validation results; successful examples do not mean every test criterion passed.

To reproduce v2 from random initialization, use the [local training guide](docs/selftrained/TRAINING.md) and [fixed v2 data manifest](docs/selftrained/v2-manifest.json). The Git LFS archive is 67,862,431 bytes with 8,962 files; train/validation/test contain 28,876/2,435/3,734 records. Each stage continues from your own full checkpoint, without the author's private Volume. Public safetensors inference weights omit optimizer and other training state and cannot reproduce an exact resumed trajectory.

For local chapter comparisons, use the [public student weights](https://huggingface.co/birdhackor/tiny-perceptron-course-models): the 30 experiment groups contain 120 weight files, including different sizes and teacher/student comparisons. The [download manifest](docs/course-experiments/public-models.json) preserves pinned revisions and file hashes; the [synthetic-world capstone manifest](docs/course-experiments/capstone-public.json) retains another 11 stage, quantization and Dense-student files. Model cards document scope, data licenses and measured results. From the installed CPU environment above:

```bash
python scripts/fetch_course_models.py --list
python scripts/check_course_models.py --model text_foundation
```

The first command lists models without downloading. The second anonymously downloads `text_foundation`, verifies files and runs CPU inference without another input file: it probes each checkpoint with a fixed start token, then generates from the public manifest's fixed text prompt `color=blue;shape=circle;`. This run produced `side=right.` for that prompt. The input and answer are retained in the check report; one answer is not an accuracy estimate.

The terminal's last line includes the following fields; the omitted `evidence` path varies with the working directory and timestamp:

```json
{"model": "text_foundation", "status": "passed"}
```

`status: passed` and a normal exit confirm file verification, loading and CPU execution. For `failed`, inspect the error message and the `evidence` report. A normal exit has exit code 0; immediately after the command, use `echo $?` on macOS/Linux or `$LASTEXITCODE` in PowerShell. Task accuracy still comes from validation and test evaluation. Public inference exports omit optimizer and RNG state and cannot reproduce an exact resumed training trajectory. Full training backups are stored separately.

## Maintenance

`pytest` runs tests; `ruff check` checks code rules, and `ruff format --check` checks formatting.

Edit `course/chapters/`, format that file’s Python examples, then regenerate notebooks. Each changed section receives three independent stages in order: readability and understanding, correctness and evidence, then continuity with the surrounding lessons. See the [editorial protocol](docs/editorial-guide.md) and [technical review protocol](docs/technical-review-guide.md). Each PR records concise conclusions for the affected material; deployment does not load historical review records. Components live in `tiny_perceptron/`; SVGs live in `course/figures/`, with generated examples in `scripts/build_visuals.py`.

```bash
python scripts/build_visuals.py
python scripts/format_course_code.py course/chapters/01.md
python scripts/build_course.py
python scripts/build_course.py --check
python scripts/check_notebooks.py --mode python
python scripts/check_notebooks.py --mode kernel --lesson 3.6
pytest -ra
ruff check .
ruff format --check .
```

Python mode runs each lesson in a fresh Python namespace for a quick check. Kernel mode starts a fresh kernel per lesson and saves executed copies under `outputs/`. Existing GitHub Actions run core checks on Linux, macOS and Windows; complete notebook validation is a separate workflow.

## References and license

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat), [MiniMind](https://github.com/jingyaogong/minimind), [MiniMind-V](https://github.com/jingyaogong/minimind-v) and [nanoVLM](https://github.com/huggingface/nanoVLM). Research sources and limitations are in [curriculum notes](docs/curriculum.md).

Code, course text and original diagrams: [MIT](LICENSE). External data has separate licensing.
