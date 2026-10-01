# Tiny Perceptron 小小感知機

> A tiny multimodal model that can see, hear and read, built from scratch with as little code as possible.

[繁體中文](README.md) | English

> [!NOTE]
> 🚧 Early days: only the **environment setup** is done. No model code yet.

## What is this?

Tiny Perceptron is a light-hearted tutorial project. We implement a small multimodal model from scratch with minimal code, and explain every step along the way.

- **Stage 1**: text + image + audio → text (multimodal in, text out)
- **Later**: video + audio → text / image, and other richer multimodal tasks

It follows the spirit of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) / [nanochat](https://github.com/karpathy/nanochat) and [MiniMind](https://github.com/jingyaogong/minimind) / [MiniMind-V](https://github.com/jingyaogong/minimind-v): small, trainable from scratch, built for learning.

**Who is it for**: people with basic Python / PyTorch experience who want to understand how a multimodal model is built from the ground up.
The primary docs are in Traditional Chinese; this page is the English summary.

## Roadmap

- [x] Environment setup: uv, PyTorch, environment check, CI on three platforms
- [ ] Stage 1: text + image + audio → text
- [ ] Later: video + audio → text / image

## Requirements

| Hardware | Good for |
| --- | --- |
| Any CPU (a laptop is fine) | Running the whole pipeline, toy-scale experiments |
| Apple Silicon (M-series, macOS 14+) | Same, accelerated with MPS |
| NVIDIA GPU | Real training; the goal is a single consumer GPU |
| Google Colab | When you don't have a GPU |

You only need [uv](https://docs.astral.sh/uv/) and Git. uv downloads Python 3.13 for you and installs everything into the project's `.venv`.

The reasoning behind each choice (PyTorch / CUDA builds, why not torchaudio, candidate datasets, ...) is in [docs/environment.md](docs/environment.md) (Chinese).

## Installation

**1. Install uv**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Clone and install**

```bash
git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
uv sync
```

`uv sync` installs PyTorch from PyPI, which behaves differently on each platform. Pick the command for your machine:

| Your machine | Command | Notes |
| --- | --- | --- |
| macOS (Apple Silicon) | `uv sync` | The default build supports MPS |
| Linux + NVIDIA, driver ≥ 580 | `uv sync` | The default build is CUDA 13.0 |
| Windows + NVIDIA, driver ≥ 580 | `uv sync --extra cu130` | The default PyPI build is CPU-only on Windows |
| NVIDIA driver < 580, or older GPUs (GTX 9/10 series, V100) | `uv sync --extra cu126` | No RTX 50 series support |
| Linux without an NVIDIA GPU | `uv sync --extra cpu` | Skips ~2.5 GB of CUDA packages |
| Windows without an NVIDIA GPU | `uv sync` | The default build is already CPU-only |

Check your driver version with `nvidia-smi`.

> [!WARNING]
> If you installed with `--extra`, pass the same `--extra` to every `uv run` (e.g. `uv run --extra cu130 python ...`). Otherwise uv will helpfully swap PyTorch back to the default build.
> Alternatively, activate the virtual environment (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows) and use `python` directly.

**3. Check the environment**

```bash
uv run python scripts/check_env.py   # versions, device (CUDA / MPS / CPU), and a real computation on it
uv run pytest                        # offline smoke tests
```

`check_env.py` prints hints for common problems, such as "an NVIDIA driver is present but the CPU-only PyTorch is installed". Please include its output when reporting issues.

### Google Colab

Colab already ships PyTorch, so install the rest with pip:

```python
!git clone https://github.com/birdhackor/tiny-perceptron-vlm.git
%cd tiny-perceptron-vlm
!pip install -e .
!python scripts/check_env.py
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

GitHub Actions runs the environment check and tests on Linux, macOS and Windows.

## Acknowledgements

[nanoGPT](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat), [MiniMind](https://github.com/jingyaogong/minimind), [MiniMind-V](https://github.com/jingyaogong/minimind-v), [nanoVLM](https://github.com/huggingface/nanoVLM)

## License

[MIT](LICENSE)
