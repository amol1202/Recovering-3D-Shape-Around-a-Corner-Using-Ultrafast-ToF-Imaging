
# 🚀 Filtered Back Projection — Reference Implementation

> Clean, well-tested, and user-friendly implementation of Filtered Back Projection (FBP) for tomography.
> Provides a Python API, CLI, notebooks, and reproducible CI for research and teaching.

[![Original Paper](https://img.shields.io/badge/Original_Paper-link-blue)](https://www.nature.com/articles/ncomms1747) [![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/OWNER/REPO/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

---

## 📑 Table of Contents

- [About](#about)  
- [Features](#features)  
- [Installation](#installation)  
- [Quickstart](#quickstart)  
- [Usage](#usage)  
  - [Python API](#python-api)  
  - [Command Line](#command-line)  
- [Examples & Notebooks](#examples--notebooks)  
- [Screenshots](#screenshots)  
- [Project Structure](#project-structure)  
- [Testing & CI](#testing--ci)  
- [Contributing](#contributing)  
- [Code of Conduct](#code-of-conduct)  
- [License](#license)  
- [Citation](#citation)  
- [Acknowledgements](#acknowledgements)  
- [Contact](#contact)

---

## 🧠 About

This repository contains a clean, reference implementation of Filtered Back Projection (FBP).  
It includes a straightforward Python API, a command-line interface, example datasets, Jupyter notebooks for learning and visualization, and CI-driven tests to ensure reproducibility.

---

## ✨ Features

- Reference-quality implementation of FBP  
- Multiple filter choices: Ram-Lak, Shepp–Logan, Hamming, Hann  
- CPU implementation (NumPy/SciPy) and optional GPU backend (PyTorch/CuPy)  
- CLI + Python API + Jupyter notebooks  
- Example datasets and automated unit tests  
- Preconfigured CI and code formatting / linting (black, isort, ruff/flake8)

---

## ⚙️ Installation

**Install from PyPI (if published)**

```bash
pip install fbp-repo-name
```

**Install directly from GitHub**

```bash
pip install git+https://github.com/OWNER/REPO.git
```

**From source (developer)**

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
python -m venv .venv
# Activate the venv:
# macOS / Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

**Docker (reproducible environment)**

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -e ".[cpu]"
CMD ["python", "-m", "fbp.cli", "--help"]
```

---

## ⚡ Quickstart

### CLI

```bash
fbp reconstruct --input data/example_sinogram.npy --output results/recon.png \
  --filter shepp-logan --angles 0:179 --size 512
```

### Python

```python
import numpy as np
from fbp import filtered_back_projection

sinogram = np.load("data/example_sinogram.npy")  # shape (n_angles, n_detectors)
angles = np.linspace(0, 180, sinogram.shape[0], endpoint=False)

recon = filtered_back_projection(sinogram,
                                 angles=angles,
                                 filter_name="shepp-logan",
                                 output_size=512)

# Save result
from imageio import imwrite
imwrite("results/recon.png", (recon * 255).astype("uint8"))
```

---

## 🧩 Usage Details

### Python API (example)

```python
from fbp import filtered_back_projection

reconstruction = filtered_back_projection(
    sinogram,                   # ndarray (n_angles, n_detectors)
    angles=angles,              # 1D array of projection angles in degrees
    filter_name="ram-lak",      # "ram-lak" | "shepp-logan" | "hann" | "hamming"
    output_size=512,            # output image size in pixels
    backend="numpy",            # "numpy" | "pytorch" | "cupy"
    clip=True                   # clip values to [0,1]
)
```

### CLI

```
Usage: fbp reconstruct [OPTIONS]

Options:
  --input PATH            Path to sinogram (.npy, .mat, .csv)
  --output PATH           Output image (.png) or .npy
  --filter TEXT           Filter to use (ram-lak|shepp-logan|hamming|hann)
  --angles TEXT           Angles spec (e.g., '0:179' or path to angles file)
  --size INTEGER          Output image size (pixels)
  --backend TEXT          backend (numpy|pytorch|cupy)
  --normalize             Normalize sinogram before reconstruction
  --help                  Show this message and exit
```

---

## 📚 Examples & Notebooks

- `notebooks/01_basic_reconstruction.ipynb` — Simple reconstruction tutorial  
- `notebooks/02_filter_comparison.ipynb` — Visualize filter effects and artifacts  
- `notebooks/03_gpu_acceleration.ipynb` — GPU acceleration with PyTorch/CuPy  

Run examples with:

```bash
bash examples/run_example.sh
```

---

## 🖼️ Screenshots

Add `assets/sinogram.png` and `assets/reconstruction.png` to show results. Example layout:

```
assets/
  ├─ sinogram.png
  └─ reconstruction.png
```

---

## 🗂️ Project Structure

```
REPO/
├─ fbp/                      # package code
│  ├─ __init__.py
│  ├─ core.py                # main algorithms
│  ├─ filters.py             # filter implementations
│  └─ cli.py                 # CLI entrypoint
├─ notebooks/                # Jupyter tutorials
├─ data/                     # example sinograms and phantoms
├─ tests/                    # unit and integration tests
├─ examples/                 # runnable scripts / demos
├─ assets/                   # images for README and docs
├─ .github/workflows/ci.yml  # CI configuration
├─ pyproject.toml
├─ README.md
└─ LICENSE
```

---

## ✅ Testing & CI

Run tests:

```bash
pytest -q
```

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Minimal GitHub Actions CI (`.github/workflows/ci.yml`):

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test]"
      - name: Run tests
        run: pytest -q
      - name: Lint
        run: |
          pip install ruff
          ruff check fbp
```

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repo  
2. Create a feature branch: `git checkout -b feat/your-change`  
3. Implement and add tests  
4. Run formatters & linters: `black . && isort . && ruff check .`  
5. Open a pull request with a clear description of your change

Please read `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` if available.

---

## 🧭 Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/). Please be respectful and constructive.

---

## 📜 License

Released under the MIT License. See `LICENSE` for details.

---

## 📖 Citation

If you use this implementation, please cite:

```
@misc{REPO2025,
  title = {Filtered Back Projection — reference implementation},
  author = {Your Name and Contributors},
  year = {2025},
  howpublished = {\url{https://github.com/OWNER/REPO}},
}
```

Also cite the original paper linked in the badges.

---

## 🙏 Acknowledgements

Thanks to the original algorithm authors, contributors, and the open-source libraries used (NumPy, SciPy, Matplotlib, PyTorch).

---

## 📬 Contact

Maintainer — **Your Name**  
Email — youremail@example.com  
GitHub — https://github.com/OWNER/REPO