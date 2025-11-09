# SCATNet Codebase (Runnable)
This is a **self-contained, runnable codebase** with:
- A minimal SCATNet-like model (toy) and a training script
- Reproducible figure scripts for Figures 1–5 and A1–A2
- One-command runners

## Quick Start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate All Figures (saved to `outputs/`)
```bash
python scripts/run_all_figures.py
```

### Run Toy Training (saves `outputs/scatnet_toy.pth`)
```bash
python scripts/run_toy_training.py
```

> Replace the toy dataset with your real loaders when ready (`src/data/dataset_loader.py`).

## Structure
- `figures/` — individual scripts for each figure.
- `scripts/run_all_figures.py` — runs every figure in order.
- `scripts/run_toy_training.py` — quick training sanity-check.
- `src/models/` — minimal model.
- `src/data/` — toy dataset.
- `outputs/` — where all results/images are written.
