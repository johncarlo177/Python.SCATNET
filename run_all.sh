#!/bin/bash
echo "==============================="
echo " Running SCATNet Codebase"
echo "==============================="
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all_figures.py
python scripts/run_toy_training.py
echo "All tasks completed. Check the 'outputs' folder."
