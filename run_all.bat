@echo off
echo ===============================
echo  Running SCATNet Codebase
echo ===============================
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python scripts\run_all_figures.py
python scripts\run_toy_training.py
echo.
echo All tasks completed. Outputs are saved in the 'outputs' folder.
pause
