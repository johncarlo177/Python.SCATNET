import os, sys, subprocess

here = os.path.dirname(__file__)
fig_dir = os.path.join(here, "..", "figures")

def run_py(script, *args):
    path = os.path.join(here, script)
    cmd = [sys.executable, path] + list(args)
    print("\n>>> Running:", " ".join(cmd))
    subprocess.check_call(cmd)

run_py("train_eval.py", "--dataset", "indian_pines", "--epochs", "3", "--fpca_k", "64", "--use_mixup", "--use_cb_focal")

for fig in ["Fig_1.py", "Fig_2.py", "Fig_3.py", "Fig_4.py", "Fig_5.py", "Fig_A1.py", "Fig_A2.py"]:
    fig_path = os.path.join(fig_dir, fig)
    print("\n>>> Generating:", fig)
    subprocess.check_call([sys.executable, fig_path])

print("\n All figures generated. Check outputs/.")
