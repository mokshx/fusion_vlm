# Step 0 on Kaggle: transformers (bf16/fp16) reference run for LFM2.5-VL-450M and -3B.
#
# Setup (once):
#   1. Locally: `bash kaggle/make_bundle.sh` -> dist/pdvlm_step0_bundle.zip
#   2. Kaggle -> Datasets -> New -> upload the zip, visibility PRIVATE
#      (UR Fall frames are CC BY-NC-SA; do not make them public).
#   3. New notebook: Accelerator = GPU (T4 x1 or P100), Internet = ON,
#      Add Input -> your private dataset.
#   4. Paste this file into one cell (or upload it) and run.
#
# Output: /kaggle/working/results/step0/kaggle_<gpu>.csv + run-*.json -> download them
# into results/step0/ locally and run scripts/summarize.py.

import glob
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.0", "accelerate", "pandas", "psutil", "requests"], check=True)

bundle = glob.glob("/kaggle/input/**/pdvlm_step0_bundle", recursive=True) or glob.glob("/kaggle/input/*")
root = bundle[0]
print("bundle:", root)
os.chdir("/kaggle/working")
subprocess.run(["cp", "-r", f"{root}/.", "."], check=True)

import torch  # noqa: E402

gpu = torch.cuda.get_device_name().replace(" ", "_") if torch.cuda.is_available() else "cpu"
env = {**os.environ, "PYTHONPATH": "src"}
for labels, tag in [("labels.csv", "main"), ("labels_floor_deliberate.csv", "floor")]:
    subprocess.run([
        sys.executable, "scripts/run_eval.py",
        "--labels", f"data/benchmark/step0/{labels}",
        "--images-root", "data/benchmark/step0",
        "--models", "lfm-450m:hf", "lfm-3b:hf",
        "--prompts", "A", "B", "B2",
        "--out", f"results/step0/kaggle_{gpu}_{tag}.csv",
    ], check=True, env=env)
subprocess.run([sys.executable, "scripts/summarize.py", f"results/step0/kaggle_{gpu}_main.csv"], check=True, env=env)
