import torch
import numpy as np
import yaml
import json
from pathlib import Path
import importlib.util

# -------------------------------------------------------------
# PATHS — EDIT TO MATCH YOUR RUN
# -------------------------------------------------------------

RUN_DIR = Path(r"checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_att_sudoku_l40s_first")
CKPT = RUN_DIR / "step_10416"
CONFIG = RUN_DIR / "all_config.yaml"

DATA_ROOT = Path("data/sudoku-extreme-1k-aug-1000/test")
INPUTS = DATA_ROOT / "all__inputs.npy"
LABELS = DATA_ROOT / "all__labels.npy"
PUZZLE_IDS = DATA_ROOT / "all__puzzle_identifiers.npy"
IDENTIFIERS_JSON = Path("data/sudoku-extreme-1k-aug-1000/identifiers.json")

INDEX = 0

# -------------------------------------------------------------
# Load dynamic dataset-required values
# -------------------------------------------------------------
inputs_np = np.load(INPUTS)
labels_np = np.load(LABELS)
pids_np = np.load(PUZZLE_IDS)

seq_len = inputs_np.shape[1]
vocab_size = int(inputs_np.max()) + 1

with open(IDENTIFIERS_JSON) as f:
    num_puzzle_identifiers = len(json.load(f))

print("seq_len:", seq_len)
print("vocab_size:", vocab_size)
print("num_puzzle_identifiers:", num_puzzle_identifiers)

# -------------------------------------------------------------
# Load config.yaml
# -------------------------------------------------------------
with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

arch = cfg["arch"]

# Add missing required fields
arch["seq_len"] = seq_len
arch["vocab_size"] = vocab_size
arch["num_puzzle_identifiers"] = num_puzzle_identifiers
arch["batch_size"] = 1   # inference uses batch 1

# -------------------------------------------------------------
# Load TRM model definition from run’s trm.py
# -------------------------------------------------------------
trm_py = RUN_DIR / "trm.py"
spec = importlib.util.spec_from_file_location("trm_run", trm_py)
trm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trm_mod)
TRM = trm_mod.TinyRecursiveReasoningModel_ACTV1

# -------------------------------------------------------------
# Initialize model
# -------------------------------------------------------------
model = TRM(arch)
sd = torch.load(CKPT, map_location="cpu")

if "state_dict" in sd:
    sd = sd["state_dict"]

sd = {k.replace("module.", ""): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)

model.cpu()
model.eval()

# -------------------------------------------------------------
# Prepare batch
# -------------------------------------------------------------
x = torch.tensor(inputs_np[INDEX:INDEX+1]).long().cpu()
y = torch.tensor(labels_np[INDEX:INDEX+1]).long().cpu()
pid = torch.tensor(pids_np[INDEX:INDEX+1]).long().cpu()

batch = {
    "inputs": x,
    "labels": y,
    "puzzle_identifiers": pid
}

# -------------------------------------------------------------
# Run TRM autoregressive halting inference
# -------------------------------------------------------------
carry = model.initial_carry(batch)
# ---- FIX: Move carry to cpu ----
carry.inner_carry.z_H = carry.inner_carry.z_H.to("cpu")
carry.inner_carry.z_L = carry.inner_carry.z_L.to("cpu")

carry.steps   = carry.steps.to("cpu")
carry.halted  = carry.halted.to("cpu")

# also move current_data tensors
for k in carry.current_data:
    carry.current_data[k] = carry.current_data[k].to("cpu")

final_logits = None
for step in range(model.config.halt_max_steps):
    carry, out = model(carry, batch)
    final_logits = out["logits"]
    if carry.halted.all():
        break

pred = final_logits.argmax(dim=-1).squeeze(0).cpu().numpy()
true = y.squeeze(0).cpu().numpy()

def print_sudoku(vec, title=None):
    """vec: 1D numpy array of length 81"""
    if title:
        print("\n" + title)

    grid = vec.reshape(9, 9)

    for row in grid:
        print(" ".join(f"{int(v):2d}" for v in row))


print("\nPrediction:")
print_sudoku(pred)
print("\nGround Truth:")
print_sudoku(true)

acc = (pred == true).mean()
print("Accuracy:", acc)