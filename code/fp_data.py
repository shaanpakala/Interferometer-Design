"""Load Fabry–Perot simulation tensor from ``data/tensors/fp_simulation.npz``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = REPO_ROOT / "data" / "tensors" / "fp_simulation.npz"

AXIS_NAMES = ("LX", "m1_R", "m1_Rc", "m2_R", "m2_Rc")


def load_fp_simulation(path: Path | None = None) -> dict:
    path = path or DEFAULT_NPZ
    d = np.load(path, allow_pickle=True)
    powers = d["powers"]
    valid_mask = d["valid_mask"]
    axes = {name: np.asarray(d[name], dtype=np.float64) for name in AXIS_NAMES}
    port_names = list(d["port_names_active"])
    return {
        "powers": powers,
        "valid_mask": valid_mask,
        "axes": axes,
        "port_names": port_names,
        "path": path,
    }
