from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np


def _as_ndarray_list(rows):
    return np.asarray(rows, dtype=float)


def load_calibrated_home(path: str | Path) -> Dict[str, np.ndarray]:
    p = Path(path)
    data = json.loads(p.read_text())
    return {
        "p_home": _as_ndarray_list(data["p_home"]),
        "a_home": _as_ndarray_list(data["a_home"]),
    }


def load_measurements(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text())
    return data


__all__ = ["load_calibrated_home", "load_measurements"]


