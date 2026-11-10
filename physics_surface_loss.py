"""Differentiable surface-consistency loss for SERS mixture composition models.

This is a soft gray-box constraint, not a PDE-style PINN.  A network predicts
solution composition; the known total concentration and calibration parameters
map that prediction through a competitive Sips/Langmuir layer.  The resulting
surface fraction is compared with the measured NNLS surface fraction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfacePhysics:
    K: np.ndarray
    gA: np.ndarray
    m: np.ndarray
    weight: float = 0.03
    min_uM: float = 3.0
    max_uM: float = 24.0


def working_range_mask(concentrations_M, min_uM=3.0, max_uM=24.0):
    """Rows whose present analytes all lie inside the validated working range."""
    C = np.asarray(concentrations_M, float)
    present = C > 0
    has_analyte = present.any(axis=1)
    lo = float(min_uM) * 1e-6
    hi = float(max_uM) * 1e-6
    in_range = ((~present) | ((C >= lo) & (C <= hi))).all(axis=1)
    return has_analyte & in_range


def surface_fraction_numpy(solution_fraction, total_concentration_M, K, gA, m):
    sol = np.asarray(solution_fraction, float)
    sol = sol / (sol.sum(axis=1, keepdims=True) + 1e-12)
    C = sol * np.asarray(total_concentration_M, float)[:, None]
    q = np.power(np.asarray(K, float)[None, :] * C + 1e-12,
                 np.asarray(m, float)[None, :])
    theta = q / (1.0 + q.sum(axis=1, keepdims=True))
    B = np.asarray(gA, float)[None, :] * theta
    return B / (B.sum(axis=1, keepdims=True) + 1e-12)


def prepare_surface_physics(concentrations_M, measured_surface_fraction, physics):
    """Build torch tensors aligned with the training rows."""
    import torch

    C = np.asarray(concentrations_M, float)
    surf = np.asarray(measured_surface_fraction, float)
    mask = working_range_mask(C, physics.min_uM, physics.max_uM)
    return {
        "K": torch.tensor(np.asarray(physics.K), dtype=torch.float32),
        "gA": torch.tensor(np.asarray(physics.gA), dtype=torch.float32),
        "m": torch.tensor(np.asarray(physics.m), dtype=torch.float32),
        "surface": torch.tensor(surf, dtype=torch.float32),
        "total_C": torch.tensor(C.sum(axis=1), dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "weight": float(physics.weight),
        "n_analytes": int(C.shape[1]),
    }


def surface_consistency_loss(predicted_composition, row_indices, tensors):
    """Differentiable Huber penalty for selected low-concentration training rows."""
    import torch
    import torch.nn.functional as F

    valid = tensors["mask"][row_indices]
    if not bool(valid.any()):
        return predicted_composition.sum() * 0.0
    n = tensors["n_analytes"]
    sol = predicted_composition[valid, :n]
    sol = sol / (sol.sum(dim=1, keepdim=True) + 1e-12)
    total = tensors["total_C"][row_indices][valid, None]
    C = sol * total
    q = torch.pow(tensors["K"][None, :] * C + 1e-12,
                  tensors["m"][None, :])
    theta = q / (1.0 + q.sum(dim=1, keepdim=True))
    B = tensors["gA"][None, :] * theta
    surface = B / (B.sum(dim=1, keepdim=True) + 1e-12)
    return F.smooth_l1_loss(surface, tensors["surface"][row_indices][valid])

