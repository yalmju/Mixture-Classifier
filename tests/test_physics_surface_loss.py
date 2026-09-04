import numpy as np

from physics_surface_loss import (
    SurfacePhysics,
    prepare_surface_physics,
    surface_consistency_loss,
    surface_fraction_numpy,
    working_range_mask,
)


def test_working_range_excludes_high_and_below_range():
    C = np.array([[3, 6, 24], [3, 6, 100], [1, 6, 12], [0, 6, 12]]) * 1e-6
    assert working_range_mask(C).tolist() == [True, False, False, True]


def test_matching_solution_has_near_zero_loss_and_gradient():
    import torch

    K = np.array([7.05e3, 3.39e4, 3.10e4])
    gA = np.array([1.0, 1.2, 1.1])
    m = np.array([0.66, 1.0, 1.0])
    C = np.array([[12.0, 6.0, 3.0]]) * 1e-6
    sol = C / C.sum(axis=1, keepdims=True)
    surface = surface_fraction_numpy(sol, C.sum(1), K, gA, m)
    cfg = SurfacePhysics(K=K, gA=gA, m=m)
    tensors = prepare_surface_physics(C, surface, cfg)
    pred = torch.tensor(sol, dtype=torch.float32, requires_grad=True)
    loss = surface_consistency_loss(pred, torch.arange(1), tensors)
    loss.backward()
    assert float(loss) < 1e-10
    assert pred.grad is not None


def test_loss_penalizes_wrong_solution_fraction():
    import torch

    K = np.array([7.05e3, 3.39e4, 3.10e4])
    gA = np.array([1.0, 1.2, 1.1])
    m = np.array([0.66, 1.0, 1.0])
    C = np.array([[12.0, 6.0, 3.0]]) * 1e-6
    true_sol = C / C.sum(axis=1, keepdims=True)
    surface = surface_fraction_numpy(true_sol, C.sum(1), K, gA, m)
    tensors = prepare_surface_physics(C, surface, SurfacePhysics(K, gA, m))
    wrong = torch.tensor([[0.1, 0.8, 0.1]], requires_grad=True)
    loss = surface_consistency_loss(wrong, torch.arange(1), tensors)
    loss.backward()
    assert float(loss) > 1e-4
    assert float(wrong.grad.abs().sum()) > 0
