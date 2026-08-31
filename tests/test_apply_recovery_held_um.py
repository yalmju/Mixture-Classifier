# -*- coding: utf-8 -*-
"""Regression check: apply_recovery must emit uM_pred for TEST-role maps.

Bug (fixed 2026-08-28): a map whose composition sat in held predictions
(test_eval) but which had no cached µM LOO row was silently skipped by the µM
branch whenever model["nnls_screen"] was False — the elif required the screen.
Scoring such a map with the deployed µM head is legitimately held-out (the head
never trained on it), so it must get a uM_pred. Found 2026-08-27 while building
the grid64 concentration tables: 12/64 conditions had no µM LOO rows for
exactly this reason.

Run directly (python tests/test_apply_recovery_held_um.py) or via pytest.
Skips when the local model/data are absent (they live outside the repo).
"""
import os
import re
import sys
import math
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH = "S:/Google Drive/내 드라이브/ACF_PEST_DB/260826_Model/mlp_composition_260826.dlm"


def _norm(p):
    return os.path.normcase(os.path.normpath(p))


def _find_test_role_map(model):
    """A map in test_eval that has neither a composition-LOO nor a µM-LOO row —
    the exact shape the bug dropped."""
    loo = {_norm(p) for p in (model.get("loo_eval") or {}).get("paths", [])}
    um_loo = {_norm(p) for p in
              ((model.get("uM") or {}).get("loo_eval") or {}).get("paths", [])}
    for p in (model.get("test_eval") or {}).get("paths", []):
        if _norm(p) not in loo and _norm(p) not in um_loo and os.path.exists(p):
            return p
    return None


def _ratio_from_name(path, subs):
    """'DQ10-TB10-TH10.csv' → {matching model substance: 10, ...}."""
    stem = os.path.basename(path).replace("_corrected", "").replace(".csv", "")
    ratio = {}
    for tok, val in re.findall(r"([A-Za-z]+)(\d+)", stem):
        for s in subs:
            if s.upper().startswith(tok.upper()) or tok.upper().startswith(s.upper()):
                ratio[s] = float(val)
                break
    return ratio


def test_test_role_map_gets_uM_pred():
    if not os.path.exists(MODEL_PATH):
        try:
            import pytest
            pytest.skip("local 260826 model not available on this machine")
        except ImportError:
            print("SKIP: model file not found:", MODEL_PATH)
            return
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    assert not model.get("nnls_screen"), \
        "precondition changed: this regression needs an nnls_screen=False model"
    assert model.get("uM"), "precondition changed: model has no µM head"
    path = _find_test_role_map(model)
    assert path, "precondition changed: no test-role map (test_eval minus LOO rows) left"

    from dl_model import apply_recovery
    ratio = _ratio_from_name(path, model["subs"])
    assert ratio, f"could not parse a nominal ratio from {path}"
    rows = apply_recovery(model, [(path, ratio, None)])
    assert len(rows) == 1
    row = rows[0]
    assert row.get("uM_pred"), (
        f"test-role map {os.path.basename(path)} came back without uM_pred — "
        "the nnls_screen=False held branch dropped the µM head again")
    bad = {k: v for k, v in row["uM_pred"].items()
           if not (isinstance(v, float) and math.isfinite(v))}
    assert not bad, f"non-finite µM predictions: {bad}"

    # Display-layer ND flag (NNLS surface support, threshold held-out validated
    # 2026-08-28 — documentation/UM_ND_FLAG_2026-08-28.md). Since the presence-
    # gate merge (2026-08-31) the surface flag lives under ``surface_nd``;
    # ``uM_nd`` now marks presence-gated components whose values ARE zeroed.
    # Absent components must carry surface_nd; a dominant present one must not.
    nd = row.get("surface_nd")
    assert nd is not None, "surface_nd flag missing from recovery row"
    for sub, r_ in ratio.items():
        if r_ == 0:
            assert nd.get(sub), f"absent component {sub} not flagged ND"
    dominant = max(ratio, key=ratio.get)
    assert not nd.get(dominant), f"dominant component {dominant} wrongly flagged ND"
    gated = row.get("uM_nd") or {}
    for sub, v in row["uM_pred"].items():
        if not gated.get(sub):
            assert v > 0, f"un-gated µM for {sub} must stay positive (display-only flag)"
    print("OK:", os.path.basename(path), "→ uM_pred =",
          {k: round(v, 2) for k, v in row["uM_pred"].items()},
          "| surface_nd =", {k: v for k, v in nd.items()},
          "| presence-gated =", {k: v for k, v in gated.items() if v})


if __name__ == "__main__":
    test_test_role_map_gets_uM_pred()
