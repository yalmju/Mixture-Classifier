"""논문용 PLS 모델 학습과 같은 호출에서 농도 단계가 받는 입력만 꺼낸다 (모델은 저장하지 않는다)."""
import os, sys, numpy as np
REPO = "/Users/seungki2/Documents/GitHub/Mixture-Classifier"
sys.path.insert(0, f"{REPO}/documentation/scripts")
import train_paper_pls as T
import dl_model, paths
from dataset import load_mixture_list, load_mixture_roles, load_preprocess
OUT = os.path.dirname(os.path.abspath(__file__))

def grab(Xraw, Rabs, gp, gcond, C_uM_rows, conc_subs, wn_axis, P, calib_path, lo, hi, **kw):
    bands = dl_model._marker_bands(conc_subs, wn_axis, P)
    ab, rng = dl_model._fit_loglinear_ab(calib_path, conc_subs, bands, lo, hi)
    names, F, Ccal = dl_model._residual_context(Xraw, Rabs, gp, wn_axis, bands, ab, rng)
    gp = np.asarray(gp, object); gcond = np.asarray(gcond, object)
    first = {g: np.where(gp == g)[0][0] for g in names}
    Tm = np.stack([np.asarray(C_uM_rows[first[g]], float) for g in names])
    cond = np.array([str(gcond[first[g]]) for g in names])
    np.savez(f"{OUT}/head_inputs.npz", F=F, Ccal=Ccal, T=Tm, cond=cond,
             names=np.array([str(n) for n in names]), subs=np.array(list(conc_subs)),
             bands=bands, ab=ab, rng=rng)
    print("dumped", F.shape, len(set(cond.tolist())), "conditions", flush=True)
    raise SystemExit(0)

dl_model._train_calibration_residual_head = grab
cfg = load_preprocess(T.PURE); roles = load_mixture_roles(T.PURE)
allm = load_mixture_list(T.PURE, filename_truth=True)
train = [T._here(it) for it in allm if roles.get(it[0], "train") != "test"]
dl_model.train_model(
    T.PURE, train, calib_path=T.CALIB, baseline=cfg["baseline"], trim=cfg["trim"],
    method="pls", n_components=T.N_COMPONENTS, seed=0, epochs=300, use_pretrain=False,
    px_per_map=int(cfg.get("sample_pixels", 100)),
    pixel_sampling=cfg.get("pixel_sampling", "representative"),
    sampling_seed=int(cfg.get("sampling_seed", 0)),
    include_blank=True, nnls_screen=False, equal_volume_mix=False,
    loo=False, progress=lambda s: print("  ", s, flush=True))
