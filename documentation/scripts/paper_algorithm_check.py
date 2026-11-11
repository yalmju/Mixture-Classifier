"""paper_algorithm.py 가 저장된 논문 모델 · 기존 파이프라인과 같은 답을 내는지 확인한다.

  1  학습 집합을 paper_algorithm 만으로 다시 맞춘 가중치 = 저장 모델(pls_composition_261110.dlm)의 가중치
  2  전처리 · 조성 · 농도 판독 = dl_model 의 적용 경로 (시험 20 맵)
  3  게이트 = unmix.unmix_map 의 hit, 그 맵의 판독 = Fig 4c 자료 (260812 맵)
  4  조건을 통째로 뺀 판독 수치 = 모델에 저장된 기록 (조성 12.3 %p, 농도 2배 이내 95.8 %)

  모델 파일은 읽기만 한다. 다시 학습하거나 덮어쓰지 않는다.

  실행:  ../../.venv/bin/python3 -u paper_algorithm_check.py [학습 표를 넣어 둘 .npz]
         (학습 표 = 95 맵 전처리. 몇 분 걸려서, 경로를 주면 한 번 만든 것을 다시 쓴다)
"""
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import paper_algorithm as A

sys.path.insert(0, paths.REPO)
import dataset
import dl_model
import unmix

PURE = paths.PURE
FINAL = f"{paths.DATA}/Ratio/260814_mixture_final"
MODEL = f"{paths.RESULTS}/261110_Model_PLS_paper/pls_composition_261110.dlm"
LETTER = f"{paths.DATA}/Pest/260812_12 trio_THI_TBZ_DQ.csv"
FIG4C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "fig4_paper_pls_2026-11-10",
                     "lettering_pixels.npz")
OUT = f"{paths.REPO}/documentation/results/paper_algorithm_2026-11-11/check.txt"
GRID = {3.0, 6.0, 12.0, 24.0}
SKIP_REF = {"LEAF"}                 # 배포 모델 학습 뒤에 들어온 배경 — train_paper_pls.py 와 같게
_NAME = re.compile(r"DQ(\d+)-TB(\d+)-THI?(\d+)")

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def base(p):
    return os.path.basename(str(p).replace("\\", "/"))


def samples():
    """train_paper_pls.py 가 dl_model.train_model 에 넘기는 것과 같은 맵 · 라벨."""
    roles = dataset.load_mixture_roles(PURE)
    train, test = [], []
    for it in dataset.load_mixture_list(PURE, filename_truth=True):     # 라벨 = 파일명 (µM)
        (test if roles.get(it[0], "train") == "test" else train).append((f"{FINAL}/{base(it[0])}", dict(it[1])))
    manifest = dataset.load_manifest(PURE)
    refs = [(dataset.base_and_batch(c)[0], p) for c, p in dataset.discover_references(PURE)]
    refs = [(n, p) for n, p in refs if n.upper() not in SKIP_REF]
    pure = [(p, n) for n, p in refs if n in A.ANALYTES
            and manifest.get(os.path.abspath(p), (None, None, "train"))[2] == "train"]
    blank = [(p, None) for n, p in refs if dataset.is_blank(n)]
    return train + pure + blank, test


def load_table(fit_maps, cache=None):
    """학습 표(95 맵 전처리). ``cache`` 경로를 주면 한 번 만든 것을 다시 쓴다."""
    if cache and os.path.isfile(cache):
        table = dict(np.load(cache, allow_pickle=True))
        table["paths"] = list(table["paths"]); table["analytes"] = list(table["analytes"])
    else:
        table = A.training_table(fit_maps, progress=lambda s: print("  ", s, flush=True))
        if cache:
            np.savez(cache, **{k: np.asarray(v, object) if k in ("paths", "group") else np.asarray(v)
                               for k, v in table.items()})
    if table["paths"] != [p for p, _ in fit_maps]:
        raise SystemExit("저장해 둔 학습 표가 지금 맵 목록과 다르다 — 지우고 다시 만들 것")
    return table


def grid_uM(path):
    m = _NAME.match(base(path))
    c = tuple(float(m.group(i)) for i in (1, 2, 3)) if m else None
    return c if c and all(v in GRID for v in c) else None


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    fit_maps, test_maps = samples()
    say(f"학습 {len(fit_maps)} 맵 · 시험 {len(test_maps)} 맵")

    table = load_table(fit_maps, cache)
    say(f"학습 표: {len(table['paths'])} 맵 · {len(table['X'])} 스펙트럼 · {table['X'].shape[1]} 점")

    saved = A.load_model(MODEL)
    with open(MODEL, "rb") as fh:
        legacy = pickle.load(fh)
    say(f"저장 모델 기록: {legacy['n_maps']} 맵 · {legacy['n_train']} 스펙트럼")

    # ── 1  다시 맞춘 가중치 = 저장 모델 ────────────────────────────────────
    say("\n[1] paper_algorithm.fit 으로 다시 맞춘 것과 저장 모델의 차이")
    new = A.fit(table)
    sh_new, sh_old = A.pixel_shares(new["composition"], table["X"]), A.pixel_shares(saved["composition"], table["X"])
    F = A.map_rows(table)
    lab = A.in_window(table["uM"])
    um_new, um_old = A.concentration(new["concentration"], F[lab]), A.concentration(saved["concentration"], F[lab])
    say(f"    조성 가중치 최대 차이 {np.abs(new['composition'][0] - saved['composition'][0]).max():.2e}"
        f" · 학습 픽셀의 몫 최대 차이 {np.abs(sh_new - sh_old).max():.2e}")
    say(f"    농도 가중치 최대 차이 {np.abs(new['concentration'][0] - saved['concentration'][0]).max():.2e}"
        f" · 맞춤 {int(lab.sum())} 맵의 µM 최대 상대 차이 {np.abs(um_new / um_old - 1).max():.2e}")

    # ── 2  판독 = dl_model 적용 경로 ───────────────────────────────────────
    say("\n[2] 시험 맵 판독: paper_algorithm.read_map 과 dl_model 적용 경로의 차이")
    d_prep = d_share = d_um = d_train = 0.0
    test_read = {}
    for k, (path, label) in enumerate(test_maps):
        print(f"   시험 맵 {k + 1}/{len(test_maps)}", flush=True)
        wn, cube = A.load_map(path)
        ax, X = A.prepare(wn, cube)
        r = A.read_map(wn, cube, saved)
        test_read[path] = r
        wn_l, X_l = dl_model.model_input(legacy, wn, cube, corrected=False)
        d_prep = max(d_prep, float(np.abs(X - X_l).max()))
        d_share = max(d_share, float(np.abs(r["pixel_shares"] - dl_model.apply_model_pixels(legacy, wn_l, X_l)).max()))
        _um, _subs, meta = dl_model.apply_uM_pixels(legacy, wn_l, X_l, return_meta=True)
        d_um = max(d_um, max(abs(r["uM"][s] / meta["map_uM"][s] - 1) for s in saved["analytes"]))
        mask = (wn >= legacy["lo"]) & (wn <= legacy["hi"])
        S_l = np.array(dl_model._map_spectra(cube, mask, 100, baseline_correct=True, sampling="representative",
                                             sampling_seed=0, map_id=path))
        S = A.training_spectra(wn, cube, map_id=path)[1]
        d_train = max(d_train, float(np.abs(S - S_l).max()) if S.shape == S_l.shape else np.inf)
    say(f"    전처리(prepare) 최대 차이 {d_prep:.2e} · 맞춤용 픽셀(training_spectra) 최대 차이 {d_train:.2e}")
    say(f"    픽셀 몫 최대 차이 {d_share:.2e} · 맵 µM 최대 상대 차이 {d_um:.2e}")

    # ── 3  게이트 = unmix_map ──────────────────────────────────────────────
    say("\n[3] 글씨맵 게이트: paper_algorithm.gate 와 unmix.unmix_map")
    cfg = dataset.load_preprocess(PURE)
    templates = A.class_templates({c: [p for _b, p, _r in maps] for c, maps in dataset.discover_dataset(PURE)})
    wn, cube = A.load_map(LETTER)
    hit = A.gate(wn, cube, templates)
    ref = unmix.unmix_map(data_dir=PURE, test_path=LETTER, method="nnls", baseline=bool(cfg["baseline"]),
                          trim=cfg["trim"], min_frac=A.GATE_MIN_SHARE, hit_mode="threshold")
    say(f"    템플릿 {list(templates)} · 통과 픽셀 {int(hit.sum())}/{len(hit)}"
        f" (unmix_map {int(np.asarray(ref.hit).sum())}) · 다른 픽셀 {int((hit != np.asarray(ref.hit, bool)).sum())}")
    r = A.read_map(wn, cube, saved, templates)
    fmt = lambda v: " · ".join(f"{s} {100 * x:.1f} %" for s, x in zip(saved["analytes"], v))
    say("    맵 판독(read_map, 통과 픽셀의 몫을 평균한 뒤 합 1) " + fmt(r["composition"].values()))
    px = A.pixel_composition(r["pixel_shares"])
    say("    픽셀 조성(픽셀마다 합 1)의 통과 픽셀 평균 " + fmt(px[r["gate"]].mean(axis=0)))
    z = np.load(FIG4C)                                  # fig4_paper_data.py 가 Fig 4c 에 쓰는 값
    cols = [list(z["comp"]).index(s) for s in saved["analytes"]]
    say(f"    Fig 4c 자료와의 최대 차이: 맵 판독 {np.abs(z['pls_map'][cols] - list(r['composition'].values())).max():.2e}"
        f" · 픽셀 조성 {np.abs(z['pls'][:, cols] - px).max():.2e}")

    # ── 4  조건을 뺀 판독 수치 ─────────────────────────────────────────────
    say("\n[4] 조건을 통째로 뺀 판독 (학습 격자 52 조건은 그 조건 없이 다시 맞춤, 시험 격자 12 조건은 저장 모델)")
    n_an = len(A.ANALYTES)
    P = A.held_out(A.fit_composition, A.pixel_shares, table["X"], table["Y"], table["group"][table["map"]])
    rec = {base(p): np.asarray(v, float) for p, v in zip(legacy["loo_eval"]["paths"], legacy["loo_eval"]["pred"])}
    comp_t, comp_p, d_rec, where = [], [], 0.0, []
    for i, path in enumerate(table["paths"]):
        c = grid_uM(path)
        mean = P[table["map"] == i].mean(axis=0)
        d_rec = max(d_rec, float(np.abs(mean - rec[base(path)]).max()))
        if c:
            comp_t.append(np.array(c) / sum(c)); comp_p.append(A.map_composition(P[table["map"] == i], n_an)); where.append("fit")
    for path, label in test_maps:
        c = grid_uM(path)
        if c:
            comp_t.append(np.array(c) / sum(c))
            comp_p.append(np.array([test_read[path]["composition"][s] for s in A.ANALYTES])); where.append("test")
    err = A.composition_error(comp_p, comp_t); where = np.array(where)
    say(f"    조성: 64 조건 {err.mean():.1f} %p (n={len(err)}) · 학습 격자 {err[where == 'fit'].mean():.1f} %p"
        f" (n={(where == 'fit').sum()}) · 시험 격자 {err[where == 'test'].mean():.1f} %p (n={(where == 'test').sum()})"
        f" · 저장 기록과 최대 차이 {d_rec:.2e}")

    T = table["uM"][lab]
    cond = np.array([",".join(f"{v:g}" for v in t) for t in T], object)
    fit_c = lambda X_, Y_: A.fit_concentration(table["wn"], X_, 10.0 ** Y_)
    C = 10.0 ** A.held_out(fit_c, lambda w, X_: np.log10(A.concentration(w, X_)), F[lab], np.log10(T), cond)
    rec_u = legacy["uM"]["loo_eval"]
    rec_u = {base(p): np.asarray(v, float) for p, v in zip(rec_u["paths"], rec_u["pred_uM"])}
    d_rec = max(float(np.abs(C[j] / rec_u[base(p)] - 1).max())
                for j, p in enumerate(np.array(table["paths"], object)[lab]))
    t_test = np.array([grid_uM(p) for p, _ in test_maps if grid_uM(p)])
    c_test = np.array([[test_read[p]["uM"][s] for s in A.ANALYTES] for p, _ in test_maps if grid_uM(p)])
    t_all, c_all = np.vstack([T, t_test]), np.vstack([C, c_test])
    fold = A.fold_error(c_all, t_all)
    const = 10.0 ** np.log10(T).mean()
    fold_c = A.fold_error(np.full_like(t_all, const), t_all)
    rmse = float(np.sqrt(np.mean(np.log10(c_all / t_all) ** 2)))
    say(f"    농도: 64 조건 2배 이내 {100 * (fold <= 2).mean():.1f} % ({int((fold <= 2).sum())}/{fold.size})"
        f" · 1.5배 이내 {100 * (fold <= 1.5).mean():.1f} % · 중앙 {np.median(fold):.2f}× · RMSE {rmse:.3f} dex")
    say("          성분별 2배 이내 " + " · ".join(f"{s} {100 * (fold[:, j] <= 2).mean():.1f} %" for j, s in enumerate(A.ANALYTES))
        + f" · 세 성분 모두 {int((fold <= 2).all(axis=1).sum())}/{len(fold)}")
    say(f"          대조(상수 {const:.1f} µM) 2배 이내 {100 * (fold_c <= 2).mean():.1f} %"
        f" · 세 성분 모두 {int((fold_c <= 2).all(axis=1).sum())}/{len(fold_c)}")
    ft, fct = A.fold_error(c_test, t_test), A.fold_error(np.full_like(t_test, const), t_test)
    say(f"          시험 격자 {len(t_test)} 맵: 2배 이내 {100 * (ft <= 2).mean():.1f} % · 1.5배 이내 {100 * (ft <= 1.5).mean():.1f} %"
        f" · 중앙 {np.median(ft):.2f}× (대조 {100 * (fct <= 2).mean():.1f} %) · 저장 기록과 최대 상대 차이 {d_rec:.2e}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n기록: {OUT}")


if __name__ == "__main__":
    main()
