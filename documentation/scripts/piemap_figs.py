"""파이 · 격자 · 픽셀맵 그림 — `piemap_compute.py` 가 만든 캐시만 읽는다 (빠름).

  그리는 코드는 전부 저장소 루트의 `grid_figs.py` 에 있고 앱(Model 탭 Export)이 같은 걸
  쓴다. 여기는 캐시를 그 서명에 맞춰 넘기고 파일로 떨어뜨리는 껍데기다 — 그림이 앱과
  스크립트에서 달라지는 일이 없도록.

  나오는 것
    fig5_pies65.png            조건별 조성 파이
    fig12_grid_truepred.png    격자 · 위=참 / 아래=예측
    fig9_grid_error.png        격자 오차 히트맵 (반복산포 하한 … 무정보 기준선 눈금)
    fig11_pixelmap65.png       조건별 10×10 픽셀 조성맵
    *.csv                      각 그림의 숫자 (Origin 에서 다시 그릴 때 쓴다)

  실행:  python3 -u piemap_figs.py [캐시폴더]
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os, sys, csv, pickle, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import grid_figs as GF

DB = paths.DB
CACHE = sys.argv[1] if len(sys.argv) > 1 else f"{paths.INTERP}/piemap_260812"
SAVE = dict(dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.01)
FALLBACK = {"DQ": "#008cf7", "TBZ": "#95dc2a", "THI": "#f35376"}

d = pickle.load(open(f"{CACHE}/piemap_data.pkl", "rb"))
cond, maps, SUB = d["cond"], d["maps"], d["sub"]
try:
    CO = json.load(open(f"{DB}/Pure/colors.json", encoding="utf-8"))
except Exception:
    CO = FALLBACK
COLS = [CO.get(s, FALLBACK.get(s, "#888888")) for s in SUB]

keys = sorted(cond)
rows = [("/".join(str(v) for v in k) + " µM", cond[k]["true"], cond[k]["pred"]) for k in keys]
errs = np.array([GF.comp_error(r[1], r[2]) for r in rows])
print(f"{len(keys)}조건 · {len(maps)}맵 · 구성 {d['arm']}")
print(f"평균 {errs.mean():.1f}%  중앙 {np.median(errs):.1f}%  "
      f"무정보 기준선 {GF.null_error([r[1] for r in rows]):.1f}%")


def save(fig, name):
    if fig is None:
        print(f"  (건너뜀) {name}"); return
    fig.savefig(f"{CACHE}/{name}.png", **SAVE)
    print(f"  {name}.png")


def table(name, header, body):
    with open(f"{CACHE}/{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(body)
    print(f"  {name}.csv ({len(body)}행)")


save(GF.composition_pies(rows, SUB, COLS,
                         title=f"Predicted composition per condition — "
                               f"leave-one-condition-out · {len(keys)} conditions"),
     "fig5_pies65")
save(GF.grid_truepred(rows, keys, SUB, COLS), "fig12_grid_truepred")
save(GF.grid_error(rows, keys, SUB), "fig9_grid_error")

first = [(("/".join(str(v) for v in k) + " µM"),
          maps[cond[k]["paths"][0]]["coords"], maps[cond[k]["paths"][0]]["ratio"],
          maps[cond[k]["paths"][0]]["hit"]) for k in keys]
save(GF.pixel_maps(first, SUB, COLS,
                   title="Per-pixel composition map — NNLS decides hit, the model composes"),
     "fig11_pixelmap65")

table("grid_conditions", *GF.condition_table(rows, keys, SUB))
for nm, h, b in GF.error_matrix_tables(rows, keys, SUB):
    table(f"grid_error_matrix_{nm}", h, b)
table("grid_pixel_maps", *GF.pixel_table(first, SUB))
