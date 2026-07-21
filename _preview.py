"""Render a static preview of the Mixture Classifier dashboard (Agg, no GUI)."""
import csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import brand
from sers_mixture import preprocess, SERSMixtureClassifier, AugmentConfig
from competitive_compare import additive_residual


def load_csv(path):
    rows = list(csv.reader(open(path)))
    cn = [h.strip() for h in rows[0][1:] if h.strip() != ""]
    ax, arr = [], []
    for r in rows[1:]:
        vals = [v for v in r if v.strip() != ""]
        if len(vals) < 2:
            continue
        ax.append(float(vals[0])); arr.append([float(v) for v in vals[1:len(cn)+1]])
    return np.array(ax), cn, np.array(arr).T


axis, names, rawp = load_csv("example_pure.csv")
_, _, rawu = load_csv("example_unknown.csv")
pures = preprocess(rawp); unk = preprocess(rawu)
clf = SERSMixtureClassifier(names, prob_threshold=0.30, max_components=3,
                            augment=AugmentConfig(n_per_pure=150))
clf.fit(pures)
y = unk[0]
det = clf.predict(y, return_details=True)[0]
B, yhat, res = additive_residual(y, pures)
ratio = B / (B.sum() + 1e-12)

# ---------- draw app-like dashboard ----------
L = lambda pair: pair[0]                      # light-theme value of a (light,dark)
plt.rcParams["font.size"] = 10
fig = plt.figure(figsize=(11.8, 7.4), dpi=110)
fig.patch.set_facecolor(L(brand.MAIN_FILL))
bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)

def card(x, y, w, h, fc):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec="#dfe5ec", lw=1))

# sidebar
card(0.015, 0.02, 0.225, 0.96, L(brand.SIDE_FILL))
bg.text(0.03, 0.94, brand.APP_NAME, fontsize=15, fontweight="bold", color="#1c2430")
bg.text(0.03, 0.915, brand.APP_TAGLINE, fontsize=9.5, color=L(brand.SUBTLE))
def sbtn(y, txt, fc, tc="white"):
    card(0.03, y, 0.195, 0.045, fc)
    bg.text(0.03+0.0975, y+0.0225, txt, ha="center", va="center", fontsize=9.5, color=tc)
sbtn(0.83, "Load pure references", L(brand.BTN_FILL))
bg.text(0.03, 0.805, "example_pure.csv — 3 pures: " + ", ".join(names),
        fontsize=7.5, color=L(brand.SUBTLE))
sbtn(0.74, "Load unknown CSV", L(brand.BTN_FILL))
bg.text(0.03, 0.715, "example_unknown.csv — 2 spectra", fontsize=7.5, color=L(brand.SUBTLE))
bg.text(0.03, 0.66, "detection threshold   0.30", fontsize=8.5, color="#1c2430")
bg.text(0.03, 0.63, "max components        3", fontsize=8.5, color="#1c2430")
sbtn(0.55, "Run analysis", L(brand.RUN_FILL))
bg.text(0.03, 0.50, f"detected: {', '.join(det['components'])}\nfit residual {res*100:.1f}%",
        fontsize=10, color="#1c2430", va="top")
sbtn(0.34, "Export results CSV", L(brand.BTN_FILL))
sbtn(0.285, "Export dashboard PNG", L(brand.BTN_FILL))

# spectrum card
card(0.255, 0.55, 0.73, 0.43, L(brand.CARD_FILL))
bg.text(0.27, 0.945, "Spectrum + NNLS reconstruction", fontsize=11, fontweight="bold",
        color="#1c2430")
ax1 = fig.add_axes([0.30, 0.62, 0.64, 0.29])
ax1.plot(axis, y, lw=1.3, color=brand.SERIES[0], label="measured")
ax1.plot(axis, yhat, lw=1.1, ls="--", color=brand.SERIES[3],
         label=f"reconstruction (res {res*100:.1f}%)")
ax1.set_xlabel("wavenumber (cm⁻¹)"); ax1.set_ylabel("intensity (norm.)")
ax1.legend(fontsize=8, frameon=False); ax1.tick_params(labelsize=8)

# composition pie card
card(0.255, 0.04, 0.35, 0.46, L(brand.CARD_FILL))
bg.text(0.27, 0.465, "Composition (ratio)", fontsize=11, fontweight="bold", color="#1c2430")
ax2 = fig.add_axes([0.29, 0.08, 0.26, 0.33])
keep = [(n, r) for n, r in zip(names, ratio) if r > 0.01]
labels, vals = zip(*keep)
ax2.pie(vals, labels=[f"{n}\n{v*100:.0f}%" for n, v in keep],
        colors=brand.SERIES[:len(vals)], startangle=90, textprops={"fontsize": 9})
ax2.set_aspect("equal")

# table card
card(0.63, 0.04, 0.355, 0.46, L(brand.CARD_FILL))
bg.text(0.645, 0.465, "Per-component read-out", fontsize=11, fontweight="bold", color="#1c2430")
proba = det.get("proba", {})
lines = [f"{'component':10s}{'present':>8}{'ratio':>7}{'prob':>6}", "-"*31]
for i, n in enumerate(names):
    present = "yes" if n in det["components"] else "-"
    p = proba.get(n, float("nan"))
    lines.append(f"{n:10s}{present:>8}{ratio[i]*100:6.0f}%{('' if p!=p else f'{p:.2f}'):>6}")
lines += ["-"*31, f"reconstruction residual: {res*100:.1f}%",
          "(ratio = signal-weighted; calibrate", " response factors for molar ratio)"]
bg.text(0.645, 0.41, "\n".join(lines), family="monospace", fontsize=9.2,
        color="#1c2430", va="top")

fig.savefig("preview_dashboard.png", dpi=120, facecolor=fig.get_facecolor())
print("saved preview_dashboard.png  |  detected:", det["components"],
      "| ratio:", {n: round(float(r), 2) for n, r in zip(names, ratio)})
