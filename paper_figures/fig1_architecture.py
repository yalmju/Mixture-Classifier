"""Figure 1 — the two-head pipeline as ONE horizontal row.

Hand-laid SVG (a schematic, not data), so every coordinate is explicit and the
figure re-renders identically. The numbers are read off dl_model.py:
  composition head    1,290 -> 256 -> 64 -> softmax 4     (BN/ReLU/dropout 0.15)
  concentration head     16 -> 128 -> 32 -> log10 uM x 3  (BN/ReLU/dropout 0.25)
  16-D context = 3 ratios + 1 log intensity + 9 ratio quantiles + 3 intensity quantiles
Run:  python paper_figures/fig1_architecture.py
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
W, H = 1450, 400
YC = 200                                   # the row's optical centre

INK, MUTE, LINE = "#12202e", "#7b8a9b", "#2b3b4d"
DQ, TBZ, THI, BLK = "#1a73e8", "#27a679", "#e8467c", "#98a2b0"
BFILL, BEDGE = "#eaf1fb", "#a9c6ec"        # composition head
CFILL, CEDGE = "#e9f5ef", "#a4d5bf"        # concentration head

s = []
add = s.append

def txt(x, y, t, size=10, fill=MUTE, anchor="middle", weight=400, ls=0, italic=False, tr=""):
    st = "font-style:italic" if italic else ""
    tr = f' transform="{tr}"' if tr else ""
    add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}" letter-spacing="{ls}" style="{st}"{tr}>{t}</text>')

def stack(x, h, w, fill, edge, n=3, d=3.5):
    """A layer drawn as n offset sheets — the 'many units' look."""
    for i in range(n - 1, 0, -1):
        add(f'<rect x="{x+i*d:.1f}" y="{YC-h/2-i*d:.1f}" width="{w}" height="{h}" rx="3" '
            f'fill="{fill}" fill-opacity="0.5" stroke="{edge}" stroke-width="1"/>')
    add(f'<rect x="{x}" y="{YC-h/2:.1f}" width="{w}" height="{h}" rx="3" '
        f'fill="{fill}" stroke="{edge}" stroke-width="1.4"/>')

def arrow(x1, y1, x2, y2, w=1.4):
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="{w}" '
        'marker-end="url(#ah)"/>')

add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    'font-family="Arial, Liberation Sans, Helvetica, sans-serif">')
add('<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto"><path d="M0 1 L9 5 L0 9 z" fill="{LINE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ---------------------------------------------------------------- a  SPECTRA
txt(22, 40, "a", 17, INK, "start", 700)
txt(39, 40, "SPECTRA", 14, INK, "start", 700, 1.2)
txt(39, 55, "one pixel is one spectrum", 11.5, MUTE, "start", 400, 0, True)

mx, my, ms = 28, 146, 96                                   # a map = a grid of pixels
for i in (2, 1):
    add(f'<rect x="{mx+i*6}" y="{my-i*6}" width="{ms}" height="{ms}" fill="#f4f8fd" '
        f'stroke="{BEDGE}" stroke-width="1"/>')
cell, alpha = ms / 6, [.15,.55,.30,.85,.20,.45, .60,.25,.75,.35,.90,.15, .30,.80,.40,.20,.55,.70,
                       .85,.35,.60,.45,.25,.30, .20,.65,.35,.75,.40,.55, .50,.30,.85,.25,.60,.20]
for r in range(6):
    for c in range(6):
        add(f'<rect x="{mx+c*cell:.1f}" y="{my+r*cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
            f'fill="{DQ}" fill-opacity="{alpha[r*6+c]}"/>')
add(f'<rect x="{mx}" y="{my}" width="{ms}" height="{ms}" fill="none" stroke="{LINE}" stroke-width="1.2"/>')
hr, hc = 2, 4                                              # the pixel the spectrum comes from
add(f'<rect x="{mx+hc*cell:.1f}" y="{my+hr*cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
    f'fill="none" stroke="{INK}" stroke-width="2"/>')
txt(mx + ms/2, my + ms + 21, "109 maps", 13, INK, "middle", 600)
txt(mx + ms/2, my + ms + 34, "10×10–20×20 px", 11.5)

sx0, sx1, sy0, sy1 = 176, 356, 128, 248                    # one example spectrum
peaks = [(555, 1.00, 11), (720, .16, 14), (950, .34, 16), (1140, .40, 14), (1250, .30, 12),
         (1380, .78, 12), (1470, .34, 12), (1600, .20, 16), (2100, .05, 30), (2300, .04, 30)]
pts = []
for i in range(320):
    wn = 500 + 2000 * i / 319
    y = 0.03 + 0.92 * sum(a * math.exp(-((wn - c) / w) ** 2 / 2) for c, a, w in peaks)
    pts.append(f"{sx0 + (sx1-sx0)*i/319:.1f},{sy1 - (sy1-sy0)*min(y,1.05):.1f}")
add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{INK}" stroke-width="1.3" '
    'stroke-linejoin="round"/>')
add(f'<line x1="{sx0}" y1="{sy1}" x2="{sx1}" y2="{sy1}" stroke="{LINE}" stroke-width="1"/>')
add(f'<line x1="{sx0}" y1="{sy1}" x2="{sx0}" y2="{sy0}" stroke="{LINE}" stroke-width="1"/>')
for wn in (500, 1000, 1500, 2000, 2500):                   # x ticks
    x = sx0 + (sx1 - sx0) * (wn - 500) / 2000
    add(f'<line x1="{x:.1f}" y1="{sy1}" x2="{x:.1f}" y2="{sy1+4}" stroke="{LINE}" stroke-width="1"/>')
    if wn in (500, 1500, 2500):
        txt(x, sy1 + 15, f"{wn:,}", 11)
iy = (sy0 + sy1) / 2 + 16
txt(sx0 - 9, iy, "Intensity (a.u.)", 11, MUTE, "middle", 400, 0, False,
    f"rotate(-90 {sx0-9} {iy})")
txt((sx0 + sx1) / 2, sy1 + 30, "Raman shift (cm⁻¹)", 11.5)
txt((sx0 + sx1) / 2, sy1 + 45, "1,290 channels", 13, INK, "middle", 600)

px, py = mx + (hc + 1) * cell, my + (hr + 0.5) * cell      # pixel -> spectrum
add(f'<path d="M{px:.1f} {py:.1f} H{px+22:.1f} V{sy0+16} H{sx0-9}" fill="none" '
    f'stroke="{LINE}" stroke-width="1.2" marker-end="url(#ah)"/>')

# ------------------------------------------------- preprocessing / NNLS gate
gx, gw = 372, 78
txt(gx + gw/2, YC - 26, "log(1 + x)", 12, INK, "middle", 600)
txt(gx + gw/2, YC - 13, "NNLS gate ≥ 0.15", 11.5)
arrow(gx, YC, gx + gw, YC)
txt(gx + gw/2, YC + 20, "23,125 hit pixels", 11.5)
txt(gx + gw/2, YC + 32, "screening only", 11, MUTE, "middle", 400, 0, True)

# ------------------------------------------------------- b  COMPOSITION HEAD
bx = 472
txt(bx, 40, "b", 17, INK, "start", 700)
txt(bx + 17, 40, "COMPOSITION HEAD", 14, INK, "start", 700, 1.2)
txt(bx + 17, 55, "one prediction per pixel", 11.5, MUTE, "start", 400, 0, True)

for x, w, h, lab in ((bx + 8, 32, 156, "1,290"), (bx + 84, 28, 114, "256"), (bx + 152, 24, 82, "64")):
    stack(x, h, w, BFILL, BEDGE)
    txt(x + w/2 + 3.5, YC + 96, lab, 13, INK, "middle", 700)
arrow(bx + 44, YC, bx + 80, YC); arrow(bx + 116, YC, bx + 148, YC)
txt(bx + 100, YC - 100, "BN · ReLU · dropout 0.15", 11.5)

smx, smw, smh = bx + 216, 38, 92                            # softmax over 4 references
add(f'<rect x="{smx}" y="{YC-smh/2}" width="{smw}" height="{smh}" rx="3" fill="#fff" '
    f'stroke="{LINE}" stroke-width="1.2"/>')
for i, (col, lab) in enumerate(((DQ, "DQ"), (TBZ, "TBZ"), (THI, "THI"), (BLK, "BLK"))):
    yy = YC - smh/2 + 1 + i*(smh-2)/4
    add(f'<rect x="{smx+1}" y="{yy:.1f}" width="{smw-2}" height="{(smh-2)/4:.1f}" fill="{col}"/>')
    txt(smx + smw/2, yy + 15, lab, 11, "#ffffff", "middle", 700)
arrow(bx + 184, YC, smx - 4, YC)
txt(smx + smw/2, YC + 96, "softmax 4", 13, INK, "middle", 700)
txt(bx + 120, YC + 122, "L1 per pixel against the map's label ratio · minor components ×3", 11.5)

# readout 1 — mean pool over the pixels of one map
rx, ry, cw = smx + 66, 116, 104
add(f'<path d="M{smx+smw/2} {YC-smh/2-4} V{ry+12} H{rx-6}" fill="none" stroke="{LINE}" '
    'stroke-width="1.4" marker-end="url(#ah)"/>')
cx = rx
for f, col in ((.34, DQ), (.29, TBZ), (.33, THI), (.04, BLK)):
    add(f'<rect x="{cx:.1f}" y="{ry+2}" width="{cw*f:.1f}" height="20" fill="{col}"/>'); cx += cw*f
add(f'<rect x="{rx}" y="{ry+2}" width="{cw}" height="20" fill="none" stroke="{LINE}" stroke-width="1"/>')
txt(rx + cw/2, ry - 6, "mean pool → composition (%)", 12, INK, "middle", 600)
txt(rx + cw/2, ry + 36, "map-level readout, not in the loss", 11, MUTE, "middle", 400, 0, True)

# ----------------------------------------------------- c  CONCENTRATION HEAD
cx0 = 904
txt(cx0, 40, "c", 17, INK, "start", 700)
txt(cx0 + 15, 40, "CONCENTRATION HEAD", 14, INK, "start", 700, 1.2)
txt(cx0 + 15, 55, "one prediction per map", 11.5, MUTE, "start", 400, 0, True)

arrow(smx + smw + 6, YC, cx0 + 4, YC)
txt((smx + smw + cx0) / 2 + 4, YC - 13, "3 ratios, renormalised", 11.5)

ctx_x, ctx_w, ctx_h = cx0 + 10, 28, 136
add(f'<rect x="{ctx_x}" y="{YC-ctx_h/2}" width="{ctx_w}" height="{ctx_h}" rx="3" '
    f'fill="{CFILL}" stroke="{CEDGE}" stroke-width="1.4"/>')
for frac in (3/16, 4/16, 13/16):
    yy = YC - ctx_h/2 + ctx_h*frac
    add(f'<line x1="{ctx_x}" y1="{yy:.1f}" x2="{ctx_x+ctx_w}" y2="{yy:.1f}" stroke="{CEDGE}" stroke-width="1"/>')
txt(ctx_x + ctx_w/2, YC + 96, "16-D context", 13, INK, "middle", 700)
txt(ctx_x - 4, YC + 110, "3 ratios · 1 intensity · 9 + 3 map quantiles (P10/50/90)", 11.5, MUTE, "start")

nx = ctx_x + 92
stack(nx, 112, 26, CFILL, CEDGE); txt(nx + 16.5, YC + 96, "128", 13, INK, "middle", 700)
stack(nx + 70, 78, 22, CFILL, CEDGE); txt(nx + 84, YC + 96, "32", 13, INK, "middle", 700)
arrow(ctx_x + ctx_w + 6, YC, nx - 4, YC); txt(ctx_x + ctx_w + 30, YC - 13, "z-score", 11.5)
arrow(nx + 34, YC, nx + 66, YC)
txt(nx + 46, YC - 78, "BN · ReLU · dropout 0.25", 11.5)

lx, lw, lh = nx + 122, 30, 70                               # three log10 concentrations
for i, col in enumerate((DQ, TBZ, THI)):
    add(f'<rect x="{lx+i*(lw+3)}" y="{YC-lh/2}" width="{lw}" height="{lh}" fill="{col}"/>')
arrow(nx + 96, YC, lx - 4, YC)
txt(lx + (3*lw + 6)/2, YC + 96, "log₁₀(µM) × 3", 13, INK, "middle", 700)
txt(lx + (3*lw + 6)/2, YC + 122, "smooth-L1 on the per-map median", 11.5)

mo = lx + 3*lw + 6 + 76                                     # readout 2 — median pool
arrow(lx + 3*lw + 12, YC, mo - 10, YC)
txt((lx + 3*lw + 12 + mo - 10) / 2, YC - 13, "median pool", 11.5)
for i, (col, lab, f) in enumerate(((DQ, "DQ", .78), (TBZ, "TBZ", .52), (THI, "THI", 1.0))):
    bh = 60 * f
    add(f'<rect x="{mo+i*25}" y="{YC+30-bh:.1f}" width="18" height="{bh:.1f}" fill="{col}"/>')
    txt(mo + i*25 + 9, YC + 43, lab, 11, INK, "middle", 600)
add(f'<line x1="{mo-4}" y1="{YC+30}" x2="{mo+3*25}" y2="{YC+30}" stroke="{MUTE}" stroke-width="0.9"/>')
txt(mo + 34, YC - 46, "µM per map", 12, INK, "middle", 600)

# -------------------------------------------------------------------- footer
add(f'<line x1="24" y1="356" x2="{W-24}" y2="356" stroke="#d7dee6" stroke-width="1"/>')
txt(W/2, 374, "Leave-one-condition-out · 71 distinct compositions, the model refit once for each", 13, INK, "middle", 600)
txt(W/2, 388, "every map of a held-out composition leaves together, so a repeat measurement is never scored against its own twin", 11.5)
add('</svg>')

svg = "\n".join(s)
(OUT / "fig1_architecture.svg").write_text(svg, encoding="utf-8")
(OUT / "fig1_architecture.html").write_text(
    "<!doctype html><meta charset='utf-8'><style>html,body{margin:0;background:#fff}</style>\n" + svg,
    encoding="utf-8")
print("wrote", OUT / "fig1_architecture.svg")
