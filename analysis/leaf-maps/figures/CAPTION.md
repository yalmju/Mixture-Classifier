# 캡션 원문

약어 DQ · TBZ · THI 는 캡션 첫 등장에서 정식 화합물명으로 풀어 쓸 것 — 프로젝트에서
쓰는 표기와 맞춰야 한다. 특히 THI 는 문서마다 갈리니 확인하고 넣을 것.

---

## Fig_leaf_panel

**Peeled mixture series — maps and representative spectra.** (**a**) Raman maps of the
ink-only leaf control and two peels of the mixture-treated leaves; rows are preparations
and columns are readouts. Each panel is plotted after asymmetric-least-squares baseline
removal (λ = 1 × 10⁵, p = 0.01). The first column is the integrated intensity of the ink
reporter band in the cellular-silent region (2100–2180 cm⁻¹), which reports how much ink
is probed in each pixel; the remaining columns are the marker-band intensity of each
analyte (DQ 1179/1521/1576, TBZ 779/1011/1549, THI 556/1138/1365 cm⁻¹), averaged over
the three bands (± 10 cm⁻¹) and divided by the ink reporter intensity of the same pixel,
so that pixel-to-pixel differences in substrate enhancement cancel. The colour bar under
each column applies to all three rows. Grid and mapped side length are given beside each
row — panels drawn at the same size cover 1900 µm for the first peel and 450 µm for the
control and the second peel. (**b**) Median spectrum of the pixels of each map,
normalised to the integrated ink reporter band of the same map and offset vertically;
shading spans the interquartile range of the pixels within that one preparation, and
vertical lines mark the marker bands used in (a).

## Fig_leaf_maps

**Raman maps of the peeled mixture series.** Each panel is one map, plotted after
asymmetric-least-squares baseline removal (λ = 1 × 10⁵, p = 0.01); the grid and the side
length of the mapped area are given above each column, so panels drawn at the same size
cover different areas — the peeled mixture map spans 1900 µm at 100 µm steps while the
control and the second peel span 450 µm at 50 µm steps. Columns are the ink-only leaf
control and two peels of the mixture-treated leaves. **Top row**, integrated intensity of the ink reporter band in the
cellular-silent region (2100–2180 cm⁻¹), which reports how much ink is probed in each
pixel. **Lower rows**, the marker-band intensity of each analyte (DQ 1179/1521/1576,
TBZ 779/1011/1549, THI 556/1138/1365 cm⁻¹), averaged over the three bands (± 10 cm⁻¹)
and divided by the ink reporter intensity of the same pixel, so that pixel-to-pixel
differences in substrate enhancement cancel. The intensity scale is shared across the
three maps within a row (limits given by the colour bar) but not between rows.

## Fig_leaf_spectra

**Representative spectra of the peeled mixture series.** (**a**) Median spectrum of the
pixels of each map in the fingerprint region, normalised to the integrated ink reporter
band (2100–2180 cm⁻¹) of the same map and offset vertically for display; shading spans
the interquartile range of the pixels within that one preparation. Vertical lines mark
the marker bands used in Fig_leaf_maps (colours as in the legend). (**b**) The same
spectra in the silent region, plotted as raw counts; the dashed line marks 2137 cm⁻¹.
The reporter band is present in every preparation, which is what makes the ratio readout
in (a) and in Fig_leaf_maps meaningful.

## Fig_leaf_contrast

**Analyte signal above the ink-only leaf control.** The control is a leaf carrying the
ink alone, mapped under the same conditions; its own pixels provide the blank in both
panels. (**a**) Ink-normalised median spectrum of each preparation minus that of the
control, offset vertically; thin horizontal lines mark zero for each trace and vertical
lines mark the marker bands. (**b**) Percentage of pixels in each map whose
ink-normalised marker intensity exceeds the control mean + 3 SD, computed separately for
each analyte. The control itself returns 2–4 %, which sets the false-positive floor of
the threshold. All three analytes clear the threshold on the peeled mixture leaves
(DQ 10 %, TBZ 36 %, THI 48 %) and on the second peel (19 %, 22 %, 29 %).

> 산포 표기 주의 — 음영과 픽셀 통계는 **한 조제 안**(10 × 10 = 100, 또는 20 × 20 = 400)
> 픽셀에서 나온 것이지 독립 반복이 아니다. 수식어 없는 `n = 100` 은 쓰지 말 것
> (`analysis/dq9-sus-reproducibility/README.md` §3). 조제간 재현성 오차막대가 필요하면
> 같은 조건을 한 번 더 조제해야 한다.
>
> 두 peel 맵의 검출률을 나란히 쓸 때 — `leavesmix2peel` 은 1900 µm 변, `leavesmix2peel2`
> 는 450 µm 변이라 뒤엣것이 보는 면적이 1/18 이다. 48% → 29% 를 "한 번 더 벗겨서 줄었다"
> 로 읽으면 안 된다. 캡션에 면적을 밝혀 두었고, 주장하려면 같은 면적으로 다시 잴 것.
