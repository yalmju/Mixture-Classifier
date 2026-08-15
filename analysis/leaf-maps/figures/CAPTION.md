# 캡션 원문

약어 DQ · TBZ · THI 는 캡션 첫 등장에서 정식 화합물명으로 풀어 쓸 것 — 프로젝트에서
쓰는 표기와 맞춰야 한다. 특히 THI 는 문서마다 갈리니 확인하고 넣을 것.

---

## Fig_leaf_panel

**Mixture on leaves, before and after peeling.** (**a**) Raman maps of the same spot on
the same mixture-treated leaf, as applied and after the peel; rows are preparations and
columns are readouts. Both maps cover 20 × 20 pixels at 100 µm steps (1900 × 1900 µm),
so the two rows are directly comparable. The peel lifts the ink together with the
surface residue, and the ink reporter falls by 26 % between the rows; analyte removal
should therefore be read against that baseline (THI 61 %, DQ 46 %, TBZ 31 % of the
marker intensity removed, i.e. 0.53, 0.73 and 0.94 of the ink-normalised level retained). Each panel is plotted after asymmetric-least-squares baseline
removal (λ = 1 × 10⁵, p = 0.01). The first column is the integrated intensity of the ink
reporter band in the cellular-silent region (2100–2180 cm⁻¹), which reports how much ink
is probed in each pixel; the remaining columns are the marker-band intensity of each
analyte (DQ 1179/1521/1576, TBZ 779/1011/1549, THI 556/1138/1365 cm⁻¹), averaged over
the three bands (± 10 cm⁻¹) and divided by the ink reporter intensity of the same pixel,
so that pixel-to-pixel differences in substrate enhancement cancel. The colour bar under
each column applies to both rows, and grid and mapped side length are repeated beside
each row. (**b**) Median spectrum of the pixels of each map,
normalised to the integrated ink reporter band of the same map and offset vertically;
shading spans the interquartile range of the pixels within that one preparation, and
vertical lines mark the marker bands used in (a).

## Fig_leaf_maps

**Raman maps of the mixture series, before and after peeling.** Each panel is one map,
plotted after asymmetric-least-squares baseline removal (λ = 1 × 10⁵, p = 0.01); both
maps span 20 × 20 pixels at 100 µm steps (1900 × 1900 µm), as printed above each column.
**Top row**, integrated intensity of the ink reporter band in the
cellular-silent region (2100–2180 cm⁻¹), which reports how much ink is probed in each
pixel. **Lower rows**, the marker-band intensity of each analyte (DQ 1179/1521/1576,
TBZ 779/1011/1549, THI 556/1138/1365 cm⁻¹), averaged over the three bands (± 10 cm⁻¹)
and divided by the ink reporter intensity of the same pixel, so that pixel-to-pixel
differences in substrate enhancement cancel. The intensity scale is shared across the
two maps within a row (limits given by the colour bar) but not between rows.

## Fig_leaf_spectra

**Representative spectra of the mixture series.** (**a**) Median spectrum of the
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
each analyte; the control row is included because its 2–4 % sets the false-positive floor
of the threshold. All three analytes clear the threshold on the mixture-treated leaves
(DQ 53 %, TBZ 43 %, THI 81 %) and fall after the peel (10 %, 36 %, 48 %).

> 산포 표기 주의 — 음영과 픽셀 통계는 **한 조제 안**(10 × 10 = 100, 또는 20 × 20 = 400)
> 픽셀에서 나온 것이지 독립 반복이 아니다. 수식어 없는 `n = 100` 은 쓰지 말 것
> (`analysis/dq9-sus-reproducibility/README.md` §3). 조제간 재현성 오차막대가 필요하면
> 같은 조건을 한 번 더 조제해야 한다.
>
> peel 전후 감소를 쓸 때 — 두 맵은 같은 잎 같은 자리이고 면적도 같다(확인됨). 다만
> 필름이 잉크째 떼어내므로 **잉크는 두 맵 사이의 고정 기준물질이 아니다.** 마커/잉크
> 비가 내려간 것은 "남은 잉크 한 단위당 분석물이 줄었다"는 뜻이니, 크기를 말할 때는
> 절대 제거율(README §3-3)을 같이 쓸 것.
> `leavesmix2peel2` 는 면적이 1/18 이라 같은 계열로 못 쓴다 (CSV 에는 있다).
