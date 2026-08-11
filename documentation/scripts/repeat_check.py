"""내일용 — 6조합 재현 세트가 들어오면 이거 하나만 돌리면 됩니다.

    python3 repeat_check.py <새_폴더> [파일패턴]

    예)  python3 repeat_check.py /Volumes/Seungki/260806/tert-new2
         python3 repeat_check.py /Volumes/Seungki/260806/set2 "{n}-2.csv"

세 가지를 판정합니다:
  ① 배제항 잔차 1.93배가 시료 산포인가 모델 misfit인가   -> 33맵 설계를 할지 말지
  ② 잉크의 조성 의존성 (#6/#5 = 0.52배)이 재현되는가      -> 가설 A/B 확정
  ③ 분석물 정량이 기판을 건너가는가                        -> 절대 uM 가능 여부
"""
import sys, os, math, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ink6

names, wn_t, P = ink6.templates()
iDQ, iTBZ, iTHI = (names.index(s) for s in ("DQ", "TBZ", "THI"))
IK = names.index("INK")

COMP = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}          # DQ / TBZ / THI, 최종 uM
SET1 = "/Volumes/Seungki/260805/tert-new/{n}-1.csv"


def read_set(pattern):
    out = {}
    for n in COMP:
        p = pattern.format(n=n)
        if not os.path.exists(p):
            cand = sorted(glob.glob(os.path.join(os.path.dirname(p), f"{n}-*.csv")))
            cand = [c for c in cand if "(bg)" not in c and "_corrected" not in c]
            if not cand:
                print(f"  !! #{n} 파일 못 찾음: {p}")
                continue
            p = cand[0]
        a, _ = ink6.ink_of(p, names, wn_t, P)
        out[n] = a
    return out


def summarise(A):
    r = {}
    for n, a in A.items():
        dq, tb, th = COMP[n]
        r[n] = dict(
            dq=a[iDQ], tbz=a[iTBZ], thi=a[iTHI], ink=a[IK], tot_sig=a.sum(),
            frac=a[IK] / a.sum(),
            R_dq=(a[iDQ] / a[iTHI]) / (dq / th),      # Langmuir 상수여야 하는 값
            R_tbz=(a[iTBZ] / a[iTHI]) / (tb / th),
        )
    return r


if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
newdir = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else "{n}-2.csv"
if not os.path.isabs(pat):
    pat = os.path.join(newdir, pat)

print("=== set 1 (2026-08-05) ===")
s1 = summarise(read_set(SET1))
print("=== set 2 (재현) ===")
s2 = summarise(read_set(pat))
common = sorted(set(s1) & set(s2))
if not common:
    print("겹치는 맵이 없습니다."); sys.exit(1)

print(f"\n{'#':>3} {'조성':>10} | {'R_DQ/R_THI':>22} | {'R_TBZ/R_THI':>22} | {'ink frac':>16}")
print(f"{'':>3} {'':>10} | {'set1':>9}{'set2':>9}{'배':>4} | {'set1':>9}{'set2':>9}{'배':>4} | {'set1':>7}{'set2':>7}")
d_dq, d_tbz, d_ink = [], [], []
for n in common:
    a, b = s1[n], s2[n]
    fd, ft = b["R_dq"] / a["R_dq"], b["R_tbz"] / a["R_tbz"]
    d_dq.append(math.log10(fd)); d_tbz.append(math.log10(ft))
    d_ink.append(math.log10(b["frac"] / a["frac"]))
    print(f"{n:>3} {'/'.join(str(v) for v in COMP[n]):>10} | "
          f"{a['R_dq']:9.3f}{b['R_dq']:9.3f}{fd:4.1f} | "
          f"{a['R_tbz']:9.3f}{b['R_tbz']:9.3f}{ft:4.1f} | "
          f"{a['frac']:7.3f}{b['frac']:7.3f}")

print("\n" + "=" * 78)
print("① 재현성 — 같은 조성을 다시 만들었을 때 게인-무관 비가 얼마나 흔들리는가")
for lbl, d in (("DQ/THI", d_dq), ("TBZ/THI", d_tbz)):
    sd = np.std(d, ddof=1)
    print(f"   {lbl:8} 재현 산포 SD = {sd:.3f} dex ({10**sd:.2f}배)   "
          f"[배제항 적합 잔차 = 0.286 dex (1.93배)]")
sd_dq = np.std(d_dq, ddof=1)
print("   판정:")
if sd_dq >= 0.20:
    print(f"     -> {10**sd_dq:.2f}배. 잔차 1.93배의 대부분이 시료 산포다.")
    print("        배제항을 아무리 정확히 구해도 맵 하나의 조성 오차는 안 줄어든다.")
    print("        33맵 설계는 값어치가 없다. 여기서 멈추는 게 맞다.")
elif sd_dq <= 0.12:
    print(f"     -> {10**sd_dq:.2f}배. 시료는 잘 재현된다. 잔차 1.93배는 모델 misfit이다.")
    print("        배제항의 진짜 모양을 찾으면 조성 오차가 내려간다.")
    print("        총농도 9-144 uM 양끝에 몰아 33맵 설계할 값어치가 있다.")
else:
    print(f"     -> {10**sd_dq:.2f}배. 애매하다 (0.12~0.20 dex). 반복을 2세트 더 넣어 SD를 확정할 것.")

print("\n② 잉크의 조성 의존성 — #6(THI 편중) / #5(기준), 둘 다 총 72 uM")
for tag, s in (("set1", s1), ("set2", s2)):
    if 5 in s and 6 in s:
        print(f"   {tag}: 잉크분율 비 = {s[6]['frac']/s[5]['frac']:.2f}배    "
              f"(가설 B 예측 0.54배 / 가설 A 예측 1.00배)")
print("   판정: set2도 0.4~0.7배면 -> 가설 B 확정, 잉크 총농도 판독에 조성 보정 필요.")
print("         set2가 1.0배 근처면 -> set1의 0.52배는 기판 산포였다, 가설 A 유지.")

print("\n③ 기판 전이성 — set1으로 맞춘 응답계수로 set2의 조성을 예측")
S1 = np.array([[s1[n]["dq"], s1[n]["tbz"], s1[n]["thi"]] for n in common])
S2 = np.array([[s2[n]["dq"], s2[n]["tbz"], s2[n]["thi"]] for n in common])
C = np.array([COMP[n] for n in common], float)
r = (np.log10(S1) - np.log10(C)).mean(axis=0)
r -= r.mean()
errs = []
for k, n in enumerate(common):
    est = S2[k] / 10 ** r
    est = est / est.sum()
    true = C[k] / C[k].sum()
    e = np.abs(est - true).sum() / 2 * 100
    errs.append(e)
    print(f"   #{n} true " + "/".join(f"{v*100:4.1f}" for v in true) +
          "   est " + "/".join(f"{v*100:4.1f}" for v in est) + f"   err {e:4.1f}%")
print(f"   평균 {np.mean(errs):.1f}%   [set1 내부 LOO = 19.2%]")
print("   판정: 25% 이내면 기판을 건너간다. 40% 넘으면 절대 uM은 기판마다 재검량 필요.")

print("\n④ 성분별 절대농도 오차 — §8.8 등급표(THI 1.3 / TBZ 1.6 / DQ 2.6배)가 재현되는가")
SUB = ["DQ", "TBZ", "THI"]
L = np.log10(S1) - np.log10(C)
g_tr = L.mean()
rr = L.mean(axis=0) - L.mean()
fold = 10 ** np.abs(np.log10((S2 / 10 ** (g_tr + rr)) / C))
for k, n in enumerate(common):
    print(f"   #{n} " + "  ".join(f"{SUB[j]}:{fold[k, j]:5.2f}x" for j in range(3)))
REF = {"DQ": 2.6, "TBZ": 1.6, "THI": 1.3}
print("   중앙값: " + "   ".join(
    f"{SUB[j]} {np.median(fold[:, j]):.2f}배 (set1 {REF[SUB[j]]:.1f}배)" for j in range(3)))
print("   판정: THI가 1.5배 이내로 재현되면 'THI 정량'을 그대로 주장할 수 있다.")
print("         THI가 2배를 넘으면 정량 주장도 접고 셋 다 반정량으로 내려야 한다.")

gain = np.array([s2[n]["tot_sig"] / s1[n]["tot_sig"] for n in common])
print(f"\n   참고 — 기판 게인 비 (set2/set1): " + "  ".join(f"#{n}:{g:.2f}" for n, g in zip(common, gain)) +
      f"   중앙 {np.median(gain):.2f}배")
