# -*- coding: utf-8 -*-
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

pdfmetrics.registerFont(TTFont("KR", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("KRB", "C:/Windows/Fonts/malgunbd.ttf"))
DOCS = r"S:/Google Drive/내 드라이브/github/Mixture Classifier/docs"
OUT = os.path.join(DOCS, "update_2026-07-30.pdf")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="KRB", fontSize=17, leading=21, textColor=colors.HexColor("#1c2430"), spaceAfter=4)
SUB = ParagraphStyle("SUB", fontName="KR", fontSize=9.5, leading=13, textColor=colors.HexColor("#5b6673"), spaceAfter=10)
H2 = ParagraphStyle("H2", fontName="KRB", fontSize=12.5, leading=16, textColor=colors.HexColor("#1a73e8"), spaceBefore=12, spaceAfter=5)
BODY = ParagraphStyle("BODY", fontName="KR", fontSize=10, leading=15, textColor=colors.HexColor("#25303b"), spaceAfter=5)
CAP = ParagraphStyle("CAP", fontName="KR", fontSize=8.5, leading=12, textColor=colors.HexColor("#8b95a1"), spaceAfter=10, alignment=1)

def img(path, w=16*cm):
    iw, ih = ImageReader(path).getSize(); return Image(path, width=w, height=w*ih/iw)
def tbl(data, colw):
    t = Table(data, colWidths=colw)
    t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"KR"),("FONTNAME",(0,0),(-1,0),"KRB"),
        ("FONTSIZE",(0,0),(-1,-1),8.5),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#25303b")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f2f4f6")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c7ccd2")),("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    return t

S = []
S += [Paragraph("SERS 혼합물 정량 — 업데이트 2", H1),
      Paragraph("2026-07-30 · DL 모델 벤치마크·해석성·농도예측 + Recovery 앱 개선 (drawable SERS ink 프로젝트)", SUB)]

S += [Paragraph("1. Recovery 앱 개선 · 버그 수정", H2),
      Paragraph("• <b>recovery% 폭발 수정</b>: ternary 미량 성분(true 1%)에서 pred/true가 폭발(2000%+)하던 것 → 3% 미만 성분을 recovery 지표에서 제외. DL 기준 DQ103/TBZ102/THI158% 로 정상화.", BODY),
      Paragraph("• <b>드리프트 삼각형</b>: 예측점을 정확도 컬러맵(초록=정확)으로, 범례 겹침 해소, ✓ 글리프 깨짐 수정. Composition-maps 패널 제거.", BODY),
      Paragraph("• <b>UX</b>: 입력부 접이식(완료 시 자동 접힘) · 진행바 + 파일 카운트([5/35] … N left) · 작업표시줄 아이콘(AppUserModelID).", BODY),
      Paragraph("• <b>DL 통합</b>: 'DL predict' 체크박스(조성 뷰를 물리기반 DL LOO로) + 'DL explain' 버튼(해석성). export README에 DL 사용 여부 명시.", BODY)]

S += [Paragraph("2. 모델 벤치마크 — 왜 MLP인가 (35-map LOO, 3 seed ±SD)", H2),
      tbl([["방법","조성오차↓","RMSE↓","R²↑","ROC-AUC↑","F1↑","묻힘↓"],
           ["NNLS","31.0%","0.322","0.13","0.911","0.82","0.35"],
           ["linear RF","23.9%","0.249","0.48","0.917","0.87","0.27"],
           ["PLS","26.5%","0.233","0.55","0.877","0.84","0.37"],
           ["SVR","26.0%","0.221","0.59","0.877","0.77","0.32"],
           ["RandomForest","27.9%","0.225","0.58","0.877","0.82","0.40"],
           ["1D-CNN","29.5%±4.2","0.284","0.32","0.801","0.80","0.44"],
           ["MLP-DL (ours)","19.1%±0.6","0.193","0.69","0.912","0.88","0.26"]],
          [3.4*cm]+[2*cm]*6),
      Spacer(1,3),
      Paragraph("MLP-DL이 7개 지표 중 6개 1위, AUC 공동 1위. 검출(AUC)은 다 양호 → 변별점은 정량 정확도(조성오차 19.1% vs 24%+, R² 0.69). 1D-CNN은 파라미터 과다로 최악·불안정(±4.2%).", BODY),
      img(os.path.join(DOCS,"model_benchmark.png"), 15*cm),
      Paragraph("좌: 정량오차(낮을수록) · 우: 검출 ROC-AUC(높을수록). MLP-DL 최저 오차 + 최고 AUC.", CAP),
      img(os.path.join(DOCS,"roc_curves.png"), 9.5*cm),
      Paragraph("검출 ROC (성분 present/absent, micro-avg). MLP-DL AUC 0.917 최고.", CAP)]

S += [Paragraph("3. 해석성 — 화학적으로 타당한 모델인가", H2),
      Paragraph("세 방법이 <b>같은 마커밴드</b>를 가리킴 = 스퓨리어스가 아닌 실제 SERS 지문에 의존.", BODY),
      img(os.path.join(DOCS,"dl_interpretability.png"), 16*cm),
      Paragraph("(1) IG attribution이 각 성분 VIP밴드에 집중 (THI 1368/550 날카로움) · (2) permutation importance도 같은 밴드 · (3) ligand ablation 시 예측 붕괴 (THI −100%, TBZ −38%, DQ −22%).", CAP)]

S += [Paragraph("4. 절대농도(µM) 예측 — 실패 (정직한 결과)", H2),
      Paragraph("Binary + Tertiary + Ratio_mix 합쳐 <b>69개 혼합물, 0.1~1000µM 전 범위</b> 검증.", BODY),
      tbl([["방법 / 지표","R²(log) DQ/TBZ/THI","within-2×","within-order"],
           ["물리역산","−3.8 / −9.7 / −0.6","6~20%","45~64%"],
           ["DL 회귀","−1.2 / −1.1 / −1.1","8~16%","45~49%"]],
          [3.6*cm,4.8*cm,3*cm,3*cm]),
      img(os.path.join(DOCS,"concentration_pred.png"), 15*cm),
      Paragraph("predicted vs true µM (log-log). 두 방법 다 대각선을 못 따라감.", CAP),
      Paragraph("<b>두 방법 다 실패</b> (R²<0, within-2× 6~20%). DL도 물리역산을 못 이김. <b>원인</b>: 절대 µM는 신호 크기(magnitude)=gain에 의존하는데, 서로 다른 측정 세션(Binary/Ratio_mix)은 gain이 달라 magnitude→µM 매핑이 불일치. 조성 <b>비율</b>은 gain-무관이라 잘 되지만(R² 0.69), 절대 µM는 <b>internal standard</b> 또는 단일 세션 gain 일관성이 필요. (drawable ink 재현성이 관건)", BODY)]

S += [Paragraph("5. 종합 · 다음", H2),
      Paragraph("• 조성 정량: <b>full-spectrum MLP</b>가 고전기법·CNN 모두 압도, 해석성까지 확보. THI 지배 혼합물의 묻힌 성분 복원.", BODY),
      Paragraph("• 범용성: pure+calibration+소수 mixture 워크플로는 시스템 무관 재사용, 모델은 시스템별 재보정. ink 재현성이 전이 담보.", BODY),
      Paragraph("• 다음: (1) Ratio_mix 농도축으로 µM·경쟁 재검증, (2) 독립 배치 blind 검증, (3) internal standard.", BODY),
      Spacer(1,5),
      Paragraph("코드: dl_quantify.py · dl_recovery.py · dl_explain.py · experiments/dl/ · 상세: docs/dl_quantification_findings.md · docs/dl_interpretability.md", CAP)]

SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.6*cm).build(S)
print("wrote", OUT)
