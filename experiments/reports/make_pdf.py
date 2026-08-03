# -*- coding: utf-8 -*-
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

pdfmetrics.registerFont(TTFont("KR", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("KRB", "C:/Windows/Fonts/malgunbd.ttf"))
DOCS = r"S:/Google Drive/내 드라이브/github/Mixture Classifier/docs"
OUT = os.path.join(DOCS, "update_2026-07-29.pdf")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="KRB", fontSize=17, leading=21, textColor=colors.HexColor("#1c2430"), spaceAfter=4)
SUB = ParagraphStyle("SUB", fontName="KR", fontSize=9.5, leading=13, textColor=colors.HexColor("#5b6673"), spaceAfter=10)
H2 = ParagraphStyle("H2", fontName="KRB", fontSize=12.5, leading=16, textColor=colors.HexColor("#1a73e8"), spaceBefore=12, spaceAfter=5)
BODY = ParagraphStyle("BODY", fontName="KR", fontSize=10, leading=15, textColor=colors.HexColor("#25303b"), spaceAfter=5)
CAP = ParagraphStyle("CAP", fontName="KR", fontSize=8.5, leading=12, textColor=colors.HexColor("#8b95a1"), spaceAfter=10, alignment=1)

def img(path, w=16*cm):
    iw, ih = ImageReader(path).getSize()
    return Image(path, width=w, height=w*ih/iw)

def tbl(data, colw):
    t = Table(data, colWidths=colw)
    t.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"KR"), ("FONTNAME",(0,0),(-1,0),"KRB"),
        ("FONTSIZE",(0,0),(-1,-1),9), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#25303b")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f2f4f6")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c7ccd2")),
        ("ALIGN",(1,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    return t

S = []
S += [Paragraph("SERS 혼합물 정량 — 업데이트", H1),
      Paragraph("2026-07-29 · Recovery 패널 개선 + physics-informed DL 정량 조사 (drawable SERS ink 프로젝트)", SUB)]

S += [Paragraph("1. Recovery 패널 개선 (UNMIXR 앱)", H2),
      Paragraph("• <b>Validate → Recovery</b> 개명, nav에서 Quantify(절대정량) 앞으로 이동 (조성 복원이 상위 단계).", BODY),
      Paragraph("• <b>VIP-band NNLS</b>: 각 물질의 cross-talk 최소 판별 밴드로만 조성을 분해하는 옵션 추가.", BODY),
      Paragraph("• <b>자기설명 export</b>: Recovery·Real·Quantify 모든 export에 README(무엇을·어떻게·결과) 동봉. 드리프트 삼각형은 export 시 제목·범례·읽기가이드 주입, 앱에서는 깔끔.", BODY),
      Paragraph("• <b>calibration CSV 오류 방지</b>: 잘못된 파일에 명확한 에러 + ratio-only 폴백(안 죽음).", BODY)]

S += [Paragraph("2. Physics-informed DL 혼합물 정량", H2),
      Paragraph("<b>목표</b>: 모든 조성·농도를 실험으로 측정할 수 없으므로, <b>pure 스펙트럼 + calibration + 소수 known-ratio 혼합물</b>만으로 임의 혼합물의 조성·농도를 예측하는 범용 도구. 물리 시뮬레이터(경쟁 Langmuir)가 학습 공간을 생성하고, DL이 역산을 학습, 실제 혼합물이 sim-to-real 보정.", BODY),
      Paragraph("Track A(스펙트럼→µM 회귀) · Track B(surface→solution 잔차 보정) 구현 (dl_quantify.py). 평가는 leave-one-map-out CV, 지표 = ½·Σ|예측−실제| 조성오차.", BODY)]

S += [Paragraph("주요 결과", H2),
      tbl([["체제","NNLS","선형 RF","PLS","full-spectrum DL"],
           ["Binary LOO (28)","27.5%","21.3%","26.5%","21.9%"],
           ["Binary→Ternary 일반화","45.2%","42.7%","26.6%","33.4%"],
           ["전체 35 LOO (ternary 포함)","31.0%","23.9%","26.5%","20.5%"]],
          [5.2*cm,2.7*cm,2.7*cm,2.4*cm,3*cm]),
      Spacer(1,4),
      Paragraph("• 쉬운 <b>binary는 선형 RF/PLS로 충분</b> — DL 동급. 하지만 <b>full 스펙트럼 입력 + 어려운(ternary) 예제를 학습에 포함</b>하면 DL이 전체 조성에서 1등(20.5%).", BODY),
      Paragraph("• <b>절대 µM</b>: 단일물질 calibration이 혼합 경쟁을 못 잡아 THI에서 폭발 → 현재 신뢰 불가. 조성 비율이 신뢰 가능한 결과.", BODY)]

S += [Paragraph("핵심: THI에 묻힌 성분 복원 (연구 목표)", H2),
      Paragraph("THI는 표면을 항상 지배 → 고전 NNLS는 묻힌 co-성분을 0으로 소실(검출 44%). 실제 합성 비율로 학습한 모델은 이를 되살림. <b>묻힌 성분 정보는 3개 B값이 아니라 full 스펙트럼 모양에 있음</b> — 그래서 full-spectrum 방법만 성공.", BODY),
      img(os.path.join(DOCS,"classify_triangle.png")),
      Paragraph("Ternary simplex 분류: 점 색=예측 정확도(초록=정확), 꼭짓점=recovery±SE. NNLS는 THI 287%/DQ 48%로 붕괴, DL은 셋 다 ~100%로 균형 복원.", CAP),
      img(os.path.join(DOCS,"buried_recovery.png")),
      Paragraph("THI 밑 묻힌 성분(빨간선=실제) 복원: NNLS는 0으로 소실, 학습 모델(PLS/DL)은 되살림.", CAP)]

S += [Paragraph("3. 빨간(hard) 케이스 개선 벤치마크", H2),
      Paragraph("잔여 오류 진단: 극단비율에서 <b>DQ↔TBZ 혼동</b> + THI 잔여 묻힘 (9/35, median은 12%).", BODY),
      tbl([["방법","전체 조성오차","red>0.35","묻힘 검출"],
           ["단일 DL","19.2%","7/35","82%"],
           ["(1) 앙상블 ×5 (bagging)","20.3%","7/35","88%"],
           ["(2) PLS+앙상블 하이브리드","22.0%","6/35","100%"]],
          [5.6*cm,3.6*cm,3*cm,3*cm]),
      Spacer(1,4),
      Paragraph("→ 앙상블·하이브리드는 <b>marginal</b> (검출↑, 정확도 소폭↓). 빨간 점은 분산이 아니라 체계적 hard case라 앙상블로 안 사라짐. <b>진짜 해법 = 데이터</b>(극단비율·다양 조성) + 불확실성 플래깅. 하이브리드는 '성분 절대 놓치면 안 될 때'에 유효.", BODY)]

S += [Paragraph("4. 결론 & 다음 단계", H2),
      Paragraph("• <b>방법론적 결론</b>: full-spectrum + 어려운 경쟁 예제 학습이면 DL이 고전기법을 넘음. 제약은 모델·픽셀 수가 아니라 <b>서로 다른 조성의 수</b>.", BODY),
      Paragraph("• <b>범용성</b>: 워크플로(pure+calib+소수 mixture→학습→예측)는 시스템 무관하게 재사용 가능. 학습된 모델은 시스템별(재보정 필요). SERS 재현성이 전이의 관건 — <b>drawable ink가 그 기반</b>을 제공.", BODY),
      Paragraph("• <b>다음</b>: (1) 독립 배치 blind 검증(내부 LOO를 넘어 real-sample 주장), (2) 농도축·조성 다양성 확대(DL 우위 확대), (3) 불확실성 플래깅.", BODY),
      Spacer(1,6),
      Paragraph("코드: dl_quantify.py · 상세: docs/dl_quantification_findings.md · 예측표: docs/mixture_predictions.csv", CAP)]

SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm).build(S)
print("wrote", OUT)
