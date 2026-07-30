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

S += [Paragraph("4. 절대농도(µM) — order-of-magnitude(자릿수)만 가능 (정직한 결론)", H2),
      Paragraph("<b>왜 단일성분 검량선(standard curve)은 혼합물에 틀리나</b>: 순수물질 검량선은 자기 혼자만의 흡착 θ=KC/(1+KC)를 따르지만, 혼합물에선 경쟁흡착 θ_i=K_iC_i/(1+Σ K_jC_j)라 같은 농도라도 다른 성분이 표면을 뺏어 커버리지가 달라짐. 그래서 <b>검량선 R²이 아무리 좋아도 혼합물 농도는 틀림</b>(모델 자체가 competition을 무시). → <b>혼합물로 학습해야</b> competition이 담김. DL에게 검량선 역할은 혼합물 데이터 자체이고, 단일성분 표준곡선은 선택적 보조.", BODY),
      Paragraph("Binary + Tertiary + Ratio_mix 합쳐 <b>69개 혼합물, 0.1~1000µM 전 범위</b> leave-one-out 검증:", BODY),
      tbl([["방법 / 지표","R²(log µM)","within-2×","within-order"],
           ["물리역산 (검량선 기반)","−0.6 ~ −19.6","6~20%","45~64%"],
           ["DL 회귀 (혼합물 학습)","−0.6 ~ −1.3","8~40%","45~80%"]],
          [4.2*cm,3.6*cm,3*cm,3*cm]),
      img(os.path.join(DOCS,"concentration_pred.png"), 14*cm),
      Paragraph("predicted vs true µM (log-log, 합친 세트). 대각선(이상값)을 잘 못 따라감.", CAP),
      Paragraph("<b>단일 측정 세트 내</b>에서는 DL이 <b>within-order ~70%</b>(median ~3×)로 준정량 가능. <b>세트를 합치면 붕괴</b>(48%)하는데, 이는 단순 gain 차이가 아님 — 기판 피크(~2100 cm⁻¹)가 세션 간 거의 동일(1.09×)하고, 그걸로 정규화해도 안 나아짐. <b>정밀 µM(2배 이내)는 물리적 한계</b>(competition + magnitude=gain 의존). 정직한 제공물 = <b>order-of-magnitude 준정량 농도</b>('~10 vs ~100 vs ~1000µM'), 묻힌 성분 포함 — 고전기법은 이걸 아예 잃음.", BODY)]

S += [Paragraph("5. 앱(UNMIXR)에 반영된 기능", H2),
      Paragraph("• <b>Recovery 탭</b>: response factor(응답계수)를 <b>평균±표준오차</b>로 표시(추정치라 불확실성 있음), 보정 solution ratio, 정확도 컬러 드리프트 삼각형, recovery±SE(미량<3% 제외). <b>DL predict</b>(leave-one-out 조성 + order-of-magnitude µM) · <b>DL explain</b>(attribution/permutation/ablation) 앱 내장. <b>Save DL model</b>(로드한 혼합물로 학습·저장).", BODY),
      Paragraph("• <b>Real data 탭</b>: <b>Load DL model</b>로 그 모델을 미지 시료 맵에 적용 → NNLS unmix 위에 <b>DL 조성 + 근사 µM</b> 표시.", BODY),
      Paragraph("• 범용성: pure+혼합물 워크플로는 시스템 무관 재사용, 모델은 시스템별 재학습. 잉크 재현성이 절대농도 전이의 관건.", BODY),
      Paragraph("• 다음: (1) 독립 배치 blind 검증, (2) 정밀 µM은 internal standard 필요.", BODY)]

# ---- Glossary (spell out abbreviations) ----
GL = ParagraphStyle("GL", fontName="KR", fontSize=8.5, leading=12, textColor=colors.HexColor("#5b6673"), spaceAfter=2)
S += [Paragraph("6. 용어 (약어 풀이)", H2),
      Paragraph("<b>DL</b> deep learning(딥러닝) · <b>MLP</b> multilayer perceptron(다층 퍼셉트론, 완전연결 신경망) · <b>1D-CNN</b> 1차원 합성곱 신경망 · <b>NNLS</b> non-negative least squares(비음수 최소제곱 unmixing) · <b>PLS</b> partial least squares regression · <b>SVR</b> support vector regression · <b>RF(RandomForest)</b> 랜덤포레스트", GL),
      Paragraph("<b>LOO</b> leave-one-out(하나 빼고 학습, 뺀 것으로 검증하는 교차검증) · <b>R²</b> 결정계수(1=완벽) · <b>RMSE</b> root-mean-square error · <b>ROC-AUC</b> receiver-operating-characteristic area under curve(검출 성능, 1=완벽·0.5=무작위) · <b>PR-AUC</b> precision-recall AUC · <b>F1</b> 정밀도·재현율 조화평균 · <b>SE</b> standard error(표준오차) · <b>SD</b> standard deviation(표준편차)", GL),
      Paragraph("<b>IG</b> Integrated Gradients(적분 기울기 — 각 파수의 예측 기여도) · <b>permutation importance</b> 특정 구간을 섞어 정확도 하락으로 중요도 측정 · <b>ligand ablation</b> 성분 마커밴드를 지워 예측 붕괴로 인과 확인 · <b>VIP</b> variable-importance-in-projection(판별 마커밴드) · <b>SERS</b> surface-enhanced Raman spectroscopy", GL),
      Paragraph("<b>µM</b> 마이크로몰(농도) · <b>order-of-magnitude / within-order</b> 자릿수 정확도(예측/실제가 10배 이내) · <b>within-2×</b> 2배 이내 · <b>recovery</b> 복원율=측정/실제×100% · <b>response factor(응답계수)</b> 단위 농도당 표면 신호 세기(THI가 크면 표면 지배) · <b>gain</b> 기판/장비 신호 배율 · <b>competition(경쟁흡착)</b> 성분들이 표면 자리를 다투는 것", GL),
      Spacer(1,4),
      Paragraph("코드: dl_quantify.py · dl_recovery.py · dl_explain.py · dl_model.py · experiments/dl/ · 상세: docs/dl_quantification_findings.md · docs/dl_interpretability.md", CAP)]

SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.6*cm).build(S)
print("wrote", OUT)
