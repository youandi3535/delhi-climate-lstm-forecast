# 🌡️ Delhi 기후 데이터 — LSTM 시계열 예측

Delhi(뉴델리)의 일별 기후 데이터를 활용해 **다음날 평균 기온(meantemp)을 예측**하는 LSTM 딥러닝 파이프라인입니다.  
데이터 탐색(EDA)부터 전처리, 모델 학습, 평가, 1년 미래 예측까지 전 과정을 단일 노트북으로 구성했습니다.

---

## 📁 프로젝트 구조

```
├── lec06_딥러닝_공모전_시계열_날씨_최종.ipynb   # 메인 분석 노트북
└── dataset/
    └── DailyDelhiClimate/
        ├── DailyDelhiClimateTrain.csv           # 학습 데이터 (2013~2016)
        └── DailyDelhiClimateTest.csv            # 평가 데이터 (2017)
```

---

## 📊 데이터셋

| 항목 | 내용 |
|---|---|
| 출처 | [Kaggle — Daily Climate time series data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) |
| Train 기간 | 2013-01-01 ~ 2016-12-31 |
| Test 기간 | 2017-01-01 ~ 2017-04-24 |
| 레코드 수 | Train 1,462행 / Test 114행 |
| 기록 주기 | 매일 (주말 포함) |

### 컬럼 설명

| 컬럼 | 설명 | 단위 |
|---|---|---|
| `date` | 날짜 | - |
| `meantemp` | 일평균 기온 **(예측 타겟)** | °C |
| `humidity` | 습도 | % |
| `wind_speed` | 풍속 | km/h |
| `meanpressure` | 평균 기압 | hPa |

---

## 🔧 파이프라인 구성

### 1. 데이터 품질 점검 (1-2 ~ 1-11)
- 컬럼 구조 및 데이터 타입 확인
- 결측치 / 중복 행 / 날짜 연속성 검사
- IQR × 1.5 기반 이상치 탐지
- 연도별 관측일 수 / 일별 기온 변화량 분석

### 2. 전처리 (2-1 ~ 2-4)
- **결측치**: 선형 보간(linear interpolation) + 경계값 bfill
- **기압 이상치**: 물리적 허용 범위(950~1050 hPa) 초과값 클리핑
- **파생 변수 생성**:

| 변수 | 설명 |
|---|---|
| `MA_7`, `MA_30` | 7일·30일 이동평균 (단기·중기 추세) |
| `Temp_Change` | 전날 대비 기온 변화량 |
| `Month`, `DayOfYear` | 월 / 연중 일수 |
| `Season` | 계절 번호 (1:봄 2:여름 3:가을 4:겨울) |
| `month_sin/cos` | 월 주기 신호 (원형 인코딩) |
| `day_sin/cos` | 연중 일수 주기 신호 (원형 인코딩) |

### 3. EDA (3-1 ~ 3-8)
- 전체 기간 기온 시계열 + 이동평균 시각화
- 4개 기후 변수 시계열 비교
- 월별·계절별 기온 분포 (BoxPlot / 히스토그램)
- 연도별 기온 추세
- 변수 간 상관관계 히트맵
- ADF 정상성 검정
- ACF / PACF 자기상관 분석

### 4. 피처 엔지니어링 (4-1 ~ 4-3)
- **피처**: `meantemp`, `humidity`, `wind_speed`, `meanpressure`, `MA_7`, `MA_30`, `Season`, `Month`, `DayOfYear`, `month_sin/cos`, `day_sin/cos` (총 13개)
- **스케일링**: MinMaxScaler — Train으로만 fit, Test는 transform만 적용 (Data Leakage 방지)
- **슬라이딩 윈도우**: `WINDOW_SIZE=30` — 과거 30일 → 다음날 기온 예측

### 5. LSTM 모델 (5-1 ~ 5-5)

```
입력: (30일, 13개 피처)
  └─ LSTM(64) + Dropout(0.2)
  └─ LSTM(32) + Dropout(0.2)
  └─ Dense(1)
출력: 다음날 meantemp
```

| 설정 | 값 |
|---|---|
| Optimizer | Adam |
| Loss | MSE |
| Metrics | MAE |
| Max Epochs | 100 |
| Batch Size | 32 |
| EarlyStopping | val_loss 기준, patience=15 |
| 가중치 초기화 | GlorotNormal(seed=42) |

### 6. 모델 평가 (6-1 ~ 6-5)
- **성능 지표**: RMSE, MAE, MAPE, RMSLE, R²
- **시각화**: 예측 vs 실제 시계열, 잔차(Residual), 산점도

### 7. 미래 예측 — 1년 (7-1 ~ 7-5)
두 가지 방법으로 365일 예측:

| 방법 | 설명 |
|---|---|
| **방법 A** 롤링 예측 | 이전 예측값을 다음 입력으로 사용 (단일 결정론적 예측) |
| **방법 B** 부트스트랩 잔차 | Test 잔차를 노이즈로 재활용해 30개 시나리오 생성 → 90% 예측 구간 추정 |

수평 수렴(flat line) 방지를 위해 `humidity`, `wind_speed`, `meanpressure`는 월평균으로 보정하고, 계절 주기 신호(`month_sin/cos`, `day_sin/cos`)는 미래 날짜 기준으로 정확히 계산해 주입합니다.

---

## ⚙️ 환경 설정

### 요구 사항

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow
```

### 권장 버전

| 라이브러리 | 권장 버전 |
|---|---|
| Python | 3.9 이상 |
| TensorFlow | 2.10 이상 |
| scikit-learn | 1.0 이상 |
| pandas | 1.4 이상 |

### 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. 데이터셋 준비
# dataset/DailyDelhiClimate/ 폴더에 CSV 파일 위치시키기

# 3. 노트북 실행
jupyter notebook lec06_딥러닝_공모전_시계열_날씨_최종.ipynb
```

---

## 📌 주요 설계 결정

**Data Leakage 방지**  
스케일러를 Train 데이터로만 `fit`하고 Test에는 `transform`만 적용합니다. Train/Val 분리도 시간순(앞 80% / 뒤 20%)으로 처리합니다.

**수평 수렴 방지**  
단일 변수(meantemp)만 롤링하면 오차가 누적돼 예측이 평탄해지는 문제가 발생합니다. 나머지 피처를 월평균으로 보정하고 sin/cos 주기 신호를 미래 날짜 기준으로 직접 계산해 주입합니다.

**원형 인코딩 (Cyclic Encoding)**  
`Month`를 정수로 쓰면 12월과 1월이 멀게 표현됩니다. `sin/cos` 쌍으로 변환하면 12월과 1월이 원 위에서 가깝게 위치해 계절 연속성을 자연스럽게 학습합니다.

---

## 📄 라이선스

MIT License
