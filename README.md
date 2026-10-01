# 이벤트 인지형 서울 주요 거점 교통 상황 예측 및 드리프트 대응 서비스

공연·경기·집회처럼 **미리 공개되는 이벤트 일정**을 "알려진 미래 정보"로 넣어, 서울 이벤트 거점
(광화문·시청, 여의도, 잠실, 상암 / 선택: 고척)의 **방향별 향후 6시간 통행속도와 평소 대비 추가 소요시간**을
예측하는 LSTM 서비스입니다. 운영 중에는 **알려진 이벤트·일시적 돌발과 예정에 없던 지속적 변화(도로 공사 등)를
구분**해, 후자만 재학습 대상으로 삼습니다.

HAIC 주가 예측 실습의 뼈대(FastAPI 서빙 → MLflow 게이트·승격 → 드리프트 감지 → warm-start fine-tuning →
aiops.log → Docker)를 그대로 쓰고, 데이터 계층과 모델 입력을 교통 예측에 맞게 바꿨습니다.

## 핵심 설계

| 항목 | 결정 | 이유 |
|---|---|---|
| 예측 대상 | **corridor(거점 도로 × 방향)** 의 통행속도 → 소요시간 | 경기·공연 종료 후 귀가 정체는 한 방향에 몰림. 양방향 평균은 신호를 희석 (실험 ④) |
| 속도 집계 | 링크별 속도를 **거리 가중 조화평균** (= Σ길이 / Σ(길이/속도)) | 실제 통과 시간 기준. 링크 길이는 TOPIS 엑셀 거리 / 링크 정보 API |
| 교통량 | 지점의 **같은 방향(유입/유출)** 만 입력으로. 0 = 결측 + 결측 마스크 | 가까운 지점이라도 반대 방향이면 부적절. 집회 시 교통량↓·정체↑ 이므로 예측 대상 아님 |
| 예보 범위 | 매시 발행, 향후 6시간 | 시작 전 도착·종료 후 귀가 정체를 몇 시간 앞서 반영 |
| 입력 | 과거 24h [속도, 교통량, 캘린더, 이벤트] + 예측 대상 시각의 [캘린더, 이벤트, corridor] | 이벤트·공휴일은 미리 알려진 미래 정보 |
| 정보 누수 방지 | 일정은 **실제 공개(게시) 시각** 이후에만, **취소는 취소 공지 시각** 이후에만 반영. 관중 수 대신 수용·신고 인원 | 학습과 운영이 같은 정보로 동작 |
| 이벤트 품질 | 회차 시각을 확정하지 못한 자동 수집 일정은 **검토 대기(review)** → 예측 미사용 | "공연 기간 ≠ 매일 공연" - 가짜 이벤트가 평상 시간 오차를 키움 |
| 드리프트 판정 | 최근 72h **평상 시간대** RMSE / 기준(val 구간) RMSE > 1.5 | 이벤트·공휴일·**일시적 돌발(사고)** 제외, **공사**는 원인으로 표시 |

## 사용 데이터

| 구분 | 출처 | 받는 방법 | 쓰임 | 상태 (2026-10-01) |
|---|---|---|---|---|
| 과거 속도 | TOPIS 자료실 "서울시 차량통행속도" 월별 엑셀 (일자 × 링크 × 01~24시, 도로명·시점/종점·방향·거리 포함) | `scripts/download_topis.py` (키 불필요) | 학습 입력·정답, corridor 링크 자동 매핑 | **적재 완료** 2025-01 ~ 2026-09, 34개 링크 |
| 과거 교통량 | TOPIS 자료실 "서울시 교통량 조사자료" 월별 엑셀 (지점 × 유입/유출 × 0~23시 + 지점 좌표·방향 설명 시트) | 같은 스크립트 | 입력 피처 (같은 도로·같은 방향 지점만) | **적재 완료** 2025-01 ~ 2026-08. 해당 지점은 세종대로 A-17 뿐 → 나머지 corridor 는 결측 마스크 |
| 경기 일정 | KBO 홈페이지 일정 (구장·시각·우천취소), K리그 홈페이지 일정 (FC서울 홈) | `scripts/import_events.py sports` (키 불필요) | 이벤트 (잠실·고척·상암) | **적재 완료** 507경기 (우천취소 22) |
| 운영 속도·교통량·링크·돌발 | 서울 열린데이터광장 `TrafficInfo`·`VolInfo`·`LinkInfo`·`SpotInfo`·`AccInfo`·`AccMainCode` | `scripts/collect_hourly.py` (`SEOUL_API_KEY`) | 운영 수집·드리프트 판정 보조 | 샘플 키로 호출·필드 확인 완료 (xml 전용, 시각 HHMM 형식 반영). 운영 수집에는 정식 키 필요 |
| 공연 | KOPIS 공연시설·공연목록·상세 | `scripts/import_events.py kopis` (`KOPIS_API_KEY`) | 대시보드 표시·드리프트 판정 제외 (피처는 실험 ⑤ 결과로 보류) | **적재 완료** 168개 공연 → 714회차. 거점 반경 1.5km·3천 석 이상 공연장 자동 선정(세종대극장, 잠실 주경기장·실내체육관·학생체육관, 서울월드컵경기장, 문화비축기지, 고척돔), 규모 = 실제 공연한 홀 객석 |
| 공공행사 | 서울시 문화행사 `culturalEventInfo` | `import_events.py culture` (`SEOUL_API_KEY`) | 대시보드 표시·드리프트 판정 제외 | **적재 완료** 279건 - 반경 안 5,559건 중 도로에 붙은 광장·공원(광화문광장·서울광장·여의도공원 등) 행사만 |
| 집회 | 서울경찰청 "오늘의 집회/시위" | CSV 템플릿 | 이벤트 피처 (csv) | 과거 게시 시각 자동 수집 불가 → 미수집 |
| 캘린더 | `config/holidays_kr.yaml` | - | 입력 피처, 드리프트 제외 | 사용 중 |
| 검증용 | `data/synthetic.py` 합성 데이터 | `TRAFFIC_DB=data/traffic_synthetic.db` | 파이프라인·실험 검증 | 실데이터와 DB·MLflow 분리 |

**가정 (리포트에 함께 표기)**
- 과거 경기 일정의 공개 시각은 남아 있지 않아 "경기 7일 전 공개"로 가정합니다(정규시즌 일정은 시즌 전에 발표되므로 보수적).
- 우천취소 공지 시각은 "경기 시작 2시간 전"으로 가정합니다.
- 경기 규모는 관중 수가 아니라 경기장 수용 인원입니다(관중 수는 경기 후에 확정되므로 정보 누수).
- KOPIS 공개 시각은 첫 공연 전 마지막 갱신 시각(`updatedate`)입니다. 첫 공연 이후에 갱신된 공연(33건)만 30일 전 공개로 가정합니다.
- KOPIS 공연 시간이 비어 있고 축제이거나 낮(15시 이전)에 시작하면 종일 행사로 보고 22시 종료로 추정합니다.

**이벤트 출처별 쓰임 (실험 ⑤, 시드 3개, 같은 이벤트 시간 기준)**: 경기만 피처로 쓸 때 이벤트 시간 RMSE 1.611 ± 0.020 으로
가장 좋았고(이벤트 정보 없음 1.657), KOPIS 를 더하면 1.664~1.682 로 나빠졌습니다. 그래서 `train.event_sources` 로 피처에는
경기·수작업 등록 일정만 쓰고, 공연·문화행사는 대시보드 표시와 드리프트 판정 제외에만 씁니다.

## 실데이터 적재 (모두 자동, 사람 판단 단계 없음)

```bash
python scripts/download_topis.py 2025-01 2026-09                 # TOPIS 월별 속도·교통량 엑셀 → data/raw/
python scripts/map_corridors.py data/raw/speed_2026_09.xlsx data/raw/volume_2026_08.xlsx
#   hubs.yaml 의 도로명·방향(상행/하행)·기준 노드(anchor)로 링크 사슬을 잇고, 같은 도로·같은 방향 교통량 지점만 골라
#   config/corridor_links.yaml 생성 + reports/corridor_mapping.md (링크 목록·근거)
python scripts/import_topis_excel.py speed  data/raw/speed_*.xlsx
python scripts/import_topis_excel.py volume data/raw/volume_*.xlsx
python scripts/import_events.py sports 20250101 20261031          # KBO·K리그
python scripts/import_events.py kopis  20250101 20261031          # KOPIS (.env 의 KOPIS_API_KEY)
python scripts/import_events.py culture 20250101 20261031         # 서울시 문화행사 (.env 의 SEOUL_API_KEY)
python scripts/run_experiments.py                                 # 설정 결정 (event_mode, 게이트)
python serving_app/train_and_register.py                          # 학습 → 게이트 → Production
python -m uvicorn serving_app.main:app --port 8077                # 대시보드에서 "최근 경기 3시간 전"으로 실측 비교

# 운영 (정식 키 발급 후, cron 매시 50분)
SEOUL_API_KEY=... python scripts/collect_hourly.py
```

## 빠른 시작 (합성 데이터)

```bash
uv sync
export TRAFFIC_DB=data/traffic_synthetic.db MLFLOW_TRACKING_URI=sqlite:///mlflow_synthetic.db
python scripts/generate_synthetic_data.py       # corridor 8개, 이벤트(취소 포함)·돌발(사고·공사) 포함
python serving_app/train_and_register.py        # MLflow 학습 → 게이트 → Production
python -m uvicorn serving_app.main:app --port 8077     # http://localhost:8077/  · /docs
python scripts/simulate_drift.py                       # 정상 → 알려진 이벤트 → 도로 공사 (기본 yeouido_up)
python scripts/run_experiments.py               # 실험 ①②③④ (--quick 은 수 분)
docker compose -f serving_app/docker-compose.yml up --build     # 컨테이너: http://localhost:8099/
```

## 실험 (`scripts/run_experiments.py` → `reports/experiments/summary.md`)

| 실험 | 비교 | 결정하는 것 |
|---|---|---|
| ① 피처 단계 | 속도만 → +교통량 → +캘린더 → +이벤트 | `train.feature_stage` |
| ② 이벤트 인코딩 | 플래그 · 규모 · 규모×시간감쇠 · 감쇠+텍스트 | `train.event_mode` |
| ③ 드리프트 대응 | 고정 · 주기적(14일) · 드리프트 감지 시 (75일 walk-forward, 돌발 제외 반영) | 운영 정책 |
| ④ 예측 단위 | 거점 평균(양방향 합침) vs 방향별 corridor — 둘 다 방향별 실측으로 평가 | 데이터 구조 |

MAE·RMSE 를 전체/평상/이벤트, corridor·방향 역할별로 보고하고, 비교 기준은 단순 예측법(지난주 같은 시각),
시드 3개 평균 ± 표준편차입니다. 합성 데이터로 실행하면 리포트 상단에 그 사실이 표시됩니다.

## 운영 설계

- **배포 게이트**: ① 평상 RMSE ≤ 기준 ② 이벤트 RMSE < 이벤트 정보 없는 모델 ③ (재학습 시) 기존 Production 보다 나빠지지 않음
- **드리프트 → 재학습**: 예측 기록(predictions) + 도착한 정답 → corridor 별 72h 평상 오차(현재 버전만) →
  이벤트·공휴일·사고 제외 → 기준 대비 1.5배 초과 시 `[WARN]`(진행 중 공사가 있으면 원인 표기) →
  최근 14일 fine-tuning(최근 가중) → 게이트 → alias `production` 이동 + 서빙 캐시 갱신, 실패 시 기존 유지
- **이벤트 상태 관리**: review(검토 대기) → 확정 / 취소는 삭제가 아니라 `cancelled` + 공지 시각

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/predict` | `{hub?, corridor?, issued_at?}` → 방향별 향후 6시간 속도·평소 속도·소요시간·추가 소요시간·관련 이벤트 |
| POST | `/predict/batch-test` | `{corridor, observations[]}` 관측 주입 → 슬라이딩 예측 → 드리프트 판정 → (재학습) |
| POST/GET/DELETE | `/events`, `/events/upload`, `/events/{id}` | 이벤트 등록(텍스트 속성 추출)·조회(`?status=review`)·삭제 |
| PATCH | `/events/{id}/status` | 검토 대기 확정 / 취소(공지 시각 기록) |
| GET | `/incidents` | 돌발 정보 |
| POST/GET | `/data/upload`, `/data/status` | corridor CSV 적재 · 현황(교통량 도착 지연 포함) |
| GET | `/hubs`, `/health`, `/monitoring/drift`, `/logs` | 거점·corridor·상태·드리프트·aiops 로그 |

## 남은 과제

- **경기별 규모 차이**: 규모가 수용 인원(고정값)이라 큰 경기와 작은 경기를 구별하지 못함 → 예매율 등 사전 공개 지표 검토
- **공연을 피처로 쓰는 방법 재설계**: 공연 유형(경기장 콘서트 / 공원 축제 / 실내 공연)을 나누고 종료 시각 정확도를 높인 뒤
  실험 ⑤ 재실행. 예: 2026-09-05 상암은 K리그와 문화비축기지 축제(MADLY MEDLEY)가 겹쳐 21시 31→18km/h 로 떨어졌지만 현재 모델은 놓침
- 집회 일정 (경찰청 게시판 게시 시각 포함) → 광화문·여의도 이벤트 반영
- 운영 서버에서 `collect_hourly.py` cron, 교통량 도착 지연 측정
- 1차 고객 결정 → 고객 지표(방향별 추가 소요시간 오차 등) 정의
