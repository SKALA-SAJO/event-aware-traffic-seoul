# 이벤트 인지형 서울 주요 거점 교통 상황 예측 및 드리프트 대응 서비스

공연·경기·집회처럼 **미리 공개되는 이벤트 일정**을 "알려진 미래 정보"로 넣어, 서울 이벤트 거점
(광화문·시청, 여의도, 잠실, 상암 / 선택: 고척)의 **방향별 향후 6시간 통행속도와 평소 대비 추가 소요시간**을
예측하는 LSTM 서비스입니다. 운영 중에는 **알려진 이벤트·일시적 돌발과 예정에 없던 지속적 변화(도로 공사 등)를
구분**해 후자만 드리프트 경보로 알리고, 재학습은 14일 주기로 합니다(실험 ③).

## 현재 상태 (2026-10-01)

- **Production v4**: 실데이터 2023-01 ~ 2026-09 (45개월, 8개 구간) + 이벤트 피처(KBO·K리그 2023~·A매치·경찰청 집회)로 학습, 배포 게이트 통과
- 성능 (마지막 56일 평가, 향후 1~6시간 속도 RMSE km/h, 이벤트 시간 = 경기·공연·행사·집회·통제 2,831시간)

| | 단순 예측 (지난주 같은 시각) | 이벤트 정보 없는 LSTM | **v4** |
|---|---|---|---|
| 평상 시간 | 2.13 | 1.56 | **1.57** |
| 이벤트 시간 | 2.80 | 1.94 | **1.89** |
| 광화문 상행 · 이벤트 | 3.54 | 2.12 | **2.05** |
| 잠실 하행 · 이벤트 | 2.00 | 1.57 | **1.40** |
| 여의도 하행 · 이벤트 | 3.04 | 2.07 | **1.78** |

- 데이터를 넣기 전과 비교 (실험 ⑥, 시드 3개, 같은 평가 기준): 이벤트 시간 1.747 → **1.704**, 광화문·여의도 4~7% 개선, 평상 시간 변화 없음
  (실험 ⑥은 2023~2024 경기 일정 적재 전에 실행. 경기 일정을 채운 v4 는 v3 대비 이벤트 시간 1.92 → 1.89 추가 개선)
- 방향별 예측이 거점 평균 예측보다 이벤트 시간 오차 약 24% 낮음 (실험 ④)
- **운영 전 필수**: 실시간 수집 cron(15분 → 시간 평균), 경찰청 집회 매일 자동 수집 - "남은 과제" 참고

HAIC 주가 예측 실습의 뼈대(FastAPI 서빙 → MLflow 게이트·승격 → 드리프트 감지 → warm-start fine-tuning →
aiops.log → Docker)를 그대로 쓰고, 데이터 계층과 모델 입력을 교통 예측에 맞게 바꿨습니다.

## 핵심 설계

| 항목 | 결정 | 이유 |
|---|---|---|
| 예측 대상 | **corridor(거점 도로 × 방향)** 의 통행속도 → 소요시간 | 경기·공연 종료 후 귀가 정체는 한 방향에 몰림. 양방향 평균은 신호를 희석 (실험 ④) |
| 속도 집계 | 링크별 속도를 **거리 가중 조화평균** (= Σ길이 / Σ(길이/속도)) | 실제 통과 시간 기준. 링크 길이는 TOPIS 엑셀 거리 / 링크 정보 API |
| 교통량 | **입력에서 제외** (`train.use_volume: false`, 2026-10-02). 켜면 지점의 같은 방향(유입/유출)만, 0 = 결측 + 결측 마스크 | 연결 지점이 광화문 A-17 하나뿐(2026-01~07 값 없음)이고 실시간 수집이 없어 운영에서 항상 결측 → 학습·운영 조건 불일치. 효과도 미확인(이벤트 RMSE 1.913 → 1.852, 시드 1개, 지점 없는 구간도 같은 크기로 변동). 집회 시 교통량↓·정체↑ 이므로 예측 대상은 아님 |
| 예보 범위 | 매시 발행, 향후 6시간 | 시작 전 도착·종료 후 귀가 정체를 몇 시간 앞서 반영 |
| 입력 | 과거 24h [속도, (교통량은 설정으로 끔), 캘린더, 이벤트] + 예측 대상 시각의 [캘린더, 이벤트, corridor] | 이벤트·공휴일은 미리 알려진 미래 정보 |
| 정보 누수 방지 | 일정은 **실제 공개(게시) 시각** 이후에만, **취소는 취소 공지 시각** 이후에만 반영. 관중 수 대신 수용·신고 인원 | 학습과 운영이 같은 정보로 동작 |
| 이벤트 품질 | 회차 시각을 확정하지 못한 자동 수집 일정은 **검토 대기(review)** → 예측 미사용 | "공연 기간 ≠ 매일 공연" - 가짜 이벤트가 평상 시간 오차를 키움 |
| 드리프트 판정 | 최근 72h **평상 시간대** RMSE / 기준(val 구간) RMSE > 1.5 | 이벤트·공휴일·**일시적 돌발(사고)** 제외, **공사**는 원인으로 표시 |
| 재학습 정책 | 드리프트는 **경보만**(`retrain.on_drift: alert`), 재학습은 **14일 주기** fine-tuning | 실데이터에서 감지 즉시 재학습은 원인 모를 일시 정체에 맞춰져 평상 오차가 커짐 (실험 ③) |
| 학습 데이터 | 속도 2023-01~ (`train` 은 DB 전체 사용) | 2025~ 보다 2023~ 가 좋음, 2020~2022 는 코로나 시기라 제외 (실험 ⑥) |
| 이벤트 피처 출처 | `train.event_sources` = 경기·A매치·경찰청 집회·수작업 등록 | 공연·문화행사·통제 공지는 넣으면 나빠져 표시·드리프트 제외에만 사용 (실험 ⑤⑥) |

## 사용 데이터

| 구분 | 출처 | 받는 방법 | 쓰임 | 상태 (2026-10-01) |
|---|---|---|---|---|
| 과거 속도 | TOPIS 자료실 "서울시 차량통행속도" 월별 엑셀 (일자 × 링크 × 01~24시, 도로명·시점/종점·방향·거리 포함) | `scripts/download_topis.py` (키 불필요), 2023~2024 는 팀 수집분 CSV `data/raw/speed_2023_*.csv` 등 | 학습 입력·정답, corridor 링크 자동 매핑 | **적재 완료** 2023-01 ~ 2026-09, 34개 링크 (팀 수집분에는 2020~ 도 있으나 코로나 시기 패턴 차이로 제외) |
| 과거 교통량 | TOPIS 자료실 "서울시 교통량 조사자료" 월별 엑셀 (지점 × 유입/유출 × 0~23시 + 지점 좌표·방향 설명 시트) | 같은 스크립트 | 입력 피처 (같은 도로·같은 방향 지점만) | **적재 완료** 2025-01 ~ 2026-08. 해당 지점은 세종대로 A-17 뿐 → 나머지 corridor 는 결측 마스크 |
| 경기 일정 | KBO 홈페이지 일정 (구장·시각·우천취소), K리그 홈페이지 일정 (FC서울 홈) | `scripts/import_events.py sports` (키 불필요) | 이벤트 (잠실·고척·상암) | **적재 완료** 507경기 (우천취소 22) |
| 운영 속도·교통량·링크·돌발 | 서울 열린데이터광장 `TrafficInfo`·`VolInfo`·`LinkInfo`·`SpotInfo`·`AccInfo`·`AccMainCode` | `scripts/collect_hourly.py` (`SEOUL_API_KEY`) | 운영 수집·드리프트 판정 보조 | 정식 키로 호출 확인 완료 (xml 전용, 시각 HHMM 형식 반영, 키는 `.env`). 상시 서버 cron 필요 |
| 공연 | KOPIS 공연시설·공연목록·상세 | `scripts/import_events.py kopis` (`KOPIS_API_KEY`) | 대시보드 표시·드리프트 판정 제외 (피처는 실험 ⑤ 결과로 보류) | **적재 완료** 168개 공연 → 714회차. 거점 반경 1.5km·3천 석 이상 공연장 자동 선정(세종대극장, 잠실 주경기장·실내체육관·학생체육관, 서울월드컵경기장, 문화비축기지, 고척돔), 규모 = 실제 공연한 홀 객석 |
| 공공행사 | 서울시 문화행사 `culturalEventInfo` | `import_events.py culture` (`SEOUL_API_KEY`) | 대시보드 표시·드리프트 판정 제외 | **적재 완료** 279건 - 반경 안 5,559건 중 도로에 붙은 광장·공원(광화문광장·서울광장·여의도공원 등) 행사만 |
| 집회 | 서울경찰청 "오늘의 주요집회" PDF 정리본 (`data/raw/events/smpa_rallies.csv`, 팀원 정리본 `smpa_rallies_team.csv` 의 행진·차로 점거·동 정보 결합) | `import_events.py smpa` | 이벤트 (광화문: 종로·남대문서, 여의도: 영등포서 중 여의도) | **적재 완료** 2023~ 신고 1천 명 이상 1,843건 (광화문 1,618 · 여의도 225), 공개 시각 = PDF 작성 시각과 게시일 중 늦은 쪽 |
| A매치 | 팀원 수동 목록 (`data/raw/events/sangam_events_manual.csv`) | `import_events.py amatch` | 이벤트 (상암, 경기) | **적재 완료** 국가대표 A매치 10경기 (2023~2025) + 콘서트 10회(표시용) |
| 캘린더 | `config/holidays_kr.yaml` | - | 입력 피처, 드리프트 제외 | 사용 중 |
| 검증용 | `data/synthetic.py` 합성 데이터 | `TRAFFIC_DB=data/traffic_synthetic.db` | 파이프라인·실험 검증 | 실데이터와 DB·MLflow 분리 |

**가정 (리포트에 함께 표기)**
- 과거 경기 일정의 공개 시각은 남아 있지 않아 "경기 7일 전 공개"로 가정합니다(정규시즌 일정은 시즌 전에 발표되므로 보수적).
- 우천취소 공지 시각은 "경기 시작 2시간 전"으로 가정합니다.
- 경기 규모는 관중 수가 아니라 경기장 수용 인원입니다(관중 수는 경기 후에 확정되므로 정보 누수).
- KOPIS 공개 시각은 첫 공연 전 마지막 갱신 시각(`updatedate`)입니다. 첫 공연 이후에 갱신된 공연(33건)만 30일 전 공개로 가정합니다.
- KOPIS 공연 시간이 비어 있고 축제이거나 낮(15시 이전)에 시작하면 종일 행사로 보고 22시 종료로 추정합니다.

**학습에 쓰는 데이터 (실험 ⑤·⑥, 시드 3개, 같은 평가 시간 기준)**

| 데이터 | 학습 피처 | 근거 (이벤트 시간 RMSE) |
|---|---|---|
| 속도 2023~2026 (과거 2년 추가) | ✅ | 2025~만: 1.747 → 2023~: 1.730 |
| 경찰청 집회 | ✅ | 2025~ 데이터만으로는 1.763 (나빠짐), 2023~ 데이터와 함께 1.704 (가장 좋음) |
| KBO·K리그·A매치 | ✅ | 경기 없음 1.657 → 경기 1.611 (⑤) / A매치를 빼면 1.872 → 1.885 (⑥-2) |
| KOPIS 공연·서울시 문화행사 | ❌ (표시·드리프트 제외만) | 넣으면 1.611 → 1.651~1.682 (⑤) |
| TOPIS 통제 공지 | ❌ (표시·드리프트 제외만) | 넣으면 1.872 → 1.882 (⑥-2) |

설정은 `train.event_sources` 하나로 바꿀 수 있습니다.

## 재현 방법 (git clone 후)

대용량 원천 데이터와 DB 는 git 에 없습니다. 목적에 따라 하나를 고르세요.

| 방법 | 결과 | 시간 |
|---|---|---|
| **A. DB 스냅샷 받기 (채점·시연 권장)** | 제출 시점과 **완전히 같은 데이터** | 1분 + 학습 7분 |
| B. 공개 출처에서 재구성 (`bootstrap_data.sh`) | 거의 같은 데이터 (아래 차이 참고) | 15~20분 + 학습 |
| C. 합성 데이터 | 파이프라인 동작 확인용 | 3분 |

**A. DB 스냅샷** (GitHub Release `data-20261001` 의 `traffic.db.gz`, 27MB)
```bash
uv sync
# 비공개 저장소라 curl 로는 받을 수 없습니다(404). 둘 중 하나로 받으세요:
#   - 브라우저(로그인 상태): 저장소 Releases → data-20261001 → traffic.db.gz 를 data/ 에 저장
#   - GitHub CLI:  gh release download data-20261001 -p traffic.db.gz -D data
gunzip data/traffic.db.gz
shasum -a 256 data/traffic.db      # 3db8201f266ce6587097fd837a1b816825424ec3298f2f12da31c0c5e5438a01
.venv/bin/python serving_app/train_and_register.py                 # 학습 → 게이트 → Production
.venv/bin/python -m uvicorn serving_app.main:app --port 8077       # http://localhost:8077
docker compose -f serving_app/docker-compose.yml up --build        # 또는 컨테이너 (http://localhost:8099, 이미지 안에서 학습)
```

**B. 재구성**: `uv sync && bash scripts/bootstrap_data.sh` - A 와 다를 수 있는 점
- 2023~2024 속도: 제출본은 팀 수집 CSV, 재구성은 같은 출처(TOPIS)의 엑셀을 새로 받음 (값 동일 여부 미검증)
- KBO·K리그 일정: 실행 시점에 홈페이지에서 수집 (이후 일정 변경 시 차이)
- KOPIS·서울시 문화행사: `.env` 에 키가 있어야 적재 - 학습 피처는 아니지만 평가의 이벤트 시간·드리프트 제외에 쓰여 지표가 조금 달라질 수 있음

**C. 합성 데이터**: DB 없이 `docker compose ... up --build` 하면 자동으로 합성 데이터로 빌드됩니다 (아래 "빠른 시작").

모든 방법에서 학습된 모델은 같은 데이터·시드라도 CPU/OS 에 따라 소수점 둘째 자리 수준으로 다를 수 있습니다
(예: 맥 이벤트 RMSE 1.89, Docker(리눅스) 1.87).

## 실데이터 적재 (모두 자동, 사람 판단 단계 없음)

**처음 받은 사람은 한 줄로**: `uv sync && bash scripts/bootstrap_data.sh` (15~20분)
- TOPIS 엑셀·KBO·K리그는 키 없이 자동 다운로드, 집회·A매치·통제 공지는 저장소의 `data/raw/events/` CSV (출처: 같은 폴더의 `SOURCES_*.md`), 구간 매핑은 `config/corridor_links.yaml` 을 씀
- KOPIS·서울시 문화행사는 `.env` 에 키가 있을 때만 (대시보드 표시용, 학습 피처 아님)
- 더 빨리: 팀 공유 드라이브/GitHub Release 의 `data/traffic.db` 를 받아 `data/` 에 두고 `python serving_app/train_and_register.py`

아래는 각 단계를 따로 실행할 때입니다.

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
python scripts/import_topis_excel.py speed data/raw/speed_2023_*.csv data/raw/speed_2024_*.csv   # 과거 2년 (팀 수집분 CSV)
python scripts/import_events.py smpa data/raw/events/smpa_rallies.csv --team data/raw/events/smpa_rallies_team.csv
python scripts/import_events.py amatch data/raw/events/sangam_events_manual.csv
python scripts/import_events.py notices data/raw/events/topis_control_notices.csv      # TOPIS 통제 공지 (표시용)
python scripts/run_experiments.py                                 # 설정 결정 (event_mode, 게이트)
python serving_app/train_and_register.py                          # 학습 → 게이트 → Production
python -m uvicorn serving_app.main:app --port 8077                # 대시보드에서 "최근 경기 3시간 전"으로 실측 비교

# 운영 (상시 서버 cron, 키는 .env 에서 자동으로 읽음)
python scripts/collect_hourly.py                                  # 매시 (운영 시 15분 간격 수집 → 시간 평균 권장)
python serving_app/train_and_register.py --fine-tune              # 14일마다
```

## 실시간 수집 (GitHub Actions)

| 워크플로 | 주기 | 내용 → `data-collect` 브랜치 |
|---|---|---|
| `collect_realtime.yml` | 15분 | 구간 링크 현재 속도(`TrafficInfo`) + 구간에 걸린 돌발(`AccInfo`) → `collected/speed/`, `incidents/`. 같은 실행에서 오늘·내일 KBO·K리그 경기 상태(우천취소 등)를 다시 받아 `collected/events/날짜_status.csv` 에 바뀐 것만 덧붙임 |
| `collect_events.yml` | 매일 19:30 | 앞으로 90일 KOPIS·KBO·K리그 일정 → `collected/events/` (일정마다 처음 보인 날 = 실제 공개 시점). 같은 실행에서 경찰청 "오늘의 주요집회" → `collected/rallies/`, `smpa_txt/` (학습 피처 smpa, `import_collected.py` 가 신고 1,000명 이상을 이벤트로 반영) |

- 준비: 저장소 Settings → Secrets 에 `SEOUL_API_KEY`, `KOPIS_API_KEY`
- `collected/speed/` 열: `collected_at, link_id, speed, corridor, segment` (예: `gwanghwamun_up`, `세종대로 광화문→세종대로사거리`). 열이 3개뿐인 예전 파일은 다음 수집 때 자동으로 5열로 바뀜. 링크 대응표 전체는 main 브랜치의 `config/corridor_links.yaml` (`data-collect` 브랜치에는 `collected/` 만 있음)
- 실행: GitHub 예약 실행(schedule)은 자주 늦거나 빠져서(10-01 21시~10-02 08시에 1회만 실행), cron-job.org 가
  같은 주기로 `POST /repos/SKALA-SAJO/event-aware-traffic-seoul/actions/workflows/<파일>/dispatches` (`{"ref":"main"}`,
  Actions Read and write 권한만 있는 fine-grained 토큰)를 호출해 실행합니다. 워크플로의 schedule 은 보조로 남겨 둠
- DB 반영(수동): `git fetch origin data-collect && python scripts/import_collected.py` → 링크별 1시간 평균(스냅숏 2회 이상) → 구간 집계
- 직전 24시간이 쌓여야 대시보드에서 "지금" 기준 실시간 예측 가능. 그 전에 import 하면 마지막 관측 시각이 바뀌어 기본(최신) 예측이
  "관측치 부족"으로 실패하므로, 처음에는 24시간이 쌓인 뒤 넣으세요 (`--require-hours 24` 가 자동으로 확인).
  10-02 새벽(01~08시)이 비어 있어 연속 24시간은 10-03 09시경부터 채워집니다

### DB 자동 반영 (macOS 백그라운드)

수동 반영(`git fetch` + `import_collected.py`)과 예측 기록(`/predict`)을 맥이 매시 20·50분에 자동으로 실행합니다.
예측 기록이 쌓여야 실데이터로 드리프트 판정(예측 vs 1시간 뒤 실측)이 돌아갑니다.

```bash
scripts/sync_collected.sh install    # 등록 (처음 한 번)
scripts/sync_collected.sh status     # 켜져 있는지 확인
scripts/sync_collected.sh            # 지금 한 번 실행
scripts/sync_collected.sh stop       # 잠깐 멈추기 (재부팅·재로그인하면 다시 켜짐)
scripts/sync_collected.sh start      # 다시 켜기
scripts/sync_collected.sh uninstall  # 완전히 해제
```

- 터미널을 열어 둘 필요 없이 macOS(launchd)가 정해진 시각에만 잠깐 실행합니다. 결과는 `logs/sync.log`
- 맥이 **켜져 있고 로그인된 상태**일 때만 돕니다. 잠자기 중 놓친 회차는 다음 회차에 밀린 수집분을 한꺼번에 가져옵니다
- 서버(8077)가 꺼져 있으면 예측 호출만 건너뜁니다 (다른 주소: `SYNC_API_URL=http://localhost:8099`)
- 실시간 속도가 아직 DB 에 없으면 수집분의 최근 24시간이 모두 찬 뒤에 처음 반영하고, 한 번 들어간 뒤에는 중간이 빠져도 계속 반영합니다.
  돌발·이벤트는 항상 반영
- 멈춰도 GitHub 수집은 계속됩니다. 수집까지 멈추려면 cron-job.org 두 작업의 Enable job 을 끄세요
- `launchctl` 로 직접: 멈추기 `launchctl bootout gui/$(id -u)/com.skala.traffic-sync`,
  해제는 그 뒤 `rm ~/Library/LaunchAgents/com.skala.traffic-sync.plist`, 확인 `launchctl list | grep traffic`
- 시연의 주력은 재현 시연(과거 시각 예측, `simulate_drift.py`)이고, 이 자동 반영 기록은 "실제로도 운영 중"이라는 보조 근거입니다
- 아직 안 되는 것: **경찰청 집회 매일 수집**(학습 피처라 운영 전 필수), A매치는 대시보드에서 수동 등록,
  실시간 속도(순간값 평균)와 TOPIS 시간 평균의 차이는 10월 TOPIS 엑셀 공개 후 겹치는 기간으로 검증 필요

## 빠른 시작 (합성 데이터)

```bash
uv sync
export TRAFFIC_DB=data/traffic_synthetic.db MLFLOW_TRACKING_URI=sqlite:///mlflow_synthetic.db
python scripts/generate_synthetic_data.py       # corridor 8개, 이벤트(취소 포함)·돌발(사고·공사) 포함
python serving_app/train_and_register.py        # MLflow 학습 → 게이트 → Production
python -m uvicorn serving_app.main:app --port 8077     # http://localhost:8077/  · /docs
python scripts/simulate_drift.py                       # 정상 → 알려진 이벤트 → 도로 공사 (기본 yeouido_up)
python scripts/run_experiments.py --exp 1 2 3 4 # 실험 ①②③④ (--quick 은 수 분)
docker compose -f serving_app/docker-compose.yml up --build     # 컨테이너: http://localhost:8099/
```

## 실험 (`scripts/run_experiments.py` → `reports/experiments/summary.md`)

| 실험 | 비교 | 결정하는 것 |
|---|---|---|
| ① 피처 단계 | 속도만 → +교통량 → +캘린더 → +이벤트 | `train.feature_stage` |
| ② 이벤트 인코딩 | 플래그 · 규모 · 규모×시간감쇠 · 감쇠+텍스트 | `train.event_mode` |
| ③ 드리프트 대응 | 고정 · 주기적(14일) · 드리프트 감지 시 (75일 walk-forward, 돌발 제외 반영) | 운영 정책 |
| ④ 예측 단위 | 거점 평균(양방향 합침) vs 방향별 corridor — 둘 다 방향별 실측으로 평가 | 데이터 구조 |
| ⑤ 이벤트 출처 | 경기만 / + KOPIS 공연 / + 서울시 문화행사 (평가 이벤트 시간 고정) | `train.event_sources` |
| ⑥ 데이터 확장 | 2025~ / + 과거 2년 / + 경찰청 집회 / A매치 제외 / + 통제 공지 (평가 이벤트 시간 고정) | 학습 기간·이벤트 출처 |

| 실험 | 결론 (실데이터) |
|---|---|
| ① | 속도 → +교통량 → +캘린더 → +이벤트 매 단계 개선 |
| ② | 감쇠(decay) 채택 - 최종 데이터에서 감쇠 1.863 < 텍스트 1.876 < 규모 1.888 < 플래그 1.907 < 이벤트 없음 1.935 |
| ③ | 14일 주기 재학습 > 고정 > 감지 시 재학습 (임계 1.5·2.0 모두) |
| ④ | 방향별이 거점 평균보다 이벤트 시간 오차 약 24% 낮음 |
| ⑤ | KOPIS 공연·문화행사는 피처로 넣으면 나빠짐 |
| ⑥ | 집회 + 과거 2년 + A매치 채택 (1.747 → 1.704), 통제 공지 제외 |

MAE·RMSE 를 전체/평상/이벤트, corridor 별로 보고하고, 비교 기준은 단순 예측법(지난주 같은 시각)입니다.
②⑥ 은 최종 설정, ①③④⑤ 는 이전 설정(경기만, 2025~) 기준이며 리포트 상단에 표시됩니다. ⑤⑥ 은 시드 3개, 나머지는 시드 1개.
병렬 실행: `EXP_OUT=<폴더>` 로 시드별 출력 폴더를 나눈 뒤 합치고 `--report-only` 로 리포트를 다시 만듭니다.

## 운영 설계

- **배포 게이트**: ① 평상 RMSE ≤ 기준 ② 이벤트 RMSE < 이벤트 정보 없는 모델 ③ (재학습 시) 기존 Production 보다 나빠지지 않음
- **드리프트 감지**: 예측 기록(predictions) + 도착한 정답 → corridor 별 72h 평상 오차(현재 버전만) →
  이벤트·공휴일·사고 제외 → 기준 대비 1.5배 초과 시 `[WARN]`(진행 중 공사가 있으면 원인 표기)
- **과거 재현 판정**: 예측 기록이 아직 없으면(실시간 운영 전) 판정 구간 72시간을 현재 모델로 매시 다시 예측해 실측과 비교합니다
  (DB 에 저장하지 않음, 대시보드에 "과거 재현"으로 표시, 재학습 트리거에는 쓰지 않음). 예: 9/28~9/30 기준 6개 구간 정상,
  상암 2개 구간은 9/28(월) 18~19시 원인 미상 정체(예측 22 → 실측 12 km/h)로 드리프트 감지
- **재학습**: 기본은 14일 주기(`train_and_register.py --fine-tune`) - 최근 14일 fine-tuning(최근 가중) → 게이트 →
  alias `production` 이동, 실패 시 기존 유지. 시연 때는 `retrain.on_drift: retrain` 으로 감지 즉시 재학습도 가능
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

**운영 전환 시 필수**
- 실시간 수집은 동작 중(cron-job.org → GitHub Actions, 15분 간격 → 시간 평균), DB 반영은 맥 자동 반영(`scripts/sync_collected.sh`)으로 매시 실행 ("실시간 수집" 참고)
- ~~경찰청 집회 매일 자동 수집~~ → `collect_events.yml` 에 추가함(`scripts/collect_rallies.py`, 2026-10-02). 러너에서 경찰청 사이트 접속이 되는지는 첫 실행 로그로 확인 필요
- 14일 주기 재학습 cron (`train_and_register.py --fine-tune`)

**모델 개선**
- 실험 ①③④⑤를 최종 설정(집회 + 2023~)으로 시드 3개 재실행 - 현재 리포트의 ①③④⑤는 이전 설정 기준
- 집회 피처 세부 조정 (신고 인원 기준, 행진·차로 점거만, 영향 시간)
- 날씨(강수량) 입력 추가
- 공연 피처 재설계 - 스타디움급(3만 석 이상)만, 콘서트·축제 구분. 예: 2026-09-05 상암은 K리그와 문화비축기지 축제가 겹쳐 21시 31→18km/h, 현재 모델은 놓침
- 경기별 규모 차이 (예매율 등 사전 공개 지표), 2020~2022 데이터 포함 여부
- 1차 고객 결정 → 고객 지표(방향별 추가 소요시간 오차 등) 정의
