#!/usr/bin/env bash
# 실데이터 DB(data/traffic.db)를 처음부터 다시 만들고 Production 모델까지 학습합니다.
# 저장소를 받은 뒤 프로젝트 루트에서:  uv sync && bash scripts/bootstrap_data.sh
#
#   - TOPIS 속도·교통량 엑셀, KBO·K리그 일정: 공개 데이터라 키 없이 자동 다운로드
#   - 경찰청 집회·A매치·TOPIS 통제 공지: 저장소에 포함된 data/raw/events/ 의 작은 CSV (팀 수집분, 출처는 SOURCES_*.md)
#   - KOPIS·서울시 문화행사: .env 에 키가 있을 때만 (학습 피처에는 쓰지 않음 - 대시보드 표시용)
#   - 구간 ↔ 링크 매핑은 config/corridor_links.yaml (저장소에 포함) 을 그대로 씀
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
START=${START:-2023-01}   # 학습 데이터 시작 (실험 ⑥: 2023~ 가 가장 좋음)
END=${END:-2026-09}

echo "[1/5] TOPIS 속도·교통량 엑셀 다운로드 ($START ~ $END, 이미 받은 파일은 건너뜀)"
$PY scripts/download_topis.py "$START" "$END"

echo "[2/5] 속도·교통량 적재 → 구간 집계"
$PY scripts/import_topis_excel.py speed  data/raw/speed_*.xlsx $(ls data/raw/speed_*.csv 2>/dev/null)
$PY scripts/import_topis_excel.py volume data/raw/volume_*.xlsx

echo "[3/5] 이벤트 적재"
S=${START/-/}01
$PY scripts/import_events.py sports "$S" 20261031
$PY scripts/import_events.py smpa data/raw/events/smpa_rallies.csv --team data/raw/events/smpa_rallies_team.csv --start "${START}-01"
$PY scripts/import_events.py amatch data/raw/events/sangam_events_manual.csv
$PY scripts/import_events.py notices data/raw/events/topis_control_notices.csv --start "${START}-01"

echo "[4/5] (선택) KOPIS·서울시 문화행사 - .env 에 키가 있을 때만"
if grep -q "^KOPIS_API_KEY=." .env 2>/dev/null; then $PY scripts/import_events.py kopis "$S" 20261031; else echo "  KOPIS_API_KEY 없음 - 건너뜀"; fi
if grep -q "^SEOUL_API_KEY=." .env 2>/dev/null; then $PY scripts/import_events.py culture "$S" 20261031; else echo "  SEOUL_API_KEY 없음 - 건너뜀"; fi

echo "[5/5] 학습 → 배포 게이트 → Production"
$PY serving_app/train_and_register.py

echo "완료: python -m uvicorn serving_app.main:app --port 8077  →  http://localhost:8077"
