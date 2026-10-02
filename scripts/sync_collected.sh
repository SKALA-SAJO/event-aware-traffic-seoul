#!/bin/bash
# 실시간 수집분 자동 반영 (macOS 백그라운드): GitHub data-collect 브랜치 → DB → 예측 기록
#
#   scripts/sync_collected.sh            # 한 번 실행 (fetch → import → 서버가 켜져 있으면 /predict → 드리프트 점검)
#   scripts/sync_collected.sh install    # 맥 백그라운드에 등록: 매시 20·50분 자동 실행 (로그인 상태일 때)
#   scripts/sync_collected.sh status     # 등록 여부 확인
#   scripts/sync_collected.sh stop       # 잠깐 멈추기 (재부팅·재로그인하면 다시 켜짐)
#   scripts/sync_collected.sh start      # 멈춘 것 다시 켜기
#   scripts/sync_collected.sh uninstall  # 완전히 해제 (예약 설정 파일까지 삭제)
#
# 같은 일을 launchctl 로 직접 할 때:
#   멈추기  launchctl bootout gui/$(id -u)/com.skala.traffic-sync
#   켜기    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.skala.traffic-sync.plist
#   해제    launchctl bootout gui/$(id -u)/com.skala.traffic-sync && rm ~/Library/LaunchAgents/com.skala.traffic-sync.plist
#   확인    launchctl list | grep traffic
#
# - 실시간 속도가 아직 DB 에 없으면 수집분의 최근 24시간이 모두 찬 뒤에 처음 반영합니다 (import_collected.py --require-hours).
# - 맥이 잠자기·꺼짐이면 그 회차는 건너뛰고, 다음 회차에 밀린 수집분을 한꺼번에 가져옵니다 (중복 없음).
# - 서버(8077)가 꺼져 있으면 예측 호출만 건너뜁니다. 서버 주소는 SYNC_API_URL 로 바꿀 수 있습니다.
# - 결과는 logs/sync.log 에 쌓입니다. GitHub 수집(cron-job.org → Actions)은 이 스크립트와 무관하게 계속 돕니다.
set -u

LABEL="com.skala.traffic-sync"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_URL="${SYNC_API_URL:-http://localhost:8077}"
DOMAIN="gui/$(id -u)"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

run() {
  cd "$ROOT" || exit 1
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') 동기화 시작"
  if ! git fetch -q origin data-collect; then
    echo "git fetch 실패 - 네트워크·GitHub 로그인 상태 확인"
    exit 1
  fi
  .venv/bin/python scripts/import_collected.py --require-hours 24 || { echo "import 실패"; exit 1; }
  if curl -sf -m 5 "$API_URL/health" > /dev/null; then
    # 기본(최신 관측 시각) 예측 → predictions 테이블에 기록 → 드리프트 판정 재료
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 120 -X POST "$API_URL/predict" \
      -H 'Content-Type: application/json' -d '{}')
    echo "예측 호출: HTTP $code"
    # 드리프트 주기 점검: 구간별 판정 → 드리프트면 aiops.log 에 [WARN] (on_drift=retrain 이면 재학습까지)
    curl -s -m 600 -X POST "$API_URL/monitoring/drift/check" | .venv/bin/python -c '
import json, sys
try:
    r = json.load(sys.stdin)
    n = {}
    for x in r["results"]:
        n[x["status"]] = n.get(x["status"], 0) + 1
    print("드리프트 점검:", n, "| 드리프트:", r["drift"] or "없음")
except Exception as e:
    print("드리프트 점검 실패:", e)'
  else
    echo "서버($API_URL) 꺼짐 - 예측 호출 건너뜀"
  fi
}

install() {
  mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/logs"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$ROOT/scripts/sync_collected.sh</string></array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Minute</key><integer>20</integer></dict>
    <dict><key>Minute</key><integer>50</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$ROOT/logs/sync.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/sync.log</string>
</dict>
</plist>
EOF
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
  launchctl bootstrap "$DOMAIN" "$PLIST" && echo "등록 완료: 매시 20·50분 실행, 로그 $ROOT/logs/sync.log"
}

case "${1:-run}" in
  run) run ;;
  install) install ;;
  status) launchctl list | grep "$LABEL" && echo "켜져 있음" || echo "꺼져 있음" ;;
  stop) launchctl bootout "$DOMAIN/$LABEL" && echo "멈춤 (다시 켜기: $0 start)" ;;
  start) launchctl bootstrap "$DOMAIN" "$PLIST" && echo "다시 켬" ;;
  uninstall) launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null; rm -f "$PLIST" && echo "해제 완료 (예약 설정 파일 삭제)" ;;
  *) echo "사용법: $0 [run|install|status|stop|start|uninstall]"; exit 2 ;;
esac
