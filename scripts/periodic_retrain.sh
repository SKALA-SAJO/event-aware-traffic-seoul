#!/bin/bash
# 재학습 자동 실행 (macOS 백그라운드): 14일 주기 + 감지 기반 조기 재학습 신청
#
#   scripts/periodic_retrain.sh            # 한 번 실행 (판정 → 필요하면 fine-tune → 서버 모델 다시 읽기)
#   scripts/periodic_retrain.sh install    # 맥 백그라운드에 등록: 매시 35분 실행 (로그인 상태일 때)
#   scripts/periodic_retrain.sh status | stop | start | uninstall
#
# 매시 도는 것은 "지금 해야 하나" 확인뿐이라 가볍습니다 (python -m serving_app.monitoring.retrain_schedule, TensorFlow 안 읽음).
#   periodic    : 마지막 base-train/fine-tune 후 retrain.period_days(14일) 경과
#   drift-early : sync_collected.sh 의 /monitoring/drift/check 가 남긴 신청(logs/retrain_request.json) + 쿨다운(3일) 경과
# 실제 학습은 이 스크립트가 별도 프로세스로 실행합니다 (서버 요청 안에서 돌지 않음). 끝나면 /admin/reload 로
# 서버의 모델 캐시를 비웁니다. 서버가 꺼져 있으면 건너뛰고, 다음 서버 기동 때 새 Production 을 읽습니다.
# 기록: logs/aiops.log([INFO] retrain triggered → [OK]/[FAIL]), logs/retrain.log(이 스크립트의 실행 기록)
# 맥이 잠자기·꺼짐이면 그 회차는 건너뛰고 다음 회차에 이어서 판단합니다. Production 모델이 없으면(base 학습 전) 아무것도 안 합니다.
set -u

LABEL="com.skala.traffic-retrain"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_URL="${SYNC_API_URL:-http://localhost:8077}"
DOMAIN="gui/$(id -u)"
LOCK="$ROOT/logs/.retrain.lock"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

run() {
  cd "$ROOT" || exit 1
  mkdir -p logs
  # 학습이 길어져 다음 회차와 겹치면 건너뜀 (3시간 넘은 잠금은 비정상 종료로 보고 지움)
  if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +180 2>/dev/null)" ]; then rmdir "$LOCK"; fi
  if ! mkdir "$LOCK" 2>/dev/null; then echo "$(date '+%F %T') 이미 실행 중 - 건너뜀"; exit 0; fi
  trap 'rmdir "$LOCK"' EXIT

  out=$(.venv/bin/python -m serving_app.monitoring.retrain_schedule 2>&1 | tail -1)
  echo "$(date '+%F %T') $out"
  # DUE 로 시작할 때만 실행 (SKIP·확인 불가·오류는 사유만 기록하고 종료)
  case "$out" in DUE\ *) ;; *) exit 0 ;; esac

  mode=$(echo "$out" | sed -E 's/^DUE mode=([^ ]+) reason=.*/\1/')
  reason=$(echo "$out" | sed -E 's/^DUE mode=[^ ]+ reason=//')
  echo "$(date '+%F %T') 재학습 시작 mode=$mode"
  if .venv/bin/python serving_app/train_and_register.py --fine-tune --mode "$mode" --reason "$reason"; then
    echo "$(date '+%F %T') 재학습 종료"
    if curl -sf -m 5 "$API_URL/health" > /dev/null; then
      echo "모델 다시 읽기: $(curl -s -m 120 -X POST "$API_URL/admin/reload")"
    else
      echo "서버($API_URL) 꺼짐 - reload 건너뜀 (다음 기동 때 새 Production 을 읽음)"
    fi
  else
    echo "$(date '+%F %T') 재학습 실패 - aiops.log 확인 (기존 Production 유지)"
  fi
}

install() {
  mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/logs"
  cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$ROOT/scripts/periodic_retrain.sh</string></array>
  <key>StartCalendarInterval</key>
  <dict><key>Minute</key><integer>35</integer></dict>
  <key>StandardOutPath</key><string>$ROOT/logs/retrain.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/retrain.log</string>
</dict>
</plist>
PLISTEOF
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
  launchctl bootstrap "$DOMAIN" "$PLIST" && echo "등록 완료: 매시 35분 판정, 로그 $ROOT/logs/retrain.log"
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
