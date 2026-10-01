"""
실시간 스냅숏 수집 (15분마다, GitHub Actions 용): corridor 링크 현재 속도 + corridor 에 걸린 돌발 정보 → CSV 추가.

    SEOUL_API_KEY=... python scripts/collect_snapshot.py --out collected

    collected/speed/YYYY-MM-DD.csv      collected_at, link_id, speed
    collected/incidents/YYYY-MM-DD.csv  collected_at, acc_id, link_id, corridor, hub, type_code, type_name,
                                        category, start, expected_end, info

DB 를 쓰지 않고 CSV 에만 쌓습니다 (Actions 러너는 매번 새로 켜지므로). 노트북에서
scripts/import_collected.py 가 링크별 1시간 평균을 내 DB 에 넣습니다.
TrafficInfo 는 호출 순간의 값이라, 15분 간격 4번의 평균이 TOPIS 과거 자료(1시간 평균)와 성격이 가깝습니다.
시각은 한국 시간 기준입니다 (Actions 러너는 UTC).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.collectors import seoul_api
from data.config import corridors

SPEED_COLS = ["collected_at", "link_id", "speed"]
INC_COLS = ["collected_at", "acc_id", "link_id", "corridor", "hub", "type_code", "type_name",
            "category", "start", "expected_end", "info"]


def _append(path: str, cols: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="collected")
    args = ap.parse_args()

    now = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).floor("min")
    day = now.strftime("%Y-%m-%d")
    cfg = corridors()

    speed, failed = [], 0
    for c in cfg.values():
        for link in c.get("links") or []:
            try:
                speed.append({"collected_at": now, "link_id": link["link_id"],
                              "speed": seoul_api.link_speed(str(link["link_id"]))})
            except Exception as e:  # 링크 하나 실패해도 나머지는 저장
                failed += 1
                print(f"  link {link['link_id']}: {e}")
    _append(os.path.join(args.out, "speed", f"{day}.csv"), SPEED_COLS, speed)

    link_map = {str(link["link_id"]): cid for cid, c in cfg.items() for link in c.get("links") or []}
    inc = []
    try:
        for r in seoul_api.incidents():
            cid = link_map.get(str(r.get("link_id")))
            if cid:
                inc.append({**r, "collected_at": now, "corridor": cid, "hub": cfg[cid]["hub"]})
    except Exception as e:
        print(f"  incidents: {e}")
    if inc:
        _append(os.path.join(args.out, "incidents", f"{day}.csv"), INC_COLS, inc)

    print(f"{now} speed {len(speed)}/{len(speed) + failed} links, incidents {len(inc)}")
    if not speed:
        sys.exit(1)  # 키 오류·접속 차단 등으로 전부 실패하면 Actions 에서 실패로 보이게


if __name__ == "__main__":
    main()
