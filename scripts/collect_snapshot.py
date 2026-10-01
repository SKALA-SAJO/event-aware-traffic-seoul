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

해외(Actions) 서버에서 서울 API 응답이 멈출 때가 있어(10-02 새벽: 34링크 × 20초 대기 → 10분 시간 초과로 취소),
요청 대기는 짧게, 링크는 동시에 부르고, 첫 호출부터 막히면 바로 실패로 끝냅니다(다음 회차가 다시 시도).
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.collectors import seoul_api
from data.config import corridors

REQUEST_TIMEOUT = float(os.getenv("SNAPSHOT_TIMEOUT", "8"))  # 팀 기본값 20초는 수집용으로 너무 김
WORKERS = 8

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

    seoul_api.TIMEOUT = REQUEST_TIMEOUT
    links = [str(link["link_id"]) for c in cfg.values() for link in c.get("links") or []]

    def fetch(link_id):
        try:
            return link_id, seoul_api.link_speed(link_id), None
        except Exception as e:  # 링크 하나 실패해도 나머지는 저장
            return link_id, None, e

    # 연결 확인: 첫 링크부터 막히면 서버가 응답하지 않는 상태 → 34번 기다리지 않고 바로 실패
    first = fetch(links[0])
    if first[2] is not None:
        print(f"{now} 서울 API 응답 없음 ({first[2]}) - 이번 회차 건너뜀")
        sys.exit(1)
    with ThreadPoolExecutor(WORKERS) as ex:
        results = [first] + list(ex.map(fetch, links[1:]))
    speed, failed = [], 0
    for link_id, v, err in results:
        if err is None:
            speed.append({"collected_at": now, "link_id": link_id, "speed": v})
        else:
            failed += 1
            print(f"  link {link_id}: {err}")
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
