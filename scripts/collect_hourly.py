"""
실시간 수집 (매시간 1회): 링크 속도 + 교통량 이력(지연분 재조회) + 돌발 정보 → 원천 테이블 → corridor 재집계.

    SEOUL_API_KEY=... python scripts/collect_hourly.py
    python scripts/collect_hourly.py --lag-report        # 교통량 도착 지연 분포 확인
    # cron (매시 50분): 50 * * * * cd /path/to/lstm-traffic && .venv/bin/python scripts/collect_hourly.py

    속도   TrafficInfo  : 현재값만 제공 → 이번 시간대(ts=HH:00) 값으로 link_speed 에 저장해 이력을 직접 축적
    교통량 VolInfo      : "교통량 이력 정보". 최신 시간대가 언제 채워지는지 아직 측정 전이므로, 매번 직전
                          collection.volume_backfill_hours 시간을 다시 조회해 늦게 들어온 값을 채우고
                          처음 값을 받은 시각(first_seen_at)을 남김 → --lag-report 로 실제 지연을 확인
    돌발   AccInfo      : 현재 돌발 목록 → corridor 링크에 매핑되는 것만 incidents 에 저장 (last_seen 갱신)

corridor 의 links 가 비어 있으면 속도·돌발 매핑을 할 수 없습니다 → scripts/setup_links.py 로 먼저 설정.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data import storage
from data.collectors import seoul_api
from data.config import corridors, section

log = logging.getLogger("collector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def collect_speed(cfg: dict, this_hour: pd.Timestamp) -> int:
    rows = []
    for cid, c in cfg.items():
        if not c.get("links"):
            log.info(f"{cid}: links 미설정 - 속도 수집 건너뜀 (scripts/setup_links.py)")
        for link in c.get("links") or []:
            try:
                v = seoul_api.link_speed(str(link["link_id"]))
                rows.append({"link_id": link["link_id"], "ts": this_hour, "speed": v})
            except Exception as e:
                log.warning(f"{cid} link {link['link_id']}: {e}")
    return storage.upsert_link_speed(pd.DataFrame(rows), source="api") if rows else 0


def collect_volume(cfg: dict, this_hour: pd.Timestamp, backfill: int) -> int:
    spots = {s["spot"] for c in cfg.values() for s in c.get("volume_spots") or []}
    rows = []
    for spot in spots:
        for k in range(1, backfill + 1):
            ts = this_hour - pd.Timedelta(hours=k)
            try:
                for r in seoul_api.spot_volume(spot, ts):
                    rows.append({"spot": spot, "ts": ts, **r})
            except Exception as e:
                log.warning(f"spot {spot} {ts}: {e}")
    return storage.upsert_spot_volume(pd.DataFrame(rows), source="api") if rows else 0


def collect_incidents(cfg: dict, now: pd.Timestamp) -> int:
    link_map = {str(link["link_id"]): cid for cid, c in cfg.items() for link in c.get("links") or []}
    rows = []
    for inc in seoul_api.incidents():
        cid = link_map.get(str(inc.get("link_id")))
        if cid is None:
            continue
        rows.append({**inc, "corridor": cid, "hub": cfg[cid]["hub"], "last_seen": now, "source": "api"})
    return storage.upsert_incidents(rows) if rows else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag-report", action="store_true")
    args = ap.parse_args()
    if args.lag_report:
        lag = storage.volume_arrival_lag()
        if lag.empty:
            print("아직 측정된 교통량 도착 기록이 없습니다 (API 로 며칠 수집 후 확인)")
        else:
            print(lag.groupby("spot")["lag_hours"].describe(percentiles=[0.5, 0.9]).round(2))
        return

    cfg = corridors()
    backfill = section("collection")["volume_backfill_hours"]
    now = pd.Timestamp.now()
    this_hour = now.floor("h")
    n_s = collect_speed(cfg, this_hour)
    n_v = collect_volume(cfg, this_hour, backfill)
    try:
        n_i = collect_incidents(cfg, now)
    except Exception as e:
        n_i = 0
        log.warning(f"incidents: {e}")
    n_o = storage.aggregate_observations(cfg, this_hour - pd.Timedelta(hours=backfill), this_hour, source="api")
    log.info(f"link_speed {n_s}, spot_volume {n_v}, incidents {n_i} → observations {n_o} rows")


if __name__ == "__main__":
    main()
