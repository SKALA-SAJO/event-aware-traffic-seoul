"""
합성 데이터로 data/traffic.db 를 채웁니다 (파이프라인 검증용, data/synthetic.py 참고).

    python scripts/generate_synthetic_data.py                     # 2025-01-01 ~ 2026-09-30, enabled 거점
    python scripts/generate_synthetic_data.py --no-drift          # 상암 공사 드리프트 없이
    python scripts/generate_synthetic_data.py --csv data/synthetic  # 업로드 실습용 CSV도 함께 저장

이전에 생성한 synthetic / simulation 데이터와 예측 기록은 지우고 다시 만듭니다
(실데이터 source 의 관측치·이벤트는 건드리지 않습니다).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data import storage
from data.config import corridors, hub_ids
from data.synthetic import DRIFT, generate_events, generate_incidents, generate_observations, public_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01 00:00")
    ap.add_argument("--end", default="2026-09-30 23:00")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--csv", default=None, help="CSV 내보내기 디렉터리 (선택)")
    args = ap.parse_args()

    hubs, cfg = hub_ids(), corridors()
    drift = None if args.no_drift else DRIFT
    events = generate_events(hubs, args.start, args.end, seed=args.seed)
    incidents = generate_incidents(cfg, args.start, args.end, seed=args.seed, drift=drift)
    obs = generate_observations(cfg, args.start, args.end, events, incidents, seed=args.seed, drift=drift)
    public = public_records(events)

    for src in ("synthetic", "simulation"):
        storage.delete_observations(source=src)
        storage.delete_events(source=src)
        storage.delete_incidents(source=src)
    with storage.connect() as conn:
        conn.execute("DELETE FROM predictions")
    storage.upsert_observations(obs, source="synthetic")
    storage.upsert_events(public)
    storage.upsert_incidents(public_records(incidents))

    n_cancel = sum(e["status"] == "cancelled" for e in public)
    print(f"observations: {len(obs):,} rows ({len(cfg)} corridors: {', '.join(cfg)})  {args.start} ~ {args.end}")
    print(f"events      : {len(public):,} ({pd.Series([e['type'] for e in public]).value_counts().to_dict()}), 취소 {n_cancel}")
    print(f"incidents   : {len(incidents):,} (사고 {sum(i['category'] == 'accident' for i in incidents)}, "
          f"공사 {sum(i['category'] == 'construction' for i in incidents)})")
    if drift:
        print(f"drift       : {DRIFT['corridor']} 도로 공사(차로 축소) from {DRIFT['start']}  ← 예정에 없던 변화")

    if args.csv:
        os.makedirs(args.csv, exist_ok=True)
        obs.rename(columns={"ts": "datetime"}).to_csv(os.path.join(args.csv, "observations.csv"), index=False)
        pd.DataFrame(public).to_csv(os.path.join(args.csv, "events.csv"), index=False)
        print(f"csv         : {args.csv}/observations.csv, events.csv")


if __name__ == "__main__":
    main()
