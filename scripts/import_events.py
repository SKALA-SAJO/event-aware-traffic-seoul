"""
이벤트 일정 일괄 적재 (서버 없이 DB 에 직접).

    python scripts/import_events.py csv data/templates/events_template.csv     # 집회·KBO·K리그 등 정리본
    python scripts/import_events.py kopis 20250101 20261231                    # KOPIS 거점 대형 공연장 공연 (31일 단위 자동 분할)
    python scripts/import_events.py culture 20260101 20261231                  # 서울시 문화행사 (광장·공원 행사)
    python scripts/import_events.py sports 20250101 20261231                   # KBO(잠실·고척)·K리그(FC서울 홈) 일정, 키 불필요

CSV 컬럼: hub, type, title, start, end, expected_size, description, announced_at[, status, status_changed_at, source]
    - 경찰청 "오늘의 집회/시위": announced_at = 게시판의 실제 게시 시각 (없으면 전날 18:00 로 가정하지 말고 비워 두면
      import 시각으로 기록 → 과거 학습에는 쓰이지 않음. 과거분을 학습에 쓰려면 실제 게시 시각을 꼭 넣으세요)
    - KBO·K리그: expected_size = 경기장 수용 인원 (관중 수는 경기 후 확정 → 정보 누수)
    - 우천취소 등: status=cancelled, status_changed_at = 취소 공지 시각
description 에서 행진·차로 통제를 추출합니다 (EVENT_NLP_BACKEND=rule|llm).
자동 수집분 중 시각을 확정하지 못한 것은 status=review 로 들어가며, 대시보드/PATCH /events/{id}/status 로 확정합니다.
"""
import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data import storage
from data.config import hubs
from data.event_nlp import extract


def _finalize(records: list[dict]) -> list[dict]:
    known = set(hubs(include_disabled=True))
    now = dt.datetime.now()
    out = []
    for r in records:
        if r["hub"] not in known:
            print(f"  건너뜀 (알 수 없는 거점 {r['hub']}): {r.get('title')}")
            continue
        flags = extract(r.get("description"))
        out.append({**r, "announced_at": r.get("announced_at") or now,
                    "march": flags["march"], "lane_control": flags["lane_control"]})
    return out


def from_csv(path: str) -> list[dict]:
    df = pd.read_csv(path, encoding="utf-8-sig", comment="#")
    df = df.astype(object).where(pd.notna(df), None)
    for c in ("start", "end", "announced_at", "status_changed_at"):
        if c in df:
            df[c] = df[c].map(lambda v: pd.Timestamp(v) if v else None)
    if "source" not in df:
        df["source"] = "csv"
    return df.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("csv")
    c.add_argument("path")
    for name in ("kopis", "culture", "sports"):
        p = sub.add_parser(name)
        p.add_argument("start", help="YYYYMMDD")
        p.add_argument("end", help="YYYYMMDD")
        if name == "sports":
            p.add_argument("--announce-days", type=int, default=7, help="과거 일정 공개 시각 가정: 경기 N일 전")
        if name == "kopis":
            p.add_argument("--fallback-days", type=int, default=30,
                           help="KOPIS 갱신 시각이 첫 공연 이후인 공연만: 첫 공연 N일 전 공개로 가정")
    args = ap.parse_args()

    if args.cmd == "csv":
        records = from_csv(args.path)
    elif args.cmd == "kopis":
        from data.collectors.kopis import events_for_hubs

        records, stats = events_for_hubs(args.start, args.end, pd.Timestamp.now(), args.fallback_days)
        print(f"  대형 공연장 {stats['venues']}")
        print(f"  공연 {stats['performances']}건 → 회차 {len(records)}건 (소규모 홀 제외 {stats['small_hall']}, "
              f"공연장 미확인 제외 {stats['unknown_hall']}, 공개 시각 가정 {stats['assumed_announce']})")
    elif args.cmd == "sports":
        from data.collectors.sports import kbo_events, kleague_events

        s, e = pd.Timestamp(args.start), pd.Timestamp(args.end)
        records = kbo_events(s, e, args.announce_days) + kleague_events(s, e, args.announce_days)
        print(f"  KBO·K리그 {len(records)}건 (취소 {sum(r['status'] == 'cancelled' for r in records)}건)")
    else:
        from data.collectors.culture import to_events
        from data.collectors.seoul_api import cultural_events

        records = to_events(cultural_events(), args.start, args.end)
    records = _finalize(records)
    ids = storage.upsert_events(records)
    n_review = sum(r.get("status") == "review" for r in records)
    print(f"registered/updated {len(ids)} events (검토 대기 {n_review}건 - 대시보드에서 확정하세요)")


if __name__ == "__main__":
    main()
