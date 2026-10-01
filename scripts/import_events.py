"""
이벤트 일정 일괄 적재 (서버 없이 DB 에 직접).

    python scripts/import_events.py csv data/templates/events_template.csv     # 집회·KBO·K리그 등 정리본
    python scripts/import_events.py kopis 20250101 20261231                    # KOPIS 거점 대형 공연장 공연 (31일 단위 자동 분할)
    python scripts/import_events.py culture 20260101 20261231                  # 서울시 문화행사 (광장·공원 행사)
    python scripts/import_events.py sports 20250101 20261231                   # KBO(잠실·고척)·K리그(FC서울 홈) 일정, 키 불필요
    python scripts/import_events.py smpa data/raw/events/smpa_rallies.csv --team data/raw/events/smpa_rallies_team.csv
    python scripts/import_events.py amatch data/raw/events/sangam_events_manual.csv   # 상암 A매치 등 수동 목록

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
import re
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
        # 원천에 행진·통제 정보가 있으면(TOPIS 통제 공지 등) 그대로, 없으면 설명문에서 추출
        out.append({**r, "announced_at": r.get("announced_at") or now,
                    "march": r.get("march", flags["march"]), "lane_control": r.get("lane_control", flags["lane_control"])})
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


SPOT_HUBS = {"광화문·시청": "gwanghwamun", "여의도": "yeouido", "잠실": "jamsil", "상암": "sangam"}


def from_topis_notices(path: str, start: str = "2023-01-01") -> list[dict]:
    """
    TOPIS 알림마당 통제안내(마라톤·퍼레이드·축제·차 없는 거리 등)를 거점·일자별로 구조화한 표
    (data/raw/events/topis_control_notices.csv 의 source=topis_notice, road_control=1).
    같은 행사에 공지가 여러 번 올라오므로 (거점, 날짜, 제목 앞부분)별로 가장 이른 공지 시각만 남긴다.
    시각이 없는 하루짜리 행사는 08~20시, 시각이 없는 여러 날 행사는 시각 불명이라 제외.
    """
    df = pd.read_csv(path)
    df = df[(df["source"] == "topis_notice") & (df["road_control"] == 1) & (df["parse_ok"] == 1)]
    df = df[df["start_date"].fillna("") >= start]
    out = {}
    for x in df.itertuples(index=False):
        hub = SPOT_HUBS.get(x.spot)
        if hub is None:
            continue
        d0, d1 = pd.Timestamp(x.start_date), pd.Timestamp(x.end_date if isinstance(x.end_date, str) else x.start_date)
        has_time = isinstance(x.start_time, str) and isinstance(x.end_time, str)
        if not has_time and d1 > d0:
            continue
        t0, t1 = (x.start_time, x.end_time) if has_time else ("08:00", "20:00")
        cat = str(x.category)
        typ = "marathon" if "마라톤" in cat else "festival"
        name = re.sub(r"^\[[^\]]*\]\s*", "", str(x.title))[:40]
        for day in pd.date_range(d0, d1, freq="D"):
            from data.collectors.rallies import _at

            s, e = _at(day, t0), _at(day, t1)
            if e <= s:
                e += pd.Timedelta(days=1)
            key = (hub, day.date(), name[:15])
            ann = pd.Timestamp(x.announced_at)
            if key in out and out[key]["announced_at"] <= ann:
                continue
            out[key] = {"hub": hub, "type": typ, "title": name, "start": s, "end": e, "expected_size": None,
                        "description": f"TOPIS 통제안내 ({cat}) {x.title}", "source": "topis_notice",
                        "announced_at": ann, "lane_control": 1, "march": int(typ == "marathon"), "status": "scheduled"}
    return list(out.values())


def from_manual_list(path: str) -> list[dict]:
    """상암 A매치·콘서트 수동 목록 (date, time, type, title, time_verified, source).
    A매치는 경기 이벤트(source=manual: 피처 사용), 콘서트는 공연(source=manual_concert: 표시·드리프트 제외만).
    공개 시각은 경기 7일 전으로 가정 (A매치 일정은 수 주 전 발표)."""
    df = pd.read_csv(path)
    out = []
    for x in df.itertuples(index=False):
        s = pd.Timestamp(f"{x.date} {x.time}")
        is_match = x.type == "amatch"
        out.append({"hub": "sangam", "type": "sports" if is_match else "concert", "title": x.title, "start": s,
                    "end": s + pd.Timedelta(hours=2 if is_match else 3), "expected_size": 66704.0,
                    "description": f"서울월드컵경기장 {x.title} (시각 확인 {'O' if x.time_verified else 'X'})",
                    "source": "manual" if is_match else "manual_concert", "status": "scheduled",
                    "announced_at": s.normalize() - pd.Timedelta(days=7)})
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("csv")
    c.add_argument("path")
    sm = sub.add_parser("smpa")
    sm.add_argument("path")
    sm.add_argument("--team", default=None, help="행진·차로 점거·동 정보가 있는 팀원 정리본")
    sm.add_argument("--min-count", type=int, default=1000)
    sm.add_argument("--start", default="2023-01-01")
    nt = sub.add_parser("notices")
    nt.add_argument("path")
    nt.add_argument("--start", default="2023-01-01")
    am = sub.add_parser("amatch")
    am.add_argument("path")
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
    elif args.cmd == "smpa":
        from data.collectors.rallies import to_events as rally_events

        records = rally_events(args.path, args.team, args.min_count, start=args.start)
        print(f"  집회 {len(records)}건 (신고 {args.min_count:,}명 이상) - "
              + ", ".join(f"{h} {sum(r['hub'] == h for r in records)}" for h in sorted({r['hub'] for r in records})))
        ids = storage.upsert_events(records)  # march·lane_control 은 원천 값 그대로 (텍스트 추출로 덮지 않음)
        print(f"registered/updated {len(ids)} events")
        return
    elif args.cmd == "notices":
        records = from_topis_notices(args.path, args.start)
        print(f"  TOPIS 통제 공지 {len(records)}건 - "
              + ", ".join(f"{h} {sum(r['hub'] == h for r in records)}" for h in sorted({r['hub'] for r in records})))
    elif args.cmd == "amatch":
        records = from_manual_list(args.path)
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
