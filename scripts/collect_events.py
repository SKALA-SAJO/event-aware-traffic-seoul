"""
예정 이벤트 스냅숏 수집 (하루 1회, GitHub Actions 용): 앞으로 N일의 KOPIS 공연·KBO·K리그 일정 → CSV.

    KOPIS_API_KEY=... python scripts/collect_events.py --out collected [--days 90]

    collected/events/YYYY-MM-DD.csv   그날 받은 예정 일정 전체 (fetched_at 포함)

    python scripts/collect_events.py --out collected --games-status      # 오늘·내일 KBO·K리그 경기 상태만 (30분 수집에 곁들임)

--games-status : 하루 1회(19:37) 수집으로는 오후 경기의 우천취소가 늦게 보이므로, 30분마다 도는 속도 수집 워크플로에서
    오늘·내일 경기만 다시 받아 collected/events/날짜_status.csv 에 "새로 보인 (일정, 상태)"만 덧붙인다. 취소는 처음 '취소'로
    보인 시각이 status_changed_at 이 되고(import_collected.py), 가정값(경기 2시간 전)보다 늦으면 이쪽을 쓴다.
    KBO·K리그는 키가 필요 없고 호출이 몇 건뿐이라 실행 시간이 거의 늘지 않는다. 취소가 실제로 몇 시에 공지되는지는 아직
    확인하지 못했다 - 이 파일이 쌓이면 확인할 수 있다.

매일 "그날 보인 목록"을 통째로 남기므로, 일정마다 처음 보인 날 = 실제로 알 수 있었던 시점의 근거가 됩니다.
scripts/import_collected.py 가 announced_at = min(수집기 가정값, 처음 보인 시각) 으로 DB 에 넣습니다.
(지금까지는 공개 시각을 '경기 7일 전', 'KOPIS 갱신 시각 또는 30일 전'으로 가정)
집회는 경찰청 게시판 수집기가 생기면 여기에 추가합니다.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.config import hubs

COLS = ["fetched_at", "source", "hub", "type", "title", "start", "end", "expected_size", "announced_at", "status",
        "status_changed_at", "external_id", "venue", "lat", "lon", "end_estimated", "runtime_text", "description"]


def games_status(out: str, now: pd.Timestamp) -> None:
    from data.collectors.sports import kbo_events, kleague_events

    start, end = now.normalize(), now.normalize() + pd.Timedelta(days=2)
    records = kbo_events(start, end) + kleague_events(start, end)
    known = set(hubs(include_disabled=True))
    df = pd.DataFrame([r for r in records if r.get("hub") in known]).reindex(columns=COLS)
    df["fetched_at"] = now
    path = os.path.join(out, "events", f"{now:%Y-%m-%d}_status.csv")
    key = ["hub", "title", "start", "status"]
    if os.path.exists(path):  # 이미 기록한 (일정, 상태)는 건너뜀 → 바뀐 것(취소 등)만 쌓임
        old = pd.read_csv(path, dtype=str)
        seen = set(map(tuple, old.reindex(columns=key).astype(str).itertuples(index=False)))
        df = df[[tuple(map(str, r)) not in seen for r in df[key].itertuples(index=False)]]
    if len(df):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    n_cancel = int((df["status"] == "cancelled").sum()) if len(df) else 0
    print(f"{now} 경기 상태 {len(records)}건 중 새로 기록 {len(df)}건 (취소 {n_cancel}) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="collected")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--games-status", action="store_true", help="오늘·내일 경기 상태만 받아 _status 파일에 덧붙임")
    args = ap.parse_args()

    now = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).floor("min")
    if args.games_status:
        return games_status(args.out, now)
    start, end = now.normalize(), now.normalize() + pd.Timedelta(days=args.days)
    records, ok = [], 0

    try:
        from data.collectors.sports import kbo_events, kleague_events

        sports = kbo_events(start, end) + kleague_events(start, end)
        records += sports
        ok += 1
        print(f"  KBO·K리그 {len(sports)}건")
    except Exception as e:
        print(f"  sports: {e}")

    if os.getenv("KOPIS_API_KEY"):
        try:
            from data.collectors.kopis import events_for_hubs

            kopis, stats = events_for_hubs(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), now)
            records += kopis
            ok += 1
            print(f"  KOPIS {len(kopis)}회차 (공연 {stats['performances']}건)")
        except Exception as e:
            print(f"  kopis: {e}")
    else:
        print("  KOPIS_API_KEY 없음 - KOPIS 건너뜀")

    known = set(hubs(include_disabled=True))
    df = pd.DataFrame([r for r in records if r.get("hub") in known]).reindex(columns=COLS)
    df["fetched_at"] = now
    path = os.path.join(args.out, "events", f"{now:%Y-%m-%d}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)  # 하루 한 파일: 같은 날 다시 돌리면 최신 목록으로 덮어씀
    print(f"{now} 예정 일정 {len(df)}건 ({start:%m-%d} ~ {end:%m-%d}) -> {path}")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
