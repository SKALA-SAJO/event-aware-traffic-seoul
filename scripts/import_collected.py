"""
GitHub Actions 가 15분마다 쌓은 스냅숏(data-collect 브랜치) → 링크별 1시간 평균 → DB → corridor 재집계.

    git fetch origin data-collect
    python scripts/import_collected.py                         # origin/data-collect 브랜치에서 바로 읽음
    python scripts/import_collected.py --dir collected         # 로컬 폴더에서 읽기 (collect_snapshot.py --out)
    python scripts/import_collected.py --require-hours 24      # 자동 실행용 (scripts/sync_collected.sh)

    속도  : collected_at 을 시간대(HH:00)로 내려 링크별 평균 → link_speed(source=api)
            한 시간대에 스냅숏이 min_snapshots 개 미만이면 건너뜀 (아직 수집 중인 시간대·누락 대비)
    돌발  : acc_id 별 마지막으로 본 시각을 last_seen 으로 → incidents
    집회  : 하루 1회 경찰청 게시판 수집(collect_rallies.py) → 신고 1,000명 이상을 이벤트(source=smpa)로. 과거 집회와 같은
            출처·제목 규칙이라 같은 일정은 덮어쓴다. 행진·차로 통제·동(洞)은 본문 텍스트에서 규칙으로 추출
    이벤트: 하루 1회 스냅숏(collect_events.py). 일정마다 가장 최근 스냅숏의 내용을 쓰고,
            announced_at = min(수집기의 가정값, 처음 보인 시각) - 실제로 먼저 보였으면 그 시각이 근거가 됨.
            취소는 처음 '취소'로 보인 시각을 status_changed_at 으로 (수집기 가정값보다 늦으면 이쪽을 씀)
다시 실행해도 같은 시간대는 덮어쓰므로 중복되지 않습니다.

--require-hours N : 실시간 속도가 아직 DB 에 한 번도 안 들어갔으면, 수집분의 최근 N시간이 모두 차 있을 때만 넣습니다.
    너무 일찍 넣으면 '마지막 관측 시각'이 지금으로 바뀌는데 그 앞이 비어 있어 기본(최신) 예측이 실패하기 때문.
    한 번 들어간 뒤에는 중간에 몇 시간 빠져도 그대로 넣습니다 (수집 누락 때문에 반영이 멈추지 않도록).
    돌발·이벤트는 마지막 관측 시각과 무관하므로 항상 넣습니다.
"""
import argparse
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data import storage
from data.config import corridors

BRANCH = "origin/data-collect"


def _read_branch(kind: str) -> pd.DataFrame:
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", BRANCH, f"collected/{kind}/"],
                           capture_output=True, text=True, check=True).stdout.split()
    parts = [pd.read_csv(io.StringIO(subprocess.run(["git", "show", f"{BRANCH}:{n}"], capture_output=True,
                                                     text=True, check=True).stdout), dtype=str) for n in names]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _read_dir(root: str, kind: str) -> pd.DataFrame:
    d = os.path.join(root, kind)
    if not os.path.isdir(d):
        return pd.DataFrame()
    parts = [pd.read_csv(os.path.join(d, f), dtype=str) for f in sorted(os.listdir(d)) if f.endswith(".csv")]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="브랜치 대신 로컬 폴더에서 읽기")
    # 2: 15분 간격(최대 4회)에서 일부가 빠져도 반영하고, 30분 간격 시절 시간대(2회)도 그대로 반영되도록
    ap.add_argument("--min-snapshots", type=int, default=2, help="시간대별 최소 스냅숏 수 (15분 간격이면 최대 4)")
    ap.add_argument("--require-hours", type=int, default=0, help="처음 반영 전 최근 N시간이 모두 차 있어야 함 (0=검사 안 함)")
    args = ap.parse_args()
    read = (lambda k: _read_dir(args.dir, k)) if args.dir else _read_branch

    sp = read("speed").drop_duplicates()
    if sp.empty:
        print("수집된 속도가 없습니다")
        return
    sp["collected_at"] = pd.to_datetime(sp["collected_at"])
    sp["speed"] = pd.to_numeric(sp["speed"], errors="coerce")
    sp["ts"] = sp["collected_at"].dt.floor("h")
    n_snap = sp.groupby("ts")["collected_at"].nunique()
    done = n_snap[n_snap >= args.min_snapshots].index
    if args.require_hours and not _live_started() and not _ready(done, args.require_hours):
        _import_rest(read)
        return
    hourly = (sp[sp["ts"].isin(done) & (sp["speed"] > 0)]
              .groupby(["link_id", "ts"], as_index=False)["speed"].mean())
    n = storage.upsert_link_speed(hourly, source="api")
    lo, hi = hourly["ts"].min(), hourly["ts"].max()
    n_obs = storage.aggregate_observations(corridors(), lo, hi, source="api") if n else 0
    print(f"속도: 스냅숏 {sp['collected_at'].nunique()}회 → {len(done)}개 시간대({lo} ~ {hi}) "
          f"link_speed {n}행 → observations {n_obs}행  (스냅숏 {args.min_snapshots}회 미만 시간대 {len(n_snap) - len(done)}개 보류)")

    _import_rest(read)


def _live_started() -> bool:
    """실시간 속도(source=api)가 DB 에 이미 들어간 적이 있는지."""
    with storage.connect() as conn:
        return conn.execute("SELECT 1 FROM observations WHERE source = 'api' LIMIT 1").fetchone() is not None


def _ready(done, hours: int) -> bool:
    if len(done) == 0:
        print("속도: 완성된 시간대가 아직 없음 - 반영 보류")
        return False
    have = pd.DatetimeIndex(done)
    need = pd.date_range(have.max() - pd.Timedelta(hours=hours - 1), have.max(), freq="h")
    missing = need.difference(have)
    if len(missing):
        print(f"속도: 최근 {hours}시간 중 {len(missing)}시간이 비어 있어 첫 반영 보류 "
              f"(빈 시간 {missing.min():%m-%d %H시} ~ {missing.max():%m-%d %H시})")
        return False
    return True


def _import_rest(read) -> None:
    inc = read("incidents")
    if not inc.empty:
        # 돌발 한 줄 형식(last_seen)과 15분마다 줄이 쌓이던 예전 형식(collected_at)을 함께 읽음
        seen = inc["last_seen"] if "last_seen" in inc else pd.Series(None, index=inc.index, dtype=object)
        if "collected_at" in inc:
            seen = seen.fillna(inc["collected_at"])
        inc["last_seen"] = pd.to_datetime(seen)
        last = inc.sort_values("last_seen").groupby("acc_id").tail(1)
        last = last.astype(object).where(last.notna(), None)
        rows = [{**r, "source": "api"} for r in last.to_dict("records")]
        print(f"돌발: {storage.upsert_incidents(rows)}건")

    rl = read("rallies")
    if not rl.empty:
        import_rallies(rl)

    ev = read("events")
    if not ev.empty:
        import_events(ev)


def import_rallies(rl: pd.DataFrame, min_count: int = 1000) -> None:
    """collected/rallies 스냅숏 → 집회 이벤트. 같은 건은 가장 최근 스냅숏의 내용을 쓴다."""
    import re
    import tempfile

    from data.collectors.rallies import to_events
    from data.event_nlp import extract_rule

    key = ["date", "start", "end", "count", "station", "place"]
    rl = rl.sort_values("fetched_at")
    first_seen = rl.groupby(key)["fetched_at"].transform("min")
    latest = rl.assign(first_seen=first_seen).groupby(key).tail(1).reset_index(drop=True)
    # 공개 시각이 비어 있으면(작성 시각·게시일을 못 읽은 PDF) 처음 받은 시각 - 그 전에는 알 수 없었으므로 보수적
    latest["announced_at"] = latest["announced_at"].where(latest["announced_at"].notna() & (latest["announced_at"] != ""),
                                                          latest["first_seen"])
    flags = latest["text"].fillna("").map(extract_rule)
    latest["march"] = flags.map(lambda f: int(f["march"]))
    latest["lane"] = flags.map(lambda f: int(f["lane_control"]))
    # 영등포서 관할 중 여의도 집회만 거점에 붙이는 규칙(rally_dong_keywords)이 쓰는 동 표기: 장소 뒤 "<여의도동 등>"
    latest["dong"] = latest["place"].fillna("").map(lambda s: (re.search(r"<([^<>]*?)>", s) or [None, ""])[1])
    with tempfile.NamedTemporaryFile("w", suffix=".csv", encoding="utf-8", newline="") as f:
        latest.drop(columns=["text", "first_seen", "fetched_at"]).to_csv(f.name, index=False)
        records = to_events(f.name, None, min_count)
    ids = storage.upsert_events(records)  # march·lane_control 은 원천 값 그대로 (텍스트 추출로 덮지 않음)
    print(f"집회: 스냅숏 {rl['fetched_at'].nunique()}회 → {len(latest)}건 중 신고 {min_count:,}명 이상 {len(ids)}건 반영")


def import_events(ev: pd.DataFrame) -> None:
    key = ["source", "hub", "type", "title", "start"]
    for c in ("fetched_at", "announced_at", "status_changed_at", "start", "end"):
        ev[c] = pd.to_datetime(ev[c], errors="coerce")
    ev = ev.sort_values("fetched_at")
    first_seen = ev.groupby(key)["fetched_at"].transform("min")
    first_cancel = ev["fetched_at"].where(ev["status"] == "cancelled").groupby([ev[k] for k in key]).transform("min")
    ev["announced_at"] = pd.concat([ev["announced_at"], first_seen], axis=1).min(axis=1)
    ev["status_changed_at"] = pd.concat([ev["status_changed_at"], first_cancel], axis=1).max(axis=1)
    latest = ev.groupby(key).tail(1).drop(columns="fetched_at")
    latest = latest.astype(object).where(latest.notna(), None)
    ids = storage.upsert_events(latest.to_dict("records"))
    print(f"이벤트: 스냅숏 {ev['fetched_at'].nunique()}회 → 일정 {len(ids)}건 "
          f"{latest.groupby('source').size().to_dict()}")


if __name__ == "__main__":
    main()
