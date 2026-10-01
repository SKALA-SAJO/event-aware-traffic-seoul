"""
GitHub Actions 가 15분마다 쌓은 스냅숏(data-collect 브랜치) → 링크별 1시간 평균 → DB → corridor 재집계.

    git fetch origin data-collect
    python scripts/import_collected.py                         # origin/data-collect 브랜치에서 바로 읽음
    python scripts/import_collected.py --dir collected         # 로컬 폴더에서 읽기 (collect_snapshot.py --out)

    속도  : collected_at 을 시간대(HH:00)로 내려 링크별 평균 → link_speed(source=api)
            한 시간대에 스냅숏이 min_snapshots 개 미만이면 건너뜀 (아직 수집 중인 시간대·누락 대비)
    돌발  : acc_id 별 마지막으로 본 시각을 last_seen 으로 → incidents
다시 실행해도 같은 시간대는 덮어쓰므로 중복되지 않습니다.
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
    ap.add_argument("--min-snapshots", type=int, default=2, help="시간대별 최소 스냅숏 수 (15분 간격이면 최대 4)")
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
    hourly = (sp[sp["ts"].isin(done) & (sp["speed"] > 0)]
              .groupby(["link_id", "ts"], as_index=False)["speed"].mean())
    n = storage.upsert_link_speed(hourly, source="api")
    lo, hi = hourly["ts"].min(), hourly["ts"].max()
    n_obs = storage.aggregate_observations(corridors(), lo, hi, source="api") if n else 0
    print(f"속도: 스냅숏 {sp['collected_at'].nunique()}회 → {len(done)}개 시간대({lo} ~ {hi}) "
          f"link_speed {n}행 → observations {n_obs}행  (스냅숏 {args.min_snapshots}회 미만 시간대 {len(n_snap) - len(done)}개 보류)")

    inc = read("incidents")
    if not inc.empty:
        inc["collected_at"] = pd.to_datetime(inc["collected_at"])
        last = inc.sort_values("collected_at").groupby("acc_id").tail(1)
        last = last.astype(object).where(last.notna(), None)
        rows = [{**r, "last_seen": r["collected_at"], "source": "api"} for r in last.to_dict("records")]
        print(f"돌발: {storage.upsert_incidents(rows)}건")


if __name__ == "__main__":
    main()
