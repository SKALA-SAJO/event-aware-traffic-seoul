"""
TOPIS 자료실 월별 엑셀 → 원천 테이블(link_speed / spot_volume) → corridor 관측치 재집계.

    python scripts/import_topis_excel.py speed  data/raw/2026년_1월_도로별_일자별_통행속도.xlsx ...
    python scripts/import_topis_excel.py volume data/raw/2026년_1월_지점별_일자별_교통량.xlsx ...

    speed  : 행 = 일자 × 링크(링크아이디·도로명·시점/종점·방향·거리), 열 = 01시 … 24시
             → 설정된 corridor 링크만 link_speed 로 (링크·방향 단위 그대로 보존)
    volume : 행 = 일자 × 지점 × 방향/구분(유입·유출), 열 = 0시 … 23시
             → 설정된 지점만 spot_volume 으로 (0 = 결측)
    이후 aggregate_observations() 가 corridor 속도(거리 가중 조화평균)·교통량(방향별 합)을 만듭니다.

엑셀 양식은 해마다 조금씩 달라 컬럼명을 유연하게 찾고, 못 찾으면 실제 컬럼 목록을 출력하고 멈춥니다.
※ 2026년 1월 속도 파일의 열 구성(일자·도로명·링크아이디·시점/종점·방향·거리·01시~24시)은 확인된 형식이며,
   교통량 파일의 방향/구분 열 값은 실제 파일로 확인이 필요합니다.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data import storage
from data.config import corridors


def find_col(cols, *cands, required: bool = True):
    for c in cols:
        if any(k in str(c).replace(" ", "") for k in cands):
            return c
    if required:
        raise SystemExit(f"컬럼을 찾을 수 없습니다 {cands} - 실제 컬럼: {list(cols)}")
    return None


def _hour_cols(cols) -> dict:
    # 속도: '~01시' … '~24시' (또는 '01시'), 교통량: '0시' … '23시'
    out = {c: int(m.group(1)) for c in cols if (m := re.fullmatch(r"\s*~?\s*(\d{1,2})\s*시\s*", str(c)))}
    if not out:
        raise SystemExit(f"'N시' 형태의 시간 열이 없습니다 - 실제 컬럼: {list(cols)}")
    return out


def _melt(df, id_cols, hour_cols):
    one_based = max(hour_cols.values()) == 24  # 속도 엑셀은 01~24시 (01시 = 00:00~00:59)
    long = df.melt(id_vars=id_cols, value_vars=list(hour_cols), var_name="hcol", value_name="value")
    long["hour"] = long["hcol"].map(hour_cols) - (1 if one_based else 0)
    long["ts"] = pd.to_datetime(long[id_cols[0]].astype(str).str.replace(r"\D", "", regex=True), format="%Y%m%d") \
        + pd.to_timedelta(long["hour"], unit="h")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long


def _read(path: str, data_sheet: bool = False) -> pd.DataFrame:
    """교통량 파일은 '범례' 시트가 앞에 있으므로 '일자' 열이 있는 첫 시트를 데이터로 씀."""
    if not data_sheet:
        return pd.read_excel(path, engine="calamine")
    for name, df in pd.read_excel(path, sheet_name=None, engine="calamine").items():
        if "일자" in map(str, df.columns):
            return df
    raise SystemExit(f"{path}: '일자' 열이 있는 시트가 없습니다")


def import_speed(path: str) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    df = _read(path)
    date_c, link_c = find_col(df.columns, "일자"), find_col(df.columns, "링크아이디", "링크ID")
    wanted = {str(link["link_id"]) for c in corridors().values() for link in c.get("links") or []}
    if not wanted:
        raise SystemExit("설정된 corridor 링크가 없습니다 → scripts/map_corridors.py 로 먼저 매핑하세요")
    df = df[df[link_c].astype(str).isin(wanted)]
    long = _melt(df, [date_c, link_c], _hour_cols(df.columns))
    out = long.rename(columns={link_c: "link_id", "value": "speed"})[["link_id", "ts", "speed"]]
    n = storage.upsert_link_speed(out, source="topis_excel")
    print(f"  link_speed {n} rows ({out['link_id'].nunique()} links)")
    return n, out["ts"].min(), out["ts"].max()


def _io_type(v) -> int:
    s = str(v)
    if "유입" in s or s.strip() in ("1", "상행"):
        return 1
    if "유출" in s or s.strip() in ("2", "하행"):
        return 2
    return 0


def import_volume(path: str) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    df = _read(path, data_sheet=True)
    date_c, spot_c = find_col(df.columns, "일자"), find_col(df.columns, "지점번호")
    io_c = find_col(df.columns, "방향", "유입유출", required=False)  # 값: 유입 / 유출
    wanted = {s["spot"] for c in corridors().values() for s in c.get("volume_spots") or []}
    df = df[df[spot_c].astype(str).isin(wanted)]
    if df.empty:
        print("  설정된 교통량 지점 데이터 없음")
        return 0, None, None
    if io_c is None:
        print("  ⚠ 유입/유출 구분 열이 없어 io_type=0 으로 저장합니다 - 방향별 집계가 되지 않습니다")
    long = _melt(df, [date_c, spot_c] + ([io_c] if io_c else []), _hour_cols(df.columns))
    long["io_type"] = long[io_c].map(_io_type) if io_c else 0
    out = (long.groupby([spot_c, "ts", "io_type"], as_index=False)["value"].sum(min_count=1)
           .rename(columns={spot_c: "spot", "value": "volume"}))
    n = storage.upsert_spot_volume(out[["spot", "ts", "io_type", "volume"]], source="topis_excel")
    print(f"  spot_volume {n} rows (결측 {int(out['volume'].isna().sum() + (out['volume'] == 0).sum())})")
    return n, out["ts"].min(), out["ts"].max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["speed", "volume"])
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    lo, hi = None, None
    for path in args.paths:
        print(path)
        _, s, e = (import_speed if args.kind == "speed" else import_volume)(path)
        if s is not None:
            lo, hi = min(lo or s, s), max(hi or e, e)
    if lo is not None:
        n = storage.aggregate_observations(corridors(), lo, hi, source="topis")
        print(f"→ observations {n} rows 재집계 ({lo} ~ {hi})")


if __name__ == "__main__":
    main()
