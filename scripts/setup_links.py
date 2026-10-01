"""
corridor ↔ TOPIS 링크 매핑 도우미 (이름이 아니라 링크 ID·방향·거리로 연결).

1) 후보 찾기 - TOPIS "도로별 일자별 통행속도" 엑셀에는 링크아이디·도로명·시점/종점·방향·거리가 함께 있으므로,
   설정의 도로명(road)으로 링크를 방향별로 묶어 보여줍니다. 결과를 보고 corridor 의 links 에 옮겨 적으세요.

    python scripts/setup_links.py candidates data/raw/2026년_1월_도로별_일자별_통행속도.xlsx

2) 길이 채우기 - 설정에 적은 링크의 지도거리·시종점명을 LinkInfo API 로 받아 links 테이블에 저장합니다.
   (엑셀의 거리 열로도 채울 수 있음: --from-excel)

    SEOUL_API_KEY=... python scripts/setup_links.py fill
    python scripts/setup_links.py fill --from-excel data/raw/...xlsx

corridor 길이·속도는 이 링크 길이로 계산됩니다: 속도 = Σ길이 / Σ(길이/링크속도).
교통량 지점은 같은 corridor 의 방향(io_type)과 맞는 것만 설정하세요 - 가까운 지점이라도 반대 방향이면 부적절합니다.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data import storage
from data.config import corridors
from scripts.import_topis_excel import find_col


def _read_links_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {
        "link_id": find_col(df.columns, "링크아이디", "링크ID"), "road": find_col(df.columns, "도로명"),
        "st": find_col(df.columns, "시점명", "시점"), "ed": find_col(df.columns, "종점명", "종점"),
        "dir": find_col(df.columns, "방향"), "dist": find_col(df.columns, "거리"),
    }
    out = df[list(cols.values())].drop_duplicates(subset=[cols["link_id"]])
    out.columns = list(cols)
    return out


def candidates(path: str):
    links = _read_links_excel(path)
    for cid, c in corridors().items():
        sel = links[links["road"].astype(str).str.contains(c["road"], na=False)]
        print(f"\n[{cid}] {c['name']} - 도로명 '{c['road']}' 후보 {len(sel)}개 링크")
        for (d, st), g in sel.groupby(["dir", "st"]):
            for r in g.itertuples():
                print(f"   방향={d}  {r.st} → {r.ed}  link_id={r.link_id}  거리={r.dist}m")


def fill(from_excel: str | None):
    ids = {str(link["link_id"]) for c in corridors().values() for link in c.get("links") or []}
    if not ids:
        print("설정된 링크가 없습니다. 먼저 candidates 결과를 config/hubs.yaml 의 corridor links 에 적으세요.")
        return
    rows = []
    if from_excel:
        links = _read_links_excel(from_excel)
        for r in links[links["link_id"].astype(str).isin(ids)].itertuples():
            rows.append({"link_id": str(r.link_id), "road_name": r.road, "st_node": r.st, "ed_node": r.ed,
                         "length_m": float(r.dist)})
    else:
        from data.collectors.seoul_api import link_info

        for lid in sorted(ids):
            info = link_info(lid)
            if info:
                rows.append(info)
    storage.upsert_links(rows)
    missing = ids - {r["link_id"] for r in rows}
    print(f"links 테이블 {len(rows)}개 갱신" + (f", 찾지 못한 링크: {sorted(missing)}" if missing else ""))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("excel")
    f = sub.add_parser("fill")
    f.add_argument("--from-excel")
    args = ap.parse_args()
    if args.cmd == "candidates":
        candidates(args.excel)
    else:
        fill(args.from_excel)


if __name__ == "__main__":
    main()
