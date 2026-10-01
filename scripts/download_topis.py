"""
TOPIS 자료실 월별 엑셀 자동 다운로드 (인증키 불필요).

    속도   : 서울시 차량통행속도 (게시판 06)  - 일자 × 링크 × 01~24시
    교통량 : 서울시 교통량 조사자료 (게시판 08) - 일자 × 지점 × 방향 × 0~23시 + 지점 좌표 시트

실행: python scripts/download_topis.py 2025-01 2026-09 [--kind speed volume] [--out data/raw]
"""
import argparse
import os
import sys

import pandas as pd
import requests

BASE = "https://topis.seoul.go.kr"
BOARDS = {"speed": "06", "volume": "08"}


def monthly_files(board: str, year: int) -> list[dict]:
    r = requests.post(f"{BASE}/refroom/selectRefRoomListASC.do",
                      data={"blbdDivCd": board, "bdwrDivCd": year, "mainBdwrRowNum": 12}, timeout=30)
    r.raise_for_status()
    return [row for row in r.json().get("rows", []) if row.get("apndFilePathNm")]


def download(row: dict, board: str, path: str) -> None:
    with requests.post(f"{BASE}/downloadFileRefRoom.do", stream=True, timeout=600, data={
        "apndFileNm": row["apndFileNm"], "apndFilePathNm": row["apndFilePathNm"],
        "bdwrSeq": row["bdwrSeq"], "blbdDivCd": board,
    }) as r:
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")  # YYYY-MM
    ap.add_argument("end")
    ap.add_argument("--kind", nargs="+", default=list(BOARDS), choices=list(BOARDS))
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    months = set(pd.period_range(args.start, args.end, freq="M"))
    for kind in args.kind:
        board = BOARDS[kind]
        for year in sorted({m.year for m in months}):
            for row in monthly_files(board, year):
                month = int(str(row.get("monthsTxt", "")).rstrip("월") or 0)
                if pd.Period(year=year, month=month, freq="M") not in months:
                    continue
                path = os.path.join(args.out, f"{kind}_{year}_{month:02d}.xlsx")
                if os.path.exists(path):
                    print(f"[skip] {path}")
                    continue
                download(row, board, path)
                print(f"[ok] {path} {os.path.getsize(path) / 1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
