"""
경찰청 "오늘의 주요집회" 일일 수집 (GitHub Actions 용, 하루 1회).

    python scripts/collect_rallies.py --out collected          # pdftotext (poppler-utils) 필요

    collected/smpa_txt/YYMMDD.txt        PDF 를 텍스트로 바꾼 원문 (나중에 파서를 고쳐 다시 읽을 수 있게 보관)
    collected/rallies/YYYY-MM-DD.csv     그날 받은 집회 목록 (date, start, end, count, station, place, text,
                                         written_at, posted_date, announced_at, fetched_at). 같은 날 다시 돌리면 덮어씀

학습 피처(event_sources 의 smpa)인데 지금까지는 10-01 까지만 들어 있어, 10-02 부터 집회가 비는 문제를 막는다.
게시판 최신 글(행사일이 어제 이후인 것)만 받으며, 이벤트 변환은 scripts/import_collected.py 가 맡는다
(announced_at 규칙·날짜 점검은 data/collectors/smpa_live.py 설명 참고).
"""
import argparse
import csv
import datetime as dt
import glob
import os
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collectors import smpa_live as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="collected")
    ap.add_argument("--since-days", type=int, default=1, help="행사일이 오늘 기준 N일 전 이후인 글만")
    ap.add_argument("--pages", type=int, default=2, help="게시판 목록에서 볼 쪽 수")
    args = ap.parse_args()

    now = dt.datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None, second=0, microsecond=0)
    cutoff = (now.date() - dt.timedelta(days=args.since_days)).strftime("%y%m%d")
    txt_dir = os.path.join(args.out, "smpa_txt")
    os.makedirs(txt_dir, exist_ok=True)

    posts = S.list_posts(args.pages)
    print(f"게시글 {len(posts)}건 (최신 {args.pages}쪽), 행사일 {cutoff} 이후만 받음")
    by_day: dict[str, dict] = {}
    for p in posts:
        if p["event_ymd"] >= cutoff:
            old = by_day.get(p["event_ymd"])
            if old is None or (p["posted"] and (old["posted"] is None or p["posted"] < old["posted"])):
                by_day[p["event_ymd"]] = p  # 같은 날짜 글이 여럿이면 가장 먼저 올라온 것

    # 이미 보관한 다른 날짜 원문과 내용이 같으면 "그 날짜의 실제 자료가 없어 복사한 것"이므로 제외하기 위한 지문
    seen: dict[str, list[str]] = {}
    for path in glob.glob(os.path.join(txt_dir, "*.txt")):
        seen.setdefault(S.digest(open(path, encoding="utf-8", errors="ignore").read()), []).append(os.path.basename(path)[:6])

    rows, skipped, failed = [], [], 0
    for ymd, post in sorted(by_day.items()):
        try:
            got = S.fetch_text(post)
        except Exception as e:  # 한 건 실패가 나머지를 막지 않게
            failed += 1
            print(f"  {ymd}: 받기 실패 {e!r}")
            continue
        if got is None:
            skipped.append((ymd, "PDF 없음"))
            continue
        ymd, text = got
        same = [o for o in seen.get(S.digest(text), []) if o != ymd]
        info = S.header_info(ymd, text, post["posted"], same)
        if info["date_check"] in ("duplicate", "mismatch"):
            skipped.append((ymd, info["date_check"]))
            continue
        with open(os.path.join(txt_dir, f"{ymd}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        seen.setdefault(S.digest(text), []).append(ymd)
        for r in S.parse_blocks(text, info["date"]):
            rows.append({**r, **{k: info[k] for k in ("written_at", "posted_date", "announced_at")},
                         "fetched_at": now.isoformat(sep=" ", timespec="minutes")})

    if rows:
        path = os.path.join(args.out, "rallies", f"{now:%Y-%m-%d}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=S.ROW_COLS + ["fetched_at"])
            w.writeheader()
            w.writerows(rows)
    days = sorted({r["date"] for r in rows})
    print(f"{now} 집회 {len(rows)}건 ({len(days)}일: {days[0] if days else '-'} ~ {days[-1] if days else '-'}), "
          f"신고 1,000명 이상 {sum(r['count'] >= 1000 for r in rows)}건, 제외 {skipped}, 실패 {failed}")
    if not by_day or (failed and not rows):
        sys.exit(1)  # 접속 차단·게시판 구조 변경이면 Actions 에서 실패로 보이게


if __name__ == "__main__":
    main()
