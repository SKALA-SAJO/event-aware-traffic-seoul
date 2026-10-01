"""
corridor ↔ TOPIS 링크·교통량 지점 자동 매핑 → config/corridor_links.yaml (생성 파일) + reports/corridor_mapping.md

규칙 (사람이 지도를 보지 않아도 재현 가능하도록 이름·방향·거리만 사용)
    링크  : hubs.yaml 의 road(도로명) · direction(상행/하행) 링크를 "앞 링크 종점명 = 뒤 링크 시점명" 으로 이어
            사슬을 만들고, anchor(경기장·광장 등 노드명)에서 앞뒤로 span_m 안에 중점이 들어오는 링크까지 포함.
    교통량: 교통량 엑셀 '수집지점 주소 및 좌표' 시트의 유출입 방향 "[도로명] A→B" 가 같은 도로이고
            A·B 가 이 corridor 사슬에 같은 순서로 있을 때만 사용 (가까워도 다른 도로·반대 방향이면 쓰지 않음).
            io_type: 엑셀 방향 '유입' = 1, '유출' = 2 (교통량 이력 API 의 io_type 과 같은 의미).

실행: python scripts/map_corridors.py data/raw/speed_2026_09.xlsx data/raw/volume_2026_08.xlsx
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml

from data.config import LINKS_PATH, hubs

IO_TYPE = {"유입": 1, "유출": 2}


def read_links(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="calamine", usecols=["도로명", "링크아이디", "시점명", "종점명", "방향", "거리"])
    df = df.drop_duplicates("링크아이디")
    df.columns = ["road", "link_id", "st", "ed", "dir", "length_m"]
    return df


def read_spots(path: str) -> pd.DataFrame:
    s = pd.read_excel(path, sheet_name="수집지점 주소 및 좌표", engine="calamine")
    s["지점번호"] = s["지점번호"].ffill()
    out = []
    for r in s.itertuples(index=False):
        m = re.match(r"\s*\[([^\]]+)\]\s*(.+?)\s*→\s*(.+?)\s*$", str(r[8]))
        if m and r[1] in IO_TYPE:
            out.append({"spot": r[0], "io_type": IO_TYPE[r[1]], "road": m.group(1), "a": m.group(2), "b": m.group(3),
                        "name": r[2], "lat": r[4], "lon": r[5]})
    return pd.DataFrame(out)


def chain(links: pd.DataFrame, road: str, direction: str, anchor: str, span_m: float) -> list[dict]:
    sel = links[(links["road"] == road) & (links["dir"] == direction)]
    by_st = {r.st: r._asdict() for r in sel.itertuples(index=False)}
    by_ed = {r.ed: r._asdict() for r in sel.itertuples(index=False)}
    if anchor not in by_st and anchor not in by_ed:
        raise SystemExit(f"{road} {direction} 에 '{anchor}' 노드가 없습니다. 노드 후보: {sorted(set(by_st) | set(by_ed))}")
    before, cum, node = [], 0.0, anchor
    while node in by_ed and cum + by_ed[node]["length_m"] / 2 <= span_m and len(before) < 20:
        link = by_ed[node]
        before.insert(0, link)
        cum += link["length_m"]
        node = link["st"]
    after, cum, node = [], 0.0, anchor
    while node in by_st and cum + by_st[node]["length_m"] / 2 <= span_m and len(after) < 20:
        link = by_st[node]
        after.append(link)
        cum += link["length_m"]
        node = link["ed"]
    return before + after


def spots_for(spots: pd.DataFrame, road: str, nodes: list[str]) -> list[dict]:
    pos = {n: i for i, n in enumerate(nodes)}
    out = []
    for r in spots[spots["road"] == road].itertuples(index=False):
        if r.a in pos and r.b in pos and pos[r.a] < pos[r.b]:
            out.append({"spot": r.spot, "io_type": int(r.io_type), "desc": f"[{r.road}] {r.a}→{r.b}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speed_xlsx")
    ap.add_argument("volume_xlsx")
    ap.add_argument("--report", default="reports/corridor_mapping.md")
    args = ap.parse_args()
    links, spots = read_links(args.speed_xlsx), read_spots(args.volume_xlsx)

    out, lines = {}, ["# corridor 자동 매핑 결과", "",
                      f"원천: `{os.path.basename(args.speed_xlsx)}`, `{os.path.basename(args.volume_xlsx)}` "
                      "(규칙은 scripts/map_corridors.py 상단 참고)", ""]
    for hub_id, h in hubs(include_disabled=True).items():
        for cid, c in (h.get("corridors") or {}).items():
            seq = chain(links, c["road"], c["direction"], c["anchor"], float(c.get("span_m", 1000)))
            nodes = [seq[0]["st"]] + [link["ed"] for link in seq]
            vols = spots_for(spots, c["road"], nodes)
            length = sum(link["length_m"] for link in seq)
            out[cid] = {
                "name": f"{c['road']} {nodes[0]}→{nodes[-1]}",
                "short": c.get("short") or f"→{nodes[-1]}",
                "links": [{"link_id": str(link["link_id"]), "length_m": int(link["length_m"]),
                           "st": link["st"], "ed": link["ed"]} for link in seq],
                "volume_spots": [{"spot": v["spot"], "io_type": v["io_type"]} for v in vols],
            }
            lines += [f"## {cid} ({h['name']}) - {out[cid]['name']} ({c['direction']}, {length / 1000:.2f}km)", "",
                      "| 순서 | 링크 ID | 시점 → 종점 | 거리(m) |", "|---|---|---|---|"]
            lines += [f"| {i + 1} | {link['link_id']} | {link['st']} → {link['ed']} | {link['length_m']} |"
                      for i, link in enumerate(seq)]
            lines += ["", "교통량 지점: " + (", ".join(f"{v['spot']} io_type={v['io_type']} ({v['desc']})" for v in vols)
                                         or "같은 도로·같은 방향 지점 없음 → 교통량 결측(마스크)으로 학습"), ""]
            print(f"{cid}: {out[cid]['name']} {len(seq)} links {length / 1000:.2f}km, volume={[v['spot'] for v in vols]}")

    with open(LINKS_PATH, "w", encoding="utf-8") as f:
        f.write("# 자동 생성 파일 - scripts/map_corridors.py 로 다시 만드세요 (직접 수정하지 말 것)\n")
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"→ {LINKS_PATH}, {args.report}")


if __name__ == "__main__":
    main()
