> 이 문서는 팀 수집 폴더(data2)의 원본 README 입니다(2026-10-01). 이 프로젝트에는 실제로 쓰는 파일만 옮겼습니다:
> `smpa/rallies.csv` → `smpa_rallies.csv`, `team_additional/smpa_rallies_team.csv` → `smpa_rallies_team.csv`,
> `team_additional/sangam_events_manual.csv` → `sangam_events_manual.csv`, `festivals/festivals.csv` → `topis_control_notices.csv`,
> `topis/speed/2023-*.csv`·`2024-*.csv` → `data/raw/speed_YYYY_MM.csv` (커밋 제외). 나머지는 프로젝트 밖에 보관합니다.

# 팀원 추가분 (내 수집분과 중복 제외)

원본: `../data_team공유/` (2026-10-01). 아래 파일만 새로 분리함.

| 파일 | 내용 | 비고 |
|---|---|---|
| seoul_stadium_events.csv | 서울시설공단 체육시설 행사(2,426행, 목동 등) | 내 수집분에 없음 |
| sangam_events_manual.csv | 상암 A매치·콘서트 수동 목록(20행) | `time_verified` 0은 시간 미확인 |
| topis_events.csv | TOPIS 공지에서 거점별 이벤트 시간 파싱(207행) | 팀원 가공본 |
| site_hourly.csv | 거점×시간 학습용 데이터셋(2023~2026-09, 131,424행, train/validation/test 분할 포함) | 팀원 가공본 |
| smpa_rallies_team.csv | 집회 정리본(13,192행). 추가 컬럼 dong, march, lane | 내 `smpa/rallies.csv`와 스키마 다름(내 쪽엔 announced_at 있음) |
| topis_speed_추가도로.csv | 경인로 속도(2023~2026, 52,022행) | 내 25개 도로에 없는 도로만 추출 |

## 중복이라 제외한 것
- holidays_kr.csv, topis_notices.jsonl: 내 파일과 바이트 단위로 동일
- smpa_txt/(2,444개), smpa_index.jsonl: 내 smpa/txt, index.jsonl과 동일
- kbo_jamsil_games.csv: 내 kbo/jamsil_games.csv와 동일
- kleague_sangam_games.csv: 2023년 이후 72경기로, 내 파일(2020~, 122경기, 관중 99건)에 모두 포함
- topis_speed.csv(2023~): 내 topis/speed 월별 파일과 동일 출처. 경인로 외 도로는 중복
