"""
이벤트 텍스트(집회 신고 내용, 통제 공지 등)에서 교통에 직접 영향을 주는 속성을 추출합니다.

    march        : 행진(이동) 여부   - 정지 집회보다 넓은 구간을 순차적으로 막음
    lane_control : 차로·도로 통제 여부

추출은 이벤트를 등록할 때 한 번만 수행해 events 테이블에 저장합니다. 학습·서빙 때마다
텍스트를 다시 해석하지 않으므로 결과가 재현 가능하고, LLM 호출 비용도 등록 건수만큼만 듭니다.

백엔드 (환경변수 EVENT_NLP_BACKEND)
    rule (기본) : 키워드 규칙. 외부 호출 없음. 과거 집회 12,000여 건 일괄 적재처럼 대량·재현성이 필요할 때 씁니다.
    openai      : OpenAI API로 구조화 추출. OPENAI_API_KEY 필요, 모델은 EVENT_NLP_OPENAI_MODEL (기본 gpt-5.4-mini).
    anthropic   : Claude API로 구조화 추출. ANTHROPIC_API_KEY 필요.
    llm         : OPENAI_API_KEY 가 있으면 openai, 없으면 anthropic.
    LLM 호출이 실패·거절되면 규칙 기반 결과로 대체합니다.
"""
import logging
import os
import re

from pydantic import BaseModel, Field

logger = logging.getLogger("aiops")

# 2023년 이후 경찰청 원문은 "행진"이라는 단어 없이 "A → B", "A ⇄ B", "※경로(2km, 1개차로)"로 이동 경로만 적는 경우가 많다
# (80건 표본에서 단어만 찾으면 행진 2건, 화살표·경로 표기까지 보면 LLM 추출과 대부분 일치 - reports/rally_nlp_check.md)
_MARCH = re.compile(r"행진|행렬|거리\s*행동|도보\s*이동|경로\s*[(:：]|[→⇄↔]")
_LANE = re.compile(  # 경찰청 원문은 "편도 全차로"처럼 한자(全)를 쓴다
    r"차로\s*(통제|점거|이용|확보)|[전全]\s*차로|\d+\s*개?\s*차로|도로\s*(통제|점거)|교통\s*통제|차량\s*통제|통제\s*구간"
)

LLM_MODEL = os.getenv("EVENT_NLP_MODEL", "claude-opus-5-5")             # anthropic 백엔드
OPENAI_MODEL = os.getenv("EVENT_NLP_OPENAI_MODEL", "gpt-5.4-mini")      # openai 백엔드
_SYSTEM = (
    "서울 도심 행사·집회 안내문에서 교통 영향 속성을 추출합니다. "
    "경찰청 원문의 약어: R=로, 出=출구, 全차로=전 차로, ※행진경로=행진. "
    "본문에 명시된 내용만 근거로 판단하고, 언급이 없으면 false로 둡니다. 집회의 성격이나 주최는 판단하지 않습니다."
)


class EventTextFeatures(BaseModel):
    march: bool = Field(description="행진·이동 행렬이 계획되어 있으면 true")
    lane_control: bool = Field(description="차로 점거, 도로·교통 통제가 언급되면 true")
    march_route: str = Field(default="", description="행진 경로가 있으면 '출발 → 도착' 형식, 없으면 빈 문자열")


def extract_rule(text: str | None) -> dict:
    text = text or ""
    return {"march": bool(_MARCH.search(text)), "lane_control": bool(_LANE.search(text)), "march_route": ""}


def extract_llm(text: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=LLM_MODEL,
        max_tokens=1024,
        output_config={"effort": "low"},
        system=_SYSTEM,
        messages=[{"role": "user", "content": text}],
        output_format=EventTextFeatures,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"LLM extraction returned no result (stop_reason={response.stop_reason})")
    return response.parsed_output.model_dump()


def extract_openai(text: str) -> dict:
    from openai import OpenAI

    response = OpenAI().responses.parse(
        model=OPENAI_MODEL,
        input=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": text}],
        text_format=EventTextFeatures,
    )
    if response.output_parsed is None:
        raise RuntimeError(f"OpenAI extraction returned no result (status={response.status})")
    return response.output_parsed.model_dump()


def extract(text: str | None, backend: str | None = None) -> dict:
    backend = backend or os.getenv("EVENT_NLP_BACKEND", "rule")
    if backend == "llm":
        backend = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
    if backend in ("openai", "anthropic") and text:
        try:
            return extract_openai(text) if backend == "openai" else extract_llm(text)
        except Exception as e:  # 추출 실패가 이벤트 등록을 막으면 안 됨 → 규칙 기반으로 대체
            logger.warning(f"[WARN] event NLP ({backend}) failed, falling back to rules: {e}")
    return extract_rule(text)
