# 서빙 컨테이너 (단일 컨테이너 - MLflow 도 sqlite 로 컨테이너 안에서 동작)
# 빌드는 프로젝트 루트에서: docker build -f serving_app/Dockerfile -t traffic-serving .
# 또는:                     docker compose -f serving_app/docker-compose.yml up --build
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY data/*.py data/
COPY data/collectors/ data/collectors/
COPY data/templates/ data/templates/
COPY serving_app/ serving_app/
COPY scripts/ scripts/

# 실데이터 DB(data/traffic.db: TOPIS 속도·교통량 + KBO·K리그 일정, README "실데이터 적재" 로 생성)를 넣고
# 빌드 시점에 base 학습 → 게이트 → Production 승격까지 끝내 둔 자기 완결형 이미지.
# 합성 데이터로 만들려면: docker build --build-arg DATA=synthetic ...
ARG DATA=real
COPY data/traffic.db data/traffic.db
RUN if [ "$DATA" = "synthetic" ]; then rm data/traffic.db && python scripts/generate_synthetic_data.py; fi \
 && python serving_app/train_and_register.py

EXPOSE 8099
ENV MODEL_SOURCE=mlflow \
    LOADING_MODE=eager

CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8099"]
