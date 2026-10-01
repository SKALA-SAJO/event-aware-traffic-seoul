"""
[Day1 사전 준비] 로컬 baseline 모델 만들기  —  scripts/train_baseline_v1.py
【실습용】 ___ (밑줄 3개)만 채우세요. 채울 곳은 [빈칸 N] 으로 표시되어 있습니다.
   ___ 가 남은 채 실행하면 "name '___' is not defined" 에러가 나며, 그 줄이 채울 곳입니다.

■ 이 파일이 하는 일 (한 줄 요약)
   업로드된 주가 CSV로 LSTM 모델을 학습시키고,
   서버가 읽어 갈 파일 2개(스케일러, 모델)를 만들어 둡니다.

■ 왜 필요한가요?
   Day1 서버(FastAPI)는 직접 학습하지 않고, "이미 학습된 모델 파일"을 읽어서 예측만 합니다.
   Day2에서 MLflow를 배우기 전까지는 이 스크립트가 만든 파일이 서버의 유일한 모델입니다.

■ 만들어지는 파일 (이름·위치를 바꾸지 마세요 — 서버가 이 경로로 찾습니다)
   serving_app/models/scaler.pkl      ← 숫자를 0~1로 바꿔 주는 '자' (Day1~3 내내 계속 사용)
   serving_app/models/haic_v1.keras   ← 학습된 LSTM 모델

■ 실행 순서
   1) 서버 실행       : uvicorn serving_app.main:app --host 0.0.0.0 --port 8077
   2) 데이터 업로드   : 대시보드(http://localhost:8077)에서 data/sample_haic_prices.csv 업로드
   3) 이 파일 실행    : (다른 터미널에서) python scripts/train_baseline_v1.py

■ 다 됐는지 확인하는 법
   약 1분 뒤 터미널에 아래 3줄이 나오고, RMSE 가 대략 2~4 사이면 성공입니다.
     scaler fit on 756행 -> serving_app/models/scaler.pkl
     baseline v1 RMSE = x.xx  (배포 게이트: $4.00)
     saved -> serving_app/models/haic_v1.keras
   ※ RMSE 가 수십 이상으로 크게 나오면 [빈칸 1]을 다시 보세요.

■ 이 파일의 빈칸 : [빈칸 1]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 이 파일에서 쓰는 도구들 (모두 이미 만들어져 있습니다)
#   load_rows        : CSV 파일 → [{"Date":..., "Close":..., "Volume":...}, ...]
#   build_sequences  : 행 목록 → 학습용 문제(X)와 정답(y)
#   train_test_split : 앞 80%는 공부용(train), 뒤 20%는 시험용(test)으로 나누기
#   HAICScaler       : 0~1로 바꾸기(transform_point, scale_close) / 달러로 되돌리기(inverse_close)
#   latest_upload    : data/uploads/ 에서 가장 최근 올린 CSV 경로
#   build_model      : LSTM 모델 뼈대 만들기
from data.features import load_rows, build_sequences, train_test_split, HAICScaler
from data.storage import latest_upload
from serving_app.lstm_model import build_model

MODEL_PATH = "serving_app/models/haic_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = 100  # 전체 문제를 100번 반복해서 학습


def rmse(y_true, y_pred) -> float:
    """예측이 실제보다 '평균 몇 달러' 틀렸는지 계산합니다."""
    return (sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)) ** 0.5


def main():
    import numpy as np

    # STEP 1. 데이터 읽기 — 가장 최근 업로드한 CSV를 행 목록으로 (756행)
    rows = load_rows(latest_upload())

    # STEP 2. 스케일러 만들고 저장하기
    #   종가(100~200)와 거래량(100만 단위)은 크기 차이가 너무 커서, 그대로 넣으면
    #   LSTM이 큰 숫자(거래량)에만 끌려갑니다. 그래서 둘 다 0~1로 맞춰 줍니다.
    #   스케일러는 "여기서 딱 한 번만" fit 하고 저장합니다. 서버·Day2·Day3 모두 이 파일을 씁니다.
    scaler = HAICScaler().fit(rows)
    scaler.save(SCALER_PATH)
    print(f"scaler fit on {len(rows)}행 -> {SCALER_PATH}")

    # STEP 3. 문제(X)와 정답(y) 만들기
    #   "최근 20일치 (종가, 거래량)을 보고 → 21일째 종가를 맞혀라" 는 문제를 하루씩 밀며 만듭니다.
    #   X = 문제 목록 (0~1로 변환됨),  y = 정답 종가 목록 (주의: 아직 "달러" 그대로!)
    X, y = build_sequences(rows, scaler)

    # STEP 4. 공부용 / 시험용 나누기
    #   주가는 시간 순서가 중요하므로 섞지 않고 "앞 80% 공부 / 뒤 20% 시험"으로 나눕니다.
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")

    # STEP 5. 공부용 정답(y_train)도 0~1로 바꾸기   ← [빈칸 1]을 풀 때 꼭 다시 읽으세요
    #   문제(X_train)는 0~1 범위인데, 정답(y_train)은 아직 "달러"입니다. (STEP 3 주석 참고)
    #   문제는 0.4인데 정답은 150이면, 모델 출력이 150까지 억지로 끌려가야 해서 학습이 불안정해집니다.
    #   그래서 정답도 scale_close 로 0~1로 바꾼 y_train_scaled 를 따로 만들어 둡니다.
    y_train_scaled = np.array([scaler.scale_close(v) for v in y_train], dtype="float32")

    # STEP 6. 빈 LSTM 모델 만들기
    model = build_model()

    # ════════════════════════════ [빈칸 1] ════════════════════════════
    # 모델을 학습시킬 "문제"와 "정답"을 넣어 주세요.  model.fit(문제, 정답, ...)
    #
    #   생각해 볼 질문
    #     · 지금 쓸 수 있는 변수: X, y, X_train, y_train, X_test, y_test, y_train_scaled
    #     · 시험용 데이터(X_test, y_test)가 학습에 섞이면 시험 점수(RMSE)는 믿을 수 있을까요?
    #     · 정답은 STEP 5에서 만든 y_train_scaled 와 원래 y_train 중 무엇이어야 할까요?
    model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

    # STEP 7. 시험 보기 — 모델 출력(0~1)을 달러로 되돌린 뒤 실제 정답(y_test, 달러)과 비교
    preds_scaled = model.predict(X_test, verbose=0).flatten()
    preds = [scaler.inverse_close(p) for p in preds_scaled]  # 실제 달러 단위로 복원
    score = rmse(y_test, preds)
    print(f"baseline v1 RMSE = {score:.2f}  (배포 게이트: $4.00)")

    # STEP 8. 모델 저장
    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}")
    if score > 4.00:
        print(
            "※ 참고: 이 RMSE는 Day1 로컬 모델이며 배포 게이트($4.00) 통과 여부는 "
            "Day2에서 MLflow로 다시 정식 검증합니다."
        )


if __name__ == "__main__":
    main()
