"""
거점별 향후 HORIZON시간 통행속도 예측 LSTM (base 학습 · fine-tuning · 실험이 공유).

    past   (LOOKBACK, F_past)  ─ LSTM(64) ─ LSTM(32) ─────────┐
                                                             ├─ Dense(64) ─ Dense(HORIZON)
    future (HORIZON, F_future) ─ TimeDistributed(Dense(16)) ─ Flatten ┘

과거 시퀀스는 LSTM이 "지금 도로 상태"를 요약하고, 예측 대상 시각의 미리 알려진 정보
(공휴일·이벤트 일정·거점)는 별도 경로로 붙입니다. 이벤트 피처를 과거 시퀀스에만 넣으면
"6시간 뒤 경기가 끝난다"는 정보를 모델이 받을 길이 없기 때문입니다.
"""
from tensorflow import keras

from data.features import FeatureSpec


def build_model(spec: FeatureSpec, lr: float = 1e-3) -> keras.Model:
    past = keras.layers.Input(shape=(spec.lookback, len(spec.past_names())), name="past")
    future = keras.layers.Input(shape=(spec.horizon, len(spec.future_names())), name="future")

    h = keras.layers.LSTM(64, return_sequences=True)(past)
    h = keras.layers.LSTM(32)(h)
    f = keras.layers.TimeDistributed(keras.layers.Dense(16, activation="relu"))(future)
    f = keras.layers.Flatten()(f)

    z = keras.layers.Concatenate()([h, f])
    z = keras.layers.Dense(64, activation="relu")(z)
    out = keras.layers.Dense(spec.horizon, name="speed_z")(z)

    model = keras.Model(inputs=[past, future], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model
