class ServingAppError(Exception):
    """서빙 앱에서 사용자에게 메시지를 보여주기 위한 도메인 에러."""


class ModelLoadError(ServingAppError):
    """서빙용 모델(MLflow Production 또는 로컬)을 로드하지 못했을 때."""

