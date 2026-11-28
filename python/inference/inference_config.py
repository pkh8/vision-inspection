from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InferenceConfig:
    """딥러닝 추론에 필요한 실행 설정."""

    # 학습된 모델 파일 경로
    model_path: Path

    # 추론할 이미지 목록 CSV
    predict_csv: Path

    # CSV의 상대 이미지 경로 기준 폴더
    dataset_root: Path

    # 모델 입력 이미지 크기
    image_size: int = 480

    # None이면 모든 라벨을 추론한다.
    # 예: {3, 4, 6}
    allowed_labels: set[int] | None = None

    # 이 값보다 낮으면 재검토로 처리한다.
    confidence_threshold: float = 0.85

    # "cuda", "cpu", 또는 None
    device: str | None = None

    # 모델 내부 클래스 번호와 원본 라벨의 매핑
    label_map_reverse: dict[int, int] = field(
        default_factory=lambda: {
            0: 3,
            1: 4,
            2: 6,
        }
    )

    # 오답 이미지와 평가 결과를 저장할 폴더
    result_dir: Path | None = None

