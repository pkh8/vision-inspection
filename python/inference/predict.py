"""학습된 ResNet18 모델로 CSV 이미지들을 추론한다."""

from collections import defaultdict
import csv
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

from inference_config import InferenceConfig
from inference_result_writer import InferenceResultWriter


def create_inference_config() -> InferenceConfig:
    """현재 추론에 사용할 설정을 생성한다."""

    project_root = Path(__file__).resolve().parents[2]

    return InferenceConfig(
        model_path=(
            project_root
            / "models"
            # / "defect_resnet18_best.pth"
            / "defect_resnet18_480.pth"
        ),
        predict_csv=(
            project_root
            / "data"
            / "sample"
            / "splits"
            / "val.csv"
        ),
        dataset_root=(
            project_root
            / "data"
            / "sample"
        ),
        result_dir=(
            project_root
            / "results"
            # / "dl_validation"
            / "inference_480"
        ),
        image_size=480,
        # None이면 CSV의 모든 라벨을 추론한다.
        allowed_labels=None,
        confidence_threshold=0.8,
        device=None,
    )


def create_device(
    config: InferenceConfig,
) -> torch.device:
    """설정과 GPU 사용 가능 여부에 따라 실행 장치를 선택한다."""

    if config.device is not None:
        return torch.device(config.device)

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def create_transform(
    config: InferenceConfig,
):
    """학습 당시와 동일한 이미지 전처리를 생성한다."""

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    config.image_size,
                    config.image_size,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def create_model(
    config: InferenceConfig,
    device: torch.device,
):
    """ResNet18 구조를 만들고 학습된 가중치를 불러온다."""

    checkpoint = torch.load(
        config.model_path,
        map_location=device,
        weights_only=False,
    )

    ckpt_image_size = checkpoint.get("image_size")
    print("체크포인트 저장 image_size:", ckpt_image_size)

    if (
        ckpt_image_size is not None
        and ckpt_image_size != config.image_size
    ):
        raise ValueError(
            "모델 크기 불일치: "
            f"checkpoint={ckpt_image_size}, "
            f"config={config.image_size}"
        )

    model = models.resnet18(
        weights=None
    )

    num_classes = len(
        config.label_map_reverse
    )

    # 학습 당시와 동일한 분류층을 생성한다.
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    # 추론 모드에서는 Dropout과 BatchNorm의 학습 동작을 끈다.
    model.eval()

    return model


def load_samples(
    config: InferenceConfig,
) -> list[dict]:
    """CSV에서 추론할 이미지 경로와 실제 라벨을 읽는다."""

    samples = []

    with config.predict_csv.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        for row in reader:
            original_label = int(row["label"])

            # None이면 모든 라벨을 포함한다.
            if (
                config.allowed_labels is not None
                and original_label
                not in config.allowed_labels
            ):
                continue

            image_path = (
                config.dataset_root
                / row["image_path"]
            )

            samples.append(
                {
                    "image_path": image_path,
                    "label": original_label,
                }
            )

    return samples


def predict_image(
    model,
    image_path: Path,
    image_transform,
    device: torch.device,
    config: InferenceConfig,
):
    """이미지 한 장을 추론하고 예측 라벨과 신뢰도를 반환한다."""

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = image_transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # 추론 단계에서는 Gradient를 계산하지 않는다.
    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

    confidence, predicted_index = (
        probabilities[0].max(dim=0)
    )

    predicted_class = config.label_map_reverse[
        predicted_index.item()
    ]

    return (
        predicted_class,
        confidence.item(),
    )


def main():
    """CSV 이미지들을 추론하고 실제 라벨과 결과를 비교한다."""

    config = create_inference_config()
    device = create_device(config)
    image_transform = create_transform(config)

    print("사용 장치:", device)
    print("추론 CSV:", config.predict_csv)
    print("모델 경로:", config.model_path)
    print("입력 크기:", config.image_size)
    print("기준 결과 폴더:", config.result_dir)

    samples = load_samples(config)
    if not samples:
        print("추론할 이미지가 없습니다.")
        return

    model = create_model(
        config,
        device,
    )

    # 실행 시각 기준 하위 폴더를 생성한다.
    result_writer = InferenceResultWriter(
        base_output_directory=config.result_dir,
    )
    print("이번 실행 저장 폴더:", result_writer.run_dir)

    # 클래스별 전체 개수와 정답 개수를 저장한다.
    class_total = defaultdict(int)
    class_correct = defaultdict(int)
    records: list[dict] = []

    correct = 0
    total = 0

    for sample in samples:
        image_path = sample["image_path"]
        actual_class = sample["label"]

        predicted_class, confidence = predict_image(
            model,
            image_path,
            image_transform,
            device,
            config,
        )

        is_correct = (
            predicted_class == actual_class
        )

        total += 1
        class_total[actual_class] += 1

        if is_correct:
            # 전체 정답 수와 클래스별 정답 수를 증가시킨다.
            correct += 1
            class_correct[actual_class] += 1
        else:
            # 오답 이미지는 현재 실행 시간 폴더에 저장한다.
            result_writer.save_wrong_image(
                image_path=image_path,
                actual_class=actual_class,
                predicted_class=predicted_class,
                confidence=confidence,
                sample_index=total,
            )

        # 신뢰도 기준에 따른 추가 판정
        if confidence < config.confidence_threshold:
            decision = "재검토 필요"
        else:
            decision = "자동 판정"

        records.append(
            {
                "run_id": result_writer.run_id,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "actual_class": actual_class,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 6),
                "decision": decision,
                "is_correct": is_correct,
            }
        )

        print(
            f"{image_path.name} | "
            f"예측: {predicted_class} | "
            f"실제: {actual_class} | "
            f"신뢰도: {confidence:.4f} | "
            f"{decision} | "
            f"{'정답' if is_correct else '오답'}"
        )

    accuracy = correct / total

    csv_saved_path = result_writer.save_predictions_csv(records)
    json_saved_path = result_writer.save_summary_json(
        config=config,
        total=total,
        correct=correct,
        accuracy=accuracy,
        class_total=class_total,
        class_correct=class_correct,
    )

    print(
        f"\n추론 정확도: "
        f"{correct}/{total} "
        f"({accuracy:.4f})"
    )

    if csv_saved_path is not None:
        print("예측 로그 CSV 저장:", csv_saved_path)
    if json_saved_path is not None:
        print("요약 JSON 저장:", json_saved_path)

    print("\n클래스별 정확도:")

    for class_label in sorted(class_total):
        total_count = class_total[class_label]
        correct_count = class_correct[class_label]

        class_accuracy = (
            correct_count / total_count
        )

        print(
            f"라벨 {class_label}: "
            f"{correct_count}/{total_count} "
            f"({class_accuracy:.4f})"
        )


if __name__ == "__main__":
    main()