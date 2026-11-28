"""ResNet18 기반 결함 분류 모델 학습 모듈."""

from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from dataset import CarinthiaDataset


# 현재 파일 위치를 기준으로 프로젝트 루트 경로를 계산한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 학습·검증 데이터 CSV 경로
TRAIN_CSV = PROJECT_ROOT / "data" / "sample" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "sample" / "splits" / "val.csv"

# CSV의 상대 이미지 경로를 조합할 기준 폴더
DATASET_ROOT = PROJECT_ROOT / "data" / "sample"

# 학습된 모델을 저장할 경로
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "defect_resnet18_best.pth"

# 학습 기본 설정
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 3


def create_transforms():
    """학습과 검증에 사용할 이미지 전처리 변환을 생성한다."""

    # ResNet18이 사전학습될 때 사용한 정규화 기준을 가져온다.
    weights = ResNet18_Weights.DEFAULT

    # 학습 데이터에는 일부 랜덤 변환을 적용해 과적합을 줄인다.
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights.transforms().mean,
                std=weights.transforms().std,
            ),
        ]
    )

    # 검증 데이터에는 랜덤 변환을 적용하지 않는다.
    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights.transforms().mean,
                std=weights.transforms().std,
            ),
        ]
    )

    return train_transform, val_transform


def create_class_weights(dataset):
    """클래스별 이미지 수 차이를 보정할 가중치를 계산한다."""

    # Dataset에 저장된 학습 라벨을 가져온다.
    labels = [
        sample["label"]
        for sample in dataset.samples
    ]

    # 각 클래스의 이미지 개수를 센다.
    counts = Counter(labels)

    weights = []

    # 이미지가 적은 클래스일수록 더 큰 가중치를 부여한다.
    for class_index in range(NUM_CLASSES):
        class_count = counts.get(class_index, 1)
        weights.append(1.0 / class_count)

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    # 가중치 평균을 클래스 수 기준으로 보정한다.
    class_weights = (
        class_weights
        / class_weights.sum()
        * NUM_CLASSES
    )

    return class_weights


def create_model():
    """ImageNet 사전학습 ResNet18을 3개 클래스 분류 모델로 변경한다."""

    # 사전학습된 ResNet18 가중치를 불러온다.
    weights = ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    # 기존 특징 추출부는 고정하고 학습하지 않는다.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 마지막 분류층만 현재 결함 클래스 수에 맞게 교체한다.
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    return model


def evaluate(model, data_loader, device):
    """검증 데이터의 평균 손실과 정확도를 계산한다."""

    # 평가 모드에서는 Dropout 등의 학습 동작을 끈다.
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # 검증에서는 클래스 가중치 없이 기본 손실을 계산한다.
    loss_function = nn.CrossEntropyLoss()

    # 검증 과정에서는 역전파를 수행하지 않는다.
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 모델의 예측 결과를 계산한다.
            outputs = model(images)

            loss = loss_function(
                outputs,
                labels,
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            # 가장 높은 점수를 가진 클래스를 예측값으로 선택한다.
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    """학습·검증을 반복하고 가장 좋은 모델을 저장한다."""

    # CUDA 사용 가능 여부에 따라 실행 장치를 선택한다.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("사용 장치:", device)

    # 학습·검증용 이미지 변환을 준비한다.
    train_transform, val_transform = (
        create_transforms()
    )

    # 학습 Dataset을 생성한다.
    train_dataset = CarinthiaDataset(
        csv_path=TRAIN_CSV,
        dataset_root=DATASET_ROOT,
        transform=train_transform,
    )

    # 검증 Dataset을 생성한다.
    val_dataset = CarinthiaDataset(
        csv_path=VAL_CSV,
        dataset_root=DATASET_ROOT,
        transform=val_transform,
    )

    print("학습 데이터:", len(train_dataset))
    print("검증 데이터:", len(val_dataset))

    # Dataset을 배치 단위로 제공하는 DataLoader를 생성한다.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 모델을 생성하고 GPU 또는 CPU로 이동한다.
    model = create_model()
    model = model.to(device)

    # 클래스 불균형 보정값을 실행 장치로 이동한다.
    class_weights = create_class_weights(
        train_dataset
    ).to(device)

    # 클래스 가중치를 적용한 손실 함수를 생성한다.
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # 학습 가능한 마지막 분류층만 Optimizer에 등록한다.
    optimizer = torch.optim.Adam(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    # 모델 저장 폴더를 생성한다.
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_accuracy = 0.0

    # 전체 학습 데이터를 여러 Epoch 동안 반복한다.
    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # 학습 데이터를 배치 단위로 처리한다.
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 이전 배치의 Gradient를 초기화한다.
            optimizer.zero_grad()

            # 모델 예측과 손실을 계산한다.
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # 손실을 이용해 Gradient를 계산하고 가중치를 갱신한다.
            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        # 현재 Epoch의 학습 손실과 정확도를 계산한다.
        train_loss = running_loss / total
        train_accuracy = correct / total

        # 학습이 끝난 뒤 검증 데이터로 성능을 확인한다.
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"[{epoch + 1:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )

        # 지금까지 가장 높은 검증 정확도일 때 모델을 저장한다.
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "class_count": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "best_val_accuracy":
                        best_accuracy,
                },
                MODEL_PATH,
            )

            print(
                "최고 모델 저장:",
                MODEL_PATH,
            )

    print(
        "학습 완료. 최고 검증 정확도:",
        best_accuracy,
    )


if __name__ == "__main__":
    main()