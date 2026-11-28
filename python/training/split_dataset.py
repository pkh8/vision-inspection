from pathlib import Path
import csv
import random
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "data"
    / "carinthia.csv"
)

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "testImages"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "splits"
)

RANDOM_SEED = 42
VALIDATION_RATIO = 0.2


def read_samples() -> list[dict]:
    """CSV에서 이미지 경로와 라벨을 읽는다."""
    with CSV_PATH.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        return list(reader)


def write_samples(
    file_path: Path,
    samples: list[dict],
) -> None:
    """분할된 샘플 목록을 CSV로 저장한다."""
    with file_path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_path",
                "file_name",
                "label",
            ],
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(samples)


def main() -> None:
    random.seed(RANDOM_SEED)

    samples = read_samples()

    test_file_names = {
        file_path.name
        for file_path in TEST_IMAGE_DIR.iterdir()
        if file_path.is_file()
    }

    train_validation_samples = [
        sample
        for sample in samples
        if sample["file_name"] not in test_file_names
    ]

    test_samples = [
        sample
        for sample in samples
        if sample["file_name"] in test_file_names
    ]

    samples_by_label = defaultdict(list)

    for sample in train_validation_samples:
        samples_by_label[sample["label"]].append(sample)

    train_samples = []
    validation_samples = []

    for label_samples in samples_by_label.values():
        random.shuffle(label_samples)

        validation_count = max(
            1,
            int(len(label_samples) * VALIDATION_RATIO),
        )

        validation_samples.extend(
            label_samples[:validation_count]
        )

        train_samples.extend(
            label_samples[validation_count:]
        )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_samples(
        SPLIT_DIR / "train.csv",
        train_samples,
    )

    write_samples(
        SPLIT_DIR / "val.csv",
        validation_samples,
    )

    write_samples(
        SPLIT_DIR / "test.csv",
        test_samples,
    )

    print("학습 이미지:", len(train_samples))
    print("검증 이미지:", len(validation_samples))
    print("테스트 이미지:", len(test_samples))


if __name__ == "__main__":
    main()