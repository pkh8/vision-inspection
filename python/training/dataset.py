from pathlib import Path
import csv

from PIL import Image
from torch.utils.data import Dataset


class CarinthiaDataset(Dataset):
    """Carinthia CSV와 이미지 파일을 연결하는 Dataset."""



    LABEL_MAP ={
        3:0,
        4:1,
        6:2,
    }

    def __init__(
        self,
        csv_path: str | Path,
        dataset_root: str | Path,
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.samples = []

        with self.csv_path.open(
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


                if original_label not in self.LABEL_MAP:
                    continue

                image_path = (
                    self.dataset_root
                    / row["image_path"]
                )

                label = self.LABEL_MAP[
                original_label]

                self.samples.append(
                    {
                        "image_path": image_path,
                        "label": label,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label