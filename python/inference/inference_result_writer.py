"""딥러닝 추론 결과를 실행 시간별 폴더에 저장하는 모듈."""

from dataclasses import asdict, is_dataclass
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil


class InferenceResultWriter:
    """실행할 때마다 현재 시간 기준 하위 폴더를 만들고 결과를 저장한다."""

    def __init__(
        self,
        base_output_directory: Path | None,
        run_folder_name: str | None = None,
    ):
        self.base_output_directory = base_output_directory

        if self.base_output_directory is None:
            self.run_dir: Path | None = None
            self.run_id: str | None = None
            return

        # 실행 시점의 로컬 시간 기준으로 폴더명을 만든다.
        self.run_id = (
            run_folder_name
            if run_folder_name
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.run_dir = self.base_output_directory / self.run_id
        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_wrong_image(
        self,
        image_path: Path,
        actual_class: int,
        predicted_class: int,
        confidence: float,
        sample_index: int | None = None,
    ) -> Path | None:
        """오답 이미지를 현재 실행 시간 폴더에 복사한다."""

        if self.run_dir is None:
            return None

        index_part = (
            f"idx_{sample_index:04d}_"
            if sample_index is not None
            else ""
        )

        output_path = (
            self.run_dir
            / (
                f"{index_part}"
                f"actual_{actual_class}_"
                f"pred_{predicted_class}_"
                f"conf_{confidence:.3f}_"
                f"{image_path.name}"
            )
        )

        shutil.copy2(
            image_path,
            output_path,
        )

        return output_path

    def save_predictions_csv(
        self,
        records: list[dict],
        filename: str = "predictions.csv",
    ) -> Path | None:
        """샘플별 추론 결과를 현재 실행 시간 폴더의 CSV로 저장한다."""

        if self.run_dir is None:
            return None

        csv_path = self.run_dir / filename
        fieldnames = [
            "run_id",
            "image_name",
            "image_path",
            "actual_class",
            "predicted_class",
            "confidence",
            "decision",
            "is_correct",
        ]

        with csv_path.open(
            mode="w",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(records)

        return csv_path

    def save_summary_json(
        self,
        config,
        total: int,
        correct: int,
        accuracy: float,
        class_total: dict[int, int],
        class_correct: dict[int, int],
        filename: str = "summary.json",
    ) -> Path | None:
        """전체 정확도와 클래스별 정확도를 현재 실행 시간 폴더의 JSON으로 저장한다."""

        if self.run_dir is None:
            return None

        class_metrics = {}
        for class_label in sorted(class_total):
            total_count = class_total[class_label]
            correct_count = class_correct[class_label]
            class_metrics[str(class_label)] = {
                "correct": correct_count,
                "total": total_count,
                "accuracy": (
                    correct_count / total_count
                    if total_count > 0
                    else 0.0
                ),
            }

        config_payload = (
            asdict(config) if is_dataclass(config) else {}
        )
        for key, value in config_payload.items():
            if isinstance(value, Path):
                config_payload[key] = str(value)
            elif isinstance(value, set):
                config_payload[key] = sorted(value)

        summary = {
            "run_id": self.run_id,
            "model_path": str(config.model_path),
            "predict_csv": str(config.predict_csv),
            "base_result_dir": str(self.base_output_directory),
            "run_result_dir": str(self.run_dir),
            "total_samples": total,
            "correct_samples": correct,
            "accuracy": accuracy,
            "class_accuracy": class_metrics,
            "config": config_payload,
        }

        json_path = self.run_dir / filename
        with json_path.open(
            mode="w",
            encoding="utf-8",
        ) as json_file:
            json.dump(
                summary,
                json_file,
                indent=2,
                ensure_ascii=False,
            )

        return json_path