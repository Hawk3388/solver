"""Consent-based local feedback and reviewed YOLO dataset contributions."""

import hashlib
import json
from pathlib import Path
import secrets
import shutil
from threading import RLock
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, StrictBool
import yaml


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=5, max_length=6000)
    store_image: StrictBool = False
    add_to_training: StrictBool = False
    annotations_reviewed: StrictBool = False
    document_version: int | None = Field(default=None, ge=0, strict=True)


class FeedbackStore:
    def __init__(self, directory, dataset):
        self.directory = Path(directory)
        self.dataset = Path(dataset)
        self.lock = RLock()

    @staticmethod
    def _atomic_text(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(path.name + ".pending")
        try:
            staging.write_text(content, encoding="utf-8")
            staging.replace(path)
        finally:
            staging.unlink(missing_ok=True)

    def submit(self, payload, document, page, filename):
        request = FeedbackRequest.model_validate(payload)
        if not request.description.strip():
            raise ValueError("Describe the error before submitting feedback.")
        if request.add_to_training and not (request.store_image and request.annotations_reviewed):
            raise ValueError("Dataset inclusion requires image-storage consent and reviewed annotations.")
        if request.add_to_training and document is None:
            raise ValueError("Open the editor and review the annotations before contributing.")
        if request.add_to_training and request.document_version != document["version"]:
            raise ValueError("The annotations changed after review. Reopen the editor and review the current document before contributing.")
        with self.lock:
            report_id = secrets.token_hex(16)
            directory = self.directory / report_id
            directory.mkdir(parents=True)
            report = {
                "id": report_id, "filename": filename,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **request.model_dump(), "dataset": None,
            }
            try:
                if request.store_image:
                    if not Path(page).is_file():
                        raise ValueError("No normalized image is available for this report.")
                    shutil.copyfile(page, directory / "worksheet.png")
                    if document is not None:
                        self._atomic_text(directory / "annotations.json", json.dumps(document, ensure_ascii=False, indent=2))
                if request.add_to_training:
                    report["dataset"] = self._add_to_dataset(document, page)
                self._atomic_text(directory / "report.json", json.dumps(report, ensure_ascii=False, indent=2))
                return {"id": report_id, "added_to_training": report["dataset"] is not None,
                        "dataset": report["dataset"]}
            except Exception:
                shutil.rmtree(directory)
                raise

    def _add_to_dataset(self, document, page):
        config_path = self.dataset / "data.yaml"
        class_ids = document["class_ids"]
        config = None
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("The existing dataset configuration is invalid.")
            names = config.get("names", {})
            names = {str(i): str(name) for i, name in enumerate(names)} if isinstance(names, list) else {str(i): str(name) for i, name in names.items()}
            if names != class_ids:
                raise ValueError("Detector classes do not match dataset/data.yaml. Review the class mapping first.")
            configured_root = Path(str(config.get("path", ".")))
            configured_root = configured_root if configured_root.is_absolute() else config_path.parent / configured_root
            if configured_root.resolve() != self.dataset.resolve():
                raise ValueError("dataset/data.yaml points to a different dataset root.")
            if config.get("train") != "images/train" or config.get("val") != "images/val":
                raise ValueError("Feedback requires the standard images/train and images/val dataset layout.")
        reverse = {name: key for key, name in class_ids.items()}
        width, height = document["width"], document["height"]
        labels = []
        for region in document["regions"]:
            x1, y1, x2, y2 = region["box"]
            if region["class_name"] not in reverse:
                raise ValueError("Unknown class in reviewed annotations.")
            labels.append(f"{reverse[region['class_name']]} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
        digest = hashlib.sha256(Path(page).read_bytes()).hexdigest()
        # Every version of the same image stays in the same split.
        split = "val" if int(digest[:8], 16) % 5 == 0 else "train"
        name = f"feedback_{digest}"
        image_target = self.dataset / "images" / split / f"{name}.png"
        label_target = self.dataset / "labels" / split / f"{name}.txt"
        label_text = "\n".join(sorted(labels)) + ("\n" if labels else "")
        for other_split in ("train", "val"):
            existing = self.dataset / "labels" / other_split / f"{name}.txt"
            if existing.exists():
                if existing.read_text(encoding="utf-8") != label_text:
                    raise ValueError("This image already has different feedback labels. Review the existing sample rather than creating conflicting duplicates.")
                return {"name": name, "split": other_split, "duplicate": True}
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        image_staging = image_target.with_suffix(".png.pending")
        try:
            shutil.copyfile(page, image_staging)
            image_staging.replace(image_target)
            self._atomic_text(label_target, label_text)
            if config is None:
                config = {"path": str(self.dataset.resolve()), "train": "images/train", "val": "images/val",
                          "nc": len(class_ids), "names": {int(i): name for i, name in class_ids.items()}}
                self._atomic_text(config_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        except Exception:
            image_target.unlink(missing_ok=True)
            label_target.unlink(missing_ok=True)
            raise
        finally:
            image_staging.unlink(missing_ok=True)
        return {"name": name, "split": split, "duplicate": False}
