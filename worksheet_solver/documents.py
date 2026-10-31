"""Editable document data and rendering without language-model calls."""

from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .detection import DetectionMixin
from .rendering import RenderingMixin


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: int = Field(gt=0, strict=True)
    answer_id: int = Field(gt=0, strict=True)
    class_name: str = Field(min_length=1, max_length=100)
    box: list[float] = Field(min_length=4, max_length=4)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: int = Field(gt=0, strict=True)
    text: str = Field(max_length=12000)
    font_size: int | None = Field(default=None, ge=6, le=96)
    offset_x: float = Field(default=0, ge=-500, le=500)
    offset_y: float = Field(default=0, ge=-500, le=500)


class DocumentEdits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=0, strict=True)
    regions: list[Region] = Field(max_length=500)
    answers: list[Answer] = Field(max_length=500)


def validate_edits(payload, document):
    edits = DocumentEdits.model_validate(payload)
    if edits.version != document["version"]:
        raise ValueError("This document changed. Reopen the editor before saving.")
    answers = {answer.id for answer in edits.answers}
    regions = {region.id for region in edits.regions}
    if len(answers) != len(edits.answers) or len(regions) != len(edits.regions):
        raise ValueError("Answer and region IDs must be unique.")
    for region in edits.regions:
        if region.answer_id not in answers:
            raise ValueError("Every region must reference an existing answer.")
        if region.class_name not in document["class_names"]:
            raise ValueError("Unknown detector class.")
        x1, y1, x2, y2 = region.box
        if not (0 <= x1 < x2 <= document["width"] and 0 <= y1 < y2 <= document["height"]):
            raise ValueError("Regions must lie inside the page and have positive size.")
    used = {region.answer_id for region in edits.regions}
    if any(answer.text.strip() and answer.id not in used for answer in edits.answers):
        raise ValueError("An answer with text must have at least one region.")
    return edits.model_dump()


def document_from_solver(solver, source_path):
    with Image.open(source_path) as image:
        width, height = image.size
    names = solver.detection_runtime.model.names
    if isinstance(names, dict):
        names = {str(int(key)): str(value) for key, value in names.items()}
    else:
        names = {str(index): str(value) for index, value in enumerate(names)}
    document = {
        "version": 0, "width": width, "height": height,
        "class_names": list(names.values()), "class_ids": names,
        "regions": [], "answers": [],
        "normalization": dict(getattr(solver, "normalization_info", {})),
        "validation_warnings": list(getattr(solver, "answer_warnings", [])),
        "warnings": [],
    }
    for answer_id, group in enumerate(solver.answer_units, start=1):
        document["answers"].append({
            "id": answer_id, "text": "", "font_size": None,
            "offset_x": 0, "offset_y": 0,
        })
        for index in group:
            gap = solver.detected_gaps[index]
            x1, y1, x2, y2 = gap[:4]
            box = [max(0, min(width, int(x1))), max(0, min(height, int(y1))),
                   max(0, min(width, int(x2))), max(0, min(height, int(y2)))]
            if box[2] > box[0] and box[3] > box[1]:
                document["regions"].append({
                    "id": index + 1, "answer_id": answer_id,
                    "class_name": str(gap[4]), "box": box,
                })
    return document


class DocumentRenderer(RenderingMixin, DetectionMixin):
    """Only detection helpers and rendering; no model initialization."""


def render_document(document, source_path, output_path):
    renderer = DocumentRenderer()
    renderer.detected_gaps = [tuple(region["box"]) + (region["class_name"],)
                              for region in document["regions"]]
    solutions = {}
    for answer in document["answers"]:
        indices = [index for index, region in enumerate(document["regions"])
                   if region["answer_id"] == answer["id"]]
        solutions[answer["id"] - 1] = {
            "gap_indices": indices, "solution": answer["text"],
            "font_size": answer.get("font_size"),
            "offset_x": answer.get("offset_x", 0), "offset_y": answer.get("offset_y", 0),
        }
    output_path = Path(output_path)
    staging = output_path.with_name("render_pending.png")
    try:
        renderer.fill_gaps_in_image(str(source_path), solutions, str(staging))
        staging.replace(output_path)
    finally:
        staging.unlink(missing_ok=True)
    warnings = list(document.get("validation_warnings", []))
    warnings.extend(renderer.render_warnings)
    warnings.extend(document.get("normalization", {}).get("warnings", []))
    for answer in document["answers"]:
        if not answer["text"].strip() or answer["text"].strip().lower() == "none":
            warnings.append({"answer_id": answer["id"], "kind": "unresolved",
                             "message": f"Answer {answer['id']} is unresolved; edit it before downloading."})
    document["warnings"] = warnings
    create_preview(output_path, output_path.with_name("result_preview.png"))


def create_preview(source, destination):
    with Image.open(source) as image:
        image.thumbnail((900, 1200), Image.Resampling.LANCZOS)
        image.save(destination, format="PNG")
