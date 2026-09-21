import cv2
import os
import ollama
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import re
import requests
import tempfile

# Define Pydantic models outside the class
class Pair(BaseModel):
    key: int
    value: str

class get_solution(BaseModel):
    solutions: List[Pair]

class WorksheetSolver():
    def __init__(self, path:str, gap_detection_model_path: str = "", llm_model_name: str = "gemini-3-flash-preview", think: bool = True, local: bool = False, thinking_budget: int = 2048, debug: bool = False, experimental: bool = False):
        self.debug = debug
        self.model_name = llm_model_name
        self.local = local
        self.path = str(path)
        self.thinking_budget = thinking_budget
        self.think = think
        self.experimental = experimental

        if self.experimental and not self.local:
            raise ValueError("Experimental mode requires local mode.")

        if gap_detection_model_path:
            self.model_path = gap_detection_model_path
        else:
            self.model_path = self.get_gap_model()

        self.image = None
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
        self.detected_gaps = []
        self.gap_groups = []  # Groups of gap indices
        self.gap_to_group = {}  # Maps gap index to group index
        self.ungrouped_gap_indices = []
        self.answer_units = []  # Line groups + single ungrouped boxes
        self.gap_to_answer_unit = {}  # Maps any gap index to answer unit index
        self.converted_image_path = None
        
        if self.debug:
            import time
            self.time = time
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Worksheet image not found: {self.path}")
        else:
            if self.is_allowed_image(self.path):
                if not self.path.lower().endswith(".png"):
                    print(f"Worksheet image found: {self.path}")
                    source_path = Path(self.path)
                    descriptor, converted_name = tempfile.mkstemp(
                        prefix=f"{source_path.stem}_",
                        suffix="_temp.png",
                        dir=source_path.parent,
                    )
                    os.close(descriptor)
                    converted_path = Path(converted_name)
                    try:
                        with Image.open(source_path) as img:
                            img.convert("RGB").save(converted_path)
                    except Exception:
                        converted_path.unlink(missing_ok=True)
                        raise
                    self.path = str(converted_path)
                    self.converted_image_path = self.path
            else:
                raise ValueError(
                    f"Invalid file type. Allowed types: {', '.join(sorted(self.allowed_extensions))}"
                )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Gap detection model not found: {self.model_path}")
        if not self.local and not self.experimental:
            if os.path.exists(".env"):
                load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Google API key not found. Set GOOGLE_API_KEY in the environment or .env file."
                )
            self.client = genai.Client(api_key=api_key)
        if self.experimental and self.local:

            from transformers.generation import LogitsProcessor
            from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            import torch

            class ThinkingTokenBudgetProcessor(LogitsProcessor):
                """
                A processor where after a maximum number of tokens are generated,
                a </think> token is added at the end to stop the thinking generation,
                and then it will continue to generate the response.
                """
                def __init__(self, tokenizer, max_thinking_tokens=None):
                    self.tokenizer = tokenizer
                    self.max_thinking_tokens = max_thinking_tokens
                    self.think_end_token = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
                    self.nl_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
                    self.tokens_generated = 0
                    self.stopped_thinking = False
                    self.neg_inf = float('-inf')

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                    self.tokens_generated += 1
                    if self.max_thinking_tokens == 0 and not self.stopped_thinking and self.tokens_generated > 0:
                        scores[:] = self.neg_inf
                        scores[0][self.nl_token] = 0
                        scores[0][self.think_end_token] = 0
                        self.stopped_thinking = True
                        return scores

                    if self.max_thinking_tokens is not None and not self.stopped_thinking:
                        if (self.tokens_generated / self.max_thinking_tokens) > .95:
                            scores[0][self.nl_token] = scores[0][self.think_end_token] * (1 + (self.tokens_generated / self.max_thinking_tokens))
                            scores[0][self.think_end_token] = (
                                scores[0][self.think_end_token] * (1 + (self.tokens_generated / self.max_thinking_tokens))
                            )

                        if self.tokens_generated >= (self.max_thinking_tokens - 1):
                            if self.tokens_generated == self.max_thinking_tokens-1:
                                scores[:] = self.neg_inf
                                scores[0][self.nl_token] = 0
                            else:
                                scores[:] = self.neg_inf
                                scores[0][self.think_end_token] = 0
                                self.stopped_thinking = True

                    return scores
                
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.think:
                processor = ThinkingTokenBudgetProcessor(tokenizer, max_thinking_tokens=self.thinking_budget)
            else:
                processor = ThinkingTokenBudgetProcessor(tokenizer, max_thinking_tokens=0)

            schema_parser = JsonSchemaParser(get_solution.model_json_schema())
            self.prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, schema_parser)

            self.pipe = pipeline(
                "image-text-to-text", 
                model=self.model_name,
                max_new_tokens=4096, 
                logits_processor=[processor], 
                device=0,
                model_kwargs={"quantization_config": quantization_config}
            )

        self.model = YOLO(self.model_path)
        
    def load_image(self, image_path: str):
        """Load image and create a copy for processing"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image {image_path} not found!")
        return self.image.copy()
    
    def get_gap_model(self) -> str:
        releases_url = "https://github.com/Hawk3388/solver/releases"
        os.makedirs("./model", exist_ok=True)
        folder_path = Path("./model")
        model_folder_names = [p.name for p in folder_path.iterdir() if p.is_dir()]
        version_pattern = re.compile(r"^v\d+\.\d+\.\d+$")
        installed_versions = [name for name in model_folder_names if version_pattern.match(name)]
        installed_versions.sort(
            key=lambda value: tuple(map(int, value.lstrip("v").split("."))),
            reverse=True,
        )
        installed_model = None
        for version in installed_versions:
            candidate = folder_path / version / "gap_detection_model.pt"
            if candidate.exists():
                installed_model = candidate
                break

        # Do not add a network round-trip to every worksheet. Releases can be
        # updated explicitly; automatic download is only needed on first use.
        if installed_model:
            return str(installed_model)

        try:
            release_response = requests.get(releases_url, timeout=8)
            release_response.raise_for_status()
            pattern = re.compile(r"<h2[^>]*>(v\d+\.\d+\.\d+)</h2>")
            remote_versions = pattern.findall(release_response.text)
            remote_versions.sort(
                key=lambda value: tuple(map(int, value.lstrip("v").split("."))),
                reverse=True,
            )

            for version in remote_versions:
                model_url = f"https://github.com/Hawk3388/solver/releases/download/{version}/gap_detection_model.pt"
                if not self.url_exists(model_url):
                    continue

                target = folder_path / version / "gap_detection_model.pt"
                temporary_target = target.with_suffix(".pt.part")
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with requests.get(model_url, stream=True, timeout=60) as response:
                        response.raise_for_status()
                        with open(temporary_target, "wb") as model_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    model_file.write(chunk)
                    temporary_target.replace(target)
                finally:
                    temporary_target.unlink(missing_ok=True)
                return str(target)
        except requests.RequestException as error:
            if self.debug:
                print(f"Model update check skipped: {error}")

        raise FileNotFoundError(
            "No gap detection model is installed and the latest model could not be downloaded."
        )


    def url_exists(self, url: str, timeout: float = 5.0) -> bool:
        try:
            r = requests.head(url, allow_redirects=True, timeout=timeout)
            return (200 <= r.status_code < 400)
        except requests.RequestException as e:
            return False
    
    def is_allowed_image(self, filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in self.allowed_extensions

    def calculate_iou(self, box1: list, box2: list):
        """
        Calculates Intersection over Union (IoU) between two boxes
        box: [x1, y1, x2, y2]
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


    def filter_overlapping_boxes(self, boxes, iou_threshold=0.5):
        """
        Filters overlapping boxes - keeps only the one with highest confidence
        
        Args:
            boxes: YOLO boxes object
            iou_threshold: Minimum IoU for overlap (0.5 = 50%)
        
        Returns:
            List of indices of boxes to keep
        """
        if len(boxes) == 0:
            return []
        
        # Extract coordinates and confidences
        coords = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = boxes.conf.cpu().numpy()
        
        # Sort by confidence (highest first)
        sorted_indices = np.argsort(-confidences)
        
        keep = []
        
        for i in sorted_indices:
            # Check if this box overlaps with already kept boxes
            should_keep = True
            
            for kept_idx in keep:
                iou = self.calculate_iou(coords[i], coords[kept_idx])
                
                if iou > iou_threshold:
                    # Overlap found - discard this box (lower confidence)
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(i)
        
        return sorted(keep)  # Back in original order
    
    def sort_reading_order(self, boxes):
        """Sort boxes in reading order: line by line from top to bottom, left to right within a line.
        
        Boxes on the same text line often have slightly different y values.
        This method groups boxes with similar y position (overlap) into lines.
        """
        if not boxes:
            return boxes
        
        # Sort roughly by y first
        boxes_sorted = sorted(boxes, key=lambda b: b[1])
        
        # Group into lines based on vertical overlap
        lines = []
        current_line = [boxes_sorted[0]]
        # y-center and height of the current line
        line_y_min = boxes_sorted[0][1]
        line_y_max = boxes_sorted[0][3]
        
        for box in boxes_sorted[1:]:
            box_y_top = box[1]
            box_y_bottom = box[3]
            box_height = box_y_bottom - box_y_top
            line_height = line_y_max - line_y_min
            
            # Check if the box overlaps vertically with the current line
            # Tolerance: at least 50% of the smaller height must overlap
            overlap = min(line_y_max, box_y_bottom) - max(line_y_min, box_y_top)
            min_height = max(min(box_height, line_height), 1)
            
            if overlap > 0 and overlap / min_height > 0.3:
                # Same line
                current_line.append(box)
                line_y_min = min(line_y_min, box_y_top)
                line_y_max = max(line_y_max, box_y_bottom)
            else:
                # New line
                lines.append(current_line)
                current_line = [box]
                line_y_min = box_y_top
                line_y_max = box_y_bottom
        
        lines.append(current_line)
        
        # Sort within each line by x, lines from top to bottom
        result = []
        for line in lines:
            line.sort(key=lambda b: b[0])  # By x coordinate
            result.extend(line)
        
        return result

    def is_line_class(self, class_name):
        """True only for the exact YOLO class name 'line'."""
        return str(class_name).strip().lower() == "line"

    def _unit_bbox(self, unit, gaps):
        """Return merged bbox (x1, y1, x2, y2) for an answer unit."""
        boxes = [gaps[i][:4] for i in unit if 0 <= i < len(gaps)]
        if not boxes:
            return (0, 0, 0, 0)
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def _boxes_to_rows(self, boxes):
        """Return writable rows in top-to-bottom order.

        YOLO boxes for consecutive ruled lines can overlap because they include
        some of the writing area above the underline. They are only the same
        row when their vertical centres are nearly identical.
        """
        rows = []
        sorted_boxes = sorted(
            boxes,
            key=lambda item: ((item[1] + item[3]) / 2, item[0]),
        )
        for box in sorted_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            if not rows:
                rows.append([x1, y1, x2, y2])
                continue

            previous = rows[-1]
            previous_height = max(1, previous[3] - previous[1])
            current_height = max(1, y2 - y1)
            previous_center = (previous[1] + previous[3]) / 2
            current_center = (y1 + y2) / 2
            same_row_tolerance = max(2, min(previous_height, current_height) * 0.25)

            if abs(current_center - previous_center) <= same_row_tolerance:
                previous[0] = min(previous[0], x1)
                previous[1] = min(previous[1], y1)
                previous[2] = max(previous[2], x2)
                previous[3] = max(previous[3], y2)
            else:
                rows.append([x1, y1, x2, y2])
        return rows

    @staticmethod
    def _normalise_solution_text(solution):
        """Remove common model boilerplate and collapse whitespace."""
        text = re.sub(r"\s+", " ", str(solution)).strip()
        text = re.sub(r"^(?:antwort|lösung)\s*:\s*", "", text, flags=re.IGNORECASE)
        return text.strip(" \t\r\n\"'")

    @staticmethod
    def _preferred_answer_font_size(image_width, rows):
        """Choose a document-scale font size instead of using line-box height.

        Detectors often return a two-pixel-high underline on large scans, so the
        detection height is not a useful proxy for the surrounding print size.
        """
        preferred = max(8, min(48, int(round(image_width / 55))))

        if len(rows) > 1:
            baselines = [row[3] for row in rows]
            spacings = [b - a for a, b in zip(baselines, baselines[1:]) if b > a]
            if spacings:
                preferred = min(preferred, max(8, int(min(spacings) * 0.76)))

        return preferred

    @staticmethod
    def _fit_words_to_rows(draw, text, rows, font, padding):
        """Greedily wrap text and report how many words were consumed."""
        words = text.split()
        lines = []
        word_index = 0

        for row in rows:
            width = max(1, row[2] - row[0] - (2 * padding))
            line_words = []

            while word_index < len(words):
                candidate = " ".join(line_words + [words[word_index]])
                bbox = draw.textbbox((0, 0), candidate, font=font)
                candidate_width = bbox[2] - bbox[0]
                if candidate_width > width:
                    break
                line_words.append(words[word_index])
                word_index += 1

            lines.append(" ".join(line_words))

        return lines, word_index

    @staticmethod
    def _ellipsize(draw, text, font, width):
        """Fit text to one row without distorting glyphs."""
        if not text:
            return ""
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= width:
            return text

        suffix = "…"
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = text[:middle].rstrip() + suffix
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= width:
                low = middle
            else:
                high = middle - 1
        return text[:low].rstrip() + suffix if low else suffix

    def _answer_capacity(self, unit, image_width):
        """Estimate how much text visibly fits in an answer unit."""
        boxes = [self.detected_gaps[idx][:4] for idx in unit]
        rows = self._boxes_to_rows(boxes)
        if not rows:
            return 0, 0

        font_size = self._preferred_answer_font_size(image_width, rows)
        padding = max(3, int(round(image_width * 0.005)))
        usable_width = sum(max(1, row[2] - row[0] - 2 * padding) for row in rows)
        # Liberation Sans averages roughly half a font-size per German character.
        max_characters = max(1, int(usable_width / max(1, font_size * 0.54)))
        return len(rows), max_characters

    @staticmethod
    def _find_underline_y(source_pixels, row):
        """Locate the actual printed underline inside a detected row."""
        image_height, image_width = source_pixels.shape[:2]
        x1 = max(0, int(row[0]))
        x2 = min(image_width, int(row[2]))
        search_margin = max(5, int(round(image_width / 110)))
        y1 = max(0, int(row[1]) - search_margin)
        y2 = min(image_height, int(row[3]) + search_margin + 1)
        if x2 <= x1 or y2 <= y1:
            return int(row[3])

        crop = source_pixels[y1:y2, x1:x2]
        grayscale = np.mean(crop[:, :, :3], axis=2)
        # Scanned worksheet rules are often medium gray rather than black.
        dark_pixels_per_y = np.sum(grayscale < 210, axis=1)
        # A worksheet line normally spans most of its detection box. Requiring
        # more than half the width rejects shorter borders from hint/callout
        # boxes, which otherwise look exactly like underlines.
        minimum_support = max(10, int((x2 - x1) * 0.55))
        candidates = np.flatnonzero(dark_pixels_per_y >= minimum_support)

        # The underline is the lowest long, dark horizontal feature in the box.
        # Choosing the lowest candidate also disambiguates overlapping row boxes.
        if candidates.size:
            return y1 + int(candidates[-1])
        return int(row[3])

    @staticmethod
    def _find_underlines_for_rows(source_pixels, rows):
        """Find and assign real worksheet lines to detected rows top-to-bottom.

        This uses the complete answer-area width, so short borders from hint
        boxes are rejected even when the detector mistakes them for a line.
        """
        if not rows:
            return []

        image_height, image_width = source_pixels.shape[:2]
        x1 = max(0, min(int(row[0]) for row in rows))
        x2 = min(image_width, max(int(row[2]) for row in rows))
        search_margin = max(8, int(round(image_width * 0.025)))
        y1 = max(0, min(int(row[1]) for row in rows) - search_margin)
        y2 = min(image_height, max(int(row[3]) for row in rows) + search_margin + 1)
        if x2 <= x1 or y2 <= y1:
            return [int(row[3]) for row in rows]

        crop = source_pixels[y1:y2, x1:x2]
        grayscale = np.mean(crop[:, :, :3], axis=2)
        dark_pixels_per_y = np.sum(grayscale < 210, axis=1)
        minimum_support = max(10, int((x2 - x1) * 0.55))
        candidate_pixels = np.flatnonzero(dark_pixels_per_y >= minimum_support)

        # Collapse the multiple pixel rows of a thick/antialiased rule into one
        # baseline, using its lower edge.
        candidate_lines = []
        for relative_y in candidate_pixels:
            absolute_y = y1 + int(relative_y)
            if candidate_lines and absolute_y <= candidate_lines[-1][-1] + 1:
                candidate_lines[-1].append(absolute_y)
            else:
                candidate_lines.append([absolute_y])
        candidate_lines = [line[-1] for line in candidate_lines]

        if len(candidate_lines) < len(rows):
            return [
                WorksheetSolver._find_underline_y(source_pixels, row)
                for row in rows
            ]

        # Assign distinct physical lines monotonically. Reserving enough
        # candidates for the remaining rows prevents multiple detections from
        # snapping to the same underline.
        assigned = []
        start = 0
        for row_index, row in enumerate(rows):
            remaining_rows = len(rows) - row_index - 1
            final_index = len(candidate_lines) - remaining_rows
            choices = range(start, final_index)
            best_index = min(choices, key=lambda index: abs(candidate_lines[index] - row[3]))
            assigned.append(candidate_lines[best_index])
            start = best_index + 1
        return assigned

    def sort_answer_units_reading_order(self, units, gaps):
        """Sort answer units globally by reading order: top->bottom, left->right."""
        if not units:
            return []

        unit_data = []
        for idx, unit in enumerate(units):
            x1, y1, x2, y2 = self._unit_bbox(unit, gaps)
            unit_data.append({
                "idx": idx,
                "unit": unit,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "h": max(1, y2 - y1),
            })

        unit_data.sort(key=lambda u: u["y1"])

        rows = []
        current_row = [unit_data[0]]
        row_y_min = unit_data[0]["y1"]
        row_y_max = unit_data[0]["y2"]

        for u in unit_data[1:]:
            overlap = min(row_y_max, u["y2"]) - max(row_y_min, u["y1"])
            row_h = max(1, row_y_max - row_y_min)
            min_h = max(1, min(row_h, u["h"]))

            if overlap > 0 and (overlap / min_h) > 0.3:
                current_row.append(u)
                row_y_min = min(row_y_min, u["y1"])
                row_y_max = max(row_y_max, u["y2"])
            else:
                rows.append(current_row)
                current_row = [u]
                row_y_min = u["y1"]
                row_y_max = u["y2"]

        rows.append(current_row)

        sorted_units = []
        for row in rows:
            row.sort(key=lambda u: u["x1"])
            sorted_units.extend([u["unit"] for u in row])

        return sorted_units
    
    def group_gaps_by_proximity(self, gaps):
        """Group gaps that are directly below each other into groups.
        
        Returns:
            List of groups, where each group is a list of gap indices (0-based) sorted by Y position
            Also returns a mapping from gap index to group index
        """
        if not gaps:
            return [], {}
        
        # Create index mapping: sorted_idx -> original_idx
        indices = list(range(len(gaps)))
        sorted_indices = sorted(indices, key=lambda i: gaps[i][1])  # Sort by Y (top to bottom)
        
        # Calculate average gap height as threshold
        heights = [(gap[3] - gap[1]) for gap in gaps]
        avg_height = sum(heights) / len(heights) if heights else 0
        
        # Distance threshold: line boxes may slightly overlap or be very close
        distance_threshold = avg_height * 1.5
        overlap_tolerance = max(5, int(avg_height * 0.15))
        
        groups = []
        gap_to_group = {}
        grouped = set()
        
        # Process gaps from top to bottom
        for sort_i, i in enumerate(sorted_indices):
            if i in grouped:
                continue
            
            gap_i = gaps[i]
            x1_i, y1_i, x2_i, y2_i = gap_i[:4]
            class_name_i = gap_i[4] if len(gap_i) > 4 else "line"
            
            # Only exact 'line' class is groupable. Other classes are ignored here.
            if not self.is_line_class(class_name_i):
                continue

            # Start new group with current line gap
            current_group = [i]
            grouped.add(i)
            
            # Look for gaps below this one
            for sort_j in range(sort_i + 1, len(sorted_indices)):
                j = sorted_indices[sort_j]
                
                if j in grouped:
                    continue
                
                gap_j = gaps[j]
                x1_j, y1_j, x2_j, y2_j = gap_j[:4]
                class_name_j = gap_j[4] if len(gap_j) > 4 else "line"
                
                # Only group if both are exact line class detections
                if not self.is_line_class(class_name_j):
                    continue
                
                # Check vertical distance (gap j should be below gap i)
                vertical_distance = y1_j - y2_i
                
                # Check horizontal alignment
                i_left, i_top, i_right, i_bottom = x1_i, y1_i, x2_i, y2_i
                j_left, j_top, j_right, j_bottom = x1_j, y1_j, x2_j, y2_j
                
                # Calculate horizontal overlap
                h_overlap_start = max(i_left, j_left)
                h_overlap_end = min(i_right, j_right)
                h_overlap = max(0, h_overlap_end - h_overlap_start)
                
                # Box widths
                i_width = i_right - i_left
                j_width = j_right - j_left
                min_width = min(i_width, j_width)
                
                # Check if box j is vertically close enough and horizontally aligned
                if -overlap_tolerance <= vertical_distance < distance_threshold:
                    # At least 30% overlap or 15px minimum
                    if h_overlap > min_width * 0.3 or h_overlap > 15:
                        current_group.append(j)
                        grouped.add(j)
                        gap_i = gap_j  # Update for next iteration
                        x1_i, y1_i, x2_i, y2_i = gap_i[:4]
                    else:
                        # Not enough overlap, end this group
                        break
                else:
                    # Distance too large, end this group
                    break
            
            # Store group (sort indices in return order)
            current_group.sort()
            for idx in current_group:
                gap_to_group[idx] = len(groups)
            
            groups.append(current_group)
        
        return groups, gap_to_group

    def detect_gaps(self):
        self.detected_gaps = []
        img = self.load_image(self.path)

        results = self.model.predict(source=self.path, conf=0.10)

        for r in results:
            if len(r.boxes) > 0:
                keep_indices = self.filter_overlapping_boxes(r.boxes, iou_threshold=0.5)
                print(f"After overlap filtering: {len(keep_indices)} boxes")
            else:
                keep_indices = []
            if len(keep_indices) == 0:
                print("\nNo gaps detected!")
                print("Check:")
                print("   - Is the image a worksheet?")
                print("   - Was the model trained correctly?")
                print("   - Try lower conf (e.g. 0.1)")
            else:
                for idx in keep_indices:
                    box = r.boxes[idx]
                    class_id = int(box.cls[0])
                    class_name = r.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    self.detected_gaps.append((int(x1), int(y1), int(x2), int(y2), class_name))
                img = r.orig_img.copy()
        
        # Sort in reading order (line by line)
        self.detected_gaps = self.sort_reading_order(self.detected_gaps)
        
        # Group gaps by proximity (vertically aligned and close together)
        self.gap_groups, self.gap_to_group = self.group_gaps_by_proximity(self.detected_gaps)
        self.ungrouped_gap_indices = [i for i in range(len(self.detected_gaps)) if i not in self.gap_to_group]

        # Build answer units for the AI:
        # - grouped line boxes stay grouped
        # - each ungrouped box (e.g. class gap) becomes its own single unit
        unsorted_units = list(self.gap_groups) + [[idx] for idx in self.ungrouped_gap_indices]
        self.answer_units = self.sort_answer_units_reading_order(unsorted_units, self.detected_gaps)
        self.gap_to_answer_unit = {}
        for unit_idx, unit in enumerate(self.answer_units):
            for gap_idx in unit:
                self.gap_to_answer_unit[gap_idx] = unit_idx
        
        print(f"Line-boxes grouped into {len(self.gap_groups)} groups")
        for i, group in enumerate(self.gap_groups):
            print(f"   Group {i+1}: {len(group)} gaps (indices: {group})")
        print(f"Ungrouped boxes (e.g. gap): {len(self.ungrouped_gap_indices)}")
        print(f"Total AI answer units: {len(self.answer_units)}")
                    
        return self.detected_gaps, img

    def mark_gaps(self, image, gaps):
        """Draw one red box per answer unit (group) instead of per single line."""

        if not self.answer_units:
            return image

        for unit_idx, unit in enumerate(self.answer_units):
            unit_boxes = [gaps[i][:4] for i in unit if 0 <= i < len(gaps)]
            if not unit_boxes:
                continue

            # Surround the whole group with one box.
            x1 = min(b[0] for b in unit_boxes)
            y1 = min(b[1] for b in unit_boxes)
            x2 = max(b[2] for b in unit_boxes)
            y2 = max(b[3] for b in unit_boxes)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            label = str(unit_idx + 1)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(image, (x1, y1 - label_size[1] - 4), (x1 + label_size[0] + 2, y1), (0, 0, 255), -1)
            cv2.putText(image, (label), (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return image
    
    def ask_ai_about_all_gaps(self, marked_image):
        """Ask Gemini about the content of ALL gap groups at once"""
        if self.debug:
            start_time = self.time.time()
        
        thinking = None
        source_path = Path(self.path)
        descriptor, marked_image_path = tempfile.mkstemp(
            prefix=f"{source_path.stem}_",
            suffix="_marked.png",
            dir=source_path.parent,
        )
        os.close(descriptor)
        if not cv2.imwrite(marked_image_path, marked_image):
            Path(marked_image_path).unlink(missing_ok=True)
            raise OSError(f"Could not write temporary marked image: {marked_image_path}")

        ocr_text = self.ocrImage(self.path)

        # Build description of answer units
        group_descriptions = []
        if self.image is not None:
            image_width = self.image.shape[1]
        else:
            with Image.open(self.path) as source_image:
                image_width = source_image.width
        for i, group in enumerate(self.answer_units):
            group_num = i + 1
            first_idx = group[0]
            class_name = str(self.detected_gaps[first_idx][4]) if len(self.detected_gaps[first_idx]) > 4 else "gap"
            row_count, max_characters = self._answer_capacity(group, image_width)
            kind = "open answer area" if row_count > 1 else f"single {class_name} gap"
            group_descriptions.append(
                f"Group {group_num}: {kind}, {row_count} writable line(s), "
                f"approximately {max_characters} characters maximum"
            )
        
        group_text = "\n".join(group_descriptions)

        prompt = f"""
Solve this German worksheet.

OCR text:
{ocr_text}

Answer groups:
{group_text}

Fill each numbered answer position with the exact text that belongs there.

Rules:
- Answer in German.
- For an inline gap, return only the missing word or short phrase. Never repeat the surrounding sentence.
- For an open answer area, give a concise direct answer, not an explanation of your reasoning.
- Stay within the approximate character limit stated for each group.
- Do not add labels such as "Antwort:" or "Lösung:".
- Match grammar, capitalization, and singular/plural.
- If unclear, answer "none".
- Return every group exactly once and use its number as the key.

Return only this JSON format:

{{
  "solutions": [
    {{
      "key": 1,
      "value": "answer"
    }}
  ]
}}
"""

        if not self.experimental:
            if not self.local:
                image = Image.open(marked_image_path)
                original_image = Image.open(self.path)
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[image, original_image, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_json_schema=get_solution.model_json_schema(),
                            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget if self.think else 0),
                        ),
                    )
                except genai.errors.ServerError:
                    fallback_model = "gemini-3-flash-preview"
                    if self.model_name == fallback_model:
                        raise

                    print(f"Gemini server error - falling back to {fallback_model}")
                    self.model_name = fallback_model
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[image, original_image, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_json_schema=get_solution.model_json_schema(),
                            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget if self.think else 0),
                        ),
                    )
                output = get_solution.model_validate_json(response.text)
            else:
                if self.model_name == "qwen3-vl:8b-thinking" and self.think:
                    print("you are using an experimantal thinking model - we will stream the response and switch to an instruct model if it seems to get stuck in thinking mode")
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt, "images": [marked_image_path, self.path]}],
                        format=get_solution.model_json_schema(),
                        options={"num_ctx": 8192},
                        stream=True
                    )
                    full_response = ""
                    thinking = ""
                    finished = True
                    for chunk in response:
                        if chunk.message.content:
                            full_response += chunk.message.content
                            print(chunk.message.content, end="", flush=True)
                        elif chunk.message.thinking:
                            print(chunk.message.thinking, end="", flush=True)
                            thinking += chunk.message.thinking
                            if len(thinking) > 12000:
                                if "\n\n" in thinking.strip()[-10:]:
                                    thinking = thinking.split("\n\n")[0]
                                    del response
                                    print(len(thinking))
                                    finished = False
                                    break
                    
                    if not finished:
                        final_response = ollama.chat(
                            model=self.model_name.replace("thinking", "instruct"),
                            messages=[{"role": "user", "content": prompt, "images": [marked_image_path, self.path]},
                                    {"role": "assistant", "content": thinking}],
                            format=get_solution.model_json_schema(),
                            options={"num_ctx": 8192}
                        )

                        output = get_solution.model_validate_json(final_response.message.content)
                    else:
                        output = get_solution.model_validate_json(full_response)
                else:
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt, "images": [marked_image_path, self.path]}],
                        format=get_solution.model_json_schema(),
                        think=None if not 'thinking' in ollama.show(self.model_name).capabilities else True if self.think else False,
                        options={
                            "num_ctx": 8192,
                            "temperature":0.1
                        }
                    )

                    thinking = getattr(response.message, "thinking", None)
                    if thinking:
                        print(thinking)
                    try:
                        output = get_solution.model_validate_json(response.message.content)
                    except Exception as e:
                        if self.debug:
                            if thinking:
                                print(f"Thinking content:\n{thinking}")
                            print(f"Full response content:\n{response.message.content}")
                            print("Debug mode ON - timing enabled")
                            end_time = self.time.time()
                            print(f"Time taken: {end_time - start_time:.2f} seconds")
                        raise ValueError(f"The AI returned an invalid response: {e}") from e
        else:
            if self.local:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image_path": marked_image_path},
                    {"type": "image", "image_path": self.path},
                    {"type": "text", "text": prompt},
                ]}]
                response = self.pipe(messages, enable_thinking=self.think, prefix_allowed_tokens_fn=self.prefix_function)[0]["generated_text"][-1]["content"]
                response = response.split("</think>")
                output = get_solution.model_validate_json(response[-1])
        
        if not self.debug:
            if self.converted_image_path and os.path.exists(self.converted_image_path):
                os.remove(self.converted_image_path)
            if os.path.exists(marked_image_path):
                os.remove(marked_image_path)
        else:  
            print("Debug mode ON - timing enabled")
            end_time = self.time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            if thinking:
                print(f"Thinking: {thinking}")
            print(f"AI output:\n{output}")

        return output
    
    def ocrImage(self, image_path):

        ocr_prompt = """
OCR this worksheet image.

Extract all visible text exactly.
Do not solve the exercises.

Replace every empty answer area (blank lines, boxes, or gaps) with:
_____

Keep the original reading order and line breaks.
Preserve capitalization, punctuation, and German characters.
Return only the OCR text.
"""

        if self.local:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": ocr_prompt, "images": [image_path]}],
                think=False,
                options={"temperature": 0.1},
            )
            return response.message.content

        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image, ocr_prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text

    def solve_all_gaps(self, marked_image):
        """Solve all gap groups with Ollama - structured!"""
        if not self.detected_gaps:
            print("No gaps found!")
            return {}
        if not self.answer_units:
            print("No answer units found to solve.")
            return {}
        
        print(f"Analyzing all {len(self.answer_units)} answer units with AI...")
        
        # Ask AI about all gap groups at once
        print("Sending image to AI...")
        solutions_data = self.ask_ai_about_all_gaps(marked_image)
        
        if solutions_data:
            print("Structured AI response received!")
            
            # Convert structured response to our format
            solutions = {}
            
            # solutions_data.solutions is now a list of GroupPair objects
            for pair in solutions_data.solutions:
                try:
                    group_id = pair.key
                    answer = pair.value
                    group_index = group_id - 1  # 0-based
                    
                    if 0 <= group_index < len(self.answer_units):
                        gap_indices = self.answer_units[group_index]
                        solutions[group_index] = {
                            'gap_indices': gap_indices,
                            'solution': answer
                        }
                except (ValueError, KeyError) as e:
                    print(f"Error processing group {group_id}: {e}")
                    continue
            
            return solutions
        else:
            print("No response received from AI.")
            return {}
    
    def fill_gaps_in_image(self, image_path: str, solutions: dict, output_path: str = "worksheet_solved.png"):
        """Render readable answers at the scale of the source document."""

        cv_image = self.load_image(image_path)
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        source_pixels = np.array(pil_image)
        draw = ImageDraw.Draw(pil_image)

        font_candidates = [
            Path(__file__).resolve().parent / "fonts" / "LiberationSans-Regular.ttf",
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        font_path = next((path for path in font_candidates if path.exists()), None)
        if font_path is None:
            raise FileNotFoundError(
                "No usable answer font found. Place LiberationSans-Regular.ttf "
                "in the fonts folder."
            )

        image_width = pil_image.width
        padding = max(3, int(round(image_width * 0.005)))
        answer_colour = (0, 0, 0)

        for group_index, solution_data in solutions.items():
            gap_indices = solution_data.get('gap_indices', [])
            solution = self._normalise_solution_text(solution_data.get('solution', ''))

            if not solution or solution.lower() == 'none':
                continue

            boxes = [
                self.detected_gaps[idx][:4]
                for idx in gap_indices
                if 0 <= idx < len(self.detected_gaps)
            ]
            rows = self._boxes_to_rows(boxes)
            if not rows:
                continue
            is_ruled_answer = all(
                self.is_line_class(self.detected_gaps[idx][4])
                for idx in gap_indices
                if 0 <= idx < len(self.detected_gaps)
            )

            preferred_size = self._preferred_answer_font_size(image_width, rows)
            minimum_size = max(7, int(round(preferred_size * 0.72)))
            lines = []
            word_index = 0
            font = None

            # Use the largest natural font that fits all available rows.
            for font_size in range(preferred_size, minimum_size - 1, -1):
                font = ImageFont.truetype(str(font_path), font_size)
                candidate_lines, consumed = self._fit_words_to_rows(
                    draw, solution, rows, font, padding
                )
                if consumed == len(solution.split()):
                    lines = candidate_lines
                    word_index = consumed
                    break

            if not lines:
                font = ImageFont.truetype(str(font_path), minimum_size)
                lines, word_index = self._fit_words_to_rows(
                    draw, solution, rows, font, padding
                )

            # Excessively verbose model output is clipped gracefully instead of
            # being squeezed into an unreadable horizontal strip.
            words = solution.split()
            if word_index < len(words):
                last_row = rows[-1]
                last_width = max(1, last_row[2] - last_row[0] - (2 * padding))
                remaining = " ".join(words[word_index:])
                combined = " ".join(part for part in [lines[-1], remaining] if part)
                lines[-1] = self._ellipsize(draw, combined, font, last_width)

            underline_positions = self._find_underlines_for_rows(source_pixels, rows)
            for row, line, underline_y in zip(rows, lines, underline_positions):
                if not line:
                    continue
                # Keep glyphs clearly above the printed underline. Descenders
                # may approach it, but the line must never cut through letters.
                baseline_y = underline_y - max(2, int(round(font.size * 0.24)))
                if is_ruled_answer:
                    x = row[0] + padding
                    anchor = "ls"
                else:
                    x = (row[0] + row[2]) / 2
                    anchor = "ms"
                draw.text(
                    (x, baseline_y),
                    line,
                    fill=answer_colour,
                    font=font,
                    anchor=anchor,
                )

        result_image = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR
        )

        if not cv2.imwrite(output_path, result_image):
            raise OSError(f"Could not write solved worksheet: {output_path}")

        print(f"Solved worksheet saved as: {output_path}")

        return result_image

# Main program
def main():
    # Best results with gemini-3-flash-preview (local: qwen3.8 for 16 GB VRAM + 32 GB RAM)
    # For Gemini you have to use a Google API-key in a .env file
    # For Ollama models you have to set local=True

    path = input("Please enter the path to the worksheet image: ").strip()
    llm_model_name = "qwen3.8"
    think = False
    local = True
    debug = True
    solver = WorksheetSolver(path, llm_model_name=llm_model_name, think=think, local=local, debug=debug)

    ask = False
    print("Loading image and detecting gaps...")
    try:
        gaps, img = solver.detect_gaps()
        
        print(f"{len(gaps)} boxes found, {len(solver.gap_groups)} line groups, {len(solver.ungrouped_gap_indices)} ungrouped!")
        
        marked_image = solver.mark_gaps(img, gaps)
        
        print("\nDetected gaps (x, y, width, height, class):")
        for i, gap in enumerate(gaps):
            unit_num = solver.gap_to_answer_unit.get(i)
            if unit_num is not None:
                print(f"  Box {i+1} (Group {unit_num + 1}): {gap}")
            else:
                print(f"  Box {i+1} (ungrouped): {gap}")
        
        print("\nGap groups:")
        for g_idx, group in enumerate(solver.gap_groups):
            print(f"  Group {g_idx+1}: gaps {[idx+1 for idx in group]}")
        
        if solver.debug:
            # Ask user if AI analysis is desired
            user_input = input("\nShould an AI analyze and fill the gaps? (y/N): ").lower().strip()
            if user_input in ['y', 'yes']:
                ask = True
        else:
            ask = True

        if ask:
            solutions = solver.solve_all_gaps(marked_image)
            
            if solutions:
                print("\nSolutions found:")
                for group_idx, sol in solutions.items():
                    group_num = group_idx + 1
                    gap_indices = [idx+1 for idx in sol['gap_indices']]
                    print(f"  Group {group_num} (gaps {gap_indices}): '{sol['solution']}'")
                
                solver.fill_gaps_in_image(path, solutions)
                
                print("\nResult saved. Press any key to exit...")
            else:
                print("No solutions received.")
        else:
            print("Gap detection only")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

# TODO:
# - better image detection with support for more kinds of worksheets
# - Add support for multiple files (batch processing)
# - Create an executable (.exe) for easy use without Python setup (Command: pyinstaller solver.spec)
