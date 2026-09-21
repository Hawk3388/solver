"""Answer layout and rendering for solved worksheet images."""

from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class RenderingMixin:
    """Rendering stage used by :class:`main.WorksheetSolver`."""

    @staticmethod
    def _boxes_to_rows(boxes):
        """Return writable rows in top-to-bottom order."""
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
            same_row_tolerance = max(
                2,
                min(previous_height, current_height) * 0.25,
            )

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
        text = re.sub(
            r"^(?:antwort|lösung)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip(" \t\r\n\"'")

    @staticmethod
    def _preferred_answer_font_size(image_width, rows):
        """Choose a font size based on document scale and line spacing."""
        preferred = max(8, min(48, int(round(image_width / 55))))

        if len(rows) > 1:
            baselines = [row[3] for row in rows]
            spacings = [
                second - first
                for first, second in zip(baselines, baselines[1:])
                if second > first
            ]
            if spacings:
                preferred = min(
                    preferred,
                    max(8, int(min(spacings) * 0.76)),
                )
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
        boxes = [self.detected_gaps[index][:4] for index in unit]
        rows = self._boxes_to_rows(boxes)
        if not rows:
            return 0, 0

        font_size = self._preferred_answer_font_size(image_width, rows)
        padding = max(3, int(round(image_width * 0.005)))
        usable_width = sum(
            max(1, row[2] - row[0] - 2 * padding)
            for row in rows
        )
        max_characters = max(
            1,
            int(usable_width / max(1, font_size * 0.54)),
        )
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
        dark_pixels_per_y = np.sum(grayscale < 210, axis=1)
        minimum_support = max(10, int((x2 - x1) * 0.55))
        candidates = np.flatnonzero(dark_pixels_per_y >= minimum_support)
        if candidates.size:
            return y1 + int(candidates[-1])
        return int(row[3])

    @classmethod
    def _find_underlines_for_rows(cls, source_pixels, rows):
        """Assign real worksheet lines to detected rows top-to-bottom."""
        if not rows:
            return []

        image_height, image_width = source_pixels.shape[:2]
        x1 = max(0, min(int(row[0]) for row in rows))
        x2 = min(image_width, max(int(row[2]) for row in rows))
        search_margin = max(8, int(round(image_width * 0.025)))
        y1 = max(0, min(int(row[1]) for row in rows) - search_margin)
        y2 = min(
            image_height,
            max(int(row[3]) for row in rows) + search_margin + 1,
        )
        if x2 <= x1 or y2 <= y1:
            return [int(row[3]) for row in rows]

        crop = source_pixels[y1:y2, x1:x2]
        grayscale = np.mean(crop[:, :, :3], axis=2)
        dark_pixels_per_y = np.sum(grayscale < 210, axis=1)
        minimum_support = max(10, int((x2 - x1) * 0.55))
        candidate_pixels = np.flatnonzero(
            dark_pixels_per_y >= minimum_support
        )

        candidate_lines = []
        for relative_y in candidate_pixels:
            absolute_y = y1 + int(relative_y)
            if candidate_lines and absolute_y <= candidate_lines[-1][-1] + 1:
                candidate_lines[-1].append(absolute_y)
            else:
                candidate_lines.append([absolute_y])
        candidate_lines = [line[-1] for line in candidate_lines]

        if len(candidate_lines) < len(rows):
            return [cls._find_underline_y(source_pixels, row) for row in rows]

        assigned = []
        start = 0
        for row_index, row in enumerate(rows):
            remaining_rows = len(rows) - row_index - 1
            final_index = len(candidate_lines) - remaining_rows
            choices = range(start, final_index)
            best_index = min(
                choices,
                key=lambda index: abs(candidate_lines[index] - row[3]),
            )
            assigned.append(candidate_lines[best_index])
            start = best_index + 1
        return assigned

    def fill_gaps_in_image(
        self,
        image_path: str,
        solutions: dict,
        output_path: str = "worksheet_solved.png",
    ):
        """Render readable answers at the scale of the source document."""
        cv_image = self.load_image(image_path)
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        source_pixels = np.array(pil_image)
        draw = ImageDraw.Draw(pil_image)

        project_root = Path(__file__).resolve().parent.parent
        font_candidates = [
            project_root / "fonts" / "LiberationSans-Regular.ttf",
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        font_path = next(
            (path for path in font_candidates if path.exists()),
            None,
        )
        if font_path is None:
            raise FileNotFoundError(
                "No usable answer font found. Place LiberationSans-Regular.ttf "
                "in the fonts folder."
            )

        image_width = pil_image.width
        padding = max(3, int(round(image_width * 0.005)))
        answer_colour = (0, 0, 0)

        for solution_data in solutions.values():
            gap_indices = solution_data.get("gap_indices", [])
            solution = self._normalise_solution_text(
                solution_data.get("solution", "")
            )
            if not solution or solution.lower() == "none":
                continue

            boxes = [
                self.detected_gaps[index][:4]
                for index in gap_indices
                if 0 <= index < len(self.detected_gaps)
            ]
            rows = self._boxes_to_rows(boxes)
            if not rows:
                continue

            is_ruled_answer = all(
                self.is_line_class(self.detected_gaps[index][4])
                for index in gap_indices
                if 0 <= index < len(self.detected_gaps)
            )
            preferred_size = self._preferred_answer_font_size(image_width, rows)
            minimum_size = max(7, int(round(preferred_size * 0.72)))
            lines = []
            word_index = 0
            font = None

            for font_size in range(preferred_size, minimum_size - 1, -1):
                font = ImageFont.truetype(str(font_path), font_size)
                candidate_lines, consumed = self._fit_words_to_rows(
                    draw,
                    solution,
                    rows,
                    font,
                    padding,
                )
                if consumed == len(solution.split()):
                    lines = candidate_lines
                    word_index = consumed
                    break

            if not lines:
                font = ImageFont.truetype(str(font_path), minimum_size)
                lines, word_index = self._fit_words_to_rows(
                    draw,
                    solution,
                    rows,
                    font,
                    padding,
                )

            words = solution.split()
            if word_index < len(words):
                last_row = rows[-1]
                last_width = max(
                    1,
                    last_row[2] - last_row[0] - (2 * padding),
                )
                remaining = " ".join(words[word_index:])
                combined = " ".join(
                    part for part in [lines[-1], remaining] if part
                )
                lines[-1] = self._ellipsize(
                    draw,
                    combined,
                    font,
                    last_width,
                )

            underline_positions = self._find_underlines_for_rows(
                source_pixels,
                rows,
            )
            for row, line, underline_y in zip(
                rows,
                lines,
                underline_positions,
            ):
                if not line:
                    continue
                baseline_y = underline_y - max(
                    2,
                    int(round(font.size * 0.24)),
                )
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

        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(output_path, result_image):
            raise OSError(f"Could not write solved worksheet: {output_path}")

        print(f"Solved worksheet saved as: {output_path}")
        return result_image
