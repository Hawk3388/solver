"""Gap detection, ordering, grouping, and debug marking."""

import cv2
import numpy as np


class DetectionMixin:
    """Detection stage used by :class:`main.WorksheetSolver`."""

    def load_image(self, image_path: str):
        """Load an image and keep the original OpenCV representation."""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image {image_path} not found!")
        return self.image.copy()

    @staticmethod
    def calculate_iou(box1: list, box2: list):
        """Calculate Intersection over Union for two ``xyxy`` boxes."""
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
        """Keep the highest-confidence detection from overlapping boxes."""
        if len(boxes) == 0:
            return []

        coords = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        sorted_indices = np.argsort(-confidences)
        keep = []

        for index in sorted_indices:
            if all(
                self.calculate_iou(coords[index], coords[kept_index]) <= iou_threshold
                for kept_index in keep
            ):
                keep.append(index)

        return sorted(keep)

    @staticmethod
    def sort_reading_order(boxes):
        """Sort boxes top-to-bottom and left-to-right within a text row."""
        if not boxes:
            return boxes

        boxes_sorted = sorted(boxes, key=lambda box: box[1])
        lines = []
        current_line = [boxes_sorted[0]]
        line_y_min = boxes_sorted[0][1]
        line_y_max = boxes_sorted[0][3]

        for box in boxes_sorted[1:]:
            box_y_top = box[1]
            box_y_bottom = box[3]
            box_height = box_y_bottom - box_y_top
            line_height = line_y_max - line_y_min
            overlap = min(line_y_max, box_y_bottom) - max(line_y_min, box_y_top)
            min_height = max(min(box_height, line_height), 1)

            if overlap > 0 and overlap / min_height > 0.3:
                current_line.append(box)
                line_y_min = min(line_y_min, box_y_top)
                line_y_max = max(line_y_max, box_y_bottom)
            else:
                lines.append(current_line)
                current_line = [box]
                line_y_min = box_y_top
                line_y_max = box_y_bottom

        lines.append(current_line)

        result = []
        for line in lines:
            line.sort(key=lambda box: box[0])
            result.extend(line)
        return result

    @staticmethod
    def is_line_class(class_name):
        """Return whether a YOLO class represents a ruled answer line."""
        return str(class_name).strip().lower() == "line"

    @staticmethod
    def _unit_bbox(unit, gaps):
        """Return a merged ``xyxy`` bounding box for an answer unit."""
        boxes = [gaps[index][:4] for index in unit if 0 <= index < len(gaps)]
        if not boxes:
            return (0, 0, 0, 0)
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    def sort_answer_units_reading_order(self, units, gaps):
        """Sort answer units globally top-to-bottom and left-to-right."""
        if not units:
            return []

        unit_data = []
        for index, unit in enumerate(units):
            x1, y1, x2, y2 = self._unit_bbox(unit, gaps)
            unit_data.append(
                {
                    "idx": index,
                    "unit": unit,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "h": max(1, y2 - y1),
                }
            )

        unit_data.sort(key=lambda unit: unit["y1"])
        rows = []
        current_row = [unit_data[0]]
        row_y_min = unit_data[0]["y1"]
        row_y_max = unit_data[0]["y2"]

        for unit in unit_data[1:]:
            overlap = min(row_y_max, unit["y2"]) - max(row_y_min, unit["y1"])
            row_height = max(1, row_y_max - row_y_min)
            min_height = max(1, min(row_height, unit["h"]))

            if overlap > 0 and overlap / min_height > 0.3:
                current_row.append(unit)
                row_y_min = min(row_y_min, unit["y1"])
                row_y_max = max(row_y_max, unit["y2"])
            else:
                rows.append(current_row)
                current_row = [unit]
                row_y_min = unit["y1"]
                row_y_max = unit["y2"]

        rows.append(current_row)

        sorted_units = []
        for row in rows:
            row.sort(key=lambda unit: unit["x1"])
            sorted_units.extend(unit["unit"] for unit in row)
        return sorted_units

    def group_gaps_by_proximity(self, gaps):
        """Group vertically adjacent and horizontally aligned line boxes."""
        if not gaps:
            return [], {}

        sorted_indices = sorted(range(len(gaps)), key=lambda index: gaps[index][1])
        heights = [gap[3] - gap[1] for gap in gaps]
        average_height = sum(heights) / len(heights) if heights else 0
        distance_threshold = average_height * 1.5
        overlap_tolerance = max(5, int(average_height * 0.15))

        groups = []
        gap_to_group = {}
        grouped = set()

        for sorted_position, index in enumerate(sorted_indices):
            if index in grouped:
                continue

            gap = gaps[index]
            x1, y1, x2, y2 = gap[:4]
            class_name = gap[4] if len(gap) > 4 else "line"
            if not self.is_line_class(class_name):
                continue

            current_group = [index]
            grouped.add(index)

            for candidate_position in range(sorted_position + 1, len(sorted_indices)):
                candidate_index = sorted_indices[candidate_position]
                if candidate_index in grouped:
                    continue

                candidate = gaps[candidate_index]
                candidate_class = candidate[4] if len(candidate) > 4 else "line"
                if not self.is_line_class(candidate_class):
                    continue

                candidate_x1, candidate_y1, candidate_x2, candidate_y2 = candidate[:4]
                vertical_distance = candidate_y1 - y2
                horizontal_overlap = max(
                    0,
                    min(x2, candidate_x2) - max(x1, candidate_x1),
                )
                minimum_width = min(x2 - x1, candidate_x2 - candidate_x1)

                if -overlap_tolerance <= vertical_distance < distance_threshold:
                    if horizontal_overlap > minimum_width * 0.3 or horizontal_overlap > 15:
                        current_group.append(candidate_index)
                        grouped.add(candidate_index)
                        x1, y1, x2, y2 = candidate[:4]
                    else:
                        break
                else:
                    break

            current_group.sort()
            for grouped_index in current_group:
                gap_to_group[grouped_index] = len(groups)
            groups.append(current_group)

        return groups, gap_to_group

    def detect_gaps(self):
        """Detect gaps and build ordered answer units."""
        self.detected_gaps = []
        image = self.load_image(self.path)
        runtime = getattr(self, "detection_runtime", None)
        if runtime is not None:
            results = runtime.predict(source=self.path, conf=0.10)
        else:
            # Compatibility for lightweight test doubles using this mixin.
            results = self.model.predict(source=self.path, conf=0.10)

        for result in results:
            if len(result.boxes) > 0:
                keep_indices = self.filter_overlapping_boxes(
                    result.boxes,
                    iou_threshold=0.5,
                )
                print(f"After overlap filtering: {len(keep_indices)} boxes")
            else:
                keep_indices = []

            if not keep_indices:
                print("\nNo gaps detected!")
                print("Check:")
                print("   - Is the image a worksheet?")
                print("   - Was the model trained correctly?")
                print("   - Try lower conf (e.g. 0.1)")
                continue

            for index in keep_indices:
                box = result.boxes[index]
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                self.detected_gaps.append(
                    (int(x1), int(y1), int(x2), int(y2), class_name)
                )
            image = result.orig_img.copy()

        self.detected_gaps = self.sort_reading_order(self.detected_gaps)
        self.gap_groups, self.gap_to_group = self.group_gaps_by_proximity(
            self.detected_gaps
        )
        self.ungrouped_gap_indices = [
            index
            for index in range(len(self.detected_gaps))
            if index not in self.gap_to_group
        ]

        unsorted_units = list(self.gap_groups) + [
            [index] for index in self.ungrouped_gap_indices
        ]
        self.answer_units = self.sort_answer_units_reading_order(
            unsorted_units,
            self.detected_gaps,
        )
        self.gap_to_answer_unit = {}
        for unit_index, unit in enumerate(self.answer_units):
            for gap_index in unit:
                self.gap_to_answer_unit[gap_index] = unit_index

        print(f"Line-boxes grouped into {len(self.gap_groups)} groups")
        for index, group in enumerate(self.gap_groups):
            print(f"   Group {index + 1}: {len(group)} gaps (indices: {group})")
        print(f"Ungrouped boxes (e.g. gap): {len(self.ungrouped_gap_indices)}")
        print(f"Total AI answer units: {len(self.answer_units)}")
        return self.detected_gaps, image

    def mark_gaps(self, image, gaps):
        """Draw one numbered red rectangle per answer unit."""
        if not self.answer_units:
            return image

        for unit_index, unit in enumerate(self.answer_units):
            unit_boxes = [gaps[index][:4] for index in unit if 0 <= index < len(gaps)]
            if not unit_boxes:
                continue

            x1 = min(box[0] for box in unit_boxes)
            y1 = min(box[1] for box in unit_boxes)
            x2 = max(box[2] for box in unit_boxes)
            y2 = max(box[3] for box in unit_boxes)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            label = str(unit_index + 1)
            label_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                1,
            )
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0] + 2, y1),
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1 + 1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
        return image
