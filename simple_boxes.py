import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def sort_reading_order(boxes):
    """Sort boxes in reading order: top to bottom, then left to right.
    
    Boxes on the same text line often have slightly different y coordinates.
    This method groups vertically overlapping boxes into rows.
    Format: (x1, y1, x2, y2)
    """
    if not boxes:
        return boxes
    
    # Start with a rough vertical sort.
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    
    # Group boxes into rows based on vertical overlap.
    lines = []
    current_line = [boxes_sorted[0]]
    line_y_min = boxes_sorted[0][1]
    line_y_max = boxes_sorted[0][3]  # y2
    
    for box in boxes_sorted[1:]:
        box_y_top = box[1]
        box_y_bottom = box[3]  # y2
        box_height = box_y_bottom - box_y_top
        line_height = line_y_max - line_y_min
        
        # Check whether the box overlaps the current row vertically.
        overlap = min(line_y_max, box_y_bottom) - max(line_y_min, box_y_top)
        min_height = max(min(box_height, line_height), 1)
        
        if overlap > 0 and overlap / min_height > 0.3:
            # Same row.
            current_line.append(box)
            line_y_min = min(line_y_min, box_y_top)
            line_y_max = max(line_y_max, box_y_bottom)
        else:
            # New row.
            lines.append(current_line)
            current_line = [box]
            line_y_min = box_y_top
            line_y_max = box_y_bottom
    
    lines.append(current_line)
    
    # Sort each row from left to right.
    result = []
    for line in lines:
        line.sort(key=lambda b: b[0])
        result.extend(line)
    
    return result


def calculate_iou(box1, box2):
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


def filter_overlapping_boxes(boxes, iou_threshold=0.5):
    """Filter overlapping boxes, keeping the highest-confidence detection."""
    if len(boxes) == 0:
        return []
    
    coords = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    sorted_indices = np.argsort(-confidences)
    
    keep = []
    for i in sorted_indices:
        should_keep = True
        for kept_idx in keep:
            iou = calculate_iou(coords[i], coords[kept_idx])
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(i)
    
    return sorted(keep)


def find_gaps_yolo(path: str, model_path: str = "best_model.pt", conf: float = 0.25):
    """Detect writable worksheet gaps with YOLO."""
    
    if not Path(path).exists():
        print(f"❌ Image not found: {path}")
        return []
    
    if not Path(model_path).exists():
        print(f"❌ YOLO model not found: {model_path}")
        return []
    
    model = YOLO(model_path)
    print(f"📊 YOLO model loaded: {model_path}")
    
    # YOLO prediction.
    results = model.predict(source=path, conf=conf)
    
    gaps = []
    img = None
    
    for r in results:
        img = r.orig_img.copy()
        if len(r.boxes) > 0:
            keep_indices = filter_overlapping_boxes(r.boxes, iou_threshold=0.5)
            print(
                f"🔍 {len(r.boxes)} boxes detected, "
                f"{len(keep_indices)} after filtering"
            )
            
            for idx in keep_indices:
                box = r.boxes[idx]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                gaps.append((int(x1), int(y1), int(x2), int(y2)))
        else:
            print("❌ No gaps detected!")
            return []
    
    # Sort in reading order, grouped by row.
    gaps = sort_reading_order(gaps)
    
    print(f"✅ {len(gaps)} gaps found!")
    
    # Mark detections.
    result = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(gaps):
        color = (0, 0, 255)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        # Number label with a filled background.
        label = str(i + 1)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(result, (x1, y1 - label_size[1] - 4), (x1 + label_size[0] + 2, y1), color, -1)
        cv2.putText(result, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        print(f"  Gap {i+1}: ({x1}, {y1}) -> ({x2}, {y2})")
    
    # Save the marked image.
    output_path = f"{Path(path).stem}_marked.png"
    cv2.imwrite(output_path, result)
    print(f"💾 Marked image saved: {output_path}")
    
    return gaps


if __name__ == "__main__":
    find_gaps_yolo("test.jpg")
