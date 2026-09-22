from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate the intersection over union (IoU) of two boxes.
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


def filter_overlapping_boxes(boxes, iou_threshold=0.5):
    """
    Filter overlapping boxes, keeping only the highest-confidence box.
    
    Args:
        boxes: YOLO boxes object.
        iou_threshold: Minimum IoU considered an overlap (0.5 = 50%).
    
    Returns:
        List of indices for the boxes to keep.
    """
    if len(boxes) == 0:
        return []
    
    # Extract coordinates and confidence scores.
    coords = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    confidences = boxes.conf.cpu().numpy()
    
    # Sort by confidence (highest first).
    sorted_indices = np.argsort(-confidences)
    
    keep = []
    
    for i in sorted_indices:
        # Check whether this box overlaps a box that has already been kept.
        should_keep = True
        
        for kept_idx in keep:
            iou = calculate_iou(coords[i], coords[kept_idx])
            
            if iou > iou_threshold:
                # Discard this box because it has the lower confidence score.
                should_keep = False
                break
        
        if should_keep:
            keep.append(i)
    
    return sorted(keep)  # Restore the original order.


def is_line_class(class_name):
    """True only for the exact YOLO class name 'line'."""
    return str(class_name).strip().lower() == "line"


def unit_bbox(unit, gaps):
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


def sort_units_reading_order(units, gaps):
    """Sort units globally by reading order: top->bottom, left->right."""
    if not units:
        return []

    unit_data = []
    for idx, unit in enumerate(units):
        x1, y1, x2, y2 = unit_bbox(unit, gaps)
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


def group_gaps_by_proximity(gaps):
    """
    Group boxes that are positioned directly below one another.
    
    Args:
        gaps: List of gap boxes as (x1, y1, x2, y2) tuples.
    
    Returns:
        groups: Groups containing gap indices in their original order.
        gap_to_group: Mapping from each gap index to its group index.
    """
    if not gaps:
        return [], {}
    
    # Create an index mapping from sorted indices to original indices.
    indices = list(range(len(gaps)))
    sorted_indices = sorted(indices, key=lambda i: gaps[i][1])  # Top to bottom.
    
    # Use the average gap height to derive the distance threshold.
    heights = [(gap[3] - gap[1]) for gap in gaps]
    avg_height = sum(heights) / len(heights) if heights else 0
    
    # Line boxes may overlap slightly or have a small vertical gap.
    distance_threshold = avg_height * 1.5
    overlap_tolerance = max(5, int(avg_height * 0.15))
    
    groups = []
    gap_to_group = {}
    grouped = set()
    
    # Process gaps from top to bottom.
    for sort_i, i in enumerate(sorted_indices):
        if i in grouped:
            continue
        
        gap_i = gaps[i]
        x1_i, y1_i, x2_i, y2_i = gap_i[:4]
        class_name_i = gap_i[4] if len(gap_i) > 4 else "line"
        
        # Only exact line-class detections are grouped.
        if not is_line_class(class_name_i):
            continue

        # Start a new group with the current line gap.
        current_group = [i]
        grouped.add(i)
        
        # Search for gaps below the current gap.
        for sort_j in range(sort_i + 1, len(sorted_indices)):
            j = sorted_indices[sort_j]
            
            if j in grouped:
                continue
            
            gap_j = gaps[j]
            x1_j, y1_j, x2_j, y2_j = gap_j[:4]
            class_name_j = gap_j[4] if len(gap_j) > 4 else "line"
            
            # Only group if both are exact line class detections
            if not is_line_class(class_name_j):
                continue
            
            # Box j may overlap slightly or sit just below box i.
            vertical_distance = y1_j - y2_i
            
            # Check horizontal alignment.
            i_left, i_top, i_right, i_bottom = x1_i, y1_i, x2_i, y2_i
            j_left, j_top, j_right, j_bottom = x1_j, y1_j, x2_j, y2_j
            
            # Calculate horizontal overlap.
            h_overlap_start = max(i_left, j_left)
            h_overlap_end = min(i_right, j_right)
            h_overlap = max(0, h_overlap_end - h_overlap_start)
            
            # Box widths.
            i_width = i_right - i_left
            j_width = j_right - j_left
            min_width = min(i_width, j_width)
            
            # Check whether box j belongs to the same vertical group and is aligned.
            if -overlap_tolerance <= vertical_distance < distance_threshold:
                # Require at least 30% overlap or a minimum visible overlap.
                if h_overlap > min_width * 0.3 or h_overlap > 15:  # 15px min overlap
                    current_group.append(j)
                    grouped.add(j)
                    gap_i = gap_j  # Use this gap in the next iteration.
                    x1_i, y1_i, x2_i, y2_i = gap_i[:4]
                else:
                    # End the group if the boxes do not overlap enough.
                    break
            else:
                # End the group if the vertical distance is too large.
                break
        
        # Store the group with indices in their original order.
        current_group.sort()
        for idx in current_group:
            gap_to_group[idx] = len(groups)
        
        groups.append(current_group)
    
    return groups, gap_to_group

# Load the trained model.
MODEL_PATH = "./model/v1.2.1/gap_detection_model.pt"

# Verify that the trained model exists.
if not Path(MODEL_PATH).exists():
    print(f"❌ Trained model not found: {MODEL_PATH}")
    print("💡 Run train_yolo.py first!")
    print("\nIf the model exists elsewhere, update MODEL_PATH accordingly.")
    exit()

print(f"✅ Loading trained model: {MODEL_PATH}\n")
model = YOLO(MODEL_PATH)

# Image to test.
IMAGE_PATH = 'test.jpg'
results = model.predict(source=IMAGE_PATH, save=True, conf=0.25)

# Process the results.
for r in results:
    print(f"📸 Image: {r.path}")
    print(f"⚡ Speed: {r.speed}")
    print(f"📦 Detections before filtering: {len(r.boxes)}")
    
    # Filter overlapping boxes.
    if len(r.boxes) > 0:
        keep_indices = filter_overlapping_boxes(r.boxes, iou_threshold=0.5)
        print(f"🔍 Boxes after overlap filtering: {len(keep_indices)}")
    else:
        keep_indices = []
    
    if len(keep_indices) == 0:
        print("\n❌ No writable areas detected!")
        print("💡 Check the following:")
        print("   - Is the image a worksheet?")
        print("   - Was the model trained correctly?")
        print("   - Try a lower confidence threshold (for example, 0.1).")
    else:
        # Extract gap boxes from the filtered indices.
        gaps = []
        gap_info = []  # Keep box information for later reference.
        
        for idx in keep_indices:
            box = r.boxes[idx]
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            gaps.append((int(x1), int(y1), int(x2), int(y2), class_name))
            gap_info.append({
                'box': box,
                'class_id': class_id,
                'confidence': float(box.conf[0])
            })
        
        # Group line boxes and build globally ordered answer units.
        groups, gap_to_group = group_gaps_by_proximity(gaps)
        grouped_indices = set(gap_to_group.keys())
        ungrouped_indices = [i for i in range(len(gaps)) if i not in grouped_indices]

        unsorted_units = list(groups) + [[idx] for idx in ungrouped_indices]
        answer_units = sort_units_reading_order(unsorted_units, gaps)
        gap_to_unit = {}
        for unit_idx, unit in enumerate(answer_units):
            for gap_idx in unit:
                gap_to_unit[gap_idx] = unit_idx

        print(f"\n✅ Writable areas after filtering: {len(gaps)}")
        print(f"📊 Line boxes grouped into {len(groups)} groups")
        print(f"📌 Ungrouped boxes (for example, gap): {len(ungrouped_indices)}\n")
        print(f"🔢 Globally numbered answer units: {len(answer_units)}\n")
        
        # Display answer units.
        for unit_idx, unit in enumerate(answer_units):
            print(f"📍 Unit {unit_idx + 1}: {len(unit)} area(s)")

            for pos_in_group, gap_idx in enumerate(unit):
                box = gap_info[gap_idx]
                gap = gaps[gap_idx]
                x1, y1, x2, y2 = gap[:4]
                
                print(f"   Area {pos_in_group + 1}:")
                print(f"     Class: {r.names[box['class_id']]}")
                print(f"     Confidence: {box['confidence']:.2%}")
                print(f"     Box: ({x1}, {y1}) → ({x2}, {y2})")
                print(f"     Size: {x2-x1} x {y2-y1} px")

        # Display ungrouped boxes separately.
        if ungrouped_indices:
            print("\n🧩 Ungrouped boxes:")
            for idx in ungrouped_indices:
                box = gap_info[idx]
                x1, y1, x2, y2 = gaps[idx][:4]
                unit_num = gap_to_unit.get(idx, -1) + 1
                print(f"   - No. {unit_num} | Class: {r.names[box['class_id']]} | Confidence: {box['confidence']:.2%} | Box: ({x1}, {y1}) → ({x2}, {y2})")
    
    # Display the image with filtered writable areas marked.
    print("\n🎨 Displaying result...")
    
    if len(keep_indices) > 0:
        # Draw one combined box per answer unit.
        img = r.orig_img.copy()
        for unit_idx, unit in enumerate(answer_units):
            x1, y1, x2, y2 = unit_bbox(unit, gaps)

            # Draw the box.
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Use a numeric label only.
            label = str(unit_idx + 1)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 4), (x1 + label_size[0] + 2, y1), (255, 0, 0), -1)
            cv2.putText(img, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        annotated = img
    else:
        annotated = r.orig_img.copy()
    
    # Save the result.
    output_path = 'yolo_detected_gaps.png'
    cv2.imwrite(output_path, annotated)
    print(f"💾 Saved: {output_path}")
    
    # Display the result.
    cv2.imshow('YOLO - Writable Area Detection', annotated)
    print("👁️  Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\n✅ Finished!")
