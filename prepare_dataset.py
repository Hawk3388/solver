"""Prepare worksheet images and YOLO labels for detector training."""

import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from ultralytics import YOLO

def find_gaps_in_image(image_path):
    """
    Find writable areas in a worksheet using the simple_boxes.py method.

    Returns: A list of (x, y, width, height) tuples.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return [], None
    
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gaps = []
    
    # Step 1: Find horizontal lines and underlines.
    inv = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inv, 100, 255, cv2.THRESH_BINARY)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    horizontal_lines = cv2.dilate(horizontal_lines, dilate_kernel, iterations=1)
    
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        if w_box > 40 and h_box < 15 and w_box > h_box * 4:
            if w_box < 350:
                text_height = 18
                text_y = max(0, y - text_height)
                gaps.append((x, text_y, w_box, text_height))
    
    # Step 2: Find empty rectangular areas.
    _, white_thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_areas = cv2.morphologyEx(white_thresh, cv2.MORPH_OPEN, clean_kernel)
    
    expand_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_areas = cv2.morphologyEx(white_areas, cv2.MORPH_CLOSE, expand_kernel)
    
    white_contours, _ = cv2.findContours(white_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in white_contours:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        area = w_box * h_box
        if 1500 < area < 6000:
            aspect_ratio = w_box / h_box if h_box > 0 else 0
            if 2 < aspect_ratio < 6:
                is_duplicate = False
                for existing_x, existing_y, existing_w, existing_h in gaps:
                    if (abs(x - existing_x) < 30 and abs(y - existing_y) < 20):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    gaps.append((x, y, w_box, h_box))
    
    # Remove duplicates and consolidate nearby detections.
    final_gaps = []
    gaps.sort(key=lambda g: (g[1], g[0]))
    
    for x, y, w_box, h_box in gaps:
        merged = False
        for i, (fx, fy, fw, fh) in enumerate(final_gaps):
            if abs(y - fy) < 10 and abs(x - fx) < 50:
                new_x = min(x, fx)
                new_y = min(y, fy)
                new_w = max(x + w_box, fx + fw) - new_x
                new_h = max(y + h_box, fy + fh) - new_y
                final_gaps[i] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        
        if not merged:
            final_gaps.append((x, y, w_box, h_box))
    
    final_gaps.sort(key=lambda gap: (gap[1], gap[0]))
    
    return final_gaps, (w, h)


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
    """
    Filter overlapping boxes, keeping the highest-confidence detection.

    Args:
        boxes: YOLO boxes object.
        iou_threshold: Minimum overlap IoU (0.5 = 50%).
    
    Returns:
        Indices of the boxes to keep.
    """
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


def find_gaps_with_yolo(image_path, model, conf=0.25, iou_threshold=0.5):
    """
    Find writable areas with a YOLO model.
    
    Args:
        image_path: Path to the image.
        model: Loaded YOLO model.
        conf: Confidence threshold.
        iou_threshold: IoU threshold used to filter overlaps.
    
    Returns: A list of boxes and the image dimensions.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return [], None
    
    img_h, img_w = image.shape[:2]
    
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    
    gaps = []
    for r in results:
        if len(r.boxes) > 0:
            keep_indices = filter_overlapping_boxes(r.boxes, iou_threshold=iou_threshold)
            
            for idx in keep_indices:
                box = r.boxes[idx]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # Convert xyxy coordinates to xywh.
                gaps.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    
    # Sort in reading order.
    gaps.sort(key=lambda gap: (gap[1], gap[0]))
    
    return gaps, (img_w, img_h)


def boxes_to_yolo_format(boxes, image_width, image_height):
    """
    Convert bounding boxes to normalized YOLO label rows.
    
    Args:
        boxes: List of (x, y, width, height) tuples.
        image_width, image_height: Source image dimensions.
        
    Returns:
        YOLO label strings.
    """
    yolo_labels = []
    
    for x, y, w, h in boxes:
        # Normalize coordinates to the 0-1 range.
        x_center = ((x + w / 2) / image_width)
        y_center = ((y + h / 2) / image_height)
        width_norm = w / image_width
        height_norm = h / image_height
        
        # YOLO Format: class_id x_center y_center width height
        # class_id = 0 because this helper creates gap labels only.
        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
    
    return yolo_labels


def prepare_yolo_dataset(source_dir, output_dir, train_split=0.8, visualize=False, yolo_model_path=None, yolo_conf=0.25):
    """
    Prepare a dataset for YOLO training.
    
    Args:
        source_dir: Directory containing worksheet images.
        output_dir: Destination directory for the YOLO dataset.
        train_split: Training share; the remainder is used for validation.
        visualize: Create marked review images when true.
        yolo_model_path: Optional detector path; otherwise use CV detection.
        yolo_conf: YOLO confidence threshold.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Confirm that the source directory exists.
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    if output_path.exists():
        print(f"⚠️  Destination directory already exists: {output_dir}")
        while True:
            choice = input("Delete and recreate the directory? (y/n): ").lower()
            if choice == 'y':
                shutil.rmtree(output_path)
                print(f"🗑️  Directory deleted: {output_dir}")
                break
            elif choice == 'n':
                print("Cancelled. Choose a different destination directory.")
                return
            else:
                print("Invalid input. Enter 'y' or 'n'.")

    # Create the dataset structure.
    folders = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]
    
    if visualize:
        folders.extend(['visualize/train', 'visualize/val'])
    
    for folder in folders:
        (output_path / folder).mkdir(parents=True, exist_ok=True)
    
    # Find all supported images.
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    images = []
    for ext in image_extensions:
        images.extend(list(source_path.glob(f'*{ext}')))
    
    if len(images) == 0:
        print(f"❌ No images found in: {source_dir}")
        return
    
    print(f"📁 {len(images)} images found")
    
    # Load the optional YOLO seed model.
    yolo_model = None
    if yolo_model_path:
        yolo_model_file = Path(yolo_model_path)
        if not yolo_model_file.exists():
            print(f"❌ YOLO model not found: {yolo_model_path}")
            return
        yolo_model = YOLO(str(yolo_model_path))
        print(f"🤖 YOLO model loaded: {yolo_model_path}")
        print(f"   Confidence Threshold: {yolo_conf}")
    else:
        print("🔧 Using CV-based detection")
    
    print("🔄 Processing images...")
    
    # Statistics.
    stats = {
        'total_images': 0,
        'total_gaps': 0,
        'train_images': 0,
        'val_images': 0,
        'skipped': 0
    }
    
    # Process images.
    for idx, img_path in enumerate(tqdm(images, desc="Processing")):
        # Find gaps with YOLO or the CV fallback.
        if yolo_model:
            gaps, size = find_gaps_with_yolo(img_path, yolo_model, conf=yolo_conf)
        else:
            gaps, size = find_gaps_in_image(img_path)
        
        if size is None:
            stats['skipped'] += 1
            continue
        img_w, img_h = size
        
        if len(gaps) == 0:
            stats['skipped'] += 1
            print(f"⚠️  No gaps found in: {img_path.name}")
            continue
        
        # Train/validation split.
        is_train = idx < int(len(images) * train_split)
        split = 'train' if is_train else 'val'
        
        # Create a filename without spaces.
        safe_name = img_path.stem.replace(' ', '_')
        safe_extension = img_path.suffix
        
        # Copy the source image.
        target_image = output_path / 'images' / split / f"{safe_name}{safe_extension}"
        shutil.copy(img_path, target_image)
        
        # Create YOLO labels.
        yolo_labels = boxes_to_yolo_format(gaps, img_w, img_h)
        label_file = output_path / 'labels' / split / f"{safe_name}.txt"
        
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        # Optionally create a marked review image.
        if visualize:
            img = cv2.imread(str(img_path))
            for x, y, w, h in gaps:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            viz_path = output_path / 'visualize' / split / f"{safe_name}_marked{safe_extension}"
            cv2.imwrite(str(viz_path), img)
        
        # Update statistics.
        stats['total_images'] += 1
        stats['total_gaps'] += len(gaps)
        if is_train:
            stats['train_images'] += 1
        else:
            stats['val_images'] += 1
    
    # Create data.yaml.
    yaml_content = f"""# Worksheet gap detection dataset
path: {output_path.absolute().as_posix()}
train: images/train
val: images/val

# Classes
nc: 1
names: ['gap']
"""
    
    with open(output_path / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    # Summary.
    print("\n✅ Dataset preparation complete!")
    print("📊 Statistics:")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Training: {stats['train_images']}")
    print(f"   Validation: {stats['val_images']}")
    print(f"   Total gaps: {stats['total_gaps']}")
    print(f"   Average per image: {stats['total_gaps']/stats['total_images']:.1f}")
    print(f"   Skipped: {stats['skipped']}")
    print(f"\n📁 Dataset saved to: {output_path}")
    print(f"📄 Configuration file: {output_path / 'data.yaml'}")
    
    if visualize:
        print(f"🎨 Visualizations: {output_path / 'visualize'}")


if __name__ == "__main__":
    # Configuration.
    SOURCE_DIR = "raw_images"       # Directory containing worksheet images.
    OUTPUT_DIR = "dataset"          # Destination for the YOLO dataset.
    TRAIN_SPLIT = 0.8               # 80% training, 20% validation.
    VISUALIZE = True                # Create marked review images.

    # Detection model; set to None to use CV-based detection.
    YOLO_MODEL = "gap_detection_model.pt"  # Path to the trained YOLO model.
    YOLO_CONF = 0.25                       # Confidence threshold.

    print("🚀 YOLO dataset preparation")
    print(f"📂 Source directory: {SOURCE_DIR}")
    print(f"📂 Destination directory: {OUTPUT_DIR}")
    print(f"📊 Train/Val Split: {TRAIN_SPLIT*100:.0f}% / {(1-TRAIN_SPLIT)*100:.0f}%")
    print(f"🎨 Visualization: {'Yes' if VISUALIZE else 'No'}")
    print(
        f"🤖 YOLO model: "
        f"{YOLO_MODEL if YOLO_MODEL else 'Not used (CV-based)'}"
    )
    print("-" * 60)
    
    prepare_yolo_dataset(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        train_split=TRAIN_SPLIT,
        visualize=VISUALIZE,
        yolo_model_path=YOLO_MODEL,
        yolo_conf=YOLO_CONF
    )
