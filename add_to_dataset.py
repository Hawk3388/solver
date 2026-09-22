"""
Add images to an existing dataset and review them in the box editor.
The script detects boxes with YOLO, opens the editor for corrections, and then
adds each image while preserving the 80/20 split.

Usage:
    python add_to_dataset.py image1.png image2.jpg ...
    python add_to_dataset.py              # asks for images interactively
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
from edit_boxes import BoxEditor

# -- Configuration -----------------------------------------------------------
DATASET_DIR = "dataset"
YOLO_MODEL = "./model/gap_detection_model.pt"
YOLO_CONF = 0.25
TRAIN_SPLIT = 0.8  # 80% train, 20% val
VISUALIZE = True
# ---------------------------------------------------------------------------


def count_dataset_images(dataset_dir):
    """Count the current dataset images in each split."""
    dataset_path = Path(dataset_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    counts = {'train': 0, 'val': 0}
    for split in ['train', 'val']:
        img_dir = dataset_path / 'images' / split
        if img_dir.exists():
            counts[split] = sum(1 for f in img_dir.iterdir() if f.suffix.lower() in image_extensions)
    
    return counts


def choose_split(dataset_dir, train_split=0.8):
    """
    Select train or validation based on the current ratio.

    The image is assigned to the split furthest below its target share.
    """
    counts = count_dataset_images(dataset_dir)
    total = counts['train'] + counts['val']
    
    if total == 0:
        return 'train'
    
    current_train_ratio = counts['train'] / (total + 1)  # Include the new image.
    
    # Add to validation when its share is too small; otherwise use training.
    if current_train_ratio >= train_split:
        return 'val'
    else:
        return 'train'


def detect_gaps(image_path, model, conf=0.25):
    """Detect gaps with YOLO and return ``xyxy`` boxes."""
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    
    boxes = []
    for r in results:
        if len(r.boxes) > 0:
            coords = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            sorted_indices = np.argsort(-confidences)
            
            keep = []
            for i in sorted_indices:
                should_keep = True
                for kept_idx in keep:
                    box1, box2 = coords[i], coords[kept_idx]
                    x1_i = max(box1[0], box2[0])
                    y1_i = max(box1[1], box2[1])
                    x2_i = min(box1[2], box2[2])
                    y2_i = min(box1[3], box2[3])
                    if x2_i > x1_i and y2_i > y1_i:
                        inter = (x2_i - x1_i) * (y2_i - y1_i)
                        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                        if inter / (a1 + a2 - inter) > 0.5:
                            should_keep = False
                            break
                if should_keep:
                    keep.append(i)
            
            for idx in keep:
                box = r.boxes[idx]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
    
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def save_to_dataset(image_path, label_path, boxes, dataset_dir, split, visualize=True):
    """Save an image and its labels to a dataset split."""
    dataset_path = Path(dataset_dir)
    img_path = Path(image_path)
    
    safe_name = img_path.stem.replace(' ', '_')
    safe_ext = img_path.suffix
    
    # Ensure that target directories exist.
    for folder in ['images', 'labels']:
        (dataset_path / folder / split).mkdir(parents=True, exist_ok=True)
    
    # Copy the image.
    target_image = dataset_path / 'images' / split / f"{safe_name}{safe_ext}"
    shutil.copy(str(img_path), str(target_image))
    
    # Copy labels saved by the editor from its temporary file.
    target_label = dataset_path / 'labels' / split / f"{safe_name}.txt"
    if Path(label_path).exists():
        shutil.copy(str(label_path), str(target_label))
    else:
        # Generate labels directly from boxes.
        image = cv2.imread(str(img_path))
        img_h, img_w = image.shape[:2]
        yolo_lines = []
        for x1, y1, x2, y2 in boxes:
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        with open(str(target_label), 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    # Create a visualization.
    if visualize:
        viz_dir = dataset_path / 'visualize' / split
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(str(img_path))
        
        # Read labels so each class can use its configured colour.
        class_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 165, 255)}  # BGR
        
        with open(str(target_label), 'r') as f:
            label_lines = f.readlines()
        
        img_h, img_w = img.shape[:2]
        for line in label_lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                width = float(parts[3]) * img_w
                height = float(parts[4]) * img_h
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                color = class_colors.get(class_id, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        viz_path = viz_dir / f"{safe_name}_marked{safe_ext}"
        cv2.imwrite(str(viz_path), img)
    
    return target_image, target_label


def resolve_input_paths(inputs):
    """
    Resolve each input as a file or directory.

    Files are accepted directly. Directories contribute their supported image
    files without recursive traversal.

    Returns:
        (resolved_images, seen_sources)
        resolved_images: Resolved image paths.
        seen_sources: Number of existing file or directory inputs.
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}
    resolved_images = []
    seen_sources = 0

    for raw in inputs:
        cleaned = str(raw).lstrip('&').strip().strip('"').strip("'")
        if not cleaned:
            continue

        path = Path(cleaned)
        if not path.exists():
            print(f"⚠️  Not found: {cleaned}")
            continue

        seen_sources += 1

        if path.is_dir():
            folder_images = [
                str(f) for f in sorted(path.iterdir())
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            if not folder_images:
                print(f"⚠️  No images found in directory: {path}")
            else:
                resolved_images.extend(folder_images)
                print(f"📁 {path.name}: {len(folder_images)} image(s) found")
        elif path.is_file():
            if path.suffix.lower() in image_extensions:
                resolved_images.append(str(path))
            else:
                print(f"⚠️  Unsupported image format: {path}")

    # Remove duplicates while retaining input order.
    resolved_images = list(dict.fromkeys(resolved_images))
    return resolved_images, seen_sources


def add_images(image_paths):
    """Add and review a collection of images."""
    dataset_path = Path(DATASET_DIR)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {DATASET_DIR}")
        print("💡 Run prepare_dataset.py first!")
        return
    
    # Load the YOLO model.
    model_path = Path(YOLO_MODEL)
    if not model_path.exists():
        print(f"❌ YOLO model not found: {YOLO_MODEL}")
        return
    
    model = YOLO(str(model_path))
    print(f"🤖 YOLO model loaded: {YOLO_MODEL}")
    
    # Show current dataset statistics.
    counts = count_dataset_images(DATASET_DIR)
    print(f"📊 Current dataset: {counts['train']} train / {counts['val']} val")
    
    # Clean and validate input paths.
    cleaned_paths = []
    for p in image_paths:
        # Remove PowerShell call operators and surrounding quotes.
        cleaned = str(p).lstrip('&').strip().strip('"').strip("'")
        if cleaned:
            cleaned_paths.append(cleaned)
    
    # Filter images.
    valid_images = []
    for p in cleaned_paths:
        path = Path(p)
        if not path.exists():
            print(f"⚠️  Not found: {p}")
        elif path.suffix.lower() not in image_extensions:
            print(f"⚠️  Unsupported image format: {p}")
        else:
            # Check whether the image is already in the dataset.
            safe_name = path.stem.replace(' ', '_')
            already_exists = False
            for split in ['train', 'val']:
                for ext in image_extensions:
                    if (dataset_path / 'images' / split / f"{safe_name}{ext}").exists():
                        already_exists = True
                        break
            if already_exists:
                print(f"⚠️  Already in the dataset: {path.name}")
            else:
                valid_images.append(path)
    
    if not valid_images:
        print("❌ No new images to add!")
        return
    
    print(f"\n📁 Processing {len(valid_images)} new image(s)")
    print("\n🎮 Editor controls:")
    print("   Left drag   = Draw a new box")
    print("   Right click = Delete a box")
    print("   Z           = Undo")
    print("   S           = Save")
    print("   N / Space   = Accept and continue")
    print("   D           = Skip image without adding it")
    print("   Q / ESC     = Cancel\n")
    
    # Temporary label directory.
    tmp_dir = Path("_tmp_add_labels")
    tmp_dir.mkdir(exist_ok=True)
    
    added = 0
    skipped = 0
    
    try:
        idx = 0
        while idx < len(valid_images):
            img_path = valid_images[idx]
            print(f"\n── [{idx + 1}/{len(valid_images)}] {img_path.name} ──")
            
            # Detect boxes.
            boxes = detect_gaps(str(img_path), model, conf=YOLO_CONF)
            print(f"🔍 {len(boxes)} boxes detected")
            
            # Create a temporary label file.
            tmp_label = tmp_dir / f"{img_path.stem}.txt"
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"❌ Failed to load image: {img_path}")
                idx += 1
                skipped += 1
                continue
            
            img_h, img_w = image.shape[:2]
            yolo_lines = []
            for x1, y1, x2, y2 in boxes:
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            with open(str(tmp_label), 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            # Open the editor.
            editor = BoxEditor(
                image_path=img_path,
                label_path=tmp_label,
                yolo_model=model,
                yolo_conf=YOLO_CONF,
                image_index=idx,
                total_images=len(valid_images)
            )
            
            action = editor.run()
            
            if action == 'quit':
                print("⏹️  Cancelled")
                break
            elif action == 'delete':
                # Skip the image.
                print(f"⏭️  Skipped: {img_path.name}")
                skipped += 1
                idx += 1
            elif action == 'prev':
                idx = max(0, idx - 1)
            else:  # 'next'
                # Add the image to the dataset.
                split = choose_split(DATASET_DIR, TRAIN_SPLIT)
                
                # Read boxes from the potentially edited label file.
                final_boxes = []
                if tmp_label.exists():
                    with open(str(tmp_label), 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                xc = float(parts[1]) * img_w
                                yc = float(parts[2]) * img_h
                                w = float(parts[3]) * img_w
                                h = float(parts[4]) * img_h
                                final_boxes.append((
                                    int(xc - w/2), int(yc - h/2),
                                    int(xc + w/2), int(yc + h/2)
                                ))
                
                target_img, target_lbl = save_to_dataset(
                    img_path, tmp_label, final_boxes, DATASET_DIR, split, VISUALIZE
                )
                
                added += 1
                print(
                    f"✅ Added to {split}: {img_path.name} "
                    f"({len(final_boxes)} boxes)"
                )
                idx += 1
    
    finally:
        # Remove temporary files.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        cv2.destroyAllWindows()
    
    # Summary.
    counts_after = count_dataset_images(DATASET_DIR)
    print(f"\n{'='*50}")
    print("✅ Finished!")
    print(f"   Added: {added}")
    print(f"   Skipped: {skipped}")
    print(
        f"   Dataset now: {counts_after['train']} train / "
        f"{counts_after['val']} val"
    )
    total = counts_after['train'] + counts_after['val']
    if total > 0:
        ratio = counts_after['train'] / total * 100
        print(f"   Training share: {ratio:.0f}%")


def main():
    import sys
    
    if len(sys.argv) > 1:
        # Accept files or directories as command-line arguments.
        image_paths, source_count = resolve_input_paths(sys.argv[1:])
        if source_count == 0:
            print("❌ No valid inputs found!")
            return
    else:
        print("📷 Add images to the dataset")
        print("   Enter file or directory paths, one per line.")
        print("   Submit an empty line to finish; directories are detected automatically.\n")

        raw_inputs = []
        while True:
            raw_input = input("  > ").strip()
            if not raw_input:
                break
            raw_inputs.append(raw_input)

        if not raw_inputs:
            print("❌ No inputs provided!")
            return

        image_paths, source_count = resolve_input_paths(raw_inputs)
        if source_count == 0:
            print("❌ No valid inputs found!")
            return

        print(f"\n✓ Accepted {len(image_paths)} image(s) in total\n")

    if not image_paths:
        print("❌ No new images to add!")
        return
    
    add_images(image_paths)


if __name__ == "__main__":
    main()
