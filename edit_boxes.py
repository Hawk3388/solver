"""
Interactive multi-class bounding-box editor.

Controls:
  - Left-click and drag: draw a new box
    - 0/1/2: select the drawing class
    - Hover over a box and press 0/1/2: change that box's class
  - Right-click: delete the box under the pointer
  - 'z': undo the last action
  - 's': save changes
  - 'n' / Space: save and open the next image
  - 'p': previous image
  - 'd': delete the current image and labels from the dataset
  - 'r': restore deleted boxes
  - 'q' / ESC: quit
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys


class BoxEditor:
    # Class definitions.
    CLASS_NAMES = {0: 'gap', 1: 'lines', 2: 'free_spaces'}
    CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 165, 255)}  # BGR
    NUM_CLASSES = 3
    
    def __init__(self, image_path, label_path, yolo_model=None, yolo_conf=0.25, image_index=0, total_images=1):
        """
        Args:
            image_path: Path to the image.
            label_path: Path to the YOLO label file.
            yolo_model: Optional preloaded YOLO model.
            yolo_conf: YOLO confidence threshold.
            image_index: Current dataset index.
            total_images: Total number of images.
        """
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)
        self.yolo_model = yolo_model
        self.yolo_conf = yolo_conf
        self.image_index = image_index
        self.total_images = total_images
        
        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            print(f"❌ Failed to load image: {image_path}")
            return
        
        self.img_h, self.img_w = self.original_image.shape[:2]
        
        # Boxes are stored as (x1, y1, x2, y2, class_id).
        self.boxes = []
        self.deleted_boxes = []
        self.undo_stack = []
        self.hover_idx = -1
        self.unsaved_changes = False
        self.current_class = 0  # Active drawing class.
        
        # Drawing state.
        self.drawing = False
        self.draw_start = None
        self.draw_current = None
        
        self._load_boxes()
    
    def _load_boxes(self):
        """Load boxes from labels or the fallback YOLO model."""
        if self.label_path.exists():
            self._load_from_labels()
        elif self.yolo_model:
            self._load_from_yolo()
        else:
            print(f"⚠️  No labels available for: {self.image_path.name}")
    
    def _load_from_labels(self):
        """Load boxes and class IDs from a YOLO label file."""
        print(f"📄 Loading labels from: {self.label_path}")
        
        with open(self.label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height, normalized.
                class_id = int(parts[0])
                x_center = float(parts[1]) * self.img_w
                y_center = float(parts[2]) * self.img_h
                width = float(parts[3]) * self.img_w
                height = float(parts[4]) * self.img_h
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Retain the class ID.
                self.boxes.append((x1, y1, x2, y2, class_id))
        
        print(f"✅ {len(self.boxes)} boxes loaded")
    
    def _load_from_yolo(self):
        """Detect boxes directly with the YOLO model."""
        print("🤖 Detecting boxes with YOLO...")
        results = self.yolo_model.predict(source=str(self.image_path), conf=self.yolo_conf, verbose=False)
        
        for r in results:
            if len(r.boxes) > 0:
                # Filter overlapping detections.
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
                            a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
                            a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
                            if inter / (a1 + a2 - inter) > 0.5:
                                should_keep = False
                                break
                    if should_keep:
                        keep.append(i)
                
                for idx in keep:
                    box = r.boxes[idx]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    # Store YOLO detections as class 0 (gap).
                    self.boxes.append((int(x1), int(y1), int(x2), int(y2), 0))
        
        # Sort in reading order.
        self.boxes.sort(key=lambda b: (b[1], b[0]))
        print(f"✅ {len(self.boxes)} boxes detected")
    
    def _point_in_box(self, px, py, box):
        """Return whether a point lies inside a box."""
        x1, y1, x2, y2 = box[:4]
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def _find_box_at(self, px, py):
        """Find the smallest box under the pointer."""
        candidates = []
        for i, box in enumerate(self.boxes):
            if self._point_in_box(px, py, box):
                x1, y1, x2, y2 = box[:4]
                area = (x2 - x1) * (y2 - y1)
                candidates.append((area, i))
        
        if candidates:
            # Prefer the smallest box for more precise selection.
            candidates.sort()
            return candidates[0][1]
        return -1
    
    def _draw(self):
        """Draw boxes and editor status onto the image."""
        display = self.original_image.copy()
        class_keys = '/'.join(str(k) for k in sorted(self.CLASS_NAMES.keys()))
        
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box[:4]
            class_id = box[4] if len(box) > 4 else 0
            color = self.CLASS_COLORS.get(class_id, (255, 255, 255))
            
            if i == self.hover_idx and not self.drawing:
                # Highlight the box under the pointer.
                thickness = 3
                # Semi-transparent overlay.
                overlay = display.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)
            else:
                thickness = 2
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            # Label with class and box number.
            class_name = self.CLASS_NAMES.get(class_id, 'unknown')
            label = f"{i+1}:{class_name}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(display, (x1, y1 - label_size[1] - 6), 
                         (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw the current left-drag selection.
        if self.drawing and self.draw_start and self.draw_current:
            sx, sy = self.draw_start
            cx, cy = self.draw_current
            x1_d, y1_d = min(sx, cx), min(sy, cy)
            x2_d, y2_d = max(sx, cx), max(sy, cy)
            color = self.CLASS_COLORS.get(self.current_class, (255, 255, 255))
            cv2.rectangle(display, (x1_d, y1_d), (x2_d, y2_d), color, 2)
            # Display dimensions and class.
            w_d, h_d = x2_d - x1_d, y2_d - y1_d
            class_name = self.CLASS_NAMES.get(self.current_class, 'unknown')
            size_text = f"{w_d}x{h_d} ({class_name})"
            cv2.putText(display, size_text, (x1_d, y1_d - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Bottom status bar.
        status_h = 50
        status_bar = np.zeros((status_h, display.shape[1], 3), dtype=np.uint8)
        status_bar[:] = (40, 40, 40)
        
        progress = f"[{self.image_index + 1}/{self.total_images}] {self.image_path.name}"
        current_class_name = self.CLASS_NAMES.get(self.current_class, 'unknown')
        info = (
            f"{progress} | Boxes: {len(self.boxes)} | "
            f"Class: {current_class_name} ({class_keys})"
        )
        if self.unsaved_changes:
            info += " | *"
        if self.drawing:
            info += " | DRAWING..."
        elif self.hover_idx >= 0:
            hover_class = self.boxes[self.hover_idx][4]
            hover_name = self.CLASS_NAMES.get(hover_class, 'unknown')
            info += f" | Box {self.hover_idx + 1}: {hover_name}"
        
        cv2.putText(status_bar, info, (10, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        controls = (
            "L-Drag=New | R=Delete | Z=Undo | S=Save | "
            "N=Next | Number=Class | Q=Quit"
        )
        cv2.putText(status_bar, controls, (display.shape[1] - 650, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        
        display = np.vstack([display, status_bar])
        return display
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Process mouse events."""
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.draw_current = (x, y)
            else:
                self.hover_idx = self._find_box_at(x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Start a left-button drawing operation.
            self.drawing = True
            self.draw_start = (x, y)
            self.draw_current = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.draw_start:
                self.drawing = False
                sx, sy = self.draw_start
                x1 = min(sx, x)
                y1 = min(sy, y)
                x2 = max(sx, x)
                y2 = max(sy, y)
                
                # Require a minimum size of 5x5 pixels.
                if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                    new_box = (x1, y1, x2, y2, self.current_class)
                    self.boxes.append(new_box)
                    self.boxes.sort(key=lambda b: (b[1], b[0]))
                    self.undo_stack.append(('add', new_box))
                    self.unsaved_changes = True
                    class_name = self.CLASS_NAMES.get(self.current_class, 'unknown')
                    print(
                        f"➕ New box ({class_name}): "
                        f"({x1}, {y1}) -> ({x2}, {y2})"
                    )
                else:
                    print("⚠️  Box is too small and was discarded")
                
                self.draw_start = None
                self.draw_current = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete the box under a right-click.
            idx = self._find_box_at(x, y)
            if idx >= 0:
                deleted_box = self.boxes.pop(idx)
                self.undo_stack.append(('delete', deleted_box))
                self.unsaved_changes = True
                self.hover_idx = -1
                print(f"🗑️  Box {idx + 1} deleted")

    def _set_hovered_box_class(self, class_id):
        """Change the class of the box under the pointer."""
        if self.hover_idx < 0 or self.hover_idx >= len(self.boxes):
            return False

        x1, y1, x2, y2, old_class = self.boxes[self.hover_idx]
        if old_class == class_id:
            return False

        self.boxes[self.hover_idx] = (x1, y1, x2, y2, class_id)
        self.undo_stack.append(('class_change', self.hover_idx, old_class, class_id))
        self.unsaved_changes = True

        old_name = self.CLASS_NAMES.get(old_class, str(old_class))
        new_name = self.CLASS_NAMES.get(class_id, str(class_id))
        print(f"🏷️  Box {self.hover_idx + 1}: {old_name} -> {new_name}")
        return True
    
    def save_labels(self):
        """Save the current boxes and class IDs as YOLO labels."""
        yolo_lines = []
        for x1, y1, x2, y2, class_id in self.boxes:
            x_center = ((x1 + x2) / 2) / self.img_w
            y_center = ((y1 + y2) / 2) / self.img_h
            width = (x2 - x1) / self.img_w
            height = (y2 - y1) / self.img_h
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        with open(self.label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # Update the class-coloured visualization when it exists.
        viz_dir = self.image_path.parent.parent.parent / 'visualize' / self.image_path.parent.name
        if viz_dir.exists():
            viz_img = self.original_image.copy()
            for x1, y1, x2, y2, class_id in self.boxes:
                color = self.CLASS_COLORS.get(class_id, (255, 255, 255))
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
            viz_path = viz_dir / f"{self.image_path.stem}_marked{self.image_path.suffix}"
            cv2.imwrite(str(viz_path), viz_img)
        
        self.unsaved_changes = False
        print(f"💾 {len(self.boxes)} boxes saved to: {self.label_path}")
    
    def delete_image(self):
        """Delete the image and label file from the dataset."""
        deleted = []
        if self.image_path.exists():
            self.image_path.unlink()
            deleted.append(str(self.image_path))
        if self.label_path.exists():
            self.label_path.unlink()
            deleted.append(str(self.label_path))
        
        # Delete the visualization when present.
        viz_path = self.image_path.parent.parent.parent / 'visualize' / self.image_path.parent.name / f"{self.image_path.stem}_marked{self.image_path.suffix}"
        if viz_path.exists():
            viz_path.unlink()
            deleted.append(str(viz_path))
        
        for f in deleted:
            print(f"  🗑️  {f}")
        print(f"❌ Image deleted: {self.image_path.name}")
    
    def run(self):
        """
        Run the main editor loop.

        Returns: 'next', 'prev', 'delete', or 'quit'.
        """
        if self.original_image is None:
            return 'next'
        
        window_name = "Box Editor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # Fit the window to a practical screen size.
        scale = min(1400 / self.img_w, 900 / self.img_h, 1.0)
        cv2.resizeWindow(window_name, int(self.img_w * scale), int(self.img_h * scale + 40))
        
        result = 'next'
        
        while True:
            display = self._draw()
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC: quit.
                if self.unsaved_changes:
                    self.save_labels()
                result = 'quit'
                break
            
            elif key == ord('n') or key == ord(' '):  # N or Space: next.
                if self.unsaved_changes:
                    self.save_labels()
                result = 'next'
                break
            
            elif key == ord('p'):  # P: previous image.
                if self.unsaved_changes:
                    self.save_labels()
                result = 'prev'
                break
            
            elif key == ord('d'):  # D: delete image.
                # Show a red confirmation overlay.
                confirm_display = self.original_image.copy()
                overlay = np.zeros_like(confirm_display)
                overlay[:] = (0, 0, 255)
                confirm_display = cv2.addWeighted(confirm_display, 0.5, overlay, 0.5, 0)
                msg = "DELETE IMAGE? D=YES / ANY OTHER KEY=NO"
                text_size, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                tx = (confirm_display.shape[1] - text_size[0]) // 2
                ty = (confirm_display.shape[0] + text_size[1]) // 2
                cv2.putText(confirm_display, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.imshow(window_name, confirm_display)
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord('d'):
                    self.delete_image()
                    result = 'delete'
                    break
                else:
                    print("↩️  Deletion cancelled")
            
            elif key == ord('z'):  # Undo
                if self.undo_stack:
                    action = self.undo_stack.pop()
                    action_type = action[0]
                    if action_type == 'delete':
                        box = action[1]
                        # Restore a deleted box.
                        self.boxes.append(box)
                        self.boxes.sort(key=lambda b: (b[1], b[0]))
                        self.unsaved_changes = True
                        print(f"↩️  Box restored: {box}")
                    elif action_type == 'add':
                        box = action[1]
                        # Remove a newly added box.
                        if box in self.boxes:
                            self.boxes.remove(box)
                            self.unsaved_changes = True
                            print(f"↩️  Addition undone: {box}")
                    elif action_type == 'class_change':
                        idx, old_class, _new_class = action[1], action[2], action[3]
                        if 0 <= idx < len(self.boxes):
                            x1, y1, x2, y2, _ = self.boxes[idx]
                            self.boxes[idx] = (x1, y1, x2, y2, old_class)
                            self.unsaved_changes = True
                            class_name = self.CLASS_NAMES.get(old_class, str(old_class))
                            print(
                                f"↩️  Class change undone: "
                                f"Box {idx + 1} -> {class_name}"
                            )
            
            elif key == ord('s'):  # Save.
                self.save_labels()
            
            elif key == ord('r'):  # Reset
                while self.deleted_boxes:
                    self.boxes.append(self.deleted_boxes.pop())
                self.boxes.sort(key=lambda b: (b[1], b[0]))
                self.unsaved_changes = True
                print("🔄 All boxes restored")
            
            elif ord('0') <= key <= ord('9'):
                selected_class = key - ord('0')
                if selected_class in self.CLASS_NAMES:
                    # Change the hovered box, or select the drawing class.
                    changed = self._set_hovered_box_class(selected_class)
                    if not changed:
                        self.current_class = selected_class
                        print(
                            f"📍 Drawing class changed to: "
                            f"{self.CLASS_NAMES[selected_class]}"
                        )
        
        return result


# -- Configuration -----------------------------------------------------------
DATASET_DIR = "dataset"          # Dataset directory.
YOLO_MODEL = "gap_detection_model.pt"  # Fallback when labels do not exist.
YOLO_CONF = 0.25
# ---------------------------------------------------------------------------


def collect_dataset_images(dataset_dir):
    """
    Collect training and validation images with their label paths.

    Returns: A list of (image_path, label_path) tuples.
    """
    dataset_path = Path(dataset_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    pairs = []
    
    for split in ['train', 'val']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if not images_dir.exists():
            continue
        
        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() in image_extensions:
                label_file = labels_dir / f"{img_file.stem}.txt"
                pairs.append((img_file, label_file))
    
    return pairs


def main():
    dataset_dir = DATASET_DIR
    
    pairs = collect_dataset_images(dataset_dir)
    
    if not pairs:
        print(f"❌ No images found in: {dataset_dir}/images/{{train,val}}/")
        return
    
    print(f"📁 Found {len(pairs)} images in the dataset")
    print("\n🎮 Controls:")
    print("   Left drag   = Draw a new box")
    print("   Number      = Select class; hover to change a box")
    print("   Right click = Delete a box")
    print("   Z           = Undo")
    print("   S           = Save")
    print("   N / Space   = Save and continue")
    print("   P           = Previous image")
    print("   D           = Delete the entire image")
    print("   R           = Reset")
    print("   Q / ESC     = Quit\n")
    
    # Load the fallback YOLO model.
    yolo_model = None
    model_path = Path(YOLO_MODEL)
    if model_path.exists():
        yolo_model = YOLO(str(model_path))
        print(f"🤖 YOLO model loaded: {YOLO_MODEL}")
    
    idx = 0
    while 0 <= idx < len(pairs):
        img_path, label_path = pairs[idx]
        print(f"\n── [{idx + 1}/{len(pairs)}] {img_path.name} ──")
        
        editor = BoxEditor(
            image_path=img_path,
            label_path=label_path,
            yolo_model=yolo_model,
            yolo_conf=YOLO_CONF,
            image_index=idx,
            total_images=len(pairs)
        )
        
        action = editor.run()
        
        if action == 'quit':
            break
        elif action == 'delete':
            # Remove the deleted image from the active list.
            pairs.pop(idx)
            if idx >= len(pairs):
                idx = len(pairs) - 1
            if len(pairs) == 0:
                print("\n⚠️  No images remain in the dataset!")
                break
        elif action == 'prev':
            idx = max(0, idx - 1)
        else:  # 'next'
            idx += 1
    
    cv2.destroyAllWindows()
    print(f"\n✅ Finished! Processed {len(pairs)} images.")


if __name__ == "__main__":
    main()
