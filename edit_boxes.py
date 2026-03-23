"""
Interaktives Skript zum Bearbeiten von Bounding Boxes mit mehreren Klassen
Unterstützt mehrere Klassen: gap, lines, free_spaces

Steuerung:
  - Linksklick + Ziehen: Neue Box zeichnen
  - 0/1/2: Wähle Klasse (0=gap, 1=lines, 2=free_spaces)
  - Rechtsklick: Box löschen (die Box unter dem Mauszeiger)
  - 'z': Letzte Aktion rückgängig machen (Undo)
  - 's': Änderungen speichern
  - 'n' / Leertaste: Speichern & nächstes Bild
  - 'p': Vorheriges Bild
  - 'd': Ganzes Bild + Labels aus Dataset löschen
  - 'r': Alle gelöschten Boxen wiederherstellen (Reset)
  - 'q' / ESC: Beenden
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys


class BoxEditor:
    # Klassen-Definition
    CLASS_NAMES = {0: 'gap', 1: 'lines', 2: 'free_spaces'}
    CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 165, 255)}  # BGR
    NUM_CLASSES = 3
    
    def __init__(self, image_path, label_path, yolo_model=None, yolo_conf=0.25, image_index=0, total_images=1):
        """
        Args:
            image_path: Pfad zum Bild
            label_path: Pfad zur YOLO Label-Datei
            yolo_model: Bereits geladenes YOLO Modell (optional)
            yolo_conf: Confidence Threshold für YOLO
            image_index: Aktueller Index im Dataset
            total_images: Gesamtanzahl Bilder
        """
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)
        self.yolo_model = yolo_model
        self.yolo_conf = yolo_conf
        self.image_index = image_index
        self.total_images = total_images
        
        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            print(f"❌ Fehler beim Laden: {image_path}")
            return
        
        self.img_h, self.img_w = self.original_image.shape[:2]
        
        # Boxen laden: Liste von (x1, y1, x2, y2, class_id)
        self.boxes = []
        self.deleted_boxes = []
        self.undo_stack = []
        self.hover_idx = -1
        self.unsaved_changes = False
        self.current_class = 0  # Aktuelle Klasse zum Zeichnen
        
        # Zeichen-Modus
        self.drawing = False
        self.draw_start = None
        self.draw_current = None
        
        self._load_boxes()
    
    def _load_boxes(self):
        """Boxen aus Label-Datei oder YOLO Modell laden"""
        if self.label_path.exists():
            self._load_from_labels()
        elif self.yolo_model:
            self._load_from_yolo()
        else:
            print(f"⚠️  Keine Labels für: {self.image_path.name}")
    
    def _load_from_labels(self):
        """Boxen aus YOLO Label-Datei laden (mit Klasse-IDs)"""
        print(f"📄 Lade Labels aus: {self.label_path}")
        
        with open(self.label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO Format: class x_center y_center width height (normalisiert)
                class_id = int(parts[0])
                x_center = float(parts[1]) * self.img_w
                y_center = float(parts[2]) * self.img_h
                width = float(parts[3]) * self.img_w
                height = float(parts[4]) * self.img_h
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Klasse-ID speichern
                self.boxes.append((x1, y1, x2, y2, class_id))
        
        print(f"✅ {len(self.boxes)} Boxen geladen")
    
    def _load_from_yolo(self):
        """Boxen direkt mit YOLO Modell erkennen"""
        print(f"🤖 Erkenne Boxen mit YOLO...")
        results = self.yolo_model.predict(source=str(self.image_path), conf=self.yolo_conf, verbose=False)
        
        for r in results:
            if len(r.boxes) > 0:
                # Überlappende filtern
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
                    # YOLO erkannte Boxen als Klasse 0 (gap) speichern
                    self.boxes.append((int(x1), int(y1), int(x2), int(y2), 0))
        
        # Sortieren in Lesereihenfolge
        self.boxes.sort(key=lambda b: (b[1], b[0]))
        print(f"✅ {len(self.boxes)} Boxen erkannt")
    
    def _point_in_box(self, px, py, box):
        """Prüfe ob ein Punkt in einer Box liegt"""
        x1, y1, x2, y2 = box[:4]
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def _find_box_at(self, px, py):
        """Finde die Box unter dem Mauszeiger (kleinste zuerst)"""
        candidates = []
        for i, box in enumerate(self.boxes):
            if self._point_in_box(px, py, box):
                x1, y1, x2, y2 = box[:4]
                area = (x2 - x1) * (y2 - y1)
                candidates.append((area, i))
        
        if candidates:
            # Kleinste Box bevorzugen (genauere Auswahl)
            candidates.sort()
            return candidates[0][1]
        return -1
    
    def _draw(self):
        """Bild mit Boxen zeichnen"""
        display = self.original_image.copy()
        
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box[:4]
            class_id = box[4] if len(box) > 4 else 0
            color = self.CLASS_COLORS.get(class_id, (255, 255, 255))
            
            if i == self.hover_idx and not self.drawing:
                # Hervorgehobene Box (Maus darüber)
                thickness = 3
                # Halbtransparentes Overlay
                overlay = display.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)
            else:
                thickness = 2
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            # Label mit Klasse und Nummer
            class_name = self.CLASS_NAMES.get(class_id, 'unknown')
            label = f"{i+1}:{class_name}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(display, (x1, y1 - label_size[1] - 6), 
                         (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Zeichne aktuelle Zeichnung (Linksklick-Drag)
        if self.drawing and self.draw_start and self.draw_current:
            sx, sy = self.draw_start
            cx, cy = self.draw_current
            x1_d, y1_d = min(sx, cx), min(sy, cy)
            x2_d, y2_d = max(sx, cx), max(sy, cy)
            color = self.CLASS_COLORS.get(self.current_class, (255, 255, 255))
            cv2.rectangle(display, (x1_d, y1_d), (x2_d, y2_d), color, 2)
            # Größe und Klasse anzeigen
            w_d, h_d = x2_d - x1_d, y2_d - y1_d
            class_name = self.CLASS_NAMES.get(self.current_class, 'unknown')
            size_text = f"{w_d}x{h_d} ({class_name})"
            cv2.putText(display, size_text, (x1_d, y1_d - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Status-Leiste unten
        status_h = 50
        status_bar = np.zeros((status_h, display.shape[1], 3), dtype=np.uint8)
        status_bar[:] = (40, 40, 40)
        
        progress = f"[{self.image_index + 1}/{self.total_images}] {self.image_path.name}"
        current_class_name = self.CLASS_NAMES.get(self.current_class, 'unknown')
        info = f"{progress} | Boxen: {len(self.boxes)} | Klasse: {current_class_name} (0/1/2)"
        if self.unsaved_changes:
            info += " | *"
        if self.drawing:
            info += " | ZEICHNEN..."
        elif self.hover_idx >= 0:
            info += f" | Box {self.hover_idx + 1}"
        
        cv2.putText(status_bar, info, (10, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        controls = "L-Drag=Neu | R=weg | Z=Undo | S=Save | N/Space=Weiter | 0/1/2=Klasse | Q=Ende"
        cv2.putText(status_bar, controls, (display.shape[1] - 650, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        
        display = np.vstack([display, status_bar])
        return display
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Maus-Events verarbeiten"""
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.draw_current = (x, y)
            else:
                self.hover_idx = self._find_box_at(x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Linksklick: Zeichnen starten
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
                
                # Mindestgröße prüfen (mind. 5x5 Pixel)
                if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                    new_box = (x1, y1, x2, y2, self.current_class)
                    self.boxes.append(new_box)
                    self.boxes.sort(key=lambda b: (b[1], b[0]))
                    self.undo_stack.append(('add', new_box))
                    self.unsaved_changes = True
                    class_name = self.CLASS_NAMES.get(self.current_class, 'unknown')
                    print(f"➕ Neue Box ({class_name}): ({x1}, {y1}) -> ({x2}, {y2})")
                else:
                    print("⚠️  Box zu klein, verworfen")
                
                self.draw_start = None
                self.draw_current = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Rechtsklick: Box löschen
            idx = self._find_box_at(x, y)
            if idx >= 0:
                deleted_box = self.boxes.pop(idx)
                self.undo_stack.append(('delete', deleted_box))
                self.unsaved_changes = True
                self.hover_idx = -1
                print(f"🗑️  Box {idx + 1} gelöscht")
    
    def save_labels(self):
        """Aktuelle Boxen als YOLO Labels speichern (mit Klassen-IDs)"""
        yolo_lines = []
        for x1, y1, x2, y2, class_id in self.boxes:
            x_center = ((x1 + x2) / 2) / self.img_w
            y_center = ((y1 + y2) / 2) / self.img_h
            width = (x2 - x1) / self.img_w
            height = (y2 - y1) / self.img_h
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        with open(self.label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # Visualisierung updaten (mit Farben nach Klasse)
        viz_dir = self.image_path.parent.parent.parent / 'visualize' / self.image_path.parent.name
        if viz_dir.exists():
            viz_img = self.original_image.copy()
            for x1, y1, x2, y2, class_id in self.boxes:
                color = self.CLASS_COLORS.get(class_id, (255, 255, 255))
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
            viz_path = viz_dir / f"{self.image_path.stem}_marked{self.image_path.suffix}"
            cv2.imwrite(str(viz_path), viz_img)
        
        self.unsaved_changes = False
        print(f"💾 {len(self.boxes)} Boxen gespeichert in: {self.label_path}")
    
    def delete_image(self):
        """Bild und Label-Datei aus dem Dataset löschen"""
        deleted = []
        if self.image_path.exists():
            self.image_path.unlink()
            deleted.append(str(self.image_path))
        if self.label_path.exists():
            self.label_path.unlink()
            deleted.append(str(self.label_path))
        
        # Auch Visualisierung löschen falls vorhanden
        viz_path = self.image_path.parent.parent.parent / 'visualize' / self.image_path.parent.name / f"{self.image_path.stem}_marked{self.image_path.suffix}"
        if viz_path.exists():
            viz_path.unlink()
            deleted.append(str(viz_path))
        
        for f in deleted:
            print(f"  🗑️  {f}")
        print(f"❌ Bild gelöscht: {self.image_path.name}")
    
    def run(self):
        """
        Hauptschleife starten.
        Returns: 'next', 'prev', 'delete', oder 'quit'
        """
        if self.original_image is None:
            return 'next'
        
        window_name = "Box Editor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # Fenster auf sinnvolle Größe setzen
        scale = min(1400 / self.img_w, 900 / self.img_h, 1.0)
        cv2.resizeWindow(window_name, int(self.img_w * scale), int(self.img_h * scale + 40))
        
        result = 'next'
        
        while True:
            display = self._draw()
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # Q oder ESC -> Beenden
                if self.unsaved_changes:
                    self.save_labels()
                result = 'quit'
                break
            
            elif key == ord('n') or key == ord(' '):  # N oder Space -> Nächstes Bild
                if self.unsaved_changes:
                    self.save_labels()
                result = 'next'
                break
            
            elif key == ord('p'):  # P -> Vorheriges Bild
                if self.unsaved_changes:
                    self.save_labels()
                result = 'prev'
                break
            
            elif key == ord('d'):  # D -> Bild löschen
                # Bestätigung: Bild rot einfärben
                confirm_display = self.original_image.copy()
                overlay = np.zeros_like(confirm_display)
                overlay[:] = (0, 0, 255)
                confirm_display = cv2.addWeighted(confirm_display, 0.5, overlay, 0.5, 0)
                msg = "BILD LOESCHEN? D=Ja / andere Taste=Nein"
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
                    print("↩️  Löschen abgebrochen")
            
            elif key == ord('z'):  # Undo
                if self.undo_stack:
                    action_type, box = self.undo_stack.pop()
                    if action_type == 'delete':
                        # Gelöschte Box wiederherstellen
                        self.boxes.append(box)
                        self.boxes.sort(key=lambda b: (b[1], b[0]))
                        self.unsaved_changes = True
                        print(f"↩️  Box wiederhergestellt: {box}")
                    elif action_type == 'add':
                        # Hinzugefügte Box wieder entfernen
                        if box in self.boxes:
                            self.boxes.remove(box)
                            self.unsaved_changes = True
                            print(f"↩️  Hinzufügen rückgängig: {box}")
            
            elif key == ord('s'):  # Speichern
                self.save_labels()
            
            elif key == ord('r'):  # Reset
                while self.deleted_boxes:
                    self.boxes.append(self.deleted_boxes.pop())
                self.boxes.sort(key=lambda b: (b[1], b[0]))
                self.unsaved_changes = True
                print("🔄 Alle Boxen wiederhergestellt")
            
            elif key == ord('0'):  # Klasse 0: gap
                self.current_class = 0
                print(f"📍 Klasse gewechselt zu: {self.CLASS_NAMES[0]}")
            
            elif key == ord('1'):  # Klasse 1: lines
                self.current_class = 1
                print(f"📍 Klasse gewechselt zu: {self.CLASS_NAMES[1]}")
            
            elif key == ord('2'):  # Klasse 2: free_spaces
                self.current_class = 2
                print(f"📍 Klasse gewechselt zu: {self.CLASS_NAMES[2]}")
        
        return result


# ── Konfiguration ──────────────────────────────────────────
DATASET_DIR = "dataset"          # Dataset-Ordner
YOLO_MODEL = "gap_detection_model.pt"  # YOLO Modell als Fallback (wenn keine Labels da sind)
YOLO_CONF = 0.25
# ───────────────────────────────────────────────────────────


def collect_dataset_images(dataset_dir):
    """
    Sammelt alle Bilder aus dem Dataset (train + val) mit zugehörigen Label-Pfaden.
    Returns: Liste von (image_path, label_path) Tupeln
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
        print(f"❌ Keine Bilder gefunden in: {dataset_dir}/images/{{train,val}}/")
        return
    
    print(f"📁 {len(pairs)} Bilder im Dataset gefunden")
    print(f"\n🎮 Steuerung:")
    print(f"   L-Ziehen    = Neue Box zeichnen")
    print(f"   Rechtsklick = Box löschen")
    print(f"   Z           = Undo")
    print(f"   S           = Speichern")
    print(f"   N / Space   = Speichern & weiter")
    print(f"   P           = Zurück")
    print(f"   D           = Ganzes Bild löschen")
    print(f"   R           = Reset")
    print(f"   Q / ESC     = Beenden\n")
    
    # YOLO Modell laden als Fallback
    yolo_model = None
    model_path = Path(YOLO_MODEL)
    if model_path.exists():
        yolo_model = YOLO(str(model_path))
        print(f"🤖 YOLO Modell geladen: {YOLO_MODEL}")
    
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
            # Bild wurde gelöscht, aus Liste entfernen
            pairs.pop(idx)
            if idx >= len(pairs):
                idx = len(pairs) - 1
            if len(pairs) == 0:
                print("\n⚠️  Keine Bilder mehr im Dataset!")
                break
        elif action == 'prev':
            idx = max(0, idx - 1)
        else:  # 'next'
            idx += 1
    
    cv2.destroyAllWindows()
    print(f"\n✅ Fertig! {len(pairs)} Bilder bearbeitet.")


if __name__ == "__main__":
    main()
