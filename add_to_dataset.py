"""
Einzelne Bilder zum bestehenden Dataset hinzufügen und direkt im Editor überprüfen.
Erkennt Boxen mit dem YOLO Modell, öffnet den Box-Editor zum Korrigieren,
und fügt das Bild dann ins Dataset ein (80/20 Split wird beibehalten).

Verwendung:
    python add_to_dataset.py bild1.png bild2.jpg ...
    python add_to_dataset.py              # fragt interaktiv nach Bildern
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
from edit_boxes import BoxEditor

# ── Konfiguration ──────────────────────────────────────────
DATASET_DIR = "dataset"
YOLO_MODEL = "gap_detection_model.pt"
YOLO_CONF = 0.25
TRAIN_SPLIT = 0.8  # 80% train, 20% val
VISUALIZE = True
# ───────────────────────────────────────────────────────────


def count_dataset_images(dataset_dir):
    """Zählt aktuelle Bilder im Dataset pro Split"""
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
    Wählt train oder val basierend auf dem aktuellen Verhältnis.
    Füllt den Split auf, der am weitesten vom Soll-Verhältnis entfernt ist.
    """
    counts = count_dataset_images(dataset_dir)
    total = counts['train'] + counts['val']
    
    if total == 0:
        return 'train'
    
    current_train_ratio = counts['train'] / (total + 1)  # +1 für das neue Bild
    
    # Wenn zu wenig val-Bilder -> val, sonst train
    if current_train_ratio >= train_split:
        return 'val'
    else:
        return 'train'


def detect_gaps(image_path, model, conf=0.25):
    """Erkennt Lücken mit YOLO und gibt (x1, y1, x2, y2) Boxen zurück"""
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
    """Speichert Bild + Labels ins Dataset"""
    dataset_path = Path(dataset_dir)
    img_path = Path(image_path)
    
    safe_name = img_path.stem.replace(' ', '_')
    safe_ext = img_path.suffix
    
    # Ordner sicherstellen
    for folder in ['images', 'labels']:
        (dataset_path / folder / split).mkdir(parents=True, exist_ok=True)
    
    # Bild kopieren
    target_image = dataset_path / 'images' / split / f"{safe_name}{safe_ext}"
    shutil.copy(str(img_path), str(target_image))
    
    # Labels aus der temporären Datei kopieren (wurde vom Editor gespeichert)
    target_label = dataset_path / 'labels' / split / f"{safe_name}.txt"
    if Path(label_path).exists():
        shutil.copy(str(label_path), str(target_label))
    else:
        # Labels aus Boxen generieren
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
    
    # Visualisierung
    if visualize:
        viz_dir = dataset_path / 'visualize' / split
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(str(img_path))
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        viz_path = viz_dir / f"{safe_name}_marked{safe_ext}"
        cv2.imwrite(str(viz_path), img)
    
    return target_image, target_label


def add_images(image_paths):
    """Hauptfunktion: Bilder zum Dataset hinzufügen"""
    dataset_path = Path(DATASET_DIR)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    if not dataset_path.exists():
        print(f"❌ Dataset nicht gefunden: {DATASET_DIR}")
        print(f"💡 Führe zuerst prepare_dataset.py aus!")
        return
    
    # YOLO Modell laden
    model_path = Path(YOLO_MODEL)
    if not model_path.exists():
        print(f"❌ YOLO Modell nicht gefunden: {YOLO_MODEL}")
        return
    
    model = YOLO(str(model_path))
    print(f"🤖 YOLO Modell geladen: {YOLO_MODEL}")
    
    # Aktuelle Dataset-Statistik
    counts = count_dataset_images(DATASET_DIR)
    print(f"📊 Aktuelles Dataset: {counts['train']} train / {counts['val']} val")
    
    # Bilder filtern
    valid_images = []
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            print(f"⚠️  Nicht gefunden: {p}")
        elif path.suffix.lower() not in image_extensions:
            print(f"⚠️  Kein unterstütztes Bildformat: {p}")
        else:
            # Prüfen ob schon im Dataset
            safe_name = path.stem.replace(' ', '_')
            already_exists = False
            for split in ['train', 'val']:
                for ext in image_extensions:
                    if (dataset_path / 'images' / split / f"{safe_name}{ext}").exists():
                        already_exists = True
                        break
            if already_exists:
                print(f"⚠️  Bereits im Dataset: {path.name}")
            else:
                valid_images.append(path)
    
    if not valid_images:
        print("❌ Keine neuen Bilder zum Hinzufügen!")
        return
    
    print(f"\n📁 {len(valid_images)} neue Bilder werden verarbeitet")
    print(f"\n🎮 Steuerung im Editor:")
    print(f"   L-Ziehen    = Neue Box zeichnen")
    print(f"   Rechtsklick = Box löschen")
    print(f"   Z           = Undo")
    print(f"   S           = Speichern")
    print(f"   N / Space   = Übernehmen & weiter")
    print(f"   D           = Bild überspringen (nicht hinzufügen)")
    print(f"   Q / ESC     = Abbrechen\n")
    
    # Temporärer Ordner für Labels
    tmp_dir = Path("_tmp_add_labels")
    tmp_dir.mkdir(exist_ok=True)
    
    added = 0
    skipped = 0
    
    try:
        idx = 0
        while idx < len(valid_images):
            img_path = valid_images[idx]
            print(f"\n── [{idx + 1}/{len(valid_images)}] {img_path.name} ──")
            
            # Boxen erkennen
            boxes = detect_gaps(str(img_path), model, conf=YOLO_CONF)
            print(f"🔍 {len(boxes)} Boxen erkannt")
            
            # Temporäre Label-Datei erstellen
            tmp_label = tmp_dir / f"{img_path.stem}.txt"
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"❌ Fehler beim Laden: {img_path}")
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
            
            # Editor öffnen
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
                print("⏹️  Abgebrochen")
                break
            elif action == 'delete':
                # Bild überspringen
                print(f"⏭️  Übersprungen: {img_path.name}")
                skipped += 1
                idx += 1
            elif action == 'prev':
                idx = max(0, idx - 1)
            else:  # 'next'
                # Ins Dataset einfügen
                split = choose_split(DATASET_DIR, TRAIN_SPLIT)
                
                # Boxen aus der (möglicherweise editierten) Label-Datei lesen
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
                print(f"✅ Hinzugefügt zu {split}: {img_path.name} ({len(final_boxes)} Boxen)")
                idx += 1
    
    finally:
        # Temporäre Dateien aufräumen
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        cv2.destroyAllWindows()
    
    # Zusammenfassung
    counts_after = count_dataset_images(DATASET_DIR)
    print(f"\n{'='*50}")
    print(f"✅ Fertig!")
    print(f"   Hinzugefügt: {added}")
    print(f"   Übersprungen: {skipped}")
    print(f"   Dataset jetzt: {counts_after['train']} train / {counts_after['val']} val")
    total = counts_after['train'] + counts_after['val']
    if total > 0:
        ratio = counts_after['train'] / total * 100
        print(f"   Train-Anteil: {ratio:.0f}%")


def main():
    import sys
    
    if len(sys.argv) > 1:
        # Bilder als Argumente
        image_paths = sys.argv[1:]
    else:
        # Interaktiv nach Bildern fragen
        print("📷 Bilder zum Dataset hinzufügen")
        print("   Gib Bildpfade ein (einer pro Zeile, leere Zeile = fertig):")
        print("   Oder ziehe Dateien hierher.\n")
        
        image_paths = []
        while True:
            path = input("  > ").strip().strip('"').strip("'")
            if not path:
                break
            image_paths.append(path)
        
        if not image_paths:
            print("❌ Keine Bilder angegeben!")
            return
    
    add_images(image_paths)


if __name__ == "__main__":
    main()
