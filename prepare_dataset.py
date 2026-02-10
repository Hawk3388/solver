"""
Dataset Vorbereitung für YOLO Training
Verarbeitet Bilder aus einem Ordner und erstellt YOLO-Labels mittels gap detection
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

def find_gaps_in_image(image_path):
    """
    Findet freie Stellen in einem Arbeitsblatt
    Basiert auf simple_boxes.py Methode
    
    Returns: Liste von (x, y, w, h) Tupeln
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Fehler beim Laden: {image_path}")
        return [], None
    
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gaps = []
    
    # Schritt 1: Horizontale Linien finden (Unterstriche)
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
    
    # Schritt 2: Freie rechteckige Bereiche finden
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
    
    # Duplikate entfernen und konsolidieren
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


def boxes_to_yolo_format(boxes, image_width, image_height):
    """
    Konvertiert Bounding Boxes zu YOLO Format
    YOLO Format: class_id x_center y_center width height (alles normalisiert 0-1)
    
    Args:
        boxes: Liste von (x, y, w, h) Tupeln
        image_width, image_height: Bildabmessungen
        
    Returns:
        Liste von YOLO Label Strings
    """
    yolo_labels = []
    
    for x, y, w, h in boxes:
        # Koordinaten normalisieren (0-1)
        x_center = ((x + w / 2) / image_width)
        y_center = ((y + h / 2) / image_height)
        width_norm = w / image_width
        height_norm = h / image_height
        
        # YOLO Format: class_id x_center y_center width height
        # class_id = 0 (nur eine Klasse: freie_stelle)
        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
    
    return yolo_labels


def prepare_yolo_dataset(source_dir, output_dir, train_split=0.8, visualize=False):
    """
    Bereitet Dataset für YOLO Training vor
    
    Args:
        source_dir: Ordner mit Arbeitsblatt-Bildern
        output_dir: Zielordner für YOLO Dataset
        train_split: Anteil für Training (Rest für Validation)
        visualize: Wenn True, erstellt markierte Bilder zur Kontrolle
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Überprüfe ob Source-Ordner existiert
    if not source_path.exists():
        print(f"❌ Ordner nicht gefunden: {source_dir}")
        return
    
    # Dataset Struktur erstellen
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
    
    # Alle Bilder finden
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(list(source_path.glob(f'*{ext}')))
        images.extend(list(source_path.glob(f'*{ext.upper()}')))
    
    if len(images) == 0:
        print(f"❌ Keine Bilder gefunden in: {source_dir}")
        return
    
    print(f"📁 {len(images)} Bilder gefunden")
    print(f"🔄 Verarbeite Bilder...")
    
    # Statistiken
    stats = {
        'total_images': 0,
        'total_gaps': 0,
        'train_images': 0,
        'val_images': 0,
        'skipped': 0
    }
    
    # Bilder verarbeiten
    for idx, img_path in enumerate(tqdm(images, desc="Verarbeite")):
        # Gaps finden
        gaps, (img_w, img_h) = find_gaps_in_image(img_path)
        
        if gaps is None or len(gaps) == 0:
            stats['skipped'] += 1
            print(f"⚠️  Keine Lücken gefunden in: {img_path.name}")
            continue
        
        # Train/Val Split
        is_train = idx < int(len(images) * train_split)
        split = 'train' if is_train else 'val'
        
        # Dateinamen (ohne Leerzeichen)
        safe_name = img_path.stem.replace(' ', '_')
        safe_extension = img_path.suffix
        
        # Bild kopieren
        target_image = output_path / 'images' / split / f"{safe_name}{safe_extension}"
        shutil.copy(img_path, target_image)
        
        # YOLO Labels erstellen
        yolo_labels = boxes_to_yolo_format(gaps, img_w, img_h)
        label_file = output_path / 'labels' / split / f"{safe_name}.txt"
        
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        # Optional: Visualisierung erstellen
        if visualize:
            img = cv2.imread(str(img_path))
            for x, y, w, h in gaps:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            viz_path = output_path / 'visualize' / split / f"{safe_name}_marked{safe_extension}"
            cv2.imwrite(str(viz_path), img)
        
        # Statistiken
        stats['total_images'] += 1
        stats['total_gaps'] += len(gaps)
        if is_train:
            stats['train_images'] += 1
        else:
            stats['val_images'] += 1
    
    # data.yaml erstellen
    yaml_content = f"""# Arbeitsblatt Freie Stellen Dataset
path: {output_path.absolute().as_posix()}
train: images/train
val: images/val

# Klassen
nc: 1
names: ['freie_stelle']
"""
    
    with open(output_path / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    # Zusammenfassung
    print(f"\n✅ Dataset Vorbereitung abgeschlossen!")
    print(f"📊 Statistiken:")
    print(f"   Gesamt Bilder: {stats['total_images']}")
    print(f"   Training: {stats['train_images']}")
    print(f"   Validation: {stats['val_images']}")
    print(f"   Freie Stellen gesamt: {stats['total_gaps']}")
    print(f"   Durchschnitt pro Bild: {stats['total_gaps']/stats['total_images']:.1f}")
    print(f"   Übersprungen: {stats['skipped']}")
    print(f"\n📁 Dataset gespeichert in: {output_path}")
    print(f"📄 Config-Datei: {output_path / 'data.yaml'}")
    
    if visualize:
        print(f"🎨 Visualisierungen in: {output_path / 'visualize'}")


if __name__ == "__main__":
    # Konfiguration
    SOURCE_DIR = "raw_images"  # Ordner mit deinen Arbeitsblatt-Bildern
    OUTPUT_DIR = "dataset"      # Zielordner für YOLO Dataset
    TRAIN_SPLIT = 0.8          # 80% Training, 20% Validation
    VISUALIZE = True           # Erstelle markierte Bilder zur Kontrolle
    
    print("🚀 YOLO Dataset Vorbereitung")
    print(f"📂 Quellordner: {SOURCE_DIR}")
    print(f"📂 Zielordner: {OUTPUT_DIR}")
    print(f"📊 Train/Val Split: {TRAIN_SPLIT*100:.0f}% / {(1-TRAIN_SPLIT)*100:.0f}%")
    print(f"🎨 Visualisierung: {'Ja' if VISUALIZE else 'Nein'}")
    print("-" * 60)
    
    prepare_yolo_dataset(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        train_split=TRAIN_SPLIT,
        visualize=VISUALIZE
    )
