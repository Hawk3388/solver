from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

def calculate_iou(box1, box2):
    """
    Berechnet Intersection over Union (IoU) zwischen zwei Boxen
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
    Filtert überlappende Boxen - behält nur die mit höchster Confidence
    
    Args:
        boxes: YOLO boxes Objekt
        iou_threshold: Mindest-IoU für Überlappung (0.5 = 50%)
    
    Returns:
        Liste von Indizes der zu behaltenden Boxen
    """
    if len(boxes) == 0:
        return []
    
    # Extrahiere Koordinaten und Confidences
    coords = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    confidences = boxes.conf.cpu().numpy()
    
    # Nach Confidence sortieren (höchste zuerst)
    sorted_indices = np.argsort(-confidences)
    
    keep = []
    
    for i in sorted_indices:
        # Prüfe ob diese Box mit bereits behaltenen überlappt
        should_keep = True
        
        for kept_idx in keep:
            iou = calculate_iou(coords[i], coords[kept_idx])
            
            if iou > iou_threshold:
                # Überlappung gefunden - verwerfe diese Box (niedrigere Confidence)
                should_keep = False
                break
        
        if should_keep:
            keep.append(i)
    
    return sorted(keep)  # Zurück in ursprünglicher Reihenfolge

# Trainiertes Modell laden
# Bestes Model: 3
MODEL_PATH = 'gap_detection_model.pt'

# Prüfe ob trainiertes Modell existiert
if not Path(MODEL_PATH).exists():
    print(f"❌ Trainiertes Modell nicht gefunden: {MODEL_PATH}")
    print(f"💡 Führe zuerst train_yolo.py aus!")
    print(f"\nFalls vorhanden, ändere MODEL_PATH zur korrekten Position")
    exit()

print(f"✅ Lade trainiertes Modell: {MODEL_PATH}\n")
model = YOLO(MODEL_PATH)

# Bild zum Testen
IMAGE_PATH = 'test.jpg'
results = model.predict(source=IMAGE_PATH, save=True, conf=0.10)

# Ergebnisse durchgehen
for r in results:
    print(f"📸 Bild: {r.path}")
    print(f"⚡ Speed: {r.speed}")
    print(f"📦 Anzahl Detektionen (vor Filterung): {len(r.boxes)}")
    
    # Überlappende Boxen filtern
    if len(r.boxes) > 0:
        keep_indices = filter_overlapping_boxes(r.boxes, iou_threshold=0.5)
        print(f"🔍 Nach Überlappungs-Filterung: {len(keep_indices)} Boxen")
    else:
        keep_indices = []
    
    if len(keep_indices) == 0:
        print("\n❌ Keine freien Stellen erkannt!")
        print("💡 Überprüfe:")
        print("   - Ist das Bild ein Arbeitsblatt?")
        print("   - Wurde das Modell richtig trainiert?")
        print("   - Versuche niedrigere conf (z.B. 0.1)")
    else:
        print("\n✅ Gefundene freie Stellen (nach Filterung):")
        # Alle erkannten freien Stellen
        for i, idx in enumerate(keep_indices):
            box = r.boxes[idx]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            print(f"  {i+1}. {r.names[class_id]}")
            print(f"     Konfidenz: {confidence:.2%}")
            print(f"     Box: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")
            print(f"     Größe: {int(x2-x1)} x {int(y2-y1)} px")
    
    # Bild mit markierten freien Stellen anzeigen (nur gefilterte)
    print(f"\n🎨 Zeige Ergebnis...")
    
    # Manuell zeichnen mit nur den gefilterten Boxen
    img = r.orig_img.copy()
    for idx in keep_indices:
        box = r.boxes[idx]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        
        # Box zeichnen
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Label mit Confidence
        # label = f"{conf:.2%}"
        # cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    annotated = img
    
    # Speichern
    output_path = 'yolo_detected_gaps.png'
    cv2.imwrite(output_path, annotated)
    print(f"💾 Gespeichert: {output_path}")
    
    # Anzeigen
    cv2.imshow('YOLO - Freie Stellen Erkennung', annotated)
    print("👁️  Drücke eine Taste um zu schließen...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\n✅ Fertig!")