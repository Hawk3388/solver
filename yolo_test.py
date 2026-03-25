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


def group_gaps_by_proximity(gaps):
    """
    Gruppiert Boxen, die direkt untereinander liegen.
    
    Args:
        gaps: Liste von Gap-Boxen als Tuples (x1, y1, x2, y2)
    
    Returns:
        groups: Liste von Gruppen, wobei jede Gruppe eine Liste von Gap-Indizes (in Original-Reihenfolge) ist
        gap_to_group: Mapping von Gap-Index zu Gruppen-Index
    """
    if not gaps:
        return [], {}
    
    # Erstelle Index-Mapping: sorted_idx -> original_idx
    indices = list(range(len(gaps)))
    sorted_indices = sorted(indices, key=lambda i: gaps[i][1])  # Sortiere nach Y (oben nach unten)
    
    # Berechne durchschnittliche Gap-Höhe als Schwellenwert
    heights = [(gap[3] - gap[1]) for gap in gaps]
    avg_height = sum(heights) / len(heights) if heights else 0
    
    # Abstands-Schwelle: Gaps sind "untereinander", wenn Abstand < durchschnittliche Höhe * 1.5
    distance_threshold = avg_height * 1.5
    
    groups = []
    gap_to_group = {}
    grouped = set()
    
    # Verarbeite Gaps von oben nach unten
    for sort_i, i in enumerate(sorted_indices):
        if i in grouped:
            continue
        
        # Starte neue Gruppe mit aktuellem Gap
        current_group = [i]
        grouped.add(i)
        gap_i = gaps[i]
        
        # Suche nach Gaps unterhalb dieses Gaps
        for sort_j in range(sort_i + 1, len(sorted_indices)):
            j = sorted_indices[sort_j]
            
            if j in grouped:
                continue
            
            gap_j = gaps[j]
            
            # Prüfe vertikalen Abstand (Gap j sollte unter Gap i sein)
            vertical_distance = gap_j[1] - gap_i[3]
            
            # Prüfe horizontale Ausrichtung
            i_left, i_top, i_right, i_bottom = gap_i
            j_left, j_top, j_right, j_bottom = gap_j
            
            # Berechne horizontale Überlappung
            h_overlap_start = max(i_left, j_left)
            h_overlap_end = min(i_right, j_right)
            h_overlap = max(0, h_overlap_end - h_overlap_start)
            
            # Breiten der Boxen
            i_width = i_right - i_left
            j_width = j_right - j_left
            min_width = min(i_width, j_width)
            
            # Prüfe ob Box j unter Box i liegt und horizontal ausgerichtet ist
            if 0 < vertical_distance < distance_threshold:
                # Mindestens 30% Überlappung oder visuell nebeneinander
                if h_overlap > min_width * 0.3 or h_overlap > 15:  # 15px min overlap
                    current_group.append(j)
                    grouped.add(j)
                    gap_i = gap_j  # Update für nächste Iteration
                else:
                    # Wenn nicht genug Überlappung, beende diese Gruppe
                    break
            else:
                # Wenn Abstand zu groß, beende diese Gruppe
                break
        
        # Speichere Gruppe (sortiere Indizes in Reihenfolge der Rückkehr)
        current_group.sort()
        for idx in current_group:
            gap_to_group[idx] = len(groups)
        
        groups.append(current_group)
    
    return groups, gap_to_group

# Trainiertes Modell laden
# Bestes Model: 3
MODEL_PATH = './model/gap_detection_model.pt'

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
results = model.predict(source=IMAGE_PATH, save=True, conf=0.25)

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
        # Extrahiere Gap-Boxen aus den gefilterten Indizes
        gaps = []
        gap_info = []  # Speichert Box-Info für spätere Referenz
        
        for idx in keep_indices:
            box = r.boxes[idx]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            gaps.append((int(x1), int(y1), int(x2), int(y2)))
            gap_info.append({
                'box': box,
                'class_id': int(box.cls[0]),
                'confidence': float(box.conf[0])
            })
        
        # Gruppiere Gaps nach Nähe
        groups, gap_to_group = group_gaps_by_proximity(gaps)
        
        print(f"\n✅ Gefundene freie Stellen (nach Filterung): {len(gaps)} Stellen")
        print(f"📊 Gruppiert in {len(groups)} Gruppen\n")
        
        # Zeige gruppierte freie Stellen
        for group_idx, group in enumerate(groups):
            print(f"📍 Gruppe {group_idx + 1}: {len(group)} Stelle(n)")
            
            for pos_in_group, gap_idx in enumerate(group):
                box = gap_info[gap_idx]
                x1, y1, x2, y2 = gaps[gap_idx]
                
                print(f"   Stelle {pos_in_group + 1}:")
                print(f"     Klasse: {r.names[box['class_id']]}")
                print(f"     Konfidenz: {box['confidence']:.2%}")
                print(f"     Box: ({x1}, {y1}) → ({x2}, {y2})")
                print(f"     Größe: {x2-x1} x {y2-y1} px")
    
    # Bild mit markierten freien Stellen anzeigen (nur gefilterte)
    print(f"\n🎨 Zeige Ergebnis...")
    
    if len(keep_indices) > 0:
        # Manuell zeichnen mit nur den gefilterten Boxen
        img = r.orig_img.copy()
        for gap_idx, gap in enumerate(gaps):
            x1, y1, x2, y2 = gap
            group_num = gap_to_group[gap_idx] + 1  # 1-based numbering
            
            # Box zeichnen
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Gruppennummer als Label
            label = f"G{group_num}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 4), (x1 + label_size[0] + 2, y1), (255, 0, 0), -1)
            cv2.putText(img, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        annotated = img
    else:
        annotated = r.orig_img.copy()
    
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