from ultralytics import YOLO
import cv2
from pathlib import Path

# Trainiertes Modell laden
MODEL_PATH = 'arbeitsblatt_yolo/transfer_learning/weights/best.pt'

# Prüfe ob trainiertes Modell existiert
if not Path(MODEL_PATH).exists():
    print(f"❌ Trainiertes Modell nicht gefunden: {MODEL_PATH}")
    print(f"💡 Führe zuerst train_yolo.py aus!")
    print(f"\nFalls vorhanden, ändere MODEL_PATH zur korrekten Position")
    exit()

print(f"✅ Lade trainiertes Modell: {MODEL_PATH}\n")
model = YOLO(MODEL_PATH)

# Bild zum Testen
IMAGE_PATH = 'arbeitsblatt.png'
results = model.predict(source=IMAGE_PATH, save=True, conf=0.25)

# Ergebnisse durchgehen
for r in results:
    print(f"📸 Bild: {r.path}")
    print(f"⚡ Speed: {r.speed}")
    print(f"📦 Anzahl freie Stellen gefunden: {len(r.boxes)}")
    
    if len(r.boxes) == 0:
        print("\n❌ Keine freien Stellen erkannt!")
        print("💡 Überprüfe:")
        print("   - Ist das Bild ein Arbeitsblatt?")
        print("   - Wurde das Modell richtig trainiert?")
        print("   - Versuche niedrigere conf (z.B. 0.1)")
    else:
        print("\n✅ Gefundene freie Stellen:")
        # Alle erkannten freien Stellen
        for i, box in enumerate(r.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            print(f"  {i+1}. {r.names[class_id]}")
            print(f"     Konfidenz: {confidence:.2%}")
            print(f"     Box: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")
            print(f"     Größe: {int(x2-x1)} x {int(y2-y1)} px")
    
    # Bild mit markierten freien Stellen anzeigen
    print(f"\n🎨 Zeige Ergebnis...")
    annotated = r.plot()  # Gibt numpy array mit gezeichneten Boxen zurück
    
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