from ultralytics import YOLO
import cv2

model = YOLO('yolo26n.pt')
results = model.predict(source='arbeitsblatt.png', save=True)

# Ergebnisse durchgehen
for r in results:
    print(f"📸 Bild: {r.path}")
    print(f"⚡ Speed: {r.speed}")
    print(f"📦 Anzahl Detektionen: {len(r.boxes)}")
    print("\n🎯 Gefundene Objekte:")
    
    # Alle erkannten Objekte
    for i, box in enumerate(r.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        print(f"  {i+1}. {r.names[class_id]}")
        print(f"     Konfidenz: {confidence:.2%}")
        print(f"     Position: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")
    
    # Bild mit Boxen anzeigen
    annotated = r.plot()  # Gibt numpy array mit gezeichneten Boxen zurück
    cv2.imshow('YOLO Detection', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Oder als PIL Image
    # from PIL import Image
    # img = Image.fromarray(annotated[..., ::-1])  # BGR zu RGB
    # img.show()