import cv2

def mark_coordinates():
    """Markiere die gegebenen Koordinaten im Bild"""
    
    # Koordinaten die markiert werden sollen
    coordinates = [
        (345, 118, 145, 18),
        (400, 168, 150, 18),
        (155, 218, 110, 18),
        (310, 268, 175, 18),
        (215, 318, 135, 18),
        (75, 368, 200, 18),
        (310, 418, 230, 18),
        (105, 468, 120, 18)
    ]
    
    # Bild laden
    image = cv2.imread('arbeitsblatt.png')
    if image is None:
        print("❌ Bild 'arbeitsblatt.png' nicht gefunden!")
        return
    
    # Kopie für Markierungen erstellen
    marked_image = image.copy()
    
    print(f"🎯 Markiere {len(coordinates)} Koordinaten...")
    
    # Jede Koordinate markieren
    for i, (x, y, w, h) in enumerate(coordinates):
        # Verschiedene Farben abwechselnd
        color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)  # Grün/Rot
        
        # Rechteck zeichnen
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), color, 2)
        
        # Nummer hinzufügen
        cv2.putText(marked_image, str(i + 1), (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        print(f"  ✅ Koordinate {i+1}: ({x}, {y}, {w}, {h})")
    
    # Ergebnis anzeigen
    cv2.imshow('Markierte Koordinaten', marked_image)
    
    # Ergebnis speichern
    output_path = 'arbeitsblatt_markiert.png'
    cv2.imwrite(output_path, marked_image)
    print(f"\n💾 Markiertes Bild gespeichert als: {output_path}")
    
    print("\n📁 Drücke eine Taste zum Schließen...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mark_coordinates()
