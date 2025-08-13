import cv2
import numpy as np

def find_gaps_smart():
    """Intelligente Lücken-Erkennung für Arbeitsblätter"""
    
    # Bild laden
    image = cv2.imread('arbeitsblatt.png')
    if image is None:
        print("❌ arbeitsblatt.png nicht gefunden!")
        return []
    
    h, w = image.shape[:2]
    print(f"📊 Bild geladen: {w} x {h} Pixel")
    
    # In Graustufen umwandeln
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gaps = []
    
    # Schritt 1: Horizontale Linien finden (Unterstriche)
    inv = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inv, 100, 255, cv2.THRESH_BINARY)
    
    # Größerer Kernel für zusammenhängende Linien
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Linien erweitern um Lücken zu schließen
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    horizontal_lines = cv2.dilate(horizontal_lines, dilate_kernel, iterations=1)
    
    # Konturen der Linien finden
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Linien zu Textbereichen konvertieren
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter für wahrscheinliche Unterstriche (nicht zu klein, nicht zu groß)
        if w > 40 and h < 15 and w > h * 4:
            # Prüfe dass es keine Tabellenlinie ist (zu lang = Tabelle)
            if w < 350:  # Keine Tabellenlinien
                # Textbereich ÜBER der Linie definieren
                text_height = 18  # Höhe für Text
                text_y = max(0, y - text_height)  # Über der Linie
                text_w = w
                text_h = text_height
                
                gaps.append((x, text_y, text_w, text_h))
    
    # Schritt 2: Freie rechteckige Bereiche finden (ohne Unterstriche)
    # Bereiche mit viel Weißraum
    _, white_thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # Morphologie um kleine Störungen zu entfernen
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_areas = cv2.morphologyEx(white_thresh, cv2.MORPH_OPEN, clean_kernel)
    
    # Größere zusammenhängende Bereiche finden
    expand_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_areas = cv2.morphologyEx(white_areas, cv2.MORPH_CLOSE, expand_kernel)
    
    # Konturen der weißen Bereiche
    white_contours, _ = cv2.findContours(white_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in white_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter für wahrscheinliche Textbereiche
        area = w * h
        if 1500 < area < 6000:  # Mittelgroße Bereiche
            aspect_ratio = w / h if h > 0 else 0
            if 2 < aspect_ratio < 6:  # Eher horizontal für Text
                # Prüfe ob schon eine Lücke in der Nähe ist
                is_duplicate = False
                for existing_x, existing_y, existing_w, existing_h in gaps:
                    if (abs(x - existing_x) < 30 and abs(y - existing_y) < 20):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    gaps.append((x, y, w, h))
    
    # Duplikate entfernen und konsolidieren
    final_gaps = []
    gaps.sort(key=lambda g: (g[1], g[0]))  # Nach y, dann x sortieren
    
    for x, y, w, h in gaps:
        # Schaue ob wir diesen Bereich mit einem existierenden kombinieren können
        merged = False
        for i, (fx, fy, fw, fh) in enumerate(final_gaps):
            # Wenn sehr nahe beieinander, kombiniere sie
            if abs(y - fy) < 10 and abs(x - fx) < 50:
                # Kombiniere zu größerem Bereich
                new_x = min(x, fx)
                new_y = min(y, fy)
                new_w = max(x + w, fx + fw) - new_x
                new_h = max(y + h, fy + fh) - new_y
                final_gaps[i] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        
        if not merged:
            final_gaps.append((x, y, w, h))
    
    # Sortieren von oben nach unten, links nach rechts
    final_gaps.sort(key=lambda gap: (gap[1], gap[0]))
    
    print(f"✅ {len(final_gaps)} Lücken gefunden!")
    
    # Markieren
    result = image.copy()
    for i, (x, y, w, h) in enumerate(final_gaps):
        color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result, str(i+1), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"  Lücke {i+1}: ({x}, {y}, {w}x{h})")
    
    # Speichern
    cv2.imwrite('arbeitsblatt_markiert_smart.png', result)
    print(f"💾 Markiertes Bild gespeichert: arbeitsblatt_markiert_smart.png")
    
    return final_gaps

if __name__ == "__main__":
    find_gaps_smart()
