import cv2
import numpy as np
import json
import base64
import os
import ollama
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List
from PIL import Image

# Pydantic Models außerhalb der Klasse definieren
class Pair(BaseModel):
    key: int
    value: str

class get_solution(BaseModel):
    solutions: List[Pair]

class ArbeitsblattSolver():
    def __init__(self, image_path: str, model_name: str = "gemini-2.5-flash"):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.image_path = image_path
        self.model_name = model_name
        
        self.image = None
        self.freie_stellen = []
        
    def load_image(self):
        """Bild laden und Kopie für Bearbeitung erstellen"""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Bild {self.image_path} nicht gefunden!")
        return self.image.copy()

    def detect_gaps(self, original_image=None):
        """Intelligente Lücken-Erkennung mit OpenCV"""
        if original_image is None:
            original_image = self.load_image()
        
        # Bildmaße ermitteln
        height, width = original_image.shape[:2]
        print(f"📊 Bild geladen: {width} x {height} Pixel")
        
        # In Graustufen umwandeln
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
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
        
        self.freie_stellen = final_gaps
        return self.freie_stellen

    def mark_gaps(self, image=None):
        """Markiere gefundene Lücken im Bild"""
        if image is None:
            image = self.load_image()
            
        for i, (x, y, w, h) in enumerate(self.freie_stellen):
            # Verschiedene Farben für bessere Sichtbarkeit
            color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)
            cv2.rectangle(image, (x-2, y-2), (x+w+2, y+h+2), color, 2)
            # Nummer hinzufügen
            cv2.putText(image, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image
    
    def extract_context_around_gap(self, gap_index: int, context_size: int = 100):
        """Extrahiere Kontext um eine Lücke für die KI"""
        if gap_index >= len(self.freie_stellen):
            return None
            
        x, y, w, h = self.freie_stellen[gap_index]
        
        # Erweiterten Bereich um die Lücke extrahieren
        x_start = max(0, x - context_size)
        y_start = max(0, y - context_size//2)
        x_end = min(self.image.shape[1], x + w + context_size)
        y_end = min(self.image.shape[0], y + h + context_size//2)
        
        context_region = self.image[y_start:y_end, x_start:x_end]
        return context_region
    
    def image_to_base64(self, image):
        """Konvertiere Bild zu Base64 für Ollama"""
        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def ask_ollama_about_all_gaps(self, marked_image):
        """Frage Gemini nach dem Inhalt ALLER Lücken auf einmal - genau wie test3"""
        try:
            # Speichere das markierte Bild (mit den Boxen) genau wie test3 es erwartet
            cv2.imwrite('./arbeitsblatt_markiert_smart.png', marked_image)
            
            # Dann lade es als PIL Image (genau wie test3)
            image = Image.open("./arbeitsblatt_markiert_smart.png")

            prompt = f"""Du siehst ein Arbeitsblatt mit {len(self.freie_stellen)} nummerierten Lücken (rote/grüne Boxen).

AUFGABE: Analysiere das Bild und fülle jede Lücke mit dem passenden Wort aus.

Gehe so vor:
1. Schaue dir den Text um jede Lücke an
2. Verstehe den Kontext und das Thema  
3. Suche nach Wortlisten oder Hinweisen im Bild
4. Gib für jede Lücke das passende Wort zurück

Achte auf:
- Grammatik und Satzstruktur
- Kontext des umgebenden Textes
- Verfügbare Wörter oder Hinweise im Arbeitsblatt
- Logischen Sinnzusammenhang

Gib für jede nummerierte Lücke das passende Wort zurück."""

            # Exakt wie in test3
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": get_solution,
                },
            )
            
            return response.parsed

        except Exception as e:
            print(f"Fehler bei Ollama-Anfrage: {str(e)}")
            return None
    
    def parse_ollama_response(self, response_text):
        """Parse die Ollama-Antwort und extrahiere die Lösungen"""
        solutions = {}
        
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Lücke ') and ':' in line:
                try:
                    # Extrahiere Lückennummer und Antwort
                    parts = line.split(':', 1)
                    luecke_part = parts[0].strip()
                    antwort = parts[1].strip()
                    
                    # Extrahiere Nummer
                    luecke_num = int(luecke_part.replace('Lücke ', ''))
                    gap_index = luecke_num - 1  # 0-basiert
                    
                    if 0 <= gap_index < len(self.freie_stellen):
                        solutions[gap_index] = {
                            'position': self.freie_stellen[gap_index],
                            'solution': antwort
                        }
                except (ValueError, IndexError):
                    continue
        
        return solutions
    
    def solve_all_gaps(self):
        """Löse alle gefundenen Lücken mit Ollama - strukturiert!"""
        if not self.freie_stellen:
            print("Keine Lücken gefunden!")
            return {}
        
        print(f"🤖 Analysiere alle {len(self.freie_stellen)} Lücken mit Ollama...")
        
        # Erstelle markiertes Bild für Ollama
        marked_image = self.mark_gaps()
        
        # Frage Ollama nach allen Lücken gleichzeitig
        print("📤 Sende Bild an Ollama...")
        solutions_data = self.ask_ollama_about_all_gaps(marked_image)
        
        if solutions_data:
            print("📥 Strukturierte Ollama-Antwort erhalten!")
            
            # Konvertiere strukturierte Antwort zu unserem Format
            solutions = {}
            
            # solutions_data.solutions ist jetzt eine Liste von Pair-Objekten
            for pair in solutions_data.solutions:
                try:
                    gap_id = pair.key
                    answer = pair.value
                    gap_index = gap_id - 1  # 0-basiert
                    
                    if 0 <= gap_index < len(self.freie_stellen):
                        solutions[gap_index] = {
                            'position': self.freie_stellen[gap_index],
                            'solution': answer
                        }
                except (ValueError, KeyError) as e:
                    print(f"Fehler beim Verarbeiten von Lücke {gap_id}: {e}")
                    continue
            
            if solutions:
                print("✨ Erkannte Lösungen:")
                for i, sol in solutions.items():
                    print(f"  Lücke {i+1}: '{sol['solution']}'")
            else:
                print("❌ Keine gültigen Lösungen gefunden.")
            
            return solutions
        else:
            print("❌ Keine Antwort von Ollama erhalten.")
            return {}
    
    def fill_gaps_in_image(self, solutions: dict, output_path: str = "arbeitsblatt_gelöst.png"):
        """Fülle die Lösungen in das Bild ein"""
        result_image = self.load_image()
        
        for gap_index, solution_data in solutions.items():
            x, y, w, h = solution_data['position']
            solution = solution_data['solution']
            
            # Text-Encoding für deutsche Umlaute korrigieren
            if isinstance(solution, str):
                solution = solution.encode('utf-8').decode('utf-8')
            
            # PowerPoint-style dynamische Schriftgröße
            # Starte mit einer großen Schrift und reduziere bis Text passt
            max_font_scale = 2.0  # Maximum
            min_font_scale = 0.3  # Minimum
            font_scale = max_font_scale
            thickness = 1
            
            # Iterativ die beste Schriftgröße finden
            while font_scale >= min_font_scale:
                (text_width, text_height), baseline = cv2.getTextSize(
                    solution, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Prüfe ob Text mit etwas Padding in die Box passt
                padding = 4  # Mindestabstand zum Rand
                if (text_width <= w - padding and text_height <= h - padding):
                    break  # Perfekte Größe gefunden
                
                font_scale -= 0.1  # Verkleinere Schrift
            
            # Nochmal final messen mit gefundener Größe
            (text_width, text_height), baseline = cv2.getTextSize(
                solution, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Text ZENTRIERT in der Lücke positionieren
            text_x = x + (w - text_width) // 2  # Horizontal zentriert
            text_y = y + (h + text_height) // 2  # Vertikal zentriert
            
            # Sicherstellen dass Text im Bild bleibt
            if text_x < x:
                text_x = x + 2
            if text_x + text_width > x + w:
                text_x = x + w - text_width - 2
            if text_y - text_height < y:
                text_y = y + text_height + 2
            if text_y > y + h:
                text_y = y + h - 2
            
            # Text in SCHWARZ einfügen (wie mit Stift geschrieben)
            cv2.putText(result_image, solution, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Speichere das Ergebnis
        cv2.imwrite(output_path, result_image)
        print(f"Gelöstes Arbeitsblatt gespeichert als: {output_path}")
        return result_image

# Hauptprogramm
def main():
    solver = ArbeitsblattSolver('arbeitsblatt.png')
    
    print("🔍 Lade Bild und erkenne Lücken...")
    try:
        original = solver.load_image()
        gaps = solver.detect_gaps()
        
        print(f"✅ {len(gaps)} Lücken gefunden!")
        
        marked_image = solver.mark_gaps()
        
        print("\n📍 Gefundene Lücken (x, y, Breite, Höhe):")
        for i, gap in enumerate(gaps):
            print(f"  Lücke {i+1}: {gap}")
        
        # Frage Benutzer, ob KI-Analyse gewünscht ist
        print("\n🤖 Soll Ollama die Lücken analysieren und ausfüllen? (j/n): ", end="")
        user_input = input().lower().strip()
        
        if user_input in ['j', 'ja', 'y', 'yes']:
            solutions = solver.solve_all_gaps()
            
            if solutions:
                print("\n✨ Lösungen gefunden:")
                for i, sol in solutions.items():
                    print(f"  Lücke {i+1}: '{sol['solution']}'")
                
                result_image = solver.fill_gaps_in_image(solutions)
                
                print("\n📁 Ergebnis gespeichert. Drücke eine Taste zum Beenden...")
            else:
                print("❌ Keine Lösungen erhalten.")
        else:
            print("📁 Nur Lückenerkennung - Drücke eine Taste zum Beenden...")
        
    except FileNotFoundError as e:
        print(f"❌ Fehler: {e}")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")

if __name__ == "__main__":
    main()
