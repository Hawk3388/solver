import cv2
import os
import ollama
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List
from PIL import Image
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Pydantic Models außerhalb der Klasse definieren
class Pair(BaseModel):
    key: int
    value: str

class get_solution(BaseModel):
    solutions: List[Pair]

class ArbeitsblattSolver():
    def __init__(self, path:str, yolo_model_path: str = "best_model.pt", llm_model_name: str = "gemini-2.5-flash", local: bool = False, experimental: bool = False):
        self.model_path = yolo_model_path
        self.model_name = llm_model_name
        self.local = local
        self.path = path
        self.experimental = experimental
        if not Path(self.path).exists():
            print(f"❌ Arbeitsblatt-Bild nicht gefunden: {self.path}")
            print(f"💡 Bitte überprüfe den Pfad zum Bild und versuche es erneut.")
            exit()
        if not Path(self.model_path).exists():
            print(f"❌ Trainiertes Modell nicht gefunden: {self.model_path}")
            print(f"💡 Führe zuerst train_yolo.py aus!")
            print(f"\nFalls vorhanden, ändere MODEL_PATH zur korrekten Position")
            exit()
        if not self.local and not self.experimental:
            load_dotenv()
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = YOLO(self.model_path)
        
        self.image = None
        self.freie_stellen = []
        
    def load_image(self, image_path: str):
        """Bild laden und Kopie für Bearbeitung erstellen"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Bild {image_path} nicht gefunden!")
        return self.image.copy()
    
    def calculate_iou(self, box1, box2):
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


    def filter_overlapping_boxes(self, boxes, iou_threshold=0.5):
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
                iou = self.calculate_iou(coords[i], coords[kept_idx])
                
                if iou > iou_threshold:
                    # Überlappung gefunden - verwerfe diese Box (niedrigere Confidence)
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(i)
        
        return sorted(keep)  # Zurück in ursprünglicher Reihenfolge
    
    def sort_reading_order(self, boxes):
        """Sortiere Boxen in Lesereihenfolge: zeilenweise von oben nach unten, innerhalb der Zeile links nach rechts.
        
        Boxen auf derselben Textzeile haben oft leicht unterschiedliche y-Werte.
        Diese Methode gruppiert Boxen mit ähnlicher y-Position (Überlappung) in Zeilen.
        """
        if not boxes:
            return boxes
        
        # Sortiere zunächst grob nach y
        boxes_sorted = sorted(boxes, key=lambda b: b[1])
        
        # Gruppiere in Zeilen basierend auf vertikaler Überlappung
        lines = []
        current_line = [boxes_sorted[0]]
        # y-Mitte und Höhe der aktuellen Zeile
        line_y_min = boxes_sorted[0][1]
        line_y_max = boxes_sorted[0][3] if len(boxes_sorted[0]) == 4 else boxes_sorted[0][1] + boxes_sorted[0][3]
        
        for box in boxes_sorted[1:]:
            box_y_top = box[1]
            box_y_bottom = box[3] if len(box) == 4 else box[1] + box[3]
            box_height = box_y_bottom - box_y_top
            line_height = line_y_max - line_y_min
            
            # Prüfe ob die Box vertikal mit der aktuellen Zeile überlappt
            # Toleranz: mindestens 50% der kleineren Höhe muss überlappen
            overlap = min(line_y_max, box_y_bottom) - max(line_y_min, box_y_top)
            min_height = max(min(box_height, line_height), 1)
            
            if overlap > 0 and overlap / min_height > 0.3:
                # Gleiche Zeile
                current_line.append(box)
                line_y_min = min(line_y_min, box_y_top)
                line_y_max = max(line_y_max, box_y_bottom)
            else:
                # Neue Zeile
                lines.append(current_line)
                current_line = [box]
                line_y_min = box_y_top
                line_y_max = box_y_bottom
        
        lines.append(current_line)
        
        # Innerhalb jeder Zeile nach x sortieren, Zeilen von oben nach unten
        result = []
        for line in lines:
            line.sort(key=lambda b: b[0])  # Nach x-Koordinate
            result.extend(line)
        
        return result

    def detect_gaps(self):
        self.freie_stellen = []

        results = self.model.predict(source=self.path, conf=0.25)

        for r in results:
            if len(r.boxes) > 0:
                keep_indices = self.filter_overlapping_boxes(r.boxes, iou_threshold=0.5)
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
                for idx in keep_indices:
                    box = r.boxes[idx]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    self.freie_stellen.append((int(x1), int(y1), int(x2), int(y2)))
                img = r.orig_img.copy()
        
        # Sortiere in Lesereihenfolge (zeilenweise)
        self.freie_stellen = self.sort_reading_order(self.freie_stellen)
                    
        return self.freie_stellen, img

    def mark_gaps(self, image, gaps):
        """Markiere gefundene Lücken im Bild mit Nummern"""

        for i, gap in enumerate(gaps):
            x1, y1, x2, y2 = gap
            # Rote Box zeichnen
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Nummer links oben an der Box
            label = str(i + 1)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            # Hintergrund für bessere Lesbarkeit
            cv2.rectangle(image, (x1, y1 - label_size[1] - 4), (x1 + label_size[0] + 2, y1), (0, 0, 255), -1)
            cv2.putText(image, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return image
    
    def ask_ollama_about_all_gaps(self, marked_image):
        """Frage Gemini nach dem Inhalt ALLER Lücken auf einmal - genau wie test3"""
        try:
            # Speichere das markierte Bild (mit den Boxen) genau wie test3 es erwartet
            marked_image_path = f"{Path(self.path).stem}_markiert.png"
            cv2.imwrite(marked_image_path, marked_image)

            prompt = f"""Look at the two images: one with red numbered boxes marking {len(self.freie_stellen)} gaps, one without markings.

For each red box, read its number label and fill in the missing word(s) from the worksheet.

Rules:
- Answer in the worksheet's language.
- Only the missing word(s), nothing else.
- Match each answer to the correct box number.
- If a box doesn't need filling, because it is already filled or is not a gap, answer with "none".
- Do NOT overthink. These are simple language exercises. Answer quickly and directly. Only reason for about 10 sentences.
- Look at the sheets carefully and use them as context for your answers.
- Output in JSON format: {{"solutions": [{{"key": box_number, "value": answer}}]}}"""

            if not self.experimental:
                if not self.local:
                    image = Image.open(marked_image_path)
                    original_image = Image.open(self.path)
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[image, original_image, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=get_solution,
                            thinking_config=types.ThinkingConfig(thinking_budget=512),
                        ),
                    )
                    output = response.parsed
                else:
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt, "images": [marked_image_path, self.path]}],
                        format=get_solution.model_json_schema(),
                        options={"num_ctx": 8192},
                        stream=True
                    )
                    full_response = ""
                    thinking = ""
                    finished = True
                    for chunk in response:
                        if chunk.message.content:
                            full_response += chunk.message.content
                            print(chunk.message.content, end="", flush=True)
                        elif chunk.message.thinking:
                            print(chunk.message.thinking, end="", flush=True)
                            thinking += chunk.message.thinking
                            if len(thinking) > 12000:
                                if "\n\n" in thinking.strip()[-10:]:
                                    thinking = thinking.split("\n\n")[0]
                                    del response
                                    print(len(thinking))
                                    finished = False
                                    break
                    
                    if not finished:
                        final_response = ollama.chat(
                            model=self.model_name.replace("thinking", "instruct"),
                            messages=[{"role": "user", "content": prompt, "images": [marked_image_path, self.path]},
                                      {"role": "assistant", "content": thinking}],
                            format=get_solution.model_json_schema(),
                            options={"num_ctx": 8192}
                        )

                        output = get_solution.model_validate_json(final_response.message.content)
                    else:
                        output = get_solution.model_validate_json(full_response)
            else:
                pass # Step 3 VL integration
            
            return output

        except Exception as e:
            print(f"Fehler bei Ollama-Anfrage: {str(e)}")
            return None
    
    def solve_all_gaps(self, marked_image):
        """Löse alle gefundenen Lücken mit Ollama - strukturiert!"""
        if not self.freie_stellen:
            print("Keine Lücken gefunden!")
            return {}
        
        print(f"🤖 Analysiere alle {len(self.freie_stellen)} Lücken mit Ollama...")
        
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
            
            return solutions
        else:
            print("❌ Keine Antwort von Ollama erhalten.")
            return {}
    
    def fill_gaps_in_image(self, image_path: str, solutions: dict, output_path: str = "arbeitsblatt_gelöst.png"):
        """Fülle die Lösungen in das Bild ein"""
        # OpenCV Bild laden und zu PIL konvertieren (für Unicode/Umlaute)
        cv_image = self.load_image(image_path)
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil_image)
        
        for gap_index, solution_data in solutions.items():
            # Position ist (x1, y1, x2, y2)
            x1, y1, x2, y2 = solution_data['position']
            w = x2 - x1
            h = y2 - y1
            solution = solution_data['solution']
            
            if not solution or solution.lower() == 'none':
                continue
            
            # Dynamische Schriftgröße finden
            font_size = 40  # Starte groß
            min_font_size = 8
            font = None
            
            while font_size >= min_font_size:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except OSError:
                    try:
                        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
                    except OSError:
                        font = ImageFont.load_default()
                        break
                
                bbox = draw.textbbox((0, 0), solution, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                padding = 4
                if text_width <= w - padding and text_height <= h - padding:
                    break
                
                font_size -= 1
            
            # Text-Größe mit finaler Schrift messen
            bbox = draw.textbbox((0, 0), solution, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Text zentriert in der Box positionieren
            text_x = x1 + (w - text_width) // 2
            text_y = y1 + (h - text_height) // 2
            
            # Text in schwarz zeichnen
            draw.text((text_x, text_y), solution, fill=(0, 0, 0), font=font)
        
        # Zurück zu OpenCV und speichern
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image)
        print(f"Gelöstes Arbeitsblatt gespeichert als: {output_path}")
        return result_image

# Hauptprogramm
def main():
    path = input("📂 Bitte den Pfad zum Arbeitsblatt-Bild eingeben: ").strip()
    # Beste Ergebnisse mit gemini-3-flash-preview
    solver = ArbeitsblattSolver(path, llm_model_name="qwen3-vl:8b-thinking", local=True)
    
    print("🔍 Lade Bild und erkenne Lücken...")
    try:
        gaps, img = solver.detect_gaps()
        
        print(f"✅ {len(gaps)} Lücken gefunden!")
        
        marked_image = solver.mark_gaps(img, gaps)
        
        print("\n📍 Gefundene Lücken (x, y, Breite, Höhe):")
        for i, gap in enumerate(gaps):
            print(f"  Lücke {i+1}: {gap}")
        
        # Frage Benutzer, ob KI-Analyse gewünscht ist
        user_input = input("\n🤖 Soll Ollama die Lücken analysieren und ausfüllen? (j/n): ").lower().strip()
        
        if user_input in ['j', 'ja', 'y', 'yes']:
            solutions = solver.solve_all_gaps(marked_image)
            
            if solutions:
                print("\n✨ Lösungen gefunden:")
                for i, sol in solutions.items():
                    print(f"  Lücke {i+1}: '{sol['solution']}'")
                
                solver.fill_gaps_in_image(path, solutions)
                
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
