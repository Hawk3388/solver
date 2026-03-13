import cv2
import os
import ollama
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Define Pydantic models outside the class
class Pair(BaseModel):
    key: int
    value: str

class get_solution(BaseModel):
    solutions: List[Pair]

class WorksheetSolver():
    def __init__(self, path:str, gap_detection_model_path: str = "./model/gap_detection_model.pt", llm_model_name: str = "gemini-2.5-flash", think: bool = True, local: bool = False, thinking_budget: int = 2048, debug: bool = False, experimental: bool = False):
        self.model_path = gap_detection_model_path
        self.model_name = llm_model_name
        self.local = local
        self.path = path
        self.debug = debug
        if think:
            self.thinking_budget = thinking_budget
        self.think = think
        self.experimental = experimental
        
        if self.debug:
            import time
            self.time = time
        if not Path(self.path).exists():
            print(f"❌ Worksheet image not found: {self.path}")
            print(f"💡 Please check the path to the image and try again.")
            exit()
        else:
            if not self.path.lower().endswith(".png"):
                print(f"✅ Worksheet image found: {self.path}")
                img = Image.open(self.path)
                img.save(f"{Path(self.path).stem}_temp.png")
                self.path = f"{Path(self.path).stem}_temp.png"
        if not Path(self.model_path).exists():
            print(f"❌ Trained model not found: {self.model_path}")
            print(f"💡 Run train_yolo.py first!")
            print(f"\nIf available, change MODEL_PATH to the correct location")
            exit()
        if not self.local and not self.experimental:
            try:
                if os.path.exists(".env"):
                    load_dotenv()
                    self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                elif os.getenv("GOOGLE_API_KEY"):
                    self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                else:
                    print(f"❌ .env file with Google API key not found!")
                    print(f"💡 Please create a .env file with your Google API key as GOOGLE_API_KEY=your_key and try again.")
            except Exception:
                print(f"❌ .env file with Google API key not found!")
                print(f"💡 Please create a .env file with your Google API key as GOOGLE_API_KEY=your_key and try again.")
        if self.experimental and self.local:

            from transformers.generation import LogitsProcessor
            from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            import torch

            class ThinkingTokenBudgetProcessor(LogitsProcessor):
                """
                A processor where after a maximum number of tokens are generated,
                a </think> token is added at the end to stop the thinking generation,
                and then it will continue to generate the response.
                """
                def __init__(self, tokenizer, max_thinking_tokens=None):
                    self.tokenizer = tokenizer
                    self.max_thinking_tokens = max_thinking_tokens
                    self.think_end_token = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
                    self.nl_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
                    self.tokens_generated = 0
                    self.stopped_thinking = False
                    self.neg_inf = float('-inf')

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                    self.tokens_generated += 1
                    if self.max_thinking_tokens == 0 and not self.stopped_thinking and self.tokens_generated > 0:
                        scores[:] = self.neg_inf
                        scores[0][self.nl_token] = 0
                        scores[0][self.think_end_token] = 0
                        self.stopped_thinking = True
                        return scores

                    if self.max_thinking_tokens is not None and not self.stopped_thinking:
                        if (self.tokens_generated / self.max_thinking_tokens) > .95:
                            scores[0][self.nl_token] = scores[0][self.think_end_token] * (1 + (self.tokens_generated / self.max_thinking_tokens))
                            scores[0][self.think_end_token] = (
                                scores[0][self.think_end_token] * (1 + (self.tokens_generated / self.max_thinking_tokens))
                            )

                        if self.tokens_generated >= (self.max_thinking_tokens - 1):
                            if self.tokens_generated == self.max_thinking_tokens-1:
                                scores[:] = self.neg_inf
                                scores[0][self.nl_token] = 0
                            else:
                                scores[:] = self.neg_inf
                                scores[0][self.think_end_token] = 0
                                self.stopped_thinking = True

                    return scores
                
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model)

            if self.think:
                processor = ThinkingTokenBudgetProcessor(tokenizer, max_thinking_tokens=self.thinking_budget)
            else:
                # print("For the experimental mode thinking will be enabled")
                processor = ThinkingTokenBudgetProcessor(tokenizer, max_thinking_tokens=self.thinking_budget)

            schema_parser = JsonSchemaParser(get_solution.model_json_schema())
            self.prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, schema_parser)

            self.pipe = pipeline(
                "image-text-to-text", 
                model=self.model, 
                max_new_tokens=4096, 
                logits_processor=[processor], 
                device=0,
                model_kwargs={"quantization_config": quantization_config}
            )

        self.model = YOLO(self.model_path)
        
        self.image = None
        self.detected_gaps = []
        
    def load_image(self, image_path: str):
        """Load image and create a copy for processing"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image {image_path} not found!")
        return self.image.copy()
    
    def calculate_iou(self, box1: list, box2: list):
        """
        Calculates Intersection over Union (IoU) between two boxes
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
        Filters overlapping boxes - keeps only the one with highest confidence
        
        Args:
            boxes: YOLO boxes object
            iou_threshold: Minimum IoU for overlap (0.5 = 50%)
        
        Returns:
            List of indices of boxes to keep
        """
        if len(boxes) == 0:
            return []
        
        # Extract coordinates and confidences
        coords = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = boxes.conf.cpu().numpy()
        
        # Sort by confidence (highest first)
        sorted_indices = np.argsort(-confidences)
        
        keep = []
        
        for i in sorted_indices:
            # Check if this box overlaps with already kept boxes
            should_keep = True
            
            for kept_idx in keep:
                iou = self.calculate_iou(coords[i], coords[kept_idx])
                
                if iou > iou_threshold:
                    # Overlap found - discard this box (lower confidence)
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(i)
        
        return sorted(keep)  # Back in original order
    
    def sort_reading_order(self, boxes):
        """Sort boxes in reading order: line by line from top to bottom, left to right within a line.
        
        Boxes on the same text line often have slightly different y values.
        This method groups boxes with similar y position (overlap) into lines.
        """
        if not boxes:
            return boxes
        
        # Sort roughly by y first
        boxes_sorted = sorted(boxes, key=lambda b: b[1])
        
        # Group into lines based on vertical overlap
        lines = []
        current_line = [boxes_sorted[0]]
        # y-center and height of the current line
        line_y_min = boxes_sorted[0][1]
        line_y_max = boxes_sorted[0][3] if len(boxes_sorted[0]) == 4 else boxes_sorted[0][1] + boxes_sorted[0][3]
        
        for box in boxes_sorted[1:]:
            box_y_top = box[1]
            box_y_bottom = box[3] if len(box) == 4 else box[1] + box[3]
            box_height = box_y_bottom - box_y_top
            line_height = line_y_max - line_y_min
            
            # Check if the box overlaps vertically with the current line
            # Tolerance: at least 50% of the smaller height must overlap
            overlap = min(line_y_max, box_y_bottom) - max(line_y_min, box_y_top)
            min_height = max(min(box_height, line_height), 1)
            
            if overlap > 0 and overlap / min_height > 0.3:
                # Same line
                current_line.append(box)
                line_y_min = min(line_y_min, box_y_top)
                line_y_max = max(line_y_max, box_y_bottom)
            else:
                # New line
                lines.append(current_line)
                current_line = [box]
                line_y_min = box_y_top
                line_y_max = box_y_bottom
        
        lines.append(current_line)
        
        # Sort within each line by x, lines from top to bottom
        result = []
        for line in lines:
            line.sort(key=lambda b: b[0])  # By x coordinate
            result.extend(line)
        
        return result

    def detect_gaps(self):
        self.detected_gaps = []

        results = self.model.predict(source=self.path, conf=0.10)

        for r in results:
            if len(r.boxes) > 0:
                keep_indices = self.filter_overlapping_boxes(r.boxes, iou_threshold=0.5)
                print(f"🔍 After overlap filtering: {len(keep_indices)} boxes")
            else:
                keep_indices = []
            if len(keep_indices) == 0:
                print("\n❌ No gaps detected!")
                print("💡 Check:")
                print("   - Is the image a worksheet?")
                print("   - Was the model trained correctly?")
                print("   - Try lower conf (e.g. 0.1)")
            else:
                for idx in keep_indices:
                    box = r.boxes[idx]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    self.detected_gaps.append((int(x1), int(y1), int(x2), int(y2)))
                img = r.orig_img.copy()
        
        # Sort in reading order (line by line)
        self.detected_gaps = self.sort_reading_order(self.detected_gaps)
                    
        return self.detected_gaps, img

    def mark_gaps(self, image, gaps):
        """Mark detected gaps in the image with numbers"""

        for i, gap in enumerate(gaps):
            x1, y1, x2, y2 = gap
            # Draw red box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Number at top left of the box
            label = str(i + 1)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            # Background for better readability
            cv2.rectangle(image, (x1, y1 - label_size[1] - 4), (x1 + label_size[0] + 2, y1), (0, 0, 255), -1)
            cv2.putText(image, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return image
    
    def ask_ai_about_all_gaps(self, marked_image):
        """Ask Gemini about the content of ALL gaps at once - just like test3"""
        if self.debug:
            start_time = self.time.time()
        # Save the marked image (with boxes) just as test3 expects
        thinking = None
        marked_image_path = f"{Path(self.path).stem}_marked.png"
        cv2.imwrite(marked_image_path, marked_image)

        prompt = f"""Look at the two images: one with red numbered boxes marking {len(self.detected_gaps)} gaps, one without markings.

For each red box, read its number label and fill in the missing word(s) from the worksheet.

Rules:
- Answer in the worksheet's language.
- Only the missing word(s), nothing else.
- Match each answer to the correct box number.
- If a box doesn't need filling, because it is already filled or is not a gap, answer with "none".
- Do NOT overthink. These are simple language exercises. Answer quickly and directly. Only reason for about 10 sentences.
- Look at the sheets carefully and use them as context for your answers.
- Only answer in this exact JSON format: {{"solutions": [{{"key": box_number, "value": answer}}]}}"""

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
                        thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget if self.think else 0),
                    ),
                )
                output = response.parsed
            else:
                if self.model_name == "qwen3-vl:8b-thinking" and self.think:
                    print("you are using an experimantal thinking model - we will stream the response and switch to an instruct model if it seems to get stuck in thinking mode")
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
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt, "images": [marked_image_path, self.path]}],
                        format=get_solution.model_json_schema(),
                        think=None if not 'thinking' in ollama.show(self.model_name).capabilities else True if self.think else False,
                        options={"num_ctx": 8192}
                    )
                    if response.message.thinking:
                        thinking = response.message.thinking
                    try:
                        output = get_solution.model_validate_json(response.message.content)
                    except Exception as e:
                        print(f"Error validating JSON response: {e}")
                        if self.debug:
                            if thinking:
                                print(f"Thinking content:\n{thinking}")
                            print(f"Full response content:\n{response.message.content}")
                            print(f"⏱️ Debug mode ON - timing enabled")
                            end_time = self.time.time()
                            print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
        else:
            if self.local:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image_path": marked_image_path},
                    {"type": "image", "image_path": self.path},
                    {"type": "text", "text": prompt},
                ]}]
                response = self.pipe(messages, enable_thinking=self.think, prefix_allowed_tokens_fn=self.prefix_function)[0]["generated_text"][-1]["content"]
                response = response.split("</think>")
                output = get_solution.model_validate_json(response[-1])
        
        if not self.debug:
            if os.path.exists(self.path) and self.path.endswith("_temp.png"):
                os.remove(self.path)
            if os.path.exists(marked_image_path):
                os.remove(marked_image_path)
        else:  
            print(f"⏱️ Debug mode ON - timing enabled")
            end_time = self.time.time()
            print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
            if thinking:
                print(f"Thinking: {thinking}")
            print(f"AI output:\n{output}")

        return output
    
    def solve_all_gaps(self, marked_image):
        """Solve all detected gaps with Ollama - structured!"""
        if not self.detected_gaps:
            print("No gaps found!")
            return {}
        
        print(f"🤖 Analyzing all {len(self.detected_gaps)} gaps with Ollama...")
        
        # Ask Ollama about all gaps at once
        print("📤 Sending image to Ollama...")
        solutions_data = self.ask_ai_about_all_gaps(marked_image)
        
        if solutions_data:
            print("📥 Structured Ollama response received!")
            
            # Convert structured response to our format
            solutions = {}
            
            # solutions_data.solutions is now a list of Pair objects
            for pair in solutions_data.solutions:
                try:
                    gap_id = pair.key
                    answer = pair.value
                    gap_index = gap_id - 1  # 0-based
                    
                    if 0 <= gap_index < len(self.detected_gaps):
                        solutions[gap_index] = {
                            'position': self.detected_gaps[gap_index],
                            'solution': answer
                        }
                except (ValueError, KeyError) as e:
                    print(f"Error processing gap {gap_id}: {e}")
                    continue
            
            return solutions
        else:
            print("❌ No response received from Ollama.")
            return {}
    
    def fill_gaps_in_image(self, image_path: str, solutions: dict, output_path: str = "worksheet_solved.png"):
        """Fill the solutions into the image"""
        # Load OpenCV image and convert to PIL (for Unicode/umlauts)
        cv_image = self.load_image(image_path)
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        draw = ImageDraw.Draw(pil_image)
        
        for gap_index, solution_data in solutions.items():
            # Position is (x1, y1, x2, y2)
            x1, y1, x2, y2 = solution_data['position']
            w = x2 - x1
            h = y2 - y1
            solution = solution_data['solution']
            
            if not solution or solution.lower() == 'none':
                continue
            
            # Find dynamic font size
            font_size = 40  # Start large
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
            
            # Measure text size with final font
            bbox = draw.textbbox((0, 0), solution, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text centered in the box
            text_x = x1 + (w - text_width) // 2
            text_y = y1 + (h - text_height) // 2
            
            # Draw text in black
            draw.text((text_x, text_y), solution, fill=(0, 0, 0), font=font)
        
        # Convert back to OpenCV and save
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image)
        print(f"Solved worksheet saved as: {output_path}")
        return result_image

# Main program
def main():
    # Best results with gemini-3-flash-preview (local: qwen3.5:35b for 16 GB VRAM + 32 GB RAM)
    # For Gemini you have to use a Google API-key in a .env file
    # For Ollama models you have to set local=True

    path = input("📂 Please enter the path to the worksheet image: ").strip()
    llm_model_name = "qwen3.5:35b"
    think = True
    local = True
    debug = True
    solver = WorksheetSolver(path, llm_model_name=llm_model_name, think=think, local=local, debug=debug)

    ask = False
    print("🔍 Loading image and detecting gaps...")
    try:
        gaps, img = solver.detect_gaps()
        
        print(f"✅ {len(gaps)} gaps found!")
        
        marked_image = solver.mark_gaps(img, gaps)
        
        print("\n📍 Detected gaps (x, y, width, height):")
        for i, gap in enumerate(gaps):
            print(f"  Gap {i+1}: {gap}")
        
        if solver.debug:
            # Ask user if AI analysis is desired
            user_input = input("\n🤖 Should an AI analyze and fill the gaps? (y/n): ").lower().strip()
            if user_input in ['y', 'yes']:
                ask = True
        else:
            ask = True

        if ask:
            solutions = solver.solve_all_gaps(marked_image)
            
            if solutions:
                print("\n✨ Solutions found:")
                for i, sol in solutions.items():
                    print(f"  Gap {i+1}: '{sol['solution']}'")
                
                solver.fill_gaps_in_image(path, solutions)
                
                print("\n📁 Result saved. Press any key to exit...")
            else:
                print("❌ No solutions received.")
        else:
            print("📁 Gap detection only")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()

# TODO:
# - better image detection with support for more kinds of worksheets
# - Add support for multiple files (batch processing)
# - Create an executable (.exe) for easy use without Python setup (Command: pyinstaller solver.spec)