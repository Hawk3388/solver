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
        self.gap_groups = []  # Groups of gap indices
        self.gap_to_group = {}  # Maps gap index to group index
        self.ungrouped_gap_indices = []
        self.answer_units = []  # Line groups + single ungrouped boxes
        self.gap_to_answer_unit = {}  # Maps any gap index to answer unit index
        
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
        line_y_max = boxes_sorted[0][3]
        
        for box in boxes_sorted[1:]:
            box_y_top = box[1]
            box_y_bottom = box[3]
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

    def is_line_class(self, class_name):
        """True only for the exact YOLO class name 'line'."""
        return str(class_name).strip().lower() == "line"

    def _unit_bbox(self, unit, gaps):
        """Return merged bbox (x1, y1, x2, y2) for an answer unit."""
        boxes = [gaps[i][:4] for i in unit if 0 <= i < len(gaps)]
        if not boxes:
            return (0, 0, 0, 0)
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def sort_answer_units_reading_order(self, units, gaps):
        """Sort answer units globally by reading order: top->bottom, left->right."""
        if not units:
            return []

        unit_data = []
        for idx, unit in enumerate(units):
            x1, y1, x2, y2 = self._unit_bbox(unit, gaps)
            unit_data.append({
                "idx": idx,
                "unit": unit,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "h": max(1, y2 - y1),
            })

        unit_data.sort(key=lambda u: u["y1"])

        rows = []
        current_row = [unit_data[0]]
        row_y_min = unit_data[0]["y1"]
        row_y_max = unit_data[0]["y2"]

        for u in unit_data[1:]:
            overlap = min(row_y_max, u["y2"]) - max(row_y_min, u["y1"])
            row_h = max(1, row_y_max - row_y_min)
            min_h = max(1, min(row_h, u["h"]))

            if overlap > 0 and (overlap / min_h) > 0.3:
                current_row.append(u)
                row_y_min = min(row_y_min, u["y1"])
                row_y_max = max(row_y_max, u["y2"])
            else:
                rows.append(current_row)
                current_row = [u]
                row_y_min = u["y1"]
                row_y_max = u["y2"]

        rows.append(current_row)

        sorted_units = []
        for row in rows:
            row.sort(key=lambda u: u["x1"])
            sorted_units.extend([u["unit"] for u in row])

        return sorted_units
    
    def group_gaps_by_proximity(self, gaps):
        """Group gaps that are directly below each other into groups.
        
        Returns:
            List of groups, where each group is a list of gap indices (0-based) sorted by Y position
            Also returns a mapping from gap index to group index
        """
        if not gaps:
            return [], {}
        
        # Create index mapping: sorted_idx -> original_idx
        indices = list(range(len(gaps)))
        sorted_indices = sorted(indices, key=lambda i: gaps[i][1])  # Sort by Y (top to bottom)
        
        # Calculate average gap height as threshold
        heights = [(gap[3] - gap[1]) for gap in gaps]
        avg_height = sum(heights) / len(heights) if heights else 0
        
        # Distance threshold: gaps are "below each other" if distance < avg_height * 1.5
        distance_threshold = avg_height * 1.5
        
        groups = []
        gap_to_group = {}
        grouped = set()
        
        # Process gaps from top to bottom
        for sort_i, i in enumerate(sorted_indices):
            if i in grouped:
                continue
            
            gap_i = gaps[i]
            x1_i, y1_i, x2_i, y2_i = gap_i[:4]
            class_name_i = gap_i[4] if len(gap_i) > 4 else "line"
            
            # Only exact 'line' class is groupable. Other classes are ignored here.
            if not self.is_line_class(class_name_i):
                continue

            # Start new group with current line gap
            current_group = [i]
            grouped.add(i)
            
            # Look for gaps below this one
            for sort_j in range(sort_i + 1, len(sorted_indices)):
                j = sorted_indices[sort_j]
                
                if j in grouped:
                    continue
                
                gap_j = gaps[j]
                x1_j, y1_j, x2_j, y2_j = gap_j[:4]
                class_name_j = gap_j[4] if len(gap_j) > 4 else "line"
                
                # Only group if both are exact line class detections
                if not self.is_line_class(class_name_j):
                    continue
                
                # Check vertical distance (gap j should be below gap i)
                vertical_distance = y1_j - y2_i
                
                # Check horizontal alignment
                i_left, i_top, i_right, i_bottom = x1_i, y1_i, x2_i, y2_i
                j_left, j_top, j_right, j_bottom = x1_j, y1_j, x2_j, y2_j
                
                # Calculate horizontal overlap
                h_overlap_start = max(i_left, j_left)
                h_overlap_end = min(i_right, j_right)
                h_overlap = max(0, h_overlap_end - h_overlap_start)
                
                # Box widths
                i_width = i_right - i_left
                j_width = j_right - j_left
                min_width = min(i_width, j_width)
                
                # Check if box j is below box i and horizontally aligned
                if 0 < vertical_distance < distance_threshold:
                    # At least 30% overlap or 15px minimum
                    if h_overlap > min_width * 0.3 or h_overlap > 15:
                        current_group.append(j)
                        grouped.add(j)
                        gap_i = gap_j  # Update for next iteration
                        x1_i, y1_i, x2_i, y2_i = gap_i[:4]
                    else:
                        # Not enough overlap, end this group
                        break
                else:
                    # Distance too large, end this group
                    break
            
            # Store group (sort indices in return order)
            current_group.sort()
            for idx in current_group:
                gap_to_group[idx] = len(groups)
            
            groups.append(current_group)
        
        return groups, gap_to_group

    def detect_gaps(self):
        self.detected_gaps = []
        img = self.load_image(self.path)

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
                    class_id = int(box.cls[0])
                    class_name = r.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    self.detected_gaps.append((int(x1), int(y1), int(x2), int(y2), class_name))
                img = r.orig_img.copy()
        
        # Sort in reading order (line by line)
        self.detected_gaps = self.sort_reading_order(self.detected_gaps)
        
        # Group gaps by proximity (vertically aligned and close together)
        self.gap_groups, self.gap_to_group = self.group_gaps_by_proximity(self.detected_gaps)
        self.ungrouped_gap_indices = [i for i in range(len(self.detected_gaps)) if i not in self.gap_to_group]

        # Build answer units for the AI:
        # - grouped line boxes stay grouped
        # - each ungrouped box (e.g. class gap) becomes its own single unit
        unsorted_units = list(self.gap_groups) + [[idx] for idx in self.ungrouped_gap_indices]
        self.answer_units = self.sort_answer_units_reading_order(unsorted_units, self.detected_gaps)
        self.gap_to_answer_unit = {}
        for unit_idx, unit in enumerate(self.answer_units):
            for gap_idx in unit:
                self.gap_to_answer_unit[gap_idx] = unit_idx
        
        print(f"📊 Line-boxes grouped into {len(self.gap_groups)} groups")
        for i, group in enumerate(self.gap_groups):
            print(f"   Group {i+1}: {len(group)} gaps (indices: {group})")
        print(f"📌 Ungrouped boxes (e.g. gap): {len(self.ungrouped_gap_indices)}")
        print(f"🧠 Total AI answer units: {len(self.answer_units)}")
                    
        return self.detected_gaps, img

    def mark_gaps(self, image, gaps):
        """Draw one red box per answer unit (group) instead of per single line."""

        if not self.answer_units:
            return image

        for unit_idx, unit in enumerate(self.answer_units):
            unit_boxes = [gaps[i][:4] for i in unit if 0 <= i < len(gaps)]
            if not unit_boxes:
                continue

            # Surround the whole group with one box.
            x1 = min(b[0] for b in unit_boxes)
            y1 = min(b[1] for b in unit_boxes)
            x2 = max(b[2] for b in unit_boxes)
            y2 = max(b[3] for b in unit_boxes)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            label = str(unit_idx + 1)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(image, (x1, y1 - label_size[1] - 4), (x1 + label_size[0] + 2, y1), (0, 0, 255), -1)
            cv2.putText(image, (label), (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return image
    
    def ask_ai_about_all_gaps(self, marked_image):
        """Ask Gemini about the content of ALL gap groups at once"""
        if self.debug:
            start_time = self.time.time()
        
        thinking = None
        marked_image_path = f"{Path(self.path).stem}_marked.png"
        cv2.imwrite(marked_image_path, marked_image)

        # Build description of answer units
        group_descriptions = []
        for i, group in enumerate(self.answer_units):
            group_num = i + 1
            first_idx = group[0]
            class_name = str(self.detected_gaps[first_idx][4]) if len(self.detected_gaps[first_idx]) > 4 else "gap"
            if len(group) > 1:
                group_descriptions.append(f"Group {group_num}: {len(group)} stacked line boxes (marked as {group_num})")
            else:
                group_descriptions.append(f"Group {group_num}: 1 single {class_name} box (marked as {group_num})")
        
        group_text = "\n".join(group_descriptions)

        prompt = f"""Look at the two images: one with red numbered boxes marking {len(self.answer_units)} answer groups, one without markings.

Answer groups to fill:
{group_text}

For each group marked with its number label, provide ONE answer that should fill that group.
The answer will be distributed across the stacked lines (first line(s) filled first, then overflow to next line).

Rules:
- Answer in the worksheet's language.
- Provide text that makes sense when distributed line by line.
- Match each answer to the correct group number.
- If a group doesn't need filling, answer with "none".
- Do NOT overthink. These are simple language exercises. Answer quickly and directly. Only reason for about 10 sentences.
- Look at the sheets carefully and use them as context for your answers.
- Only answer in this exact JSON format: {{"solutions": [{{"key": group_number, "value": answer}}]}}"""

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
        """Solve all gap groups with Ollama - structured!"""
        if not self.detected_gaps:
            print("No gaps found!")
            return {}
        if not self.answer_units:
            print("No answer units found to solve.")
            return {}
        
        print(f"🤖 Analyzing all {len(self.answer_units)} answer units with AI...")
        
        # Ask AI about all gap groups at once
        print("📤 Sending image to AI...")
        solutions_data = self.ask_ai_about_all_gaps(marked_image)
        
        if solutions_data:
            print("📥 Structured AI response received!")
            
            # Convert structured response to our format
            solutions = {}
            
            # solutions_data.solutions is now a list of GroupPair objects
            for pair in solutions_data.solutions:
                try:
                    group_id = pair.key
                    answer = pair.value
                    group_index = group_id - 1  # 0-based
                    
                    if 0 <= group_index < len(self.answer_units):
                        gap_indices = self.answer_units[group_index]
                        solutions[group_index] = {
                            'gap_indices': gap_indices,
                            'solution': answer
                        }
                except (ValueError, KeyError) as e:
                    print(f"Error processing group {group_id}: {e}")
                    continue
            
            return solutions
        else:
            print("❌ No response received from AI.")
            return {}

    def _load_font_with_fallback(self, font_size: int):
        """
        Load a scalable TrueType font across platforms.
        Returns (font, is_default_bitmap_font).
        """
        font_candidates = [
            # Repo-local fonts (if present)
            "fonts/DejaVuSans.ttf",
            "fonts/LiberationSans-Regular.ttf",

            # Linux (Hugging Face Spaces)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",

            # Windows
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",

            # macOS
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ]

        for font_path in font_candidates:
            try:
                return ImageFont.truetype(font_path, font_size), False
            except OSError:
                continue

        # Last fallback (bitmap font, limited scaling)
        return ImageFont.load_default(), True
    
    def fill_gaps_in_image(self, image_path: str, solutions: dict, output_path: str = "worksheet_solved.png"):
        """Fill the solutions into grouped gaps with text flowing across multiple boxes"""
        # Load OpenCV image and convert to PIL (for Unicode/umlauts)
        cv_image = self.load_image(image_path)
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(pil_image)

        for group_index, solution_data in solutions.items():
            gap_indices = solution_data['gap_indices']
            solution = solution_data['solution']

            if not solution or solution.lower() == 'none':
                continue

            # Get all boxes for this group
            boxes = [self.detected_gaps[idx] for idx in gap_indices]

            # Calculate total available space
            total_width = sum(box[2] - box[0] for box in boxes)
            avg_height = boxes[0][3] - boxes[0][1]
            first_box_width = boxes[0][2] - boxes[0][0]

            # Find optimal font size for this solution
            font_size = 40
            min_font_size = 8
            chosen_font = None
            is_default_bitmap = False

            while font_size >= min_font_size:
                font, is_default = self._load_font_with_fallback(font_size)

                # Bitmap-default font is not really scalable -> use it directly
                if is_default:
                    chosen_font = font
                    is_default_bitmap = True
                    break

                # Measure full solution text
                bbox = draw.textbbox((0, 0), solution, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                padding = 4
                fits_height = text_height <= (avg_height - padding)
                fits_width = (text_width <= (total_width - padding)) or (text_width <= (first_box_width - padding))

                if fits_height and fits_width:
                    chosen_font = font
                    break

                font_size -= 1

            # Absolute fallback safety
            if chosen_font is None:
                chosen_font, is_default_bitmap = self._load_font_with_fallback(min_font_size)

            # Distribute text across boxes in the group
            words = solution.split()
            current_box_idx = 0
            x_offset = boxes[current_box_idx][0]  # Start position in current box

            for word in words:
                if current_box_idx >= len(boxes):
                    break

                # Get current box dimensions
                x1, y1, x2, y2 = boxes[current_box_idx][:4]
                box_height = y2 - y1

                # Measure word with space
                word_with_space = word + " "
                bbox = draw.textbbox((0, 0), word_with_space, font=chosen_font)
                word_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Check if word fits in current box
                available_width = (x2 - x_offset) - 4  # padding

                if word_width <= available_width:
                    # Word fits in current box
                    text_y = y1 + (box_height - text_height) // 2
                    draw.text((x_offset, text_y), word_with_space, fill=(0, 0, 0), font=chosen_font)
                    x_offset += word_width
                else:
                    # Move to next box
                    current_box_idx += 1
                    if current_box_idx >= len(boxes):
                        break

                    nx1, ny1, nx2, ny2 = boxes[current_box_idx][:4]
                    nbox_height = ny2 - ny1
                    x_offset = nx1

                    # Re-check fit in new box
                    available_width = (nx2 - x_offset) - 4
                    if word_width <= available_width:
                        text_y = ny1 + (nbox_height - text_height) // 2
                        draw.text((x_offset, text_y), word_with_space, fill=(0, 0, 0), font=chosen_font)
                        x_offset += word_width
                    else:
                        # If single word is too long, draw truncated chunk
                        trimmed = word
                        trimmed_width = 0
                        while len(trimmed) > 1:
                            test = trimmed + " "
                            tb = draw.textbbox((0, 0), test, font=chosen_font)
                            tw = tb[2] - tb[0]
                            if tw <= available_width:
                                trimmed_width = tw
                                break
                            trimmed = trimmed[:-1]

                        if len(trimmed) > 0:
                            if trimmed_width == 0:
                                trimmed_width = draw.textbbox((0, 0), trimmed + " ", font=chosen_font)[2]
                            text_y = ny1 + (nbox_height - text_height) // 2
                            draw.text((x_offset, text_y), trimmed + " ", fill=(0, 0, 0), font=chosen_font)
                            x_offset += trimmed_width

        # Convert back to OpenCV BGR and save
        result_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_cv)

        if self.debug:
            print(f"✅ Filled worksheet saved to: {output_path}")

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
        
        print(f"✅ {len(gaps)} boxes found, {len(solver.gap_groups)} line groups, {len(solver.ungrouped_gap_indices)} ungrouped!")
        
        marked_image = solver.mark_gaps(img, gaps)
        
        print("\n📍 Detected gaps (x, y, width, height):")
        for i, gap in enumerate(gaps):
            unit_num = solver.gap_to_answer_unit.get(i)
            if unit_num is not None:
                print(f"  Box {i+1} (Group {unit_num + 1}): {gap}")
            else:
                print(f"  Box {i+1} (ungrouped): {gap}")
        
        print("\n📊 Gap groups:")
        for g_idx, group in enumerate(solver.gap_groups):
            print(f"  Group {g_idx+1}: gaps {[idx+1 for idx in group]}")
        
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
                for group_idx, sol in solutions.items():
                    group_num = group_idx + 1
                    gap_indices = [idx+1 for idx in sol['gap_indices']]
                    print(f"  Group {group_num} (gaps {gap_indices}): '{sol['solution']}'")
                
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
