"""OCR, multimodal model calls, and structured solution mapping."""

from pathlib import Path

import cv2
from google import genai
from google.genai import types
import ollama
from PIL import Image

from .schemas import get_solution


class SolvingMixin:
    """AI solving stage used by :class:`main.WorksheetSolver`."""

    def _build_solution_prompt(self, ocr_text):
        """Describe answer units and build the structured solving prompt."""
        if self.image is not None:
            image_width = self.image.shape[1]
        else:
            with Image.open(self.path) as source_image:
                image_width = source_image.width

        group_descriptions = []
        for index, group in enumerate(self.answer_units):
            group_number = index + 1
            first_index = group[0]
            first_gap = self.detected_gaps[first_index]
            class_name = str(first_gap[4]) if len(first_gap) > 4 else "gap"
            row_count, max_characters = self._answer_capacity(group, image_width)
            kind = (
                "open answer area"
                if row_count > 1
                else f"single {class_name} gap"
            )
            group_descriptions.append(
                f"Group {group_number}: {kind}, {row_count} writable line(s), "
                f"approximately {max_characters} characters maximum"
            )

        group_text = "\n".join(group_descriptions)
        return f"""
Solve this German worksheet.

OCR text:
{ocr_text}

Answer groups:
{group_text}

Fill each numbered answer position with the exact text that belongs there.

Rules:
- Answer in German.
- For an inline gap, return only the missing word or short phrase. Never repeat the surrounding sentence.
- For an open answer area, give a concise direct answer, not an explanation of your reasoning.
- Stay within the approximate character limit stated for each group.
- Do not add labels such as "Antwort:" or "Lösung:".
- Match grammar, capitalization, and singular/plural.
- If unclear, answer "none".
- Return every group exactly once and use its number as the key.

Return only this JSON format:

{{
  "solutions": [
    {{
      "key": 1,
      "value": "answer"
    }}
  ]
}}
"""

    def _solve_with_cloud(self, marked_image_path, prompt):
        """Request structured solutions from Gemini."""
        with (
            Image.open(marked_image_path) as marked_image,
            Image.open(self.path) as original_image,
        ):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[marked_image, original_image, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=get_solution.model_json_schema(),
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=(
                                self.thinking_budget if self.think else 0
                            )
                        ),
                    ),
                )
            except genai.errors.ServerError:
                fallback_model = "gemini-3-flash-preview"
                if self.model_name == fallback_model:
                    raise
                print(
                    f"Gemini server error - falling back to {fallback_model}"
                )
                self.model_name = fallback_model
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[marked_image, original_image, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=get_solution.model_json_schema(),
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=(
                                self.thinking_budget if self.think else 0
                            )
                        ),
                    ),
                )
        return get_solution.model_validate_json(response.text)

    def _solve_with_streaming_ollama(self, marked_image_path, prompt):
        """Handle the legacy Qwen thinking/instruct fallback flow."""
        print(
            "you are using an experimantal thinking model - we will stream "
            "the response and switch to an instruct model if it seems to get "
            "stuck in thinking mode"
        )
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [marked_image_path, self.path],
                }
            ],
            format=get_solution.model_json_schema(),
            options={"num_ctx": 8192},
            stream=True,
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
                if len(thinking) > 12000 and "\n\n" in thinking.strip()[-10:]:
                    thinking = thinking.split("\n\n")[0]
                    del response
                    print(len(thinking))
                    finished = False
                    break

        if finished:
            return get_solution.model_validate_json(full_response), thinking

        final_response = ollama.chat(
            model=self.model_name.replace("thinking", "instruct"),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [marked_image_path, self.path],
                },
                {"role": "assistant", "content": thinking},
            ],
            format=get_solution.model_json_schema(),
            options={"num_ctx": 8192},
        )
        return (
            get_solution.model_validate_json(final_response.message.content),
            thinking,
        )

    def _solve_with_ollama(self, marked_image_path, prompt, start_time):
        """Request structured solutions from a local Ollama model."""
        if self.model_name == "qwen3-vl:8b-thinking" and self.think:
            return self._solve_with_streaming_ollama(marked_image_path, prompt)

        capabilities = ollama.show(self.model_name).capabilities
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [marked_image_path, self.path],
                }
            ],
            format=get_solution.model_json_schema(),
            think=(
                None
                if "thinking" not in capabilities
                else bool(self.think)
            ),
            options={"num_ctx": 8192, "temperature": 0.1},
        )

        thinking = getattr(response.message, "thinking", None)
        if thinking:
            print(thinking)
        try:
            output = get_solution.model_validate_json(response.message.content)
        except Exception as error:
            if self.debug:
                if thinking:
                    print(f"Thinking content:\n{thinking}")
                print(f"Full response content:\n{response.message.content}")
                print("Debug mode ON - timing enabled")
                print(f"Time taken: {self.time.time() - start_time:.2f} seconds")
            raise ValueError(
                f"The AI returned an invalid response: {error}"
            ) from error
        return output, thinking

    def _solve_with_experimental_pipeline(self, marked_image_path, prompt):
        """Request solutions from the optional Transformers pipeline."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image_path": marked_image_path},
                    {"type": "image", "image_path": self.path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        response = self.pipe(
            messages,
            enable_thinking=self.think,
            prefix_allowed_tokens_fn=self.prefix_function,
        )[0]["generated_text"][-1]["content"]
        return get_solution.model_validate_json(response.split("</think>")[-1])

    def ask_ai_about_all_gaps(self, marked_image):
        """Ask the configured multimodal model about every answer unit."""
        start_time = self.time.time() if self.debug else None
        thinking = None
        source_path = Path(self.path)
        marked_image_path = self.temporary_path(
            f"{source_path.stem}_marked.png"
        )

        if not cv2.imwrite(str(marked_image_path), marked_image):
            raise OSError(
                f"Could not write temporary marked image: {marked_image_path}"
            )

        ocr_text = self.ocr_image(self.path)
        prompt = self._build_solution_prompt(ocr_text)

        if self.experimental:
            output = self._solve_with_experimental_pipeline(
                str(marked_image_path),
                prompt,
            )
        elif self.local:
            output, thinking = self._solve_with_ollama(
                str(marked_image_path),
                prompt,
                start_time,
            )
        else:
            output = self._solve_with_cloud(str(marked_image_path), prompt)

        if self.debug:
            print("Debug mode ON - timing enabled")
            print(f"Time taken: {self.time.time() - start_time:.2f} seconds")
            if thinking:
                print(f"Thinking: {thinking}")
            print(f"AI output:\n{output}")
        return output

    def ocr_image(self, image_path):
        """Extract worksheet text without solving it."""
        ocr_prompt = """
OCR this worksheet image.

Extract all visible text exactly.
Do not solve the exercises.

Replace every empty answer area (blank lines, boxes, or gaps) with:
_____

Keep the original reading order and line breaks.
Preserve capitalization, punctuation, and German characters.
Return only the OCR text.
"""

        if self.local:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": ocr_prompt,
                        "images": [image_path],
                    }
                ],
                think=False,
                options={"temperature": 0.1},
            )
            return response.message.content

        with Image.open(image_path) as image:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[image, ocr_prompt],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        return response.text

    def ocrImage(self, image_path):
        """Backward-compatible alias for the original public method."""
        return self.ocr_image(image_path)

    def solve_all_gaps(self, marked_image):
        """Solve all answer units and map results back to detected gaps."""
        if not self.detected_gaps:
            print("No gaps found!")
            return {}
        if not self.answer_units:
            print("No answer units found to solve.")
            return {}

        print(f"Analyzing all {len(self.answer_units)} answer units with AI...")
        print("Sending image to AI...")
        solutions_data = self.ask_ai_about_all_gaps(marked_image)
        if not solutions_data:
            print("No response received from AI.")
            return {}

        print("Structured AI response received!")
        solutions = {}
        for pair in solutions_data.solutions:
            group_id = pair.key
            group_index = group_id - 1
            if 0 <= group_index < len(self.answer_units):
                solutions[group_index] = {
                    "gap_indices": self.answer_units[group_index],
                    "solution": pair.value,
                }
        return solutions
