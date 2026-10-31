"""OCR, multimodal model calls, and structured solution mapping."""

from contextlib import contextmanager
from pathlib import Path
import time

import cv2
from google import genai
from google.genai import types
import ollama
import httpx
from PIL import Image

from .schemas import get_solution
from .answers import inspect_answers
from .config import DEFAULT_CLOUD_MODEL, REQUEST_TIMEOUT_SECONDS, OLLAMA_MAX_SECONDS


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
                        thinking_config=(
                            None if self.think else types.ThinkingConfig(thinking_budget=0)
                        ),
                    ),
                )
            except genai.errors.ServerError:
                fallback_model = DEFAULT_CLOUD_MODEL
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
                        thinking_config=(
                            None if self.think else types.ThinkingConfig(thinking_budget=0)
                        ),
                    ),
                )
        return get_solution.model_validate_json(response.text)

    def _solve_with_ollama(self, marked_image_path, prompt, start_time):
        """Request structured solutions from a local Ollama model."""
        capabilities = self._ollama_capabilities()
        content, thinking = self._chat_with_ollama(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [marked_image_path, self.path],
                }
            ],
            response_format=get_solution.model_json_schema(),
            think=(
                None
                if "thinking" not in capabilities
                else bool(self.think)
            ),
            options={"num_ctx": 8192, "temperature": 0.1},
            phase="Solving exercises",
        )

        try:
            output = get_solution.model_validate_json(content)
        except Exception as error:
            if self.debug:
                if thinking:
                    print(f"Thinking content:\n{thinking}")
                print(f"Full response content:\n{content}")
                print("Debug mode ON - timing enabled")
                print(f"Time taken: {self.time.time() - start_time:.2f} seconds")
            raise ValueError(
                f"The AI returned an invalid response: {error}"
            ) from error
        return output, thinking

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
        self._solution_prompt = self._build_solution_prompt(ocr_text)
        output = self._request_solutions(str(marked_image_path), self._solution_prompt)
        if self.debug:
            print(f"Time taken: {self.time.time() - start_time:.2f} seconds")
        return output

    def _request_solutions(self, marked_image_path, prompt):
        """Reuse the selected backend for a bounded answer-repair request."""
        start_time = self.time.time() if self.debug else None
        thinking = None

        if self.local:
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
            content, _thinking = self._chat_with_ollama(
                messages=[
                    {
                        "role": "user",
                        "content": ocr_prompt,
                        "images": [image_path],
                    }
                ],
                think=False,
                options={"temperature": 0.1},
                phase="Reading worksheet",
            )
            return content

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
        format_repaired = False
        try:
            solutions_data = self.ask_ai_about_all_gaps(marked_image)
        except ValueError:
            # JSON/schema failures get one repair request, not an unbounded loop.
            if not hasattr(self, "_solution_prompt"):
                raise
            marked_path = self.temporary_path(f"{Path(self.path).stem}_marked.png")
            solutions_data = self._request_solutions(
                str(marked_path), self._solution_prompt + "\nThe previous response was invalid. Return only valid JSON matching the schema, with every answer ID exactly once.",
            )
            format_repaired = True
        if not solutions_data:
            print("No response received from AI.")
            return {}

        print("Structured AI response received!")
        expected = set(range(1, len(self.answer_units) + 1))
        accepted, missing, issues = inspect_answers(solutions_data.solutions, expected)
        self.answer_warnings = []
        if missing and not format_repaired:
            marked_path = self.temporary_path(f"{Path(self.path).stem}_marked.png")
            repair_prompt = self._solution_prompt + (
                "\nRepair the previous response. Return ONLY these unresolved IDs, "
                f"each exactly once: {missing}. Do not return already resolved IDs."
            )
            try:
                repair = self._request_solutions(str(marked_path), repair_prompt)
                recovered, _, repair_issues = inspect_answers(repair.solutions, missing)
                accepted.update(recovered)
                if repair_issues:
                    self.answer_warnings.extend(repair_issues)
            except Exception as error:
                self.answer_warnings.append(f"Answer repair failed: {error}")
        unresolved = sorted(expected - accepted.keys())
        if issues and not unresolved:
            self.answer_warnings.append("The first AI response required validation/repair.")
        elif issues:
            self.answer_warnings.extend(issues)
        solutions = {}
        for number in sorted(expected):
            solutions[number - 1] = {
                "gap_indices": self.answer_units[number - 1],
                "solution": accepted.get(number, ""),
                "unresolved": number not in accepted,
            }
        if unresolved:
            self.answer_warnings.append(f"Review unresolved answers: {unresolved}.")
        for warning in self.answer_warnings:
            print(f"Warning: {warning}")
        return solutions

    def _report_model_progress(self, message):
        if not self.debug:
            return
        callback = getattr(self, "progress_callback", None)
        if callback is not None:
            callback(message)

    def _ollama_capabilities(self):
        if not hasattr(self, "_local_capabilities"):
            self._report_model_progress(f"Checking Ollama model {self.model_name}")
            try:
                with self._managed_ollama_client() as client:
                    self._local_capabilities = client.show(self.model_name).capabilities or []
            except httpx.TimeoutException as error:
                raise TimeoutError(f"Ollama did not respond while checking model {self.model_name}.") from error
        return self._local_capabilities

    def _chat_with_ollama(self, *, messages, think, options, phase, response_format=None):
        """Read active generations incrementally instead of timing out on a buffered reply."""
        started = time.monotonic()
        content, thinking = [], []
        last_update, previous_state = started, None
        self._report_model_progress(f"{phase}: waiting for Ollama (model loading / image processing)")
        stream = None
        try:
            with self._managed_ollama_client() as client:
                stream = client.chat(model=self.model_name, messages=messages, think=think,
                                     format=response_format, options=options, stream=True)
                try:
                    for chunk in stream:
                        now = time.monotonic()
                        elapsed = now - started
                        if elapsed > OLLAMA_MAX_SECONDS:
                            raise TimeoutError(
                                f"Ollama exceeded the {OLLAMA_MAX_SECONDS}-second generation limit during {phase.lower()}. "
                                "Disable Thinking or select a faster local model, then retry. "
                                "SOLVER_OLLAMA_MAX_SECONDS controls this limit."
                            )
                        message = chunk.message
                        if message.content:
                            content.append(message.content)
                        if message.thinking:
                            thinking.append(message.thinking)
                        state = "writing answer" if content else "thinking" if thinking else "processing image"
                        if state != previous_state or now - last_update >= 1:
                            self._report_model_progress(f"{phase}: {state} · {elapsed:.0f} s")
                            last_update, previous_state = now, state
                        if chunk.done:
                            if self.debug:
                                count = getattr(chunk, "eval_count", None)
                                print(f"Ollama {phase.lower()}: {elapsed:.1f} s, generated tokens: {count}")
                            break
                finally:
                    close_stream = getattr(stream, "close", None)
                    if close_stream is not None:
                        close_stream()
        except httpx.TimeoutException as error:
            raise TimeoutError(
                f"Ollama sent no data for {REQUEST_TIMEOUT_SECONDS} seconds during {phase.lower()} "
                f"(model: {self.model_name}). The model may still be loading or processing the image. "
                "Check Ollama, try disabling Thinking, or increase SOLVER_REQUEST_TIMEOUT_SECONDS."
            ) from error
        except (httpx.ConnectError, ConnectionError) as error:
            raise ConnectionError("Cannot connect to Ollama. Start the local Ollama service and retry.") from error
        return "".join(content), "".join(thinking)

    def _ollama_client(self):
        return ollama.Client(timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10.0))

    @contextmanager
    def _managed_ollama_client(self):
        """Close both older and newer SDK clients without requiring __enter__."""
        client = self._ollama_client()
        try:
            yield client
        finally:
            close = getattr(client, "close", None)
            if not callable(close):
                # Older SDKs only expose cleanup through their HTTPX client.
                close = getattr(getattr(client, "_client", None), "close", None)
            if callable(close):
                close()
