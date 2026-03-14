import os
import tempfile
import uuid
import warnings

import gradio as gr
import requests
from PIL import Image

from main import WorksheetSolver

warnings.filterwarnings("ignore")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
GAP_DETECTION_MODEL_PATH = "./model/gap_detection_model.pt"
GAP_MODEL_URL = "https://github.com/Hawk3388/solver/releases/download/v1.1.0/gap_detection_model.pt"


def ensure_gap_model() -> str:
	os.makedirs("./model", exist_ok=True)
	if os.path.exists(GAP_DETECTION_MODEL_PATH):
		return GAP_DETECTION_MODEL_PATH

	with requests.get(GAP_MODEL_URL, stream=True, timeout=60) as response:
		response.raise_for_status()
		with open(GAP_DETECTION_MODEL_PATH, "wb") as model_file:
			for chunk in response.iter_content(chunk_size=8192):
				if chunk:
					model_file.write(chunk)

	return GAP_DETECTION_MODEL_PATH


def _is_allowed_image(filename: str) -> bool:
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def solve_worksheet(image_path: str):
	if not image_path:
		raise gr.Error("Please upload an image first.")

	if not _is_allowed_image(image_path):
		raise gr.Error("Please upload a valid image file (PNG, JPG, JPEG, WEBP, BMP).")

	try:
		model_path = ensure_gap_model()
	except Exception as error:
		raise gr.Error(f"Could not load the gap detection model: {error}") from error

	with tempfile.TemporaryDirectory() as tmp_dir:
		unique_id = uuid.uuid4().hex
		input_path = os.path.join(tmp_dir, f"{unique_id}.png")
		output_path = os.path.join(tmp_dir, f"{unique_id}_solved.png")

		try:
			Image.open(image_path).convert("RGB").save(input_path)

			solver = WorksheetSolver(
				input_path,
				gap_detection_model_path=model_path,
				llm_model_name="gemini-3-flash-preview",
				think=True,
				local=False,
				thinking_budget=2048,
				debug=False,
				experimental=False,
			)

			gaps, detected_image = solver.detect_gaps()
			if not gaps:
				raise gr.Error("No gaps were detected. Please try a clearer worksheet image.")

			marked_image = solver.mark_gaps(detected_image, gaps)
			solutions = solver.solve_all_gaps(marked_image)

			if not solutions:
				raise gr.Error("The AI could not find any solutions.")

			solver.fill_gaps_in_image(input_path, solutions, output_path=output_path)

			solved_image = Image.open(output_path).copy()
			return solved_image

		except Exception as error:
			raise gr.Error(f"Processing error: {error}") from error


def build_app() -> gr.Blocks:
	with gr.Blocks(title="Worksheet Solver", css="""
		.app-shell {max-width: 1200px; margin: 0 auto;}
		.hero {text-align: center; margin: 14px 0 8px;}
		.hero h1 {font-size: 2rem; margin-bottom: 6px;}
		.hero p {opacity: 0.85;}
	""") as demo:
		gr.HTML(
			"""
			<div class='hero'>
				<h1>Worksheet Solver</h1>
				<p>Upload a worksheet image and generate the solved version.</p>
			</div>
			"""
		)

		with gr.Row(elem_classes=["app-shell"]):
			with gr.Column(scale=1):
				image_input = gr.Image(
					type="filepath",
					label="Worksheet Image",
					sources=["upload"],
				)

				solve_button = gr.Button("Solve", variant="primary")

			with gr.Column(scale=1):
				image_output = gr.Image(type="pil", label="Solved Worksheet")

		solve_button.click(
			fn=solve_worksheet,
			inputs=image_input,
			outputs=image_output,
		)

	return demo


demo = build_app()

if __name__ == "__main__":
	demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=True)
