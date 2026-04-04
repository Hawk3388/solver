import os
import tempfile
import uuid
import warnings

import gradio as gr
from PIL import Image

from main import WorksheetSolver

warnings.filterwarnings("ignore")

def solve_worksheet(image_path: str):
	if not image_path:
		raise gr.Error("Please upload an image first.")

	with tempfile.TemporaryDirectory() as tmp_dir:
		unique_id = uuid.uuid4().hex
		input_path = os.path.join(tmp_dir, f"{unique_id}.png")
		output_path = os.path.join(tmp_dir, f"{unique_id}_solved.png")

		try:
			Image.open(image_path).convert("RGB").save(input_path)

			solver = WorksheetSolver(
				input_path,
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
