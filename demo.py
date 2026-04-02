import os
import tempfile
import uuid
import warnings
import re

import gradio as gr
import requests
from PIL import Image
from pathlib import Path

from main import WorksheetSolver

warnings.filterwarnings("ignore")

def get_gap_model() -> str:
	download = False

	os.makedirs("./model", exist_ok=True)
	folder_path = Path("./model")
	model_folder_names = [p.name for p in folder_path.iterdir() if p.is_dir()]

	if model_folder_names:
		latest_version = sorted(model_folder_names, key=lambda s: list(map(int, s.lstrip("v").split("."))), reverse=True)[0]
		model_path = folder_path / latest_version / "gap_detection_model.pt"
		if not model_path.exists():
			download = True
	else:
		download = True
	
	release_response = requests.get(RELEASES_URL)
	if release_response.status_code == 200:
		pattern = re.compile(r"<h2[^>]*>(v\d+\.\d+\.\d+)</h2>")
		versions = pattern.findall(release_response.text)
		if not versions:
			raise Exception("Could not determine the latest model version from GitHub releases.")
	else:
		raise Exception(f"Failed to fetch releases from GitHub: {release_response.status_code}")

	for version in versions:
		GAP_MODEL_URL = f"https://github.com/Hawk3388/solver/releases/download/{version}/gap_detection_model.pt"
		if not url_exists(GAP_MODEL_URL):
			continue
		if download:
			gd_model_path = str(folder_path / version / "gap_detection_model.pt")
			with requests.get(GAP_MODEL_URL, stream=True, timeout=60) as response:
				with open(gd_model_path, "wb") as model_file:
					for chunk in response.iter_content(chunk_size=8192):
						if chunk:
							model_file.write(chunk)
			break
		else:
			compare_versions = sorted([latest_version, version], key=lambda s: list(map(int, s.lstrip("v").split("."))), reverse=True)
			newer_version = compare_versions[0]
			if newer_version != latest_version:
				gd_model_path = str(folder_path / newer_version / "gap_detection_model.pt")
				with requests.get(GAP_MODEL_URL, stream=True, timeout=60) as response:
					with open(gd_model_path, "wb") as model_file:
						for chunk in response.iter_content(chunk_size=8192):
							if chunk:
								model_file.write(chunk)
				break
			else:
				gd_model_path = str(model_path)

	return gd_model_path


def url_exists(url: str, timeout: float = 5.0) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return (200 <= r.status_code < 400)
    except requests.RequestException as e:
        return False


def _is_allowed_image(filename: str) -> bool:
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def solve_worksheet(image_path: str):
	if not image_path:
		raise gr.Error("Please upload an image first.")

	if not _is_allowed_image(image_path):
		raise gr.Error("Please upload a valid image file (PNG, JPG, JPEG, WEBP, BMP).")

	with tempfile.TemporaryDirectory() as tmp_dir:
		unique_id = uuid.uuid4().hex
		input_path = os.path.join(tmp_dir, f"{unique_id}.png")
		output_path = os.path.join(tmp_dir, f"{unique_id}_solved.png")

		try:
			Image.open(image_path).convert("RGB").save(input_path)

			solver = WorksheetSolver(
				input_path,
				gap_detection_model_path=MODEL_PATH,
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

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
RELEASES_URL = "https://github.com/Hawk3388/solver/releases"
MODEL_PATH = get_gap_model()

demo = build_app()

if __name__ == "__main__":
	demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=True)
