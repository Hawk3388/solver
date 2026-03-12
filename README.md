# Worksheet Solver

An AI-powered tool that automatically detects and fills in blank gaps on language worksheets (e.g., fill-in-the-blank, cloze tests, grammar exercises). It combines a custom-trained YOLO object detection model with a multimodal large language model (Google Gemini or a local Ollama model) to locate gaps and generate correct answers.

## How It Works

1. **Gap Detection** — A fine-tuned YOLOv26 model scans the worksheet image and detects all fillable blank regions (`freie_stelle`).
2. **Overlap Filtering** — Overlapping detections are filtered using IoU-based non-maximum suppression, keeping only the highest-confidence boxes.
3. **Reading Order Sorting** — Detected gaps are sorted in natural reading order (top-to-bottom, left-to-right within each line).
4. **Gap Marking** — Gaps are numbered and highlighted with red bounding boxes on the image.
5. **LLM Solving** — The marked image (plus the original) is sent to a multimodal LLM. The LLM analyzes the worksheet context and returns the correct answer for each numbered gap as structured JSON.
6. **Answer Rendering** — Solutions are rendered directly into the original image at the correct positions with dynamically sized text, producing a solved worksheet.

## Project Structure

| File / Directory | Description |
| --- | --- |
| `main.py` | Core library — `WorksheetSolver` class with the full pipeline (detect → mark → solve → fill) |
| `app.py` | Flask web application — REST API and web UI for solving worksheets in the browser |
| `train_yolo.py` | YOLO training script with transfer learning (YOLOv26) |
| `prepare_dataset.py` | Dataset preparation — generates YOLO labels from raw worksheet images |
| `add_to_dataset.py` | Adds individual images to the dataset with auto-detection and manual correction |
| `edit_boxes.py` | Interactive bounding box editor for correcting YOLO-detected boxes |
| `simple_boxes.py` | OpenCV-based rule-based gap detection (used for initial dataset labeling) |
| `yolo_test.py` | Standalone YOLO inference and IoU filtering tests |
| `solver.spec` | PyInstaller spec for building a standalone Windows executable |
| `templates/index.html` | Web UI for the Flask app |
| `model/` | Trained gap detection model (`gap_detection_model.pt`) |
| `dataset/` | YOLO dataset (images, labels, `data.yaml`) |
| `arbeitsblatt_yolo/` | Training output (weights per epoch, metrics) |
| `raw_images/` | Source worksheet images for dataset creation |

### Test Files

| File | Description |
| --- | --- |
| `test_export_mode_to_onnx.py` | Exports the YOLO model to ONNX format |
| `test_onnx_inference.py` | ONNX Runtime inference with `YOLOONNXInference` class |
| `test_onnx_model.py` | Visual testing of ONNX model predictions |
| `test_ollama.py` | Tests Ollama API integration with reasoning models |
| `test_local_thinking_budget.py` | Local LLM thinking token budget control |
| `test_mark_coordinates.py` | Coordinate visualization utility |
| `test_ocr.py` | OCR-based gap detection via Ollama grounding |
| `test_step_vl.py` | Tests Step3-VL-10B vision-language model |

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for training; CPU inference is supported)
- Google Gemini API key (for cloud-based solving) **or** a local [Ollama](https://ollama.com/) installation (for local solving)

### Key Dependencies

- [Ultralytics (YOLOv26)](https://github.com/ultralytics/ultralytics) — object detection
- [Google GenAI SDK](https://pypi.org/project/google-genai/) — Gemini API client (optional)
- [Ollama](https://ollama.com/) — local LLM inference (optional)
- [Flask](https://flask.palletsprojects.com/) — web interface
- OpenCV, Pillow, PyTorch, NumPy, Pydantic

## Web Interface

```bash
python app.py
```

Opens a server on `http://localhost:5000`. The web UI lets you:

- Upload a worksheet image (PNG, JPG, JPEG, WEBP, BMP)
- Configure model settings (model name, thinking mode, local/cloud)
- View and download the solved worksheet

The gap detection model is automatically downloaded from GitHub Releases if not present locally.

### Web Interface — Standalone Executable

1. Download and extract `solver.zip` from the [Releases](https://github.com/Hawk3388/solver/releases) page
2. Rename `.env.example` to `.env` and paste your Google API key for cloud-based solving:

   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

   > Not required when using a local Ollama model.
3. *(Optional)* Replace `model/gap_detection_model.pt` with a different model if needed
4. Run `solver.exe` — a console window will appear with the server address
5. Open the displayed URL (e.g. `http://127.0.0.1:5000`) in your browser

## Installation

```bash
# Clone the repository
git clone https://github.com/Hawk3388/solver.git
cd solver

# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root for cloud-based solving (Gemini):

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Not required when using a local Ollama model.

## Usage

### CLI — Solving a Worksheet

```bash
python main.py
```

You will be prompted to enter the path to a worksheet image. The tool will:

1. Detect all gaps using the YOLO model
2. Display the detected gap coordinates
3. In debug mode, ask whether to solve them with the LLM
4. Save the solved worksheet as `worksheet_solved.png`

### Programmatic Usage

```python
from main import WorksheetSolver

solver = WorksheetSolver(
    path="worksheet.png",
    gap_detection_model_path="./model/gap_detection_model.pt",
    llm_model_name="gemini-2.5-flash",  # or e.g. "qwen3.5:35b" for Ollama
    local=False,                         # True for Ollama, False for Gemini
    think=True,                          # Enable extended thinking/reasoning
    thinking_budget=2048,                # Max thinking tokens (Gemini / experimental)
    debug=False,                         # Debug timing and output
    experimental=False                   # Experimental local pipeline (HuggingFace)
)

gaps, img = solver.detect_gaps()
marked = solver.mark_gaps(img, gaps)
solutions = solver.solve_all_gaps(marked)
solver.fill_gaps_in_image("worksheet.png", solutions, "solved.png")
```

### Building a Standalone Executable

```bash
pyinstaller solver.spec
```

This bundles the Flask web app into a standalone Windows executable (output in `dist/`).

## Training Your Own Model

### 1. Prepare the Dataset

Place worksheet images in `raw_images/`, then run:

```bash
python prepare_dataset.py
```

This uses OpenCV-based heuristics to auto-detect gaps and generate YOLO-format labels. Review and manually correct labels in `dataset/labels/` as needed. Use `edit_boxes.py` for interactive box editing.

To add individual images to an existing dataset:

```bash
python add_to_dataset.py
```

### 2. Train the Model

```bash
python train_yolo.py
```

Default training configuration (adjustable in `train_yolo.py`):

| Parameter | Default | Description |
| --- | --- | --- |
| Model size | `l` (large) | Options: `n`, `s`, `m`, `l`, `x` |
| Epochs | 300 | More epochs = better accuracy but longer training |
| Image size | 640 | Use 1280 for high-resolution worksheets |
| Batch size | 16 | Reduce if running out of GPU memory |
| Device | `0` (GPU) | Set to `'cpu'` for CPU training |

The best model weights are saved to `arbeitsblatt_yolo/transfer_learning/weights/best.pt`.

### 3. Test the Model

```bash
python yolo_test.py
```

### 4. Export to ONNX (optional)

```bash
python test_export_mode_to_onnx.py
```

Exports the model for lightweight deployment with ONNX Runtime (CPU or GPU).

## Supported Worksheet Types

The solver works with various fill-in-the-blank worksheet formats, including:

- Cloze tests (Lückentexte)
- Grammar exercises (verb conjugation, articles, cases)
- Vocabulary exercises
- Reading comprehension worksheets

The LLM automatically detects the worksheet language and responds accordingly.

## License

This project is licensed under the [GNU General Public License v3.0 (GPLv3)](LICENSE.md).
