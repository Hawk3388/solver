# Worksheet Solver

An AI-powered tool that automatically detects and fills in blank gaps on language worksheets (e.g., fill-in-the-blank, cloze tests, grammar exercises). It combines a custom-trained YOLO object detection model with a large language model (Gemini or Ollama) to locate gaps and generate correct answers.

## How It Works

1. **Gap Detection** — A fine-tuned YOLOv26 model scans the worksheet image and detects all fillable blank regions (`freie_stelle`).
2. **Overlap Filtering** — Overlapping detections are filtered using IoU-based non-maximum suppression, keeping only the highest-confidence boxes.
3. **Gap Marking** — Detected gaps are numbered and highlighted with red bounding boxes on the image.
4. **LLM Solving** — The marked image is sent to a multimodal LLM (Google Gemini or a local Ollama model) with a detailed prompt. The LLM analyzes the worksheet context and returns the correct answer for each numbered gap.
5. **Answer Rendering** — Solutions are rendered directly into the original image at the correct positions with dynamically sized text, producing a solved worksheet.

## Project Structure

| File | Description |
|---|---|
| `main.py` | Main application — end-to-end pipeline from image input to solved output |
| `train_yolo.py` | YOLO training script with transfer learning (YOLOv26) |
| `prepare_dataset.py` | Dataset preparation — generates YOLO labels from raw worksheet images |
| `test_yolo.py` | Standalone YOLO inference test script |
| `simple_boxes.py` | OpenCV-based gap detection (rule-based, used for dataset labeling) |
| `mark_coordinates.py` | Utility to visualize specific coordinates on a worksheet |
| `ocr.py` | Experimental OCR-based gap detection using Ollama (deepseek-ocr) |
| `dataset/` | YOLO dataset (images, labels, `data.yaml`) |
| `arbeitsblatt_yolo/` | Training output (weights, metrics, plots) |
| `raw_images/` | Source worksheet images for dataset creation |

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for training; CPU inference is supported)
- Google Gemini API key (for cloud-based solving) **or** a local Ollama installation (for local solving)

### Key Dependencies

- [Ultralytics (YOLOv26)](https://github.com/ultralytics/ultralytics) — object detection
- [Google GenAI SDK](https://pypi.org/project/google-genai/) — Gemini API client
- [Ollama](https://ollama.com/) — local LLM inference (optional)
- OpenCV, Pillow, PyTorch, NumPy

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd solver

# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root for cloud-based solving:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Usage

### Solving a Worksheet

```bash
python main.py
```

You will be prompted to enter the path to a worksheet image. The tool will:
1. Detect all gaps using the YOLO model
2. Display the detected gaps
3. Ask whether to solve them with the LLM
4. Save the solved worksheet as `arbeitsblatt_gelöst.png`

### Programmatic Usage

```python
from main import ArbeitsblattSolver

solver = ArbeitsblattSolver(
    yolo_model_path="best_model.pt",
    llm_model_name="gemini-2.5-flash",  # or an Ollama model name
    local=False                          # True for Ollama, False for Gemini
)

gaps, img = solver.detect_gaps("worksheet.png")
marked = solver.mark_gaps(img, gaps)
solutions = solver.solve_all_gaps(marked)
solver.fill_gaps_in_image("worksheet.png", solutions, "solved.png")
```

## Training Your Own Model

### 1. Prepare the Dataset

Place worksheet images in `raw_images/`, then run:

```bash
python prepare_dataset.py
```

This uses OpenCV-based heuristics to auto-detect gaps and generate YOLO-format labels. Review and manually correct labels in `dataset/labels/` as needed.

### 2. Train the Model

```bash
python train_yolo.py
```

Default training configuration (adjustable in `train_yolo.py`):

| Parameter | Default | Description |
|---|---|---|
| Model size | `l` (large) | Options: `n`, `s`, `m`, `l`, `x` |
| Epochs | 1000 | More epochs = better accuracy but longer training |
| Image size | 640 | Use 1280 for high-resolution worksheets |
| Batch size | 16 | Reduce if running out of GPU memory |
| Device | `0` (GPU) | Set to `'cpu'` for CPU training |

The best model weights are saved to `arbeitsblatt_yolo/transfer_learning/weights/best.pt`.

### 3. Test the Model

```bash
python test_yolo.py
```

## Supported Worksheet Types

The solver works with various fill-in-the-blank worksheet formats, including:

- Cloze tests (Lückentexte)
- Grammar exercises (verb conjugation, articles, cases)
- Vocabulary exercises
- Reading comprehension worksheets

The LLM automatically detects the worksheet language and responds accordingly.

## License

This project is for educational and personal use.
