# Worksheet Solver

> Detect answer areas, solve worksheet exercises with a vision-language model, and render the answers back onto the original page.

Worksheet Solver combines a custom YOLO detector with Google Gemini or a local Ollama vision model. It understands inline gaps, ruled answer lines, and larger free-response areas, keeps them in reading order, and produces a clean solved worksheet image.

The project includes a modern web interface, a command-line workflow, a Python API, batch processing, and utilities for maintaining and training the detection model.

> [!WARNING]
> AI-generated answers may be incomplete or incorrect. Always review the result before using it for teaching, assessment, or grading.

## Key Features

- Detects inline gaps, ruled lines, and free answer areas with YOLO.
- Groups related writing lines into one logical answer unit.
- Orders answers from top to bottom and left to right.
- Uses structured model responses to map answers back to the correct areas.
- Supports Google Gemini and local Ollama vision models.
- Fits, wraps, and aligns black answer text to the original worksheet layout.
- Locates printed underlines to compensate for imperfect detection boxes.
- Processes up to 10 worksheets per web request.
- Downloads one solved worksheet as PNG or multiple results as a ZIP archive.
- Reuses one thread-safe YOLO runtime per server process.
- Normalizes and validates images before inference.
- Cleans up temporary files after successful and failed jobs.
- Includes dataset preparation, annotation, training, and diagnostic tools.

## Processing Pipeline

```text
Input image(s) → Validate and normalize → Detect answer areas → Group and order
               → OCR and solve          → Map answers         → Render PNG
```

| Stage | What happens |
| --- | --- |
| Validation | The image header, dimensions, pixel count, aspect ratio, and file type are checked. |
| Normalization | EXIF orientation is applied, transparency is flattened, RGB is enforced, perspective is corrected when a clear page boundary exists, and landscape pages are rotated to portrait. |
| Detection | YOLO finds candidate gaps and answer lines; overlapping detections are filtered by confidence. |
| Grouping | Related ruled lines are combined and all answer units are sorted in reading order. |
| Solving | OCR text, the normalized worksheet, and a numbered debug image are sent to the selected vision-language model. |
| Rendering | Structured answers are fitted to the available rows and written onto the worksheet in black. |

## Requirements

- Python 3.10 or newer
- Windows, Linux, or macOS
- PNG, JPG, JPEG, WEBP, or BMP worksheet images
- One solving backend:
  - a Google Gemini API key, or
  - a running Ollama installation with a vision-capable model
- Internet access on first use if no detector model is installed locally

A CUDA-capable GPU is recommended for detector training and large local models. Normal detector inference can also run on CPU.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Hawk3388/solver.git
cd solver
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Or on Linux and macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure a solving backend

For Google Gemini, copy `.env.example` to `.env` and add your API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

For local inference, install and start Ollama, download a vision-capable model, and enable **Local Mode** in the web settings.

### 5. Start the application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The detector is loaded once during server startup. If no weights are available under `model/vX.Y.Z/gap_detection_model.pt`, the application attempts to download the newest compatible release automatically.

## Web Interface

1. Select one or more worksheet images.
2. Optionally adjust the model or inference settings.
3. Click **Solve**.
4. Click a preview to inspect it at a larger size.
5. Download one result as a PNG or multiple successful results as a ZIP archive.

The web application processes worksheets sequentially. A failed file does not discard successful results from the same batch; its filename and error are reported separately.

### Available Settings

| Setting | Description |
| --- | --- |
| LLM Model Name | Gemini model identifier or exact Ollama model name |
| Local Mode | Uses Ollama instead of the Gemini API |
| Thinking | Enables reasoning output when supported by the selected backend |
| Thinking Budget | Controls the maximum reasoning budget sent to supported models |
| Debug Mode | Prints timing and model diagnostics to the server console |
| Experimental Mode | Uses the optional local Transformers pipeline; requires Local Mode |

Page rotation and conservative perspective correction are always enabled in the web application.

### Upload Limits and Image Safety

| Limit | Value |
| --- | ---: |
| Files per request | 10 |
| Size per file | 10 MB |
| Total request size | 100 MB |
| Maximum source pixels | 40 million |
| Maximum normalized output pixels | 16 million |
| Maximum aspect ratio | 6:1 |

These checks prevent highly compressed oversized images from bypassing the upload-size limit. Alpha channels are flattened onto white, all images are converted to RGB, and large images are proportionally downscaled.

> [!CAUTION]
> The server binds to `0.0.0.0` and is therefore reachable from the local network unless blocked by a firewall. Do not expose it to an untrusted network without authentication, TLS, and a reverse proxy.

## Solving Backends

### Google Gemini

Gemini is the default web backend. The application reads `GOOGLE_API_KEY` from the environment or a local `.env` file and defaults to `gemini-3-flash-preview`.

Worksheet images and prompts are sent to Google's API when this backend is used. Review the applicable privacy and retention policies before processing sensitive documents.

### Ollama

Local Mode sends OCR and solving requests to the local Ollama service:

1. Start Ollama.
2. Install a vision-capable model.
3. Enter its exact name under **LLM Model Name**.
4. Enable **Local Mode**.

Verify installed model names with:

```bash
ollama list
```

Text-only models cannot inspect worksheet images.

### Experimental Transformers Mode

Experimental Mode loads a local Hugging Face image-to-text model with 4-bit quantization and JSON-schema enforcement. It imports `torch`, `transformers`, `bitsandbytes`, and `lm-format-enforcer` at runtime; these optional packages are not included in `requirements.txt`.

This mode expects compatible CUDA hardware and significantly more memory than the standard detector workflow.

## Command-Line Interface

Run the interactive CLI:

```bash
python main.py
```

It accepts:

- one image path;
- one directory; or
- multiple image and directory paths separated by semicolons.

A single image opens an interactive detection-and-solving workflow. Directory or multi-file input is processed sequentially and written to `solved_batch/`. Output names are collision-safe and existing files are never overwritten.

The example CLI configuration currently uses the local Ollama model `qwen3.8`. Adjust the `WorksheetSolver` arguments in `main.py` before using a different backend.

## Python API

### Solve one worksheet

```python
from main import WorksheetSolver

with WorksheetSolver(
    path="worksheet.png",
    llm_model_name="gemini-3-flash-preview",
    local=False,
    think=True,
    thinking_budget=2048,
) as solver:
    gaps, image = solver.detect_gaps()
    marked_image = solver.mark_gaps(image, gaps)
    solutions = solver.solve_all_gaps(marked_image)
    solver.fill_gaps_in_image(
        solver.path,
        solutions,
        output_path="worksheet_solved.png",
    )
```

Using `WorksheetSolver` as a context manager guarantees that its private temporary directory is removed after success or failure.

### Solve a batch

```python
from main import solve_batch

results = solve_batch(
    ["worksheets", "extra_sheet.png"],
    output_directory="solved_batch",
    solver_kwargs={
        "llm_model_name": "gemini-3-flash-preview",
        "local": False,
        "think": True,
    },
)

for result in results:
    if result.success:
        print(f"Solved: {result.output_path}")
    else:
        print(f"Failed: {result.source_path}: {result.error}")
```

`solve_batch()` accepts files and directories, deduplicates resolved paths, ignores unsupported files inside directories, and continues after individual failures by default. Use `recursive=True` to include subdirectories or `continue_on_error=False` to stop at the first error.

### `WorksheetSolver` Options

| Parameter | Default | Description |
| --- | --- | --- |
| `path` | required | Source worksheet image |
| `gap_detection_model_path` | automatic | Optional explicit detector weights path |
| `llm_model_name` | `gemini-3-flash-preview` | Gemini, Ollama, or Transformers model identifier |
| `think` | `True` | Enables reasoning where supported |
| `local` | `False` | Uses a local backend instead of Gemini |
| `thinking_budget` | `2048` | Reasoning-token budget |
| `debug` | `False` | Prints diagnostic timing information |
| `experimental` | `False` | Enables the optional Transformers pipeline |
| `detection_runtime` | `None` | Reuses an explicitly supplied detector runtime |
| `max_output_pixels` | `16000000` | Downscales larger normalized images |
| `auto_rotate_page` | `True` | Rotates landscape images to portrait |
| `correct_perspective` | `True` | Applies conservative four-corner page rectification |

## Architecture

The public facade in `main.py` composes three focused pipeline mixins:

```text
main.py
└── WorksheetSolver
    ├── DetectionMixin  ── worksheet_solver/detection.py
    ├── SolvingMixin    ── worksheet_solver/solving.py
    └── RenderingMixin  ── worksheet_solver/rendering.py
```

| Path | Responsibility |
| --- | --- |
| `app.py` | Flask API, upload validation, batch orchestration, and PNG/ZIP responses |
| `templates/index.html` | Responsive web interface, previews, settings, and downloads |
| `main.py` | Public solver facade, model resolution, normalization orchestration, and CLI |
| `worksheet_solver/images.py` | Safe decoding, EXIF handling, RGB conversion, resizing, rotation, and perspective correction |
| `worksheet_solver/detection.py` | YOLO inference, overlap filtering, reading order, answer grouping, and debug marking |
| `worksheet_solver/solving.py` | OCR, prompt construction, Gemini/Ollama calls, and structured answer mapping |
| `worksheet_solver/rendering.py` | Font sizing, wrapping, underline alignment, and final PNG rendering |
| `worksheet_solver/runtime.py` | Process-wide detector cache and inference lock |
| `worksheet_solver/batch.py` | File discovery, collision-safe naming, cleanup, and fault-tolerant sequential processing |
| `worksheet_solver/schemas.py` | Pydantic schemas for structured model responses |
| `fonts/` | Liberation Sans font used for consistent answer rendering |

The detector is cached by its resolved model path. A process-wide lock serializes access to `YOLO.predict()` because the shared model is not assumed to be thread-safe. Multi-process deployments load one detector per worker process.

## Optional Gradio Demo

`demo.py` contains a minimal single-image Gradio interface. Gradio is intentionally not part of the base dependencies:

```bash
pip install gradio
python demo.py
```

| Environment variable | Default | Description |
| --- | --- | --- |
| `PORT` | `7860` | Gradio server port |
| `GRADIO_SHARE` | `false` | Enables a public Gradio share link |

## Detector Development

Detector training is optional. It is only required when adapting the model to new worksheet styles or improving detection quality.

### Tooling Overview

| Script | Purpose |
| --- | --- |
| `prepare_dataset.py` | Creates an initial YOLO dataset using OpenCV heuristics or an existing model |
| `add_to_dataset.py` | Adds files or directories through model-assisted annotation and review |
| `edit_boxes.py` | Reviews and edits existing YOLO boxes interactively |
| `train_yolo.py` | Trains or resumes the worksheet detector |
| `yolo_test.py` | Runs standalone detection, grouping, and visualization diagnostics |
| `simple_boxes.py` | Provides a smaller detector visualization and ordering utility |

The paths and model constants near the top or bottom of these scripts are project-specific and should be reviewed before use.

### 1. Prepare source images

Place images in `raw_images/`, review the configuration at the bottom of `prepare_dataset.py`, and run:

```bash
python prepare_dataset.py
```

Automatically generated annotations are starting points and should be reviewed manually.

### 2. Review or extend annotations

```bash
python edit_boxes.py
python add_to_dataset.py image1.png image2.jpg
```

`add_to_dataset.py` also accepts directories and keeps the training/validation split close to its configured 80/20 target.

> [!IMPORTANT]
> Keep the class IDs in `dataset/data.yaml`, the annotation files, and the trained detector consistent. The runtime treats the exact class name `line` as a ruled answer line; other detected classes become independent answer units.

### 3. Train

```bash
python train_yolo.py
```

Current defaults:

| Parameter | Value |
| --- | --- |
| Base model | `yolo26l.pt` |
| Epochs | `1500` |
| Image size | `640` |
| Batch size | `16` |
| Device | GPU `0` |
| Output directory | `worksheet_yolo/transfer_learning/` |

The best checkpoint is written to:

```text
worksheet_yolo/transfer_learning/weights/best.pt
```

Adjust the model size, epochs, batch size, and device for the available hardware and dataset size.

### 4. Diagnose a trained model

Review `MODEL_PATH` and `IMAGE_PATH` in `yolo_test.py`, then run:

```bash
python yolo_test.py
```

## Build the Windows Executable

Install PyInstaller and build from the project root:

```bash
pip install pyinstaller
pyinstaller solver.spec
```

The executable is written to `dist/`. The current specification bundles the HTML templates; verify that the detector model, answer font, and `.env` configuration are available in the final distribution before publishing it. Never include a real API key in a release.

## Troubleshooting

### No detector model is available

- Confirm that internet access is available for the first-run release lookup.
- Or place weights at `model/vX.Y.Z/gap_detection_model.pt`.
- Or pass `gap_detection_model_path` explicitly through the Python API.

### No answer areas are detected

- Use a clear, upright image with sufficient contrast.
- Confirm that the worksheet style resembles the detector's training data.
- Run `yolo_test.py` and inspect the marked answer units.
- Add representative examples and hard negatives to the training dataset.

### Answers appear on the wrong lines

- Enable Debug Mode and inspect the detector output.
- Check whether tables, borders, or decorative lines resemble answer areas.
- Review the reading order and class labels in the annotations.

### Gemini fails to initialize

- Confirm that `.env` is in the project root.
- Verify that `GOOGLE_API_KEY` is present and valid.
- Confirm that the selected model is available to that API key.

### Ollama requests fail

- Confirm that the Ollama service is running.
- Verify the model name with `ollama list`.
- Use a vision-capable model rather than a text-only model.

### Rendered text does not fit

The renderer reduces the font size within a readable range, wraps text across detected rows, and ellipsizes overflow. Very long answers or incorrectly grouped answer areas may still require a shorter model response or improved annotations.

## Known Limitations

- Accuracy depends on both detector quality and the selected language model.
- The solving prompt is currently tailored to German worksheets.
- Tables, colored backgrounds, handwriting, and unusual line styles may cause false or missed detections.
- Every landscape image is rotated to portrait by default, which may be unsuitable for genuine landscape worksheets.
- PDF input is not supported; convert each page to an image first.
- The web interface does not currently provide manual answer editing.
- Jobs are processed in memory and sequentially; this design favors predictable resource use over maximum throughput.

## License

Worksheet Solver is licensed under the [GNU General Public License v3.0](LICENSE.md).
