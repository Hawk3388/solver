from main import WorksheetSolver
import os
import uuid
import base64
import tempfile
from flask import Flask, render_template, request, jsonify
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected.'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Please upload a valid image file (PNG, JPG, JPEG, WEBP, BMP).'}), 400

    tmp_dir = tempfile.mkdtemp()
    unique_id = uuid.uuid4().hex
    ext = file.filename.rsplit('.', 1)[1].lower()
    input_path = os.path.join(tmp_dir, f"{unique_id}.{ext}")
    output_path = os.path.join(tmp_dir, f"{unique_id}_solved.png")

    try:
        file.save(input_path)

        solver = WorksheetSolver(input_path, llm_model_name="qwen3.5:35b", local=True)
        gaps, img = solver.detect_gaps()

        if not gaps:
            return jsonify({'error': 'No gaps detected in the worksheet.'}), 422

        marked_image = solver.mark_gaps(img, gaps)
        solutions = solver.solve_all_gaps(marked_image)

        if not solutions:
            return jsonify({'error': 'The AI could not find any solutions.'}), 422

        solver.fill_gaps_in_image(input_path, solutions, output_path=output_path)

        with open(output_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({'image': image_data})

    except Exception as e:
        return jsonify({'error': f'Processing error: {e}'}), 500

    finally:
        for f in Path(tmp_dir).glob('*'):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)