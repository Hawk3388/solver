import warnings
warnings.filterwarnings('ignore')
from main import WorksheetSolver
import os
import uuid
import base64
import tempfile
import requests
from flask import Flask, render_template, request, jsonify
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

GAP_DETECTION_MODEL_PATH = "./model/gap_detection_model.pt"
        
if not os.path.exists(GAP_DETECTION_MODEL_PATH):
    url = "https://github.com/Hawk3388/solver/releases/download/v1.1.0/gap_detection_model.pt"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open("./model/gap_detection_model.pt", 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    GAP_DETECTION_MODEL_PATH = "./model/gap_detection_model.pt"

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

        model_name = request.form.get('model_name', 'gemini-2.5-flash')
        local = request.form.get('local', 'false') == 'true'
        think = request.form.get('think', 'true') == 'true'
        debug = request.form.get('debug', 'false') == 'true'

        solver = WorksheetSolver(
            input_path,
            yolo_model_path=GAP_DETECTION_MODEL_PATH,
            llm_model_name=model_name,
            think=think,
            local=local,
            debug=debug
        )
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