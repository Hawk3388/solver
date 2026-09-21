import warnings
warnings.filterwarnings('ignore')
from main import WorksheetSolver, solve_batch
import os
import sys
import base64
import tempfile
from flask import Flask, render_template, request, jsonify
from waitress import serve
import socket
from pathlib import Path
from worksheet_solver.images import (
    ImageNormalizationError,
    validate_image_file,
)

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

app = Flask("solver", template_folder=os.path.join(base_path, 'templates'))
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_BATCH_FILES = 10
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE * MAX_BATCH_FILES
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

print('Loading shared gap detection model...')
DETECTION_RUNTIME = WorksheetSolver.preload_detection_model()
print(f'Gap detection model ready: {DETECTION_RUNTIME.model_path}')

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        'error': 'The upload exceeds the 100 MB total batch limit.'
    }), 413

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    files = request.files.getlist('files')
    if not files:
        files = request.files.getlist('file')
    files = [file for file in files if file.filename]

    if not files:
        return jsonify({'error': 'No file selected.'}), 400
    if len(files) > MAX_BATCH_FILES:
        return jsonify({
            'error': f'A maximum of {MAX_BATCH_FILES} images is allowed per batch.'
        }), 400

    temporary_directory = tempfile.TemporaryDirectory(
        prefix='worksheet_upload_'
    )
    tmp_dir = temporary_directory.name

    try:
        model_name = request.form.get('model_name', 'gemini-3-flash-preview')
        if not model_name.strip():
            return jsonify({'error': 'Model name must not be empty.'}), 400
        local = request.form.get('local', 'false') == 'true'
        think = request.form.get('think', 'true') == 'true'
        try:
            thinking_budget = int(request.form.get('thinking_budget', '2048'))
        except ValueError:
            return jsonify({'error': 'Thinking budget must be a number.'}), 400
        thinking_budget = max(0, min(thinking_budget, 32768))
        debug = request.form.get('debug', 'false') == 'true'
        experimental = request.form.get('experimental', 'false') == 'true'
        auto_rotate_page = (
            request.form.get('auto_rotate_page', 'false') == 'true'
        )
        correct_perspective = (
            request.form.get('correct_perspective', 'false') == 'true'
        )
        if experimental and not local:
            return jsonify({
                'error': 'Experimental mode requires Local Mode.'
            }), 400

        input_paths = []
        original_names = {}
        errors = []
        allowed = ', '.join(sorted(ALLOWED_EXTENSIONS))

        for index, file in enumerate(files, start=1):
            original_name = Path(file.filename).name
            ext = Path(original_name).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                errors.append({
                    'filename': original_name,
                    'error': f'Unsupported file type. Allowed: {allowed}',
                })
                continue

            input_path = Path(tmp_dir) / f"upload_{index}{ext}"
            file.save(input_path)
            if input_path.stat().st_size > MAX_FILE_SIZE:
                errors.append({
                    'filename': original_name,
                    'error': 'The image is larger than the 10 MB per-file limit.',
                })
                input_path.unlink(missing_ok=True)
                continue

            try:
                validate_image_file(input_path)
            except ImageNormalizationError as error:
                errors.append({
                    'filename': original_name,
                    'error': str(error),
                })
                input_path.unlink(missing_ok=True)
                continue

            resolved_path = input_path.resolve()
            input_paths.append(resolved_path)
            original_names[resolved_path] = original_name

        if not input_paths:
            return jsonify({
                'error': 'None of the selected files could be processed.',
                'errors': errors,
            }), 400

        batch_results = solve_batch(
            input_paths,
            tmp_dir,
            solver_kwargs={
                'llm_model_name': model_name,
                'think': think,
                'local': local,
                'thinking_budget': thinking_budget,
                'debug': debug,
                'experimental': experimental,
                'auto_rotate_page': auto_rotate_page,
                'correct_perspective': correct_perspective,
                'detection_runtime': DETECTION_RUNTIME,
            },
            solver_class=WorksheetSolver,
        )

        images = []
        used_download_names = set()
        for result in batch_results:
            original_name = original_names[result.source_path]
            if not result.success:
                errors.append({
                    'filename': original_name,
                    'error': result.error,
                })
                continue

            with open(result.output_path, 'rb') as solved_file:
                image_data = base64.b64encode(solved_file.read()).decode('utf-8')
            base_name = f'{Path(original_name).stem}_solved'
            download_name = f'{base_name}.png'
            suffix = 2
            while download_name.lower() in used_download_names:
                download_name = f'{base_name}_{suffix}.png'
                suffix += 1
            used_download_names.add(download_name.lower())
            images.append({
                'filename': download_name,
                'image': image_data,
            })

        if not images:
            return jsonify({
                'error': 'The selected worksheets could not be solved.',
                'errors': errors,
            }), 422

        response = {'images': images, 'errors': errors}
        if len(images) == 1:
            response['image'] = images[0]['image']
        return jsonify(response), 207 if errors else 200

    except Exception as e:
        return jsonify({'error': f'Processing error: {e}'}), 500

    finally:
        temporary_directory.cleanup()

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f" * Serving Flask app '{app.name}'")
    print(f" * Running on all addresses ({host})")
    print(f" * Running on http://127.0.0.1:{port}")
    print(f" * Running on http://{local_ip}:{port}")
    print("Press CTRL+C to quit")
    serve(app, host=host, port=port)
