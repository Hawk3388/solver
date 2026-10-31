'use strict';

const $ = id => document.getElementById(id);
const fileInput = $('file');
const previewGrid = $('preview-grid');
const resultGrid = $('result-grid');
const resultPanel = resultGrid.closest('.panel');
let selectionVersion = 0;
let originalUrls = [];
let currentJob = null;
let currentSnapshot = null;
let cardSignature = '';
let busy = false;
let limits = {max_files: 10, max_file_size: 10 * 1024 * 1024};
let editor = null;
let modalFocus = null;

async function api(url, options = {}) {
    const response = await fetch(url, {cache: 'no-store', ...options});
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`The server returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
}

function jsonOptions(method, data) {
    return {method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)};
}

function showError(message) {
    $('error-msg').textContent = message;
    $('error-msg').style.display = message ? 'block' : 'none';
}

function setBusy(value) {
    busy = value;
    fileInput.disabled = value;
    document.querySelectorAll('.choose-files-btn').forEach(element => { element.disabled = value; });
    previewGrid.querySelectorAll('img').forEach(image => image.setAttribute('aria-disabled', String(value)));
    $('solve-btn').disabled = value;
    $('solve-btn').textContent = value ? 'Processing worksheets…' :
        (fileInput.files.length > 1 ? `Solve ${fileInput.files.length} Worksheets` : 'Solve');
    resultPanel.classList.toggle('is-processing', value);
    $('spinner').style.display = value ? 'flex' : 'none';
}

function layout(grid, count) {
    grid.classList.toggle('single-image', count === 1);
    grid.classList.toggle('multi-image', count > 1);
    grid.style.display = count ? 'grid' : 'none';
}

function button(label, action, className = 'secondary-btn') {
    const element = document.createElement('button');
    element.type = 'button';
    element.className = className;
    element.textContent = label;
    element.addEventListener('click', () => Promise.resolve().then(action).catch(error => showError(error.message)));
    return element;
}

function feedbackLink() {
    const link = document.createElement('a');
    const editorLink = $('editor-report');
    link.className = 'secondary-btn';
    link.href = editorLink.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = 'Report error';
    link.title = 'Report an error on GitHub (opens a new tab)';
    return link;
}

function warningText(warning) {
    return typeof warning === 'string' ? warning : warning.message;
}

function imageCard(src, filename, fullSrc, downloadUrl, item = null) {
    const card = document.createElement('div');
    card.className = 'image-card';
    const image = document.createElement('img');
    image.src = src;
    image.alt = filename;
    image.title = item ? 'Click to edit answers' : 'Click to choose different images';
    image.tabIndex = 0;
    image.setAttribute('role', 'button');
    const open = () => {
        if (item) openEditor(item).catch(error => showError(error.message));
        else if (!busy) fileInput.click();
    };
    image.addEventListener('click', open);
    image.addEventListener('keydown', event => {
        if (['Enter', ' '].includes(event.key)) { event.preventDefault(); open(); }
    });
    card.append(image);
    const name = document.createElement('div');
    name.className = 'image-card-name';
    name.textContent = filename;
    card.append(name);
    const actions = document.createElement('div');
    actions.className = 'card-actions';
    if (item) {
        actions.append(button('Edit answers', () => openEditor(item)));
        actions.append(feedbackLink());
    } else {
        const choose = button('Change images', () => { if (!busy) fileInput.click(); }, 'secondary-btn choose-files-btn');
        choose.disabled = busy;
        actions.append(choose, button('Enlarge', () => openImageModal(fullSrc, filename, downloadUrl, null)));
    }
    card.append(actions);
    return card;
}

function openImageModal(src, filename, downloadUrl, item) {
    modalFocus = document.activeElement;
    $('image-modal-image').src = src;
    $('image-modal-image').alt = filename;
    $('image-modal-caption').textContent = filename;
    $('image-modal-download').href = downloadUrl;
    $('image-modal-download').download = filename;
    $('modal-actions').replaceChildren();
    if (item) {
        $('modal-actions').append(button('Edit answers', () => { closeImageModal(); return openEditor(item); }));
        $('modal-actions').append(feedbackLink());
    }
    $('image-modal').classList.add('open');
    document.body.style.overflow = 'hidden';
    $('image-modal-close').focus();
}

function closeImageModal() {
    $('image-modal').classList.remove('open');
    $('image-modal-image').removeAttribute('src');
    $('image-modal-download').removeAttribute('href');
    $('modal-actions').replaceChildren();
    document.body.style.overflow = '';
    if (modalFocus?.isConnected) modalFocus.focus();
    modalFocus = null;
}

fileInput.addEventListener('change', () => {
    if (busy) return;
    selectionVersion += 1;
    currentJob = null;
    currentSnapshot = null;
    cardSignature = '';
    originalUrls.forEach(url => URL.revokeObjectURL(url));
    originalUrls = [];
    previewGrid.replaceChildren();
    resultGrid.replaceChildren();
    $('job-status-list').replaceChildren();
    $('job-status-list').style.display = 'none';
    $('result-download').style.display = 'none';
    $('result-download').removeAttribute('href');
    $('result-placeholder').style.display = 'flex';
    resultPanel.classList.remove('has-result');
    const files = Array.from(fileInput.files);
    document.querySelector('.main-wrapper').classList.toggle('has-image', files.length > 0);
    $('upload-group').style.display = files.length ? 'none' : 'block';
    layout(previewGrid, files.length);
    layout(resultGrid, 0);
    files.forEach(file => {
        const src = URL.createObjectURL(file);
        originalUrls.push(src);
        previewGrid.append(imageCard(src, file.name, src, src));
    });
    showError(files.length > limits.max_files ? `Select no more than ${limits.max_files} images.` : '');
    setBusy(false);
});

function renderSnapshot(snapshot) {
    currentSnapshot = snapshot;
    const ready = snapshot.items.filter(item => item.state === 'ready');
    const signature = snapshot.items.map(item => `${item.id}:${item.state}:${item.revision}`).join('|');
    $('result-placeholder').style.display = snapshot.items.length ? 'none' : 'flex';
    $('job-progress').max = snapshot.total;
    $('job-progress').value = snapshot.completed;
    const active = snapshot.items.find(item => item.state === 'running') || snapshot.items.find(item => item.state === 'queued');
    $('progress-text').textContent = active ? `${snapshot.completed}/${snapshot.total} complete · ${active.filename}: ${active.stage}` : `${snapshot.completed}/${snapshot.total} processed`;
    const statuses = $('job-status-list');
    statuses.replaceChildren();
    const notices = snapshot.items.filter(item => item.state === 'failed' || item.warnings?.length);
    statuses.style.display = notices.length ? 'block' : 'none';
    notices.forEach(item => {
        const row = document.createElement('div');
        row.className = 'job-status';
        const name = document.createElement('span');
        name.className = 'job-name';
        name.textContent = item.filename;
        const stage = document.createElement('span');
        stage.className = 'job-stage';
        stage.textContent = item.error || item.warnings.map(warningText).join(' ');
        row.append(name, stage);
        if (item.state === 'failed') {
            row.append(button('Retry', () => retryItem(item)));
            if (item.editable) row.append(button('Correct areas', () => openEditor(item)));
            row.append(feedbackLink());
        }
        statuses.append(row);
    });
    if (signature !== cardSignature) {
        resultGrid.replaceChildren();
        ready.forEach(item => resultGrid.append(imageCard(item.preview_url, item.filename, item.image_url, item.download_url, item)));
        layout(resultGrid, snapshot.total);
        resultPanel.classList.toggle('has-result', ready.length > 0);
        snapshot.items.forEach((item, index) => {
            const card = previewGrid.children[index];
            if (card && item.editable) {
                const replacement = imageCard(item.source_preview_url, item.filename, item.source_url, item.source_url);
                card.replaceWith(replacement);
            }
        });
        cardSignature = signature;
    }
    const download = $('result-download');
    download.style.display = ready.length ? 'flex' : 'none';
    if (ready.length === 1) {
        download.href = ready[0].download_url;
        $('result-download-label').textContent = 'Download solved image';
    } else if (ready.length > 1) {
        download.href = snapshot.archive_url;
        $('result-download-label').textContent = `Download ${ready.length} images as ZIP`;
    }
}

async function watchJob(jobId, version) {
    let failures = 0;
    while (currentJob === jobId && selectionVersion === version) {
        try {
            const snapshot = await api(`/api/jobs/${jobId}`);
            if (currentJob !== jobId || selectionVersion !== version) return;
            failures = 0;
            renderSnapshot(snapshot);
            if (snapshot.state === 'finished') { setBusy(false); return; }
        } catch (error) {
            failures += 1;
            showError(`Progress connection interrupted: ${error.message}`);
            if (failures >= 5) { setBusy(false); return; }
        }
        await new Promise(resolve => setTimeout(resolve, Math.min(1000 * (failures + 1), 5000)));
    }
}

async function solveWorksheet() {
    if (busy) return;
    const files = Array.from(fileInput.files);
    if (!files.length || files.length > limits.max_files) return showError(`Select between 1 and ${limits.max_files} images.`);
    if (files.some(file => file.size > limits.max_file_size)) return showError('One or more images exceed the per-file size limit.');
    const version = selectionVersion;
    const data = new FormData();
    files.forEach(file => data.append('files', file));
    data.append('model_name', $('model-name').value);
    data.append('local', $('local-mode').checked);
    data.append('think', $('thinking').checked);
    data.append('debug', $('debug-mode').checked);
    showError('');
    setBusy(true);
    try {
        const snapshot = await api('/solve', {method: 'POST', body: data});
        if (version !== selectionVersion) return;
        currentJob = snapshot.id;
        cardSignature = '';
        renderSnapshot(snapshot);
        await watchJob(snapshot.id, version);
    } catch (error) { showError(error.message); setBusy(false); }
}

async function retryItem(item, geometry) {
    if (!currentJob) return;
    const payload = geometry ? {geometry} : {};
    const snapshot = await api(`/api/jobs/${currentJob}/items/${item.id}/retry`, jsonOptions('POST', payload));
    renderSnapshot(snapshot);
    if (!busy) {
        setBusy(true);
        await watchJob(currentJob, selectionVersion);
    }
}

async function openEditor(item) {
    const jobId = currentJob;
    const documentData = await api(item.document_url);
    if (jobId !== currentJob) return;
    editor = {item, jobId, document: documentData, selectedRegion: documentData.regions[0]?.id || null,
        selectedAnswer: documentData.answers[0]?.id || null, dirty: false, drawing: false, drag: null, saving: false, advanced: false};
    $('editor-dialog').classList.remove('advanced');
    $('editor-advanced').setAttribute('aria-expanded', 'false');
    $('editor-advanced').textContent = 'Adjust areas';
    $('editor-title').textContent = item.filename;
    $('editor-view').value = 'page';
    $('editor-message').textContent = documentData.warnings.length ? documentData.warnings.map(warningText).join('\n') : 'Click any answer to edit it directly on the page.';
    fillEditorControls();
    updateEditorImage();
    $('editor-dialog').showModal();
    renderAnswerFields();
}

// One field per logical answer: grouped lines retain a single complete text.
function renderAnswerFields() {
    const layer = $('answer-fields');
    layer.replaceChildren();
    if (!editor || editor.advanced) return;
    const doc = editor.document;
    const scale = $('editor-stage').getBoundingClientRect().width / doc.width || 1;
    doc.answers.forEach(answer => {
        const regions = doc.regions.filter(region => region.answer_id === answer.id);
        if (!regions.length) return;
        const left = Math.min(...regions.map(region => region.box[0]));
        const top = Math.min(...regions.map(region => region.box[1]));
        const right = Math.max(...regions.map(region => region.box[2]));
        const bottom = Math.max(...regions.map(region => region.box[3]));
        const field = document.createElement('textarea');
        field.className = 'worksheet-answer';
        field.dataset.answerId = answer.id;
        field.setAttribute('aria-label', `Answer ${answer.id}`);
        field.placeholder = 'Type your answer…';
        field.maxLength = 12000;
        field.spellcheck = true;
        field.value = answer.text;
        field.style.left = `${left / doc.width * 100}%`;
        field.style.top = `${top / doc.height * 100}%`;
        field.style.width = `${(right - left) / doc.width * 100}%`;
        field.style.height = `${(bottom - top) / doc.height * 100}%`;
        field.style.fontSize = `${Math.max(12, (answer.font_size || doc.width * .018) * scale)}px`;
        const expand = () => {
            field.style.height = `${(bottom - top) / doc.height * 100}%`;
            field.style.height = `${Math.min(Math.max(field.scrollHeight, 36), Math.max(36, (doc.height - top) * scale))}px`;
        };
        field.addEventListener('focus', expand);
        field.addEventListener('input', () => {
            answer.text = field.value;
            editor.dirty = true;
            $('editor-message').textContent = 'Unsaved changes';
            expand();
        });
        field.addEventListener('blur', () => { field.style.height = `${(bottom - top) / doc.height * 100}%`; });
        layer.append(field);
    });
}

$('editor-advanced').addEventListener('click', () => {
    if (!editor || editor.saving) return;
    editor.advanced = !editor.advanced;
    editor.drawing = false;
    editor.drag = null;
    $('editor-dialog').classList.toggle('advanced', editor.advanced);
    $('editor-advanced').setAttribute('aria-expanded', String(editor.advanced));
    $('editor-advanced').textContent = editor.advanced ? 'Back to typing' : 'Adjust areas';
    $('editor-view').value = 'page';
    fillEditorControls();
    updateEditorImage();
});

function chooseOptions(select, entries, selected) {
    select.replaceChildren();
    entries.forEach(([value, label]) => {
        const option = document.createElement('option');
        option.value = String(value); option.textContent = label;
        select.append(option);
    });
    if (selected != null) select.value = String(selected);
}

function selectedRegion() { return editor?.document.regions.find(region => region.id === editor.selectedRegion); }
function selectedAnswer() { return editor?.document.answers.find(answer => answer.id === editor.selectedAnswer); }

function fillEditorControls() {
    if (!editor) return;
    const doc = editor.document;
    chooseOptions($('answer-select'), doc.answers.map(answer => [answer.id, `Answer ${answer.id}`]), editor.selectedAnswer);
    chooseOptions($('region-select'), doc.regions.map(region => [region.id, `Area ${region.id} · Answer ${region.answer_id}`]), editor.selectedRegion);
    chooseOptions($('region-class'), doc.class_names.map(name => [name, name]), selectedRegion()?.class_name);
    chooseOptions($('region-answer'), doc.answers.map(answer => [answer.id, `Answer ${answer.id}`]), selectedRegion()?.answer_id);
    const answer = selectedAnswer();
    $('answer-text').value = answer?.text || '';
    $('answer-font').value = answer?.font_size ?? '';
    $('answer-offset-x').value = answer?.offset_x ?? 0;
    $('answer-offset-y').value = answer?.offset_y ?? 0;
    for (const id of ['answer-text', 'answer-font', 'answer-offset-x', 'answer-offset-y']) $(id).disabled = !answer;
    const region = selectedRegion();
    const values = region ? [region.box[0], region.box[1], region.box[2] - region.box[0], region.box[3] - region.box[1]] : ['', '', '', ''];
    ['region-x', 'region-y', 'region-width', 'region-height'].forEach((id, index) => { $(id).value = values[index]; $(id).disabled = !region; });
    for (const id of ['region-class', 'region-answer', 'delete-region', 'split-answer']) $(id).disabled = !region;
    $('editor-warnings').textContent = doc.warnings.map(warningText).join('\n');
    drawRegions();
}

function updateEditorImage() {
    if (!editor) return;
    const view = $('editor-view').value;
    const doc = editor.document;
    $('editor-image').src = view === 'page' ? doc.source_url : view === 'original' ? doc.original_url : doc.image_url;
    $('region-canvas').style.display = editor.advanced && view === 'page' ? 'block' : 'none';
    renderAnswerFields();
    drawRegions();
}

function drawRegions() {
    if (!editor) return;
    const canvas = $('region-canvas');
    const doc = editor.document;
    canvas.width = doc.width; canvas.height = doc.height;
    const context = canvas.getContext('2d');
    const scale = doc.width / Math.max(1, canvas.getBoundingClientRect().width);
    context.lineWidth = Math.max(2, 2 * scale);
    context.font = `${Math.max(13, 13 * scale)}px sans-serif`;
    doc.regions.forEach(region => {
        const [x1, y1, x2, y2] = region.box;
        const selected = region.id === editor.selectedRegion;
        context.fillStyle = selected ? '#22d3ee33' : '#8b5cf61c';
        context.strokeStyle = selected ? '#0891b2' : '#7c3aed';
        context.fillRect(x1, y1, x2 - x1, y2 - y1);
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);
        context.fillStyle = '#172554';
        context.fillText(`${region.answer_id} · ${region.class_name}`, x1 + 3 * scale, Math.max(15 * scale, y1 - 4 * scale));
        if (selected) { context.fillStyle = '#0891b2'; context.fillRect(x2 - 8 * scale, y2 - 8 * scale, 8 * scale, 8 * scale); }
    });
    if (editor.drag?.mode === 'new') {
        const [x, y] = editor.drag.start;
        const [endX, endY] = editor.drag.end || editor.drag.start;
        context.strokeStyle = '#0891b2';
        context.strokeRect(x, y, endX - x, endY - y);
    }
}

function canvasPoint(event) {
    const rect = $('region-canvas').getBoundingClientRect();
    return [Math.max(0, Math.min(editor.document.width, (event.clientX - rect.left) / rect.width * editor.document.width)),
        Math.max(0, Math.min(editor.document.height, (event.clientY - rect.top) / rect.height * editor.document.height))];
}

$('region-canvas').addEventListener('pointerdown', event => {
    if (!editor || editor.saving) return;
    const point = canvasPoint(event);
    $('region-canvas').setPointerCapture(event.pointerId);
    if (editor.drawing) { editor.drag = {mode: 'new', start: point, end: point}; return; }
    const hit = [...editor.document.regions].reverse().find(region => {
        const [x1, y1, x2, y2] = region.box;
        return point[0] >= x1 && point[0] <= x2 && point[1] >= y1 && point[1] <= y2;
    });
    if (!hit) return;
    editor.selectedRegion = hit.id; editor.selectedAnswer = hit.answer_id;
    const handle = editor.document.width / Math.max(1, $('region-canvas').getBoundingClientRect().width) * 14;
    const mode = point[0] > hit.box[2] - handle && point[1] > hit.box[3] - handle ? 'resize' : 'move';
    editor.drag = {mode, start: point, box: [...hit.box]};
    fillEditorControls();
});

$('region-canvas').addEventListener('pointermove', event => {
    if (!editor?.drag || editor.saving) return;
    const point = canvasPoint(event);
    const drag = editor.drag;
    if (drag.mode === 'new') { drag.end = point; drawRegions(); return; }
    const region = selectedRegion();
    const [x1, y1, x2, y2] = drag.box;
    if (drag.mode === 'resize') {
        region.box = [x1, y1, Math.min(editor.document.width, Math.max(x1 + 2, point[0])), Math.min(editor.document.height, Math.max(y1 + 2, point[1]))];
    } else {
        const left = Math.max(0, Math.min(editor.document.width - (x2 - x1), x1 + point[0] - drag.start[0]));
        const top = Math.max(0, Math.min(editor.document.height - (y2 - y1), y1 + point[1] - drag.start[1]));
        region.box = [left, top, left + x2 - x1, top + y2 - y1];
    }
    editor.dirty = true;
    fillEditorControls();
});

function endPointer() {
    if (!editor?.drag) return;
    const drag = editor.drag;
    if (drag.mode === 'new') {
        const [x, y] = drag.start; const [endX, endY] = drag.end;
        if (Math.abs(x - endX) > 3 && Math.abs(y - endY) > 3) {
            const answerId = Math.max(0, ...editor.document.answers.map(answer => answer.id)) + 1;
            const regionId = Math.max(0, ...editor.document.regions.map(region => region.id)) + 1;
            editor.document.answers.push({id: answerId, text: '', font_size: null, offset_x: 0, offset_y: 0});
            editor.document.regions.push({id: regionId, answer_id: answerId,
                class_name: editor.document.class_names.includes('line') ? 'line' : editor.document.class_names[0],
                box: [Math.min(x, endX), Math.min(y, endY), Math.max(x, endX), Math.max(y, endY)]});
            editor.selectedRegion = regionId; editor.selectedAnswer = answerId; editor.dirty = true;
        }
    }
    editor.drag = null; editor.drawing = false;
    $('add-region').textContent = 'Draw a new area';
    fillEditorControls();
}
$('region-canvas').addEventListener('pointerup', endPointer);
$('region-canvas').addEventListener('pointercancel', () => { if (editor) { editor.drag = null; drawRegions(); } });

$('answer-select').addEventListener('change', () => { editor.selectedAnswer = Number($('answer-select').value); fillEditorControls(); });
$('region-select').addEventListener('change', () => { editor.selectedRegion = Number($('region-select').value); editor.selectedAnswer = selectedRegion()?.answer_id; fillEditorControls(); });
$('answer-text').addEventListener('input', () => { if (selectedAnswer()) { selectedAnswer().text = $('answer-text').value; editor.dirty = true; } });
for (const [id, key] of [['answer-font', 'font_size'], ['answer-offset-x', 'offset_x'], ['answer-offset-y', 'offset_y']]) {
    $(id).addEventListener('change', () => {
        if (selectedAnswer() && $(id).checkValidity()) {
            selectedAnswer()[key] = $(id).value === '' && key === 'font_size' ? null : Number($(id).value);
            editor.dirty = true;
        }
    });
}
$('region-class').addEventListener('change', () => { selectedRegion().class_name = $('region-class').value; editor.dirty = true; drawRegions(); });
$('region-answer').addEventListener('change', () => {
    const region = selectedRegion();
    const oldId = region.answer_id;
    region.answer_id = Number($('region-answer').value);
    editor.selectedAnswer = region.answer_id;
    if (!editor.document.regions.some(item => item.answer_id === oldId)) {
        const oldAnswer = editor.document.answers.find(answer => answer.id === oldId);
        const target = selectedAnswer();
        target.text = [target.text, oldAnswer?.text].filter(Boolean).join(' ');
        editor.document.answers = editor.document.answers.filter(answer => answer.id !== oldId);
    }
    editor.dirty = true;
    fillEditorControls();
});
for (const id of ['region-x', 'region-y', 'region-width', 'region-height']) {
    $(id).addEventListener('change', () => {
        const region = selectedRegion();
        if (!region) return;
        const values = ['region-x', 'region-y', 'region-width', 'region-height'].map(key => Number($(key).value));
        const [x, y, width, height] = values;
        if (values.every(Number.isFinite) && x >= 0 && y >= 0 && width > 0 && height > 0 && x + width <= editor.document.width && y + height <= editor.document.height) {
            region.box = [x, y, x + width, y + height]; editor.dirty = true; drawRegions();
        } else { $('editor-message').textContent = 'Keep the entire area inside the page.'; fillEditorControls(); }
    });
}
$('add-region').addEventListener('click', () => { editor.drawing = !editor.drawing; $('editor-view').value = 'page'; updateEditorImage(); $('add-region').textContent = editor.drawing ? 'Drag on the page to draw' : 'Draw a new area'; });
$('delete-region').addEventListener('click', () => {
    const region = selectedRegion();
    if (!region || !confirm('Delete this detected area?')) return;
    editor.document.regions = editor.document.regions.filter(item => item.id !== region.id);
    if (!editor.document.regions.some(item => item.answer_id === region.answer_id)) editor.document.answers = editor.document.answers.filter(answer => answer.id !== region.answer_id);
    editor.selectedRegion = editor.document.regions[0]?.id || null;
    editor.selectedAnswer = selectedRegion()?.answer_id || editor.document.answers[0]?.id || null;
    editor.dirty = true; fillEditorControls();
});
$('split-answer').addEventListener('click', () => {
    const region = selectedRegion();
    if (!region) return;
    if (editor.document.regions.filter(item => item.answer_id === region.answer_id).length === 1) {
        $('editor-message').textContent = 'This region already belongs to a separate answer.';
        return;
    }
    const id = Math.max(0, ...editor.document.answers.map(answer => answer.id)) + 1;
    editor.document.answers.push({id, text: '', font_size: null, offset_x: 0, offset_y: 0});
    region.answer_id = id; editor.selectedAnswer = id; editor.dirty = true; fillEditorControls();
});

async function saveEditor() {
    if (!editor || editor.saving) return;
    const editing = editor;
    const used = new Set(editing.document.regions.map(region => region.answer_id));
    editing.document.answers = editing.document.answers.filter(answer => used.has(answer.id) || answer.text.trim());
    const payload = {version: editing.document.version, regions: editing.document.regions, answers: editing.document.answers};
    editing.saving = true;
    $('answer-fields').inert = true;
    $('editor-advanced').disabled = true;
    $('editor-download').disabled = true;
    document.querySelector('.editor-controls').inert = true;
    document.querySelector('.editor-toolbar').inert = true;
    $('editor-save').disabled = true;
    $('editor-message').textContent = 'Rendering your changes…';
    try {
        const saved = await api(editing.item.document_url, jsonOptions('PUT', payload));
        editing.document = saved; editing.dirty = false;
        $('editor-message').textContent = saved.warnings.length ? `Saved. ${saved.warnings.map(warningText).join('\n')}` : 'Saved. Your downloads now contain the corrected result.';
        fillEditorControls();
        updateEditorImage();
        const snapshot = await api(`/api/jobs/${editing.jobId}`);
        if (currentJob === editing.jobId) renderSnapshot(snapshot);
        return true;
    } catch (error) { $('editor-message').textContent = error.message; return false; }
    finally {
        editing.saving = false;
        $('answer-fields').inert = false;
        $('editor-advanced').disabled = false;
        $('editor-download').disabled = false;
        document.querySelector('.editor-controls').inert = false;
        document.querySelector('.editor-toolbar').inert = false;
        $('editor-save').disabled = false;
    }
}
$('editor-save').addEventListener('click', saveEditor);
$('editor-view').addEventListener('change', updateEditorImage);
$('editor-image').addEventListener('load', () => { drawRegions(); renderAnswerFields(); });
$('editor-download').addEventListener('click', async () => {
    if (!editor || editor.saving) return;
    if ((editor.dirty || editor.item.state !== 'ready') && !(await saveEditor())) return;
    const item = currentSnapshot.items.find(item => item.id === editor.item.id);
    const link = document.createElement('a');
    link.href = item.download_url;
    link.download = item.filename;
    document.body.append(link);
    link.click();
    link.remove();
});

function closeEditor() {
    if (editor?.saving) return;
    if (editor?.dirty && !confirm('Discard your unsaved corrections?')) return;
    $('editor-dialog').close(); editor = null;
}
$('editor-close').addEventListener('click', closeEditor);
$('editor-dialog').addEventListener('cancel', event => { event.preventDefault(); closeEditor(); });

async function reprocessGeometry(restore) {
    if (!editor || editor.saving) return;
    if (!confirm('This will replace the current areas and answers and run detection and AI again. Continue?')) return;
    const normalization = editor.document.normalization;
    const rotation = restore ? 0 : ((normalization.rotation || 0) + Number($('page-rotation').value)) % 360;
    const geometry = {rotation, perspective: restore ? false : Boolean(normalization.perspective_corrected)};
    const item = editor.item;
    $('editor-dialog').close(); editor = null;
    await retryItem(item, geometry);
}
$('rotate-page').addEventListener('click', () => reprocessGeometry(false).catch(error => showError(error.message)));
$('restore-page').addEventListener('click', () => reprocessGeometry(true).catch(error => showError(error.message)));

function toggleSettings() {
    const open = $('settings-panel').classList.toggle('open');
    $('settings-overlay').classList.toggle('open', open);
    $('settings-toggle').setAttribute('aria-expanded', String(open));
    $('settings-panel').inert = !open;
}
$('settings-panel').inert = true;
$('image-modal-close').addEventListener('click', closeImageModal);
$('image-modal').addEventListener('click', event => { if (event.target === $('image-modal')) closeImageModal(); });
document.addEventListener('keydown', event => {
    if (!$('image-modal').classList.contains('open')) return;
    if (event.key === 'Escape') closeImageModal();
    if (event.key === 'Tab') {
        const controls = Array.from($('image-modal').querySelectorAll('button, a[href]'));
        const first = controls[0]; const last = controls.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
});
window.addEventListener('beforeunload', event => {
    if (editor?.dirty) { event.preventDefault(); event.returnValue = ''; }
});

api('/api/config').then(config => {
    limits = config;
    document.querySelector('.upload-secondary').textContent = `Up to ${config.max_files} PNG, JPG, WEBP, or BMP images · ${Math.floor(config.max_file_size / (1024 * 1024))} MB per file`;
    document.querySelector('.info-pill').lastChild.textContent = ` Upload up to ${config.max_files} images`;
    const defaults = config.defaults;
    $('model-name').value = defaults.llm_model_name;
    $('local-mode').checked = defaults.local;
    $('thinking').checked = defaults.think;
}).catch(error => showError(`Could not load server settings: ${error.message}`));
