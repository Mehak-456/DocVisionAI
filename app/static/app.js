// DocVisionAI Dashboard Application Logic

document.addEventListener('DOMContentLoaded', () => {
    // Upload & Preprocessing Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const canvasOverlay = document.getElementById('canvasOverlay');
    const processBtn = document.getElementById('processBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const ocrInfoPanel = document.getElementById('ocrInfoPanel');
    const ocrBoxCount = document.getElementById('ocrBoxCount');
    const jsonOutput = document.getElementById('jsonOutput');
    const copyJsonBtn = document.getElementById('copyJsonBtn');
    const docTypeSelect = document.getElementById('docTypeSelect');

    // Stats Ribbon
    const statOcrTime = document.getElementById('statOcrTime');
    const statVlmTime = document.getElementById('statVlmTime');
    const statConfidence = document.getElementById('statConfidence');

    // Tab switching
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Application State
    let selectedFile = null;
    let ocrResults = [];
    let scaleX = 1;
    let scaleY = 1;
    let activeHoverBox = null;
    let lossChartInstance = null;
    let vramChartInstance = null;

    // Initialize metrics charts on page load
    initMetricsCharts();

    // --- Document Processing & Drag-Drop Handling ---

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('drag-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-active');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (event) => {
            imagePreview.src = event.target.result;
            imagePreview.onload = () => {
                setupCanvas();
                processBtn.disabled = false;
                uploadPrompt.classList.add('hidden');
                previewContainer.classList.remove('hidden');

                ocrResults = [];
                clearCanvas();
                ocrInfoPanel.classList.add('hidden');
            };
        };
        reader.readAsDataURL(file);
    }

    function setupCanvas() {
        const displayWidth = imagePreview.clientWidth;
        const displayHeight = imagePreview.clientHeight;

        canvasOverlay.width = displayWidth;
        canvasOverlay.height = displayHeight;

        scaleX = displayWidth / imagePreview.naturalWidth;
        scaleY = displayHeight / imagePreview.naturalHeight;
    }

    window.addEventListener('resize', () => {
        if (selectedFile && !previewContainer.classList.contains('hidden')) {
            setupCanvas();
            drawAllBboxes();
        }
    });

    function clearCanvas() {
        const ctx = canvasOverlay.getContext('2d');
        ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
    }

    function drawAllBboxes() {
        clearCanvas();
        ocrResults.forEach(item => {
            drawSingleBbox(item, false);
        });
        if (activeHoverBox) {
            drawSingleBbox(activeHoverBox, true);
        }
    }

    function drawSingleBbox(item, isHovered) {
        const ctx = canvasOverlay.getContext('2d');
        const box = item.box;

        ctx.beginPath();
        ctx.moveTo(box[0][0] * scaleX, box[0][1] * scaleY);
        ctx.lineTo(box[1][0] * scaleX, box[1][1] * scaleY);
        ctx.lineTo(box[2][0] * scaleX, box[2][1] * scaleY);
        ctx.lineTo(box[3][0] * scaleX, box[3][1] * scaleY);
        ctx.closePath();

        if (isHovered) {
            ctx.strokeStyle = '#8b5cf6';
            ctx.lineWidth = 3;
            ctx.fillStyle = 'rgba(139, 92, 246, 0.2)';
            ctx.fill();

            const x1 = box[0][0] * scaleX;
            const y1 = box[0][1] * scaleY;
            ctx.fillStyle = '#8b5cf6';
            ctx.font = '12px sans-serif';
            const labelWidth = ctx.measureText(item.text).width + 10;

            ctx.fillRect(x1, y1 - 20, labelWidth, 18);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(item.text, x1 + 5, y1 - 7);
        } else {
            ctx.strokeStyle = '#6366f1';
            ctx.lineWidth = 1.5;
        }
        ctx.stroke();
    }

    canvasOverlay.addEventListener('mousemove', (e) => {
        if (!ocrResults.length) return;

        const rect = canvasOverlay.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        let foundHover = null;
        for (let i = ocrResults.length - 1; i >= 0; i--) {
            const item = ocrResults[i];
            if (isPointInPolygon(mouseX, mouseY, item.box)) {
                foundHover = item;
                break;
            }
        }

        if (activeHoverBox !== foundHover) {
            activeHoverBox = foundHover;
            drawAllBboxes();
        }
    });

    function isPointInPolygon(px, py, box) {
        const x1 = box[0][0] * scaleX;
        const y1 = box[0][1] * scaleY;
        const x2 = box[2][0] * scaleX;
        const y2 = box[2][1] * scaleY;

        const minX = Math.min(x1, x2);
        const maxX = Math.max(x1, x2);
        const minY = Math.min(y1, y2);
        const maxY = Math.max(y1, y2);

        return px >= minX && px <= maxX && py >= minY && py <= maxY;
    }

    // Run Extract Pipeline Button
    processBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        processBtn.disabled = true;
        btnSpinner.classList.remove('hidden');
        processBtn.querySelector('.btn-text').innerText = "Analyzing VLM Context...";
        jsonOutput.innerHTML = `<span class="json-string">// Pipeline initiated. Running EasyOCR and fine-tuned Qwen2.5-VL inference...</span>`;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('doc_type', docTypeSelect.value);

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                ocrResults = data.ocr_results || [];
                setupCanvas();
                drawAllBboxes();

                if (data.ocr_statistics) {
                    ocrBoxCount.innerText = data.ocr_statistics.total_boxes_found;
                    ocrInfoPanel.classList.remove('hidden');
                }

                const vlmTime = data.vlm_statistics?.vlm_latency_sec || data.vlm_statistics?.vlm_latency || 0.05;
                const ocrTime = Math.max(0.05, (data.pipeline_latency_sec || 0.1) - vlmTime);
                statOcrTime.innerText = `${ocrTime.toFixed(3)}s`;
                statVlmTime.innerText = `${vlmTime.toFixed(3)}s`;
                statConfidence.innerText = `${((data.confidence || 0.95) * 100).toFixed(0)}%`;

                const fieldsObj = data.fields || data.structured_data || {};
                jsonOutput.innerHTML = highlightJSON(fieldsObj);

                if (data.authenticity) {
                    renderAuthenticityResults(data.authenticity);
                } else {
                    document.getElementById('authResultsContent').innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">Authenticity detection skipped or failed.</div>`;
                }

                switchTab('extracted-data');
            } else {
                jsonOutput.innerText = `Error: ${data.detail || data.message || 'Pipeline failed processing.'}`;
            }
        } catch (err) {
            console.error(err);
            jsonOutput.innerText = `Execution Error: ${err.message || 'Backend server is offline or failed.'}`;
        } finally {
            processBtn.disabled = false;
            btnSpinner.classList.add('hidden');
            processBtn.querySelector('.btn-text').innerText = "Run Extract Pipeline";
        }
    });

    // Syntax Highlighting for JSON
    function highlightJSON(obj) {
        if (obj === undefined || obj === null) obj = {};
        let jsonStr = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
        if (!jsonStr) jsonStr = '{}';
        jsonStr = jsonStr.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        return jsonStr.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g, function (match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            return `<span class="${cls}">${match}</span>`;
        });
    }

    function renderAuthenticityResults(auth) {
        const container = document.getElementById('authResultsContent');
        let badgeClass = auth.is_authentic ? 'badge-success' : 'badge-danger';
        if (auth.risk_level === 'MEDIUM') badgeClass = 'badge-warning';

        let html = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; color: #fff;">Verdict: <span class="${badgeClass}" style="padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.9rem; font-weight: bold; background: ${auth.is_authentic ? '#10b981' : (auth.risk_level === 'MEDIUM' ? '#f59e0b' : '#f43f5e')}; color: white;">${auth.verdict}</span></h3>
                <div style="font-size: 1.2rem; font-weight: bold; color: ${auth.is_authentic ? '#10b981' : (auth.risk_level === 'MEDIUM' ? '#f59e0b' : '#f43f5e')};">Score: ${auth.authenticity_score}/100</div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 1rem;">
        `;

        if (auth.checks && auth.checks.length) {
            auth.checks.forEach(check => {
                const icon = check.passed ? '✅' : '❌';
                const color = check.passed ? '#10b981' : '#f43f5e';
                html += `
                <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 8px; border-left: 4px solid ${color};">
                    <div style="font-weight: 600; margin-bottom: 0.25rem; color: #fff;">${icon} ${check.name}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">${check.details}</div>
                </div>
                `;
            });
        }

        html += `</div>`;
        container.innerHTML = html;
    }

    // Tabs Navigation
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            switchTab(tabId);
        });
    });

    function switchTab(tabId) {
        tabButtons.forEach(b => {
            if (b.getAttribute('data-tab') === tabId) {
                b.classList.add('active');
            } else {
                b.classList.remove('active');
            }
        });

        tabContents.forEach(content => {
            if (content.id === tabId) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
    }

    // Copy JSON button
    copyJsonBtn.addEventListener('click', () => {
        const rawText = jsonOutput.innerText;
        navigator.clipboard.writeText(rawText).then(() => {
            const originalText = copyJsonBtn.innerText;
            copyJsonBtn.innerText = "Copied!";
            copyJsonBtn.style.background = "#10b981";
            copyJsonBtn.style.color = "#ffffff";
            setTimeout(() => {
                copyJsonBtn.innerText = originalText;
                copyJsonBtn.style.background = "";
                copyJsonBtn.style.color = "";
            }, 1500);
        });
    });

    // Chart.js Metrics
    async function initMetricsCharts() {
        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();

            const steps = data.training_run.loss_history.map(item => `Step ${item.step}`);
            const losses = data.training_run.loss_history.map(item => item.loss);

            const lossCanvas = document.getElementById('lossChart');
            if (lossCanvas) {
                const lossCtx = lossCanvas.getContext('2d');
                if (lossChartInstance) lossChartInstance.destroy();
                lossChartInstance = new Chart(lossCtx, {
                    type: 'line',
                    data: {
                        labels: steps,
                        datasets: [{
                            label: 'Cross-Entropy Loss',
                            data: losses,
                            borderColor: '#6366f1',
                            backgroundColor: 'rgba(99, 102, 241, 0.05)',
                            borderWidth: 2,
                            tension: 0.3,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                            x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                        }
                    }
                });
            }

            const vramCanvas = document.getElementById('vramChart');
            if (vramCanvas) {
                const vramCtx = vramCanvas.getContext('2d');
                if (vramChartInstance) vramChartInstance.destroy();
                vramChartInstance = new Chart(vramCtx, {
                    type: 'bar',
                    data: {
                        labels: data.optimization_comparison.labels,
                        datasets: [{
                            label: 'Inference VRAM (GB)',
                            data: data.optimization_comparison.vram_gb,
                            backgroundColor: ['#f43f5e', '#6366f1', '#10b981'],
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: {
                                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                ticks: { color: '#94a3b8' },
                                title: { display: true, text: 'VRAM Usage (GB)', color: '#94a3b8' }
                            },
                            x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                        }
                    }
                });
            }
        } catch (err) {
            console.error("Failed to fetch or render metrics: ", err);
        }
    }
});
