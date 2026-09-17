# DocVisionAI

> **Vision Language Model for Invoice & Identity Document Understanding**

DocVisionAI is an enterprise-grade document extraction engine. It leverages a hybrid OCR-VLM approach, combining **EasyOCR** for spatial text localizations with **Qwen2.5-VL** fine-tuned via PEFT/LoRA (Low-Rank Adaptation) to extract structured JSON data from business documents and identity cards.

---

## 👁️‍🗨️ System Architecture

```mermaid
sequence-diagram
    autonumber
    Client ->> FastAPI Router: Upload Document (Image/PDF)
    FastAPI Router ->> OCR Engine: Extract Text & Bounding Boxes
    OCR Engine -->> FastAPI Router: OCR Text Context & Bounding Boxes
    FastAPI Router ->> Image Preprocessor: Apply Deskew/Denoise/CLAHE/Resize
    Image Preprocessor -->> FastAPI Router: Normalized Image
    FastAPI Router ->> Qwen2.5-VL Model (LoRA): Preprocessed Image + OCR Prompt
    Qwen2.5-VL Model (LoRA) -->> FastAPI Router: Generates Logits / Structured JSON
    FastAPI Router ->> Client: Returns Clean Structured JSON + Latency + Confidence
```

---

## 📦 Directory Structure

```
DocVisionAI/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── train.py                  # CLI wrapper for model fine-tuning
├── evaluate.py               # CLI wrapper for computing test metrics
├── predict.py                # CLI wrapper for single-document inference
├── inference.py              # Programmatic SDK Pipeline
├── app/
│   ├── main.py               # FastAPI server entrypoint
│   ├── api/
│   │   └── routes.py         # HTTP endpoints (/extract, /predict, /train, /health, /metrics)
│   ├── models/
│   │   └── vlm_model.py      # Qwen2.5-VL PEFT/LoRA adapter loader & confidence logic
│   ├── ocr/
│   │   └── ocr_engine.py     # EasyOCR bounding box extractor & PDF handler
│   ├── preprocessing/
│   │   └── image_processor.py# OpenCV deskewing, denoising, contrast enhancements
│   ├── datasets/
│   │   └── dataset_loader.py # Multi-modal dataset parser & Qwen conversation template
│   ├── training/
│   │   └── train_engine.py   # PyTorch training wrap & prompt masking loss
│   ├── configs/
│   │   ├── config.py         # Pydantic global settings manager
│   │   └── lora_config.json  # Low-rank adaptation hyper-parameters
│   ├── utils/
│   │   ├── helpers.py        # Spacing/JSON parse utilities
│   │   └── benchmark.py      # Speed/Memory execution benchmarker
│   └── static/
│       ├── index.html        # Premium Glassmorphic Web Dashboard
│       ├── styles.css        
│       └── app.js            
└── tests/
    ├── test_api.py           # Integration API tests
    └── test_pipeline.py      # Module pipeline tests
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (Optional, required for real deep learning training/inference. Falls back to CPU or simulation mode automatically).

### Local Installation

1. Clone the project and navigate into it:
   ```bash
   git clone <repository_url> DocVisionAI
   cd DocVisionAI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create/Configure your Environment file `.env`:
   ```env
   USE_SIMULATOR=True  # Set to False to run actual weights loading (needs PyTorch GPU)
   PORT=8000
   HOST=0.0.0.0
   ```

---

## 📁 Dataset Format

To train or fine-tune LoRA adapters, arrange your dataset in the `datasets/` root folder:

```
datasets/
├── invoice/
│   ├── inv_001.jpg
│   └── inv_001.json
├── aadhaar/
│   ├── aadhaar_001.png
│   └── aadhaar_001.json
└── pan/
    ├── pan_001.webp
    └── pan_001.json
```

### Schema Examples (`.json` Annotation)

#### Invoices:
```json
{
  "invoice_number": "INV-2026-001",
  "invoice_date": "2026-07-28",
  "vendor_name": "CloudTech Solutions",
  "GSTIN": "27AAAAA1111A1Z1",
  "buyer_name": "Vivek Chauhan",
  "total_amount": 1628.40,
  "subtotal": 1380.00,
  "tax": 248.40,
  "currency": "USD"
}
```

#### Aadhaar Cards:
```json
{
  "name": "Vivek Chauhan",
  "aadhaar_number": "1234-5678-9012",
  "DOB": "15/08/1998",
  "gender": "Male",
  "address": "Delhi, India"
}
```

#### PAN Cards:
```json
{
  "PAN Number": "ABCDE1234F",
  "Name": "VIVEK CHAUHAN",
  "Father's Name": "R. S. CHAUHAN",
  "DOB": "15/08/1998"
}
```

---

## ⚙️ Model Training

Trigger PEFT/LoRA fine-tuning using the root CLI:

```bash
python train.py --epochs 3 --lr 2e-4 --batch_size 2 --grad_accum 4 --output_dir lora_adapters
```

If no local images are discovered in `datasets/`, the script automatically generates structured mock data partitions in `datasets/splits/` to demonstrate end-to-end execution.

---

## 📊 Evaluation & Metrics

Calculate model performance metrics (Precision, Recall, F1, Exact Match) on evaluation data splits:

```bash
python evaluate.py --eval_data datasets/splits/eval.json --summary_path lora_adapters/training_summary.json
```

This generates `datasets/splits/eval_results.json` and plots training loss curves to `evaluation_loss_curve.png`.

---

## 🔮 Inference & API Usage

### 1. Programmatic SDK (`inference.py`)

Run inference easily in python programs:
```python
from inference import DocVisionPipeline

pipeline = DocVisionPipeline()
result = pipeline.run("datasets/mock_images/mock_doc.png", doc_type="invoice")

print(result["fields"])
```

### 2. Single-Sample CLI (`predict.py`)

Run inference directly from terminal:
```bash
python predict.py --input app/static/invoice.jpg --doc_type invoice --output extraction_output.json
```

### 3. FastAPI Web Server

Start the API server locally:
```bash
python app/main.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser to view the **Glassmorphism Web Dashboard**.

- **Endpoints**:
  - `GET /api/health` - Health metrics and CUDA hardware logs.
  - `POST /api/extract` - Primary document extraction API (Accepts file parameter + doc_type parameter).
  - `POST /api/predict` - Raw model runner (JSON request payload).
  - `POST /api/train` - Trigger background training.
  - `GET /api/metrics` - Fetch dashboard training analytics.
  - `GET /gui` - Interactive **Gradio Web Interface** (if gradio is installed).

---

## 🐳 Docker Deployment

1. Spin up the containerized service:
   ```bash
   docker-compose up --build
   ```

2. Open the dashboard at `http://localhost:8000`.

---

## 🧪 Testing

Execute automated unit tests validating API routes, pre-processors, and VLM operations:
```bash
pytest tests/
# OR
python -m unittest discover -s tests
```

<!-- Updated: 2026-07-29 -->
