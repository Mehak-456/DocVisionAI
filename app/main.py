import os
import sys
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.configs.config import settings

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DocVisionAI.Main")

app = FastAPI(
    title="DocVisionAI API",
    description="Vision-Language Pipeline for Invoice & ID Card Extraction",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount upload directory as static files so frontend can load processed images
app.mount("/temp_uploads", StaticFiles(directory=settings.temp_dir), name="temp_uploads")

# Include API endpoints
app.include_router(api_router, prefix="/api")

# Serve the static dashboard UI
@app.get("/")
def read_root():
    """Serves the dashboard directly when reaching root."""
    index_path = os.path.join(settings.static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({
        "message": "DocVisionAI API is running.",
        "dashboard_ui": "/static/index.html",
        "gradio_ui": "/gui (if Gradio is installed)"
    })

# Mount main frontend static files
# We do this after routes to ensure static files don't mask active endpoints
if os.path.exists(settings.static_dir):
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

# Mount Gradio Interface (Bonus feature)
try:
    import gradio as gr
    from inference import DocVisionPipeline

    def gradio_extract_fn(image_filepath, schema_type):
        """Wrapper function connecting Gradio UI to Python Pipeline SDK."""
        if not image_filepath:
            return None, {"error": "Please upload a document image."}
            
        try:
            pipeline = DocVisionPipeline()
            # Determine annotated output file name
            annotated_out = image_filepath + ".annotated.jpg"
            
            result = pipeline.run(
                file_path=image_filepath,
                doc_type=schema_type,
                output_annotated_path=annotated_out
            )
            
            return annotated_out, result
        except Exception as e:
            logger.error(f"Gradio pipeline run failed: {e}")
            return None, {"error": str(e)}

    # Build Gradio UI block
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")) as demo:
        gr.Markdown(
            "# 👁️‍🗨️ DocVisionAI Gradio Interface\n"
            "Upload an invoice, Aadhaar card, or PAN card to run deskew/contrast-enhancement, "
            "OCR bounding box detection, and fine-tuned Qwen2.5-VL extraction."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="filepath", label="Upload Document")
                schema_select = gr.Dropdown(
                    choices=["auto", "invoice", "aadhaar", "pan"],
                    value="auto",
                    label="Target Extraction Schema"
                )
                submit_btn = gr.Button("Extract Structured Fields", variant="primary")
            
            with gr.Column(scale=1):
                output_img = gr.Image(type="filepath", label="Annotated OCR Bounding Boxes")
                output_json = gr.JSON(label="Extracted Structured Fields")
                
        submit_btn.click(
            fn=gradio_extract_fn,
            inputs=[input_img, schema_select],
            outputs=[output_img, output_json]
        )

    # Mount Gradio app inside FastAPI at path '/gui'
    app = gr.mount_gradio_app(app, demo, path="/gui")
    logger.info("Gradio UI dashboard successfully mounted at /gui")

except Exception as e:
    logger.warning(f"Gradio mounting skipped: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
