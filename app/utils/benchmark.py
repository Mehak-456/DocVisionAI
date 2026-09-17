#!/usr/bin/env python
"""
DocVisionAI: Performance & Latency Benchmarking Utility.
Measures execution time and memory footprint of the OCR & VLM pipelines.
"""
import os
import sys
import time
import logging
from typing import Dict, Any

# Add root directory to PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference import DocVisionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DocVisionAI.Benchmark")

def run_benchmark(image_path: str, iterations: int = 5, use_simulator: bool = True) -> Dict[str, Any]:
    logger.info(f"Starting pipeline benchmark: Iterations={iterations}, SimulatorMode={use_simulator}")
    
    # Initialize pipeline
    t_start = time.time()
    pipeline = DocVisionPipeline(use_simulator=use_simulator)
    init_time = time.time() - t_start
    logger.info(f"Pipeline engine initialization time: {init_time:.3f}s")
    
    latencies = []
    ocr_boxes_counts = []
    
    # Warmup run
    logger.info("Executing pipeline warmup run...")
    try:
        pipeline.run(image_path, doc_type="invoice")
    except Exception as e:
        logger.error(f"Warmup run failed: {e}")
        return {"success": False, "error": str(e)}

    # Benchmark loop
    for i in range(iterations):
        logger.info(f"Benchmark Iteration {i+1}/{iterations}...")
        t0 = time.time()
        result = pipeline.run(image_path, doc_type="invoice")
        elapsed = time.time() - t0
        latencies.append(elapsed)
        ocr_boxes_counts.append(result.get("ocr_boxes_found", 0))
        
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    # Estimate VRAM/RAM usage
    import psutil
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    
    vram_usage_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram_usage_gb = torch.cuda.memory_allocated(0) / (1024**3)
    except ImportError:
        pass
        
    metrics = {
        "success": True,
        "iterations": iterations,
        "init_time_sec": round(init_time, 3),
        "latencies": [round(l, 3) for l in latencies],
        "avg_latency_sec": round(avg_latency, 3),
        "min_latency_sec": round(min_latency, 3),
        "max_latency_sec": round(max_latency, 3),
        "avg_ocr_boxes": sum(ocr_boxes_counts) / len(ocr_boxes_counts),
        "system_ram_mb": round(ram_usage_mb, 2),
        "cuda_vram_gb": round(vram_usage_gb, 3)
    }
    
    logger.info("==========================================")
    logger.info("BENCHMARK PERFORMANCE RESULTS")
    logger.info("==========================================")
    logger.info(f"Average Pipeline Latency: {metrics['avg_latency_sec']:.3f}s")
    logger.info(f"Minimum Latency:          {metrics['min_latency_sec']:.3f}s")
    logger.info(f"Maximum Latency:          {metrics['max_latency_sec']:.3f}s")
    logger.info(f"RAM Utilization:          {metrics['system_ram_mb']:.1f} MB")
    logger.info(f"CUDA VRAM Allocation:     {metrics['cuda_vram_gb']:.3f} GB")
    logger.info("==========================================")
    
    return metrics

def main():
    # Find a dummy sample image to use
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_img = os.path.join(base_dir, "app", "static", "invoice.jpg")
    
    if not os.path.exists(sample_img):
        # Create a small dummy image for testing
        from PIL import Image
        os.makedirs(os.path.dirname(sample_img), exist_ok=True)
        img = Image.new("RGB", (300, 400), color="white")
        img.save(sample_img)
        logger.info(f"Created temporary benchmark image at {sample_img}")
        
    # Run benchmark in simulator mode
    run_benchmark(sample_img, iterations=3, use_simulator=True)

if __name__ == "__main__":
    main()
