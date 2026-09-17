# Dockerfile for DocVisionAI Document Understanding Engine

# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Prevent writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by OpenCV, EasyOCR, and PDF processors
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directory to mount and save temporary uploaded files
RUN mkdir -p temp_uploads lora_adapters datasets

# Expose FastAPI server port
EXPOSE 8000

# Run server on container start
CMD ["python", "app/main.py"]
