# DocVisionAI Makefile - Development & Deployment Shortcuts

.PHONY: install run test train evaluate docker-build docker-up docker-down clean help

help:
	@echo "DocVisionAI CLI Command Shortcuts:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make run           - Run FastAPI development server"
	@echo "  make test          - Run unit and integration tests"
	@echo "  make train         - Run model fine-tuning CLI"
	@echo "  make evaluate      - Run metric evaluation calculations"
	@echo "  make docker-build  - Build local Docker image"
	@echo "  make docker-up     - Start containerized services via docker-compose"
	@echo "  make docker-down   - Stop containerized services"
	@echo "  make clean         - Clean temporary uploads, logs, and python caches"

install:
	pip install -r requirements.txt

run:
	python3 app/main.py

test:
	python3 -m unittest discover -s tests

train:
	python3 train.py

evaluate:
	python3 evaluate.py

docker-build:
	docker build -t docvisionai .

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	rm -rf app/__pycache__ app/api/__pycache__ app/models/__pycache__ app/ocr/__pycache__ app/preprocessing/__pycache__ app/training/__pycache__ app/datasets/__pycache__ app/configs/__pycache__ app/utils/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -f temp_uploads/*
