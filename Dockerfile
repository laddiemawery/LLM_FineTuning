FROM python:3.11-slim

# System dependencies for OCR and PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY configs/ configs/
COPY scripts/ scripts/
COPY training/ training/
COPY sources/registry.yaml sources/registry.yaml

# Create data directories
RUN mkdir -p \
    sources/textbooks sources/articles \
    sources/training_logs/images sources/training_logs/spreadsheets \
    sources/training_logs/databases sources/training_logs/other \
    data/extracted/textbooks data/extracted/articles data/extracted/training_logs \
    data/chunks \
    data/generated/qa_pairs data/generated/conversations \
    data/generated/completions data/generated/classification \
    data/validated data/training

# Mount points for source data and output
VOLUME ["/app/sources/textbooks", "/app/sources/articles", "/app/sources/training_logs"]
VOLUME ["/app/data"]
VOLUME ["/app/outputs"]

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
CMD ["scripts/run_pipeline.py", "--list-steps"]
