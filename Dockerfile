FROM python:3.11-slim

# Install tesseract OCR + English language data
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Render sets PORT env var; default to 10000
ENV PORT=10000
EXPOSE $PORT

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
