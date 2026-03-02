FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first (~700MB vs 2.5GB with CUDA)
# --no-cache-dir means pip doesn't keep the downloaded files after install
# This keeps the image lean
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app code (not .venv, docs, .git, etc.)
COPY api.py classifier.py name_detector.py notifier.py ./

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
