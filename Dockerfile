FROM python:3.11-slim

ENV SMARTATTEND_INSIGHTFACE_ROOT=/opt/insightface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV runtime libs + ffmpeg for RTSP / video codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU PyTorch first (keeps the image ~2 GB smaller than the default CUDA wheel pulled by ultralytics)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

COPY . .

# Pre-download the buffalo_l models into the image (optional; comment out for a smaller image)
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='/opt/insightface', providers=['CPUExecutionProvider'], allowed_modules=['detection','recognition'])"

# Persist DB, key, outputs outside the container:  -v smartattend_data:/app/data -v smartattend_out:/app/outputs
VOLUME ["/app/data", "/app/outputs"]
EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
