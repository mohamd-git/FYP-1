# syntax=docker/dockerfile:1
# CPU-only image for the AGV Rail Inspection PoC (pipeline + dashboard).
FROM python:3.12-slim

# Headless OpenCV only needs libgthread (from glib) at runtime -- NOT the heavy
# libGL / Mesa / LLVM stack -- so this layer stays tiny and flaky-network-friendly.
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel is the large CUDA build).
# Generous retries/timeout so a flaky connection can recover the large wheels.
RUN pip install --no-cache-dir --retries 10 --timeout 120 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.7.1 torchvision==0.22.1

# Remaining Python dependencies (torch/torchvision already satisfied above).
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

# ultralytics pulls in the FULL opencv-python (which needs the heavy libGL / Mesa /
# LLVM stack at runtime). Swap it for the headless build -- identical cv2 API, but
# it links no GUI/GL system libs, so the slim image needs none of them.
RUN pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir --retries 10 --timeout 120 opencv-python-headless

# Application code.
COPY . .

ENV PYTHONUNBUFFERED=1 \
    AGV_DASHBOARD_HOST=0.0.0.0

EXPOSE 5000

# Run the looping pipeline + operator dashboard together.
CMD ["python", "run.py", "--all"]
