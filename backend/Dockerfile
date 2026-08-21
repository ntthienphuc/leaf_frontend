FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONPATH="/app/backend:$PYTHONPATH"

WORKDIR /app

COPY --chown=user backend/requirements.txt requirements.txt
# CUDA-enabled wheels still work on CPU Spaces, while allowing torch.cuda to
# see a GPU immediately when this Space's hardware is upgraded.
ARG PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url ${PYTORCH_CUDA_INDEX} torch==2.5.1 torchvision==0.20.1 && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user backend backend
COPY --chown=user frontend frontend

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
