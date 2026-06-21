FROM python:3.11-slim

# System dependencies for OpenCV must be installed as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set up Hugging Face compliance (runs under non-root UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install python packages
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files, setting correct ownership
COPY --chown=user . .

# Expose Hugging Face default port
EXPOSE 7860

# Run uvicorn pointing to main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
