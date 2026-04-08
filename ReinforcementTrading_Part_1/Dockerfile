FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# Create cache directory for numba to avoid permissions errs in HF spaces
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

COPY . .

# Run the FastAPI server natively
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
