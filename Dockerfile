FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy your code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose default port (optional; useful for local testing)
EXPOSE 8080

ENTRYPOINT ["sh", "-c"]
CMD ["gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:${PORT:-8080}"]

