# Use Python 3.11 slim for a small, stable image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependecies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependecies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Train the ML model during build
RUN python ml_model/train_model.py

# Expose port
EXPOSE 8000

# Render sets $PORT=10000, Synology will use 8000 bu default
CMD gunicorn --workers 3 --bind 0.0.0.0:${PORT:-8000} run:app