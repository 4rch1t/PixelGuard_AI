FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-般-libsm6 libxext6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates/ ./templates/
COPY runs/ ./runs/

# Expose port
EXPOSE 7860

# Run Flask app
CMD ["python", "app.py"]
