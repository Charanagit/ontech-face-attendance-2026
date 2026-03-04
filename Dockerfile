FROM python:3.11-slim-bookworm

# Install minimal system deps for opencv + onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy & install requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "admin_app_embeddings.py", "--server.port=8501", "--server.address=0.0.0.0"]
