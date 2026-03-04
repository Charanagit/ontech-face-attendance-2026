FROM python:3.11-slim

# Install system deps for opencv + onnxruntime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Expose port (Streamlit default)
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "admin_app_embeddings.py", "--server.port=8501", "--server.address=0.0.0.0"]
