FROM python:3.11-slim-bookworm

# Install system deps + execstack for patch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libstdc++6 \
    libgomp1 \
    prelink \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch the .so to clear executable stack flag
RUN execstack -c /app/.venv/lib/python3.11/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "admin_app_embeddings.py", "--server.port=8501", "--server.address=0.0.0.0"]
