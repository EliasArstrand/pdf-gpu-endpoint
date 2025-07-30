FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    build-essential \
    cmake \
    git \
    curl \
    libopenblas-dev \
    && apt-get clean

# Create working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY app.py .

# Install Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download LLM model (TinyLLaMA GGUF)
RUN mkdir -p models && \
    curl -L -o models/tinyllama.gguf https://huggingface.co/cmp-nct/tinyllama-gguf/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]