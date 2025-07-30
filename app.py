from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import subprocess
import os
import urllib.request
import tempfile

# Model download settings
MODEL_PATH = "models/tinyllama.gguf"
MODEL_URL = "https://huggingface.co/cmp-nct/tinyllama-gguf/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"

# Download the model if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading TinyLLaMA model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded.")

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        # Run the model with the PDF path
        result = subprocess.run(
            ["llama", "-m", MODEL_PATH, "-p", f"Read and summarize this PDF: {tmp_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Remove file after processing
        os.remove(tmp_path)

        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"error": result.stderr.strip(), "status": "llama.cpp failed"}
            )

        return {
            "response": result.stdout.strip(),
            "status": "ok"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "exception"
        }
