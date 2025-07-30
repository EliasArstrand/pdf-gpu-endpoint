from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import fitz  # PyMuPDF
from llama_cpp import Llama
import os

app = FastAPI()

# Load the model
MODEL_PATH = "models/tinyllama.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8)

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Ask LLM to format it
def ask_llm_to_structure(text):
    prompt = (
        "Extract all artikelnummer, namn, antal sålda, and datum from the following text "
        "and format the result as a JSON array. Only return the JSON array.\n\n"
        f"{text}"
    )

    response = llm(prompt, max_tokens=1024, stop=["</s>"])
    return response['choices'][0]['text'].strip()

@app.post("/extract")
async def extract_info(file: UploadFile = File(...)):
    try:
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Get text from PDF
        text = extract_text_from_pdf(file_path)

        # Use LLM to get JSON
        raw_json = ask_llm_to_structure(text)

        return JSONResponse(content={"data": raw_json, "status": "ok"})

    except Exception as e:
        return JSONResponse(content={"error": str(e), "status": "error"}, status_code=500)

# For local testing
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)