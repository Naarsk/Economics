import requests
import fitz  # PyMuPDF
import os

def extract_pdf_text(filepath):
    """Extracts text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"[ERROR] Could not read PDF: {e}"

def query_ollama(prompt, model="llama3", pdf_path=None):
    """
    Sends a prompt (optionally with PDF content) to the local Ollama LLM and returns the response.

    Parameters:
        prompt (str): The user prompt to query.
        model (str): The model to use (default is 'llama3').
        pdf_path (str): Optional path to a PDF file to include in the context.

    Returns:
        str: The LLM's generated response.
    """
    # Add PDF content if provided
    if pdf_path:
        if not os.path.exists(pdf_path) or not pdf_path.endswith(".pdf"):
            return "[ERROR] Invalid or missing PDF file path."
        pdf_text = extract_pdf_text(pdf_path)
        prompt = f"{prompt}\n\n[The following document is provided as context:]\n{pdf_text}"

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "[No response returned]")
    except requests.exceptions.RequestException as e:
        return f"[ERROR] Request failed: {e}"
    except KeyError:
        return "[ERROR] Unexpected response format from Ollama."

response = query_ollama(
    prompt="Summarize the main arguments in the document.",
    model = "deepseek-r1",
    pdf_path=".\\files\\2e52e4b8-3cce-47d3-a571-d8d108fd139a.pdf"
)
print(response)
