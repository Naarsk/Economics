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
        prompt = f"{prompt}\n\n--- DOCUMENT START ---\n{pdf_text}\n--- DOCUMENT END ---\n \n"

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


def build_prompt(text):
    """
    Builds the final prompt for DeepSeek to extract the outlook
    strictly in the required JSON format.
    """
    return f"""
You are a financial analyst.  
Your task: summarize the following investment report, focus on the name of the fund, the date of the report, the outlook on capital distributions, its magnitude in percentage terms

{text}

remember your task: summarize the provided investment report, focus on the name of the fund, the date of the report, the outlook on capital distributions, its magnitude in percentage terms
"""


def build_final_prompt(summary_text):
    return f"""
        You are a financial analyst that must output only a valid JSON object forecasting the outlook on capital distributions in the next year.
        
        Context from the document:
        \"\"\"{summary_text}\"\"\"
        
        Based on the above context, produce strictly one JSON object following this schema:
        {json_schema}
        
        Respond with ONLY the JSON object.
        """


json_schema = """{
  "results": {
    "name": "<fund name>",
    "date": "<yyyymmdd>",
    "outlook": "<increase|stable|decrease>",
    "magnitude": <percentage>,
    "confidence": <number between 0 and 1>,
    "explanation": "<short one-line reasoning>"
  }
}"""
