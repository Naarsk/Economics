from Finance.Thesis.functions import extract_pdf_text, query_ollama


def chunk_text(text, max_chars=4000):
    """Split text into chunks of max_chars length, respecting sentence boundaries."""
    chunks = []
    while len(text) > max_chars:
        split_at = text.rfind('.', 0, max_chars) + 1
        if split_at <= 0:
            split_at = max_chars
        chunks.append(text[:split_at].strip())
        text = text[split_at:]
    if text:
        chunks.append(text.strip())
    return chunks


def summarize_chunk(chunk, model="deepseek-r1"):
    """Summarizes a chunk of text focusing only on outlook-relevant content."""
    prompt = f"""
    Summarize the following text focusing ONLY on:
    - capital distribution outlook
    - expected changes (increase/decrease/stable)
    - any magnitude or confidence information
    - relevant dates

    Text:
    {chunk}

    Provide a concise 3-4 sentence summary.
    """
    return query_ollama(prompt=prompt, model=model)


def summarize_pdf(filepath, model="deepseek-r1"):
    """
    Extracts text from a PDF, chunks it, and summarizes each chunk.
    Returns a combined summary to be used as context for the final query.
    """
    pdf_text = extract_pdf_text(filepath)
    if pdf_text.startswith("[ERROR]"):
        return pdf_text

    # Split into chunks
    chunks = chunk_text(pdf_text)

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        summary = summarize_chunk(chunk, model=model)
        summaries.append(summary)

    # Combine all summaries into one compact text
    combined_summary = "\n".join(summaries)
    return combined_summary.strip()

def build_final_prompt(summary_text):
    """
    Builds the final prompt for DeepSeek to extract the outlook
    strictly in the required JSON format.
    """
    return f"""
You are a financial analyst.  
Your task: Based on the following summarized document, extract the OUTLOOK of capital distributions for the next year.

Return ONLY a JSON object that matches this structure exactly:

{{
  "results": {{
    "name": "Example Fund Name",
    "date": "20240630",
    "outlook": "increase",
    "magnitude": "10%",
    "confidence": 0.85,
    "explanation": "Expected distributions will increase due to strong projected exits."
  }}
}}

Rules:
- Only one top-level JSON object is allowed.
- All fields must be filled based on the information available.
- If some information is missing, make a best estimate but still return valid JSON.
- Do NOT include any text, markdown, or explanations outside the JSON.

--- DOCUMENT SUMMARY START ---
{summary_text}
--- DOCUMENT SUMMARY END ---
"""
