import os
import re
import fitz
import pandas as pd
import requests
from collections import Counter
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import random

from Finance.Thesis.D_prompts import build_summary_prompt, build_json_prompt, json_schema


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


def query_ollama(prompt, url = "http://localhost:11434/api/generate", model="llama3"):
    """
    Sends a prompt to the local Ollama LLM and returns the response.

    Parameters:
        prompt (str): The user prompt to query.
        url (str): The url to Ollama (default is localhost)
        model (str): The model to use (default is 'llama3').

    Returns:
        str: The LLM's generated response.
    """

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


def make_summary(source_dir, output_dir, selected_pdfs, model="deepseek-r1", url = "http://localhost:11434/api/generate", variable_of_interest = "the outlook on capital distributions", period = "year following the report date", fund_manager = "Apollo Global Management"):

    os.makedirs(output_dir, exist_ok=True)   # ✅ ensure directory exists

    total = len(selected_pdfs)
    # Get all PDF files from the source directory
    all_pdfs = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]

    # Randomly select n PDFs

    n = len(all_pdfs)
    selected_pdfs = random.sample(all_pdfs, min(n, len(all_pdfs)))

    for i, pdf_file in enumerate(selected_pdfs, start=1):
        try:
            print(f"[{i}/{total}] Processing {pdf_file}...")

            pdf_path = os.path.join(source_dir, pdf_file)
            text = extract_pdf_text(pdf_path)
            final_prompt = build_summary_prompt(text, variable_of_interest, period, fund_manager)

            # ✅ Ensure response is converted to string
            response = query_ollama(prompt=final_prompt, url=url, model=model)
            if not isinstance(response, str):
                response = str(response)

            # ✅ Save response
            output_filename = os.path.splitext(pdf_file)[0] + "_summary.txt"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response)

            print(f"[{i}/{total}] ✅ Saved → {output_filename}")

        except Exception as e:
            print(f"[{i}/{total}] ❌ Failed for {pdf_file}: {e}")

    print(f"✅ Completed processing {total} documents.")


def remove_think_tags(text):
    """Remove all content enclosed in <think>...</think> tags."""
    text = re.sub(r"```json|```", "", text)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def make_json_from_summary(summary_dir,output_dir, model="deepseek-r1", url = "http://localhost:11434/api/generate", variable_of_interest = "the outlook on capital distributions", period = "year following the report date"):
    for file in os.listdir(summary_dir):
        if file.lower().endswith(".txt"):
            summary_path = os.path.join(summary_dir, file)

            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read()

            # ✅ Clean summaries from <think> sections
            cleaned_summary = remove_think_tags(summary_text)

            # Build prompt using the cleaned summary
            prompt = build_json_prompt(cleaned_summary, variable_of_interest=variable_of_interest, period=period, json_schema=json_schema)

            # Query DeepSeek
            response = query_ollama(prompt=prompt, model=model, url=url)

            # Save response directly (no JSON cleaning)
            output_file = os.path.splitext(file)[0] + "_parsed.json"
            output_path = os.path.join(output_dir, output_file)

            with open(output_path, "w", encoding="utf-8") as out:
                out.write(remove_think_tags(response))

            print(f"Processed {file} → {output_file}")
    print("✅ All summaries cleaned and processed into JSON.")


def get_most_used_words(df: pd.DataFrame, top_n: int = 20):
    """Returns top N most common meaningful words (lemmatized, no stopwords) from explanations."""
    # ✅ Make sure to download these once
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')

    # ✅ Combine all text
    text = " ".join(df["explanation"].dropna().astype(str))

    # ✅ Tokenize & clean
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())  # only alphabetic, min 3 chars

    # ✅ Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]

    # ✅ Lemmatize (normalize singular/plural, verb forms, etc.)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    # ✅ Count frequencies
    word_counts = Counter(words)

    # ✅ Return top N as DataFrame
    top_words = word_counts.most_common(top_n)
    return pd.DataFrame(top_words, columns=["word", "count"])


def get_least_used_words(df: pd.DataFrame, last_n: int = 20,min_count: int = 1):
    """Returns last N most common meaningful words (lemmatized, no stopwords) from explanations."""

    # ✅ Combine all text
    text = " ".join(df["explanation"].dropna().astype(str))

    # ✅ Tokenize & clean
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())  # only alphabetic, min 3 chars

    # ✅ Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]

    # ✅ Lemmatize (normalize singular/plural, verb forms, etc.)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    # ✅ Count frequencies
    word_counts = Counter(words)

    # ✅ Convert to DataFrame and sort ascending
    df_counts = pd.DataFrame(word_counts.items(), columns=["word", "count"])
    df_counts = df_counts[df_counts["count"] >= min_count]
    df_counts = df_counts.sort_values(by="count", ascending=True).head(last_n).reset_index(drop=True)

    return df_counts


