import os
import re
import fitz
import pandas as pd
import requests
from matplotlib import pyplot as plt
from collections import Counter
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from Finance.Thesis.prompts import build_summary_prompt, build_json_prompt


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


def make_summary(source_dir, output_dir, selected_pdfs):

    os.makedirs(output_dir, exist_ok=True)   # ✅ ensure directory exists

    total = len(selected_pdfs)

    for i, pdf_file in enumerate(selected_pdfs, start=1):
        try:
            print(f"[{i}/{total}] Processing {pdf_file}...")

            pdf_path = os.path.join(source_dir, pdf_file)
            text = extract_pdf_text(pdf_path)
            final_prompt = build_summary_prompt(text)

            # ✅ Ensure response is converted to string
            response = query_ollama(prompt=final_prompt, model="deepseek-r1")
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


def make_json_from_summary(summary_dir,output_dir):
    for file in os.listdir(summary_dir):
        if file.lower().endswith(".txt"):
            summary_path = os.path.join(summary_dir, file)

            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read()

            # ✅ Clean summaries from <think> sections
            cleaned_summary = remove_think_tags(summary_text)

            # Build prompt using the cleaned summary
            prompt = build_json_prompt(cleaned_summary)

            # Query DeepSeek
            response = query_ollama(prompt=prompt, model="deepseek-r1")

            # Save response directly (no JSON cleaning)
            output_file = os.path.splitext(file)[0] + "_parsed.json"
            output_path = os.path.join(output_dir, output_file)

            with open(output_path, "w", encoding="utf-8") as out:
                out.write(remove_think_tags(response))

            print(f"Processed {file} → {output_file}")
    print("✅ All summaries cleaned and processed into JSON.")


def plot_distributions(df: pd.DataFrame):
    """Plots histogram distributions for outlook_num and confidence."""

    plt.figure(figsize=(12, 5))

    # ✅ Outlook_num distribution
    plt.subplot(1, 2, 1)
    df["outlook_num"].hist(bins=3, rwidth=0.8)
    plt.xticks([-1, 0, 1], ["Decrease (-1)", "Stable (0)", "Increase (1)"])
    plt.title("Distribution of Outlook")
    plt.xlabel("Outlook")
    plt.ylabel("Count")

    # ✅ Confidence distribution
    plt.subplot(1, 2, 2)
    df["confidence"].hist(bins=20, rwidth=0.8)
    plt.title("Distribution of Confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


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


