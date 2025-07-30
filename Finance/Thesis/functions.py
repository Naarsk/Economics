import json
import os
import re
import fitz
import pandas as pd
import requests
from matplotlib import pyplot as plt
from collections import Counter
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from Finance.Thesis.schemas import build_summary_prompt, build_json_prompt


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
    total = len(selected_pdfs)
    completed = 0

    for pdf_file in selected_pdfs:
        completed += 1
        print(f"[{completed}/{total}] Processing {pdf_file}...")

        pdf_path = os.path.join(source_dir, pdf_file)
        text = extract_pdf_text(pdf_path)
        final_prompt = build_summary_prompt(text)

        # Query DeepSeek 
        response = query_ollama(prompt=final_prompt, model="deepseek-r1")

        # Save response to a .txt file
        output_filename = os.path.splitext(pdf_file)[0] + "_summary.txt"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response)

        print(f"[{completed}/{total}] ✅ Processed: {pdf_file} → {output_filename}")

    print(f"✅ All {total} documents have been processed.")


def clean_outlooks_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the outlooks DataFrame: keep required columns, normalize date,
    and add numeric outlook column."""

    # ✅ keep only required columns (ignore missing ones safely)
    required_cols = ["name", "date", "outlook", "magnitude", "confidence", "explanation", "source_file"]
    df = df[[col for col in required_cols if col in df.columns]].copy()

    # ✅ convert date to yyyymmdd
    def normalize_date(val):
        try:
            return pd.to_datetime(str(val), errors="coerce").strftime("%Y%m%d")
        except Exception:
            return None

    df["date"] = df["date"].apply(normalize_date)

    # ✅ drop rows with invalid dates
    df = df[df["date"].notna()]

    # Ensure outlook is a string (take the first element if it's a list)
    df["outlook"] = df["outlook"].apply(lambda x: x[0] if isinstance(x, list) and x else x)

    # ✅ map outlook to numeric values
    outlook_map = {"increase": 1, "stable": 0, "decrease": -1}
    df["outlook_num"] = df["outlook"].map(outlook_map)

    # ✅ drop rows where outlook is not one of the expected values
    df = df[df["outlook_num"].notna()]

    return df


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


def plot_avg_outlook_by_quarter(df: pd.DataFrame):
    """Calculates average outlook_num by quarter and plots it."""

    # ✅ Ensure date is datetime
    df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df[df["date_dt"].notna()]

    # ✅ Extract Year-Quarter
    df["quarter"] = df["date_dt"].dt.to_period("Q")

    # ✅ Calculate average per quarter
    avg_by_quarter = df.groupby("quarter")["outlook_num"].mean()

    # ✅ Plot
    plt.figure(figsize=(10, 5))
    avg_by_quarter.plot(marker="o")
    plt.title("Average Outlook by Quarter")
    plt.xlabel("Quarter")
    plt.ylabel("Average Outlook (1=Increase, 0=Stable, -1=Decrease)")
    plt.grid(True)
    plt.show()


def plot_avg_outlook_by_year(df: pd.DataFrame, start_year=2019, end_year=2025):
    """Calculates average outlook_num by year (2019–2025) and plots it."""

    # ✅ Ensure date is datetime
    df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df[df["date_dt"].notna()]

    # ✅ Extract year and filter range
    df["year"] = df["date_dt"].dt.year
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    # ✅ Calculate average per year
    avg_by_year = df.groupby("year")["outlook_num"].mean()

    # ✅ Plot
    plt.figure(figsize=(8, 5))
    avg_by_year.plot(marker="o", color="blue")
    plt.title(f"Average Outlook by Year ({start_year}–{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Average Outlook (1=Increase, 0=Stable, -1=Decrease)")
    plt.grid(True)
    plt.show()


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


def make_excel(json_dir, excel_dir, excel_name = "clean_outlooks.xlsx"):
    excel_path = os.path.join(excel_dir, excel_name)

    data = []
    for file in os.listdir(json_dir):
        print("Processing {}".format(file))
        if file.endswith("_parsed.json"):
            filepath = os.path.join(json_dir, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    results = json_data.get("results", None)

                    # ✅ only keep entries where results is a dict
                    if isinstance(results, dict):
                        results["source_file"] = file
                        data.append(results)
                    else:
                        print(f"[SKIP] {file} -> results is not a dict")

            except Exception as e:
                print(f"[WARNING] Could not parse {file}: {e}")
    # ✅ Create DataFrame from list of dicts
    df = pd.DataFrame(data)
    clean_df = clean_outlooks_df(df)
    # ✅ Save to Excel
    os.makedirs(excel_dir, exist_ok=True)
    clean_df.to_excel(excel_path, index=False)
    print(f"[INFO] Saved {len(df)} entries to {excel_path}")
