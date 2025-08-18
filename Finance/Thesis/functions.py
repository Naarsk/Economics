import json
import os
import re
import fitz
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from collections import Counter
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from Finance.Thesis.prompts import build_summary_prompt, build_json_prompt
from Finance.Thesis.palette import palette


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


def clean_outlooks_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the outlooks DataFrame: keep required columns, normalize date,
    and add numeric outlook column."""

    # ✅ keep only required columns (ignore missing ones safely)
    required_cols = ["fund_name", "report_date", "outlook", "magnitude", "confidence", "explanation", "source_file"]
    df = df[[col for col in required_cols if col in df.columns]].copy()

    # ✅ convert date to yyyymmdd
    def normalize_date(val):
        try:
            return pd.to_datetime(str(val), errors="coerce").strftime("%Y%m%d")
        except Exception:
            return None

    df["report_date"] = df["report_date"].apply(normalize_date)

    # ✅ drop rows with invalid dates
    df = df[df["report_date"].notna()]

    # Ensure outlook is a string (take the first element if it's a list)
    df["outlook"] = df["outlook"].apply(lambda x: x[0] if isinstance(x, list) and x else x)

    # ✅ map outlook to numeric values
    outlook_map = {"increase": 1, "stable": 0, "decrease": -1}
    df["outlook_num"] = df["outlook"].map(outlook_map)

    # ✅ drop rows where outlook is not one of the expected values
    df = df[df["outlook_num"].notna()]

    return df


def make_excel_from_json(json_dir, excel_dir, excel_name="clean_outlooks.xlsx"):
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

                    if isinstance(results, dict):
                        # ✅ single dict case
                        results["source_file"] = file
                        data.append(results)

                    elif isinstance(results, list):
                        # ✅ list of dicts case
                        for entry in results:
                            if isinstance(entry, dict):
                                entry["source_file"] = file
                                data.append(entry)

                    else:
                        print(f"[SKIP] {file} -> results not dict or list")

            except Exception as e:
                print(f"[WARNING] Could not parse {file}: {e}")

    # ✅ Create DataFrame from list of dicts
    df = pd.DataFrame(data)

    # Apply cleaning step if you have it
    if "clean_outlooks_df" in globals():
        clean_df = clean_outlooks_df(df)
    else:
        clean_df = df

    # ✅ Save to Excel
    os.makedirs(excel_dir, exist_ok=True)
    clean_df.to_excel(excel_path, index=False)

    print(f"Saved {len(clean_df)} rows to {excel_path}")

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



def plot_avg_outlook_by_year(df: pd.DataFrame,
                             filename: str,
                             start_year=2019,
                             end_year=2025,
                             output_dir=r"C:\Users\leocr\Projects\Economics\Finance\Thesis\Latex\img"):
    """
    Calculates average outlook_num by year (2019–2025), plots it with error bars,
    and saves to the given filename.

    Error bars are inversely proportional to sqrt(n), where n is the number of entries.
    Gaps if <3 entries, but line connects valid points.
    """

    # ✅ Ensure date is datetime
    df["date_dt"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
    df = df[df["date_dt"].notna()]

    # ✅ Extract year and filter range
    df["year"] = df["date_dt"].dt.year
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    # ✅ Compute mean + count per year
    grouped = df.groupby("year")["outlook_num"].agg(["mean", "count"])

    # ✅ Only keep mean if count >= 3
    grouped.loc[grouped["count"] < 3, "mean"] = float("nan")

    # ✅ Reindex to full year range
    years = range(start_year, end_year + 1)
    avg_by_year = grouped["mean"].reindex(years)
    counts = grouped["count"].reindex(years)

    # ✅ Define error bars (inverse sqrt of n)
    errors = 0.5 / np.sqrt(counts.replace(0, np.nan))  # avoid /0

    # ✅ Plot
    plt.figure(figsize=(8, 5))

    # Scatter with error bars
    plt.errorbar(avg_by_year.index, avg_by_year.values,
                 yerr=errors,
                 fmt="o", color=palette["primary_red"], ecolor=palette["dark_gray"], capsize=5)

    # Line only for valid points
    valid = avg_by_year.dropna()
    plt.plot(valid.index, valid.values, "-", color=palette["primary_red"])

    plt.title(f"Average Outlook by Year ({start_year}–{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Average Outlook (1=Increase, 0=Stable, -1=Decrease)")
    plt.grid(True)

    # ✅ Save to file
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {filepath}")
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


def load_sentiment(path, start_year, end_year):
    """Load and aggregate sentiment by year, returning mean, count, and error bands."""
    df = pd.read_excel(path)
    df["date_dt"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
    df = df[df["date_dt"].notna()]

    df["year"] = df["date_dt"].dt.year
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    grouped = df.groupby("year")["outlook_num"].agg(["mean", "count"])
    mean = grouped["mean"].reindex(range(start_year, end_year + 1))
    counts = grouped["count"].reindex(range(start_year, end_year + 1))

    # Errors = 1/sqrt(n), shaded area
    errors = 0.5 / np.sqrt(counts.replace(0, np.nan))
    return mean, errors


def load_pe_irr(path, years):
    """Load PE IRR Excel, return annual averages aligned with years."""
    df = pd.read_excel(path)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0])
    pe_1yr_irr = df.iloc[:, 0] / 100  # convert % to decimal if needed

    pe_annual = pe_1yr_irr.groupby(pe_1yr_irr.index.year).mean()
    return pe_annual.reindex(years)
