import re
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ✅ Make sure to download these once
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def get_most_used_words(df: pd.DataFrame, top_n: int = 20):
    """Returns top N most common meaningful words (lemmatized, no stopwords) from explanations."""

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
