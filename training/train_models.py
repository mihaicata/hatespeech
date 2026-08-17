"""One-off training script for the classical (non-transformer) methods.

Fetches two small public datasets (English: tweet_eval "hate"; German:
GermEval 2018 offensive-language), combines them into one multilingual
binary-labeled corpus, and trains/saves:

  - models/tfidf_lr.joblib     TF-IDF (char n-grams) + Logistic Regression
  - models/word2vec.model      Word2Vec embeddings trained on the corpus
  - models/word2vec_lr.joblib  Logistic Regression on mean-pooled Word2Vec vectors

Run once from the `training/` directory: `./fetch_data.sh && python train_models.py`
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_DIR = Path(__file__).parent
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

TOKEN_RE = re.compile(r"[a-zA-ZäöüßÄÖÜ]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_data() -> pd.DataFrame:
    en = pd.concat([
        pd.read_parquet(DATA_DIR / "en_train.parquet"),
        pd.read_parquet(DATA_DIR / "en_val.parquet"),
    ])[["text", "label"]]
    en["label"] = en["label"].astype(int)

    de = pd.concat([
        pd.read_parquet(DATA_DIR / "de_train.parquet"),
        pd.read_parquet(DATA_DIR / "de_test.parquet"),
    ])[["text", "binary"]].rename(columns={"binary": "label"})
    de["label"] = (de["label"] == "OFFENSE").astype(int)

    df = pd.concat([en, de], ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def main():
    df = load_data()
    print(f"Total examples: {len(df)} | positive rate: {df['label'].mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.15, random_state=42, stratify=df["label"]
    )

    # --- TF-IDF (char n-grams, language-agnostic) + Logistic Regression ---
    tfidf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=30000)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)),
    ])
    tfidf_pipeline.fit(X_train, y_train)
    print(f"TF-IDF + LogisticRegression test accuracy: {tfidf_pipeline.score(X_test, y_test):.3f}")
    joblib.dump(tfidf_pipeline, MODELS_DIR / "tfidf_lr.joblib")

    # --- Word2Vec (trained on this corpus) + Logistic Regression ---
    tokenized_train = [tokenize(t) for t in X_train]
    w2v = Word2Vec(
        sentences=tokenized_train, vector_size=100, window=5,
        min_count=2, workers=4, epochs=15, sg=1,
    )
    w2v.save(str(MODELS_DIR / "word2vec.model"))

    def doc_vector(tokens: list[str]) -> np.ndarray:
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)

    Xw_train = np.array([doc_vector(t) for t in tokenized_train])
    Xw_test = np.array([doc_vector(tokenize(t)) for t in X_test])

    w2v_lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    w2v_lr.fit(Xw_train, y_train)
    print(f"Word2Vec + LogisticRegression test accuracy: {w2v_lr.score(Xw_test, y_test):.3f}")
    joblib.dump(w2v_lr, MODELS_DIR / "word2vec_lr.joblib")

    print(f"Saved artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
