"""Exports the trained TF-IDF+LR and Word2Vec+LR models to compact
JSON + binary Float32 files for client-side (browser) inference in
docs/index.html — no server, no scikit-learn/gensim needed at runtime.

Run after train_models.py: `python export_for_web.py`
"""
import json
from pathlib import Path

import joblib
import numpy as np
from gensim.models import Word2Vec

MODELS_DIR = Path(__file__).parent.parent / "models"
OUT_DIR = Path(__file__).parent.parent / "docs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def export_tfidf():
    pipeline = joblib.load(MODELS_DIR / "tfidf_lr.joblib")
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    vocab = {ngram: int(idx) for ngram, idx in vec.vocabulary_.items()}
    idf = vec.idf_.astype("<f4")
    coef = clf.coef_[0].astype("<f4")

    (OUT_DIR / "tfidf_vocab.json").write_text(json.dumps({
        "ngram_min": vec.ngram_range[0],
        "ngram_max": vec.ngram_range[1],
        "intercept": float(clf.intercept_[0]),
        "vocab": vocab,
    }))

    with open(OUT_DIR / "tfidf_weights.bin", "wb") as f:
        f.write(idf.tobytes())
        f.write(coef.tobytes())

    print(f"tfidf: {len(vocab)} vocab entries -> "
          f"{(OUT_DIR / 'tfidf_vocab.json').stat().st_size / 1024:.0f}KB json + "
          f"{(OUT_DIR / 'tfidf_weights.bin').stat().st_size / 1024:.0f}KB bin")


def export_word2vec():
    w2v = Word2Vec.load(str(MODELS_DIR / "word2vec.model"))
    lr = joblib.load(MODELS_DIR / "word2vec_lr.joblib")

    words = w2v.wv.index_to_key
    vocab = {w: i for i, w in enumerate(words)}
    vectors = np.stack([w2v.wv[w] for w in words]).astype("<f4")

    (OUT_DIR / "word2vec_vocab.json").write_text(json.dumps({
        "dim": w2v.vector_size,
        "intercept": float(lr.intercept_[0]),
        "coef": lr.coef_[0].astype("<f4").tolist(),
        "vocab": vocab,
    }))

    with open(OUT_DIR / "word2vec_vectors.bin", "wb") as f:
        f.write(vectors.tobytes())

    print(f"word2vec: {len(vocab)} words x {w2v.vector_size}d -> "
          f"{(OUT_DIR / 'word2vec_vocab.json').stat().st_size / 1024:.0f}KB json + "
          f"{(OUT_DIR / 'word2vec_vectors.bin').stat().st_size / 1024:.0f}KB bin")


if __name__ == "__main__":
    export_tfidf()
    export_word2vec()
    print(f"Exported to {OUT_DIR}")
