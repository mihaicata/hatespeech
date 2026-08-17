"""Five independent hate-speech detection methods plus an equal-vote ensemble.

Each method returns a MethodResult with its own 0-100 confidence and a
binary vote (confidence >= 50). The ensemble in run_methods() averages the
votes with equal weight per method — no method's raw confidence scale gets
to dominate the others.
"""
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np

from hate_speech import _demojize, _detect_language, analyze_comment
from keywords import KEYWORDS, LEET_MAP
from reference_examples import REFERENCE_EXAMPLES

MODELS_DIR = Path(__file__).parent / "models"


@dataclass
class MethodResult:
    id: str
    label: str
    confidence: float
    vote: bool
    detail: str = ""


# --- Method 1: keyword / lexicon matching -----------------------------------

_REPEAT_RE = re.compile(r"(.)\1{2,}")
_NONWORD_RE = re.compile(r"[^a-zäöüßA-ZÄÖÜ\s]")


def _normalize_for_keywords(text: str) -> str:
    t = _demojize(text).lower().translate(LEET_MAP)
    t = _NONWORD_RE.sub(" ", t)
    t = _REPEAT_RE.sub(r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


def keyword_method(text: str) -> MethodResult:
    normalized = _normalize_for_keywords(text)
    matched = sorted({kw for kw in KEYWORDS if kw in normalized})
    confidence = round(100 * (1 - math.exp(-0.9 * len(matched))), 2)
    detail = ", ".join(matched) if matched else "no keyword matches"
    return MethodResult("keyword", "Keyword Matching", confidence, confidence >= 50, detail)


# --- Method 2: TF-IDF (char n-grams) + Logistic Regression ------------------

_tfidf_pipeline = None


def _get_tfidf():
    global _tfidf_pipeline
    if _tfidf_pipeline is None:
        _tfidf_pipeline = joblib.load(MODELS_DIR / "tfidf_lr.joblib")
    return _tfidf_pipeline


def tfidf_method(text: str) -> MethodResult:
    pipeline = _get_tfidf()
    proba = pipeline.predict_proba([_demojize(text)])[0][1]
    confidence = round(float(proba) * 100, 2)
    return MethodResult(
        "tfidf", "TF-IDF + Logistic Regression", confidence, confidence >= 50,
        "char n-grams, trained on EN+DE tweets",
    )


# --- Method 3: Word2Vec (trained on corpus) + Logistic Regression -----------

_TOKEN_RE = re.compile(r"[a-zA-ZäöüßÄÖÜ]+")
_w2v_model = None
_w2v_lr = None


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _get_word2vec():
    global _w2v_model, _w2v_lr
    if _w2v_model is None:
        from gensim.models import Word2Vec

        _w2v_model = Word2Vec.load(str(MODELS_DIR / "word2vec.model"))
        _w2v_lr = joblib.load(MODELS_DIR / "word2vec_lr.joblib")
    return _w2v_model, _w2v_lr


def word2vec_method(text: str) -> MethodResult:
    w2v, lr = _get_word2vec()
    tokens = _tokenize(_demojize(text))
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    vec = np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)
    proba = lr.predict_proba([vec])[0][1]
    confidence = round(float(proba) * 100, 2)
    coverage = f"{len(vecs)}/{len(tokens)} words recognized" if tokens else "no words"
    return MethodResult("word2vec", "Word2Vec + Logistic Regression", confidence, confidence >= 50, coverage)


# --- Method 4: sentence-embedding k-NN similarity vote -----------------------

_sbert_model = None
_reference_embeddings = None


def _get_sbert():
    global _sbert_model, _reference_embeddings
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer

        _sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        ref_texts = [t for t, _ in REFERENCE_EXAMPLES]
        _reference_embeddings = _sbert_model.encode(ref_texts, normalize_embeddings=True, convert_to_numpy=True)
    return _sbert_model, _reference_embeddings


def embedding_knn_method(text: str, k: int = 7) -> MethodResult:
    model, ref_embeddings = _get_sbert()
    query = model.encode([_demojize(text)], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = ref_embeddings @ query  # cosine similarity, both sides normalized
    top_idx = np.argsort(sims)[::-1][:k]
    weights = np.clip(sims[top_idx], 0, None)
    labels = np.array([REFERENCE_EXAMPLES[i][1] for i in top_idx])

    total = weights.sum()
    confidence = round(float((weights * labels).sum() / total) * 100, 2) if total > 0 else 0.0
    nearest_text = REFERENCE_EXAMPLES[top_idx[0]][0]
    detail = f'nearest match ({sims[top_idx[0]]:.2f} sim): "{nearest_text[:50]}"'
    return MethodResult("embedding_knn", "Sentence Embedding k-NN", confidence, confidence >= 50, detail)


# --- Method 5: fine-tuned transformer classifier -----------------------------

def transformer_method(text: str) -> MethodResult:
    result = analyze_comment(text)
    detail = f"language: {result.language.upper()}"
    return MethodResult("transformer", "Fine-tuned Transformer", result.confidence, result.confidence >= 50, detail)


# --- orchestrator -------------------------------------------------------------

METHOD_REGISTRY: list[tuple[str, str, callable]] = [
    ("keyword", "Keyword Matching", keyword_method),
    ("tfidf", "TF-IDF + Logistic Regression", tfidf_method),
    ("word2vec", "Word2Vec + Logistic Regression", word2vec_method),
    ("embedding_knn", "Sentence Embedding k-NN", embedding_knn_method),
    ("transformer", "Fine-tuned Transformer", transformer_method),
]
METHOD_FUNCS = {mid: fn for mid, _, fn in METHOD_REGISTRY}
METHOD_LABELS = {mid: label for mid, label, _ in METHOD_REGISTRY}


def run_methods(text: str, method_ids: list[str]) -> dict:
    valid_ids = [m for m in method_ids if m in METHOD_FUNCS] or list(METHOD_FUNCS)
    results = [METHOD_FUNCS[m](text) for m in valid_ids]

    votes = [1 if r.vote else 0 for r in results]
    vote_share = round(100 * sum(votes) / len(votes), 2) if votes else 0.0
    avg_confidence = round(sum(r.confidence for r in results) / len(results), 2) if results else 0.0

    return {
        "results": [asdict(r) for r in results],
        "ensemble": {
            "vote_share": vote_share,
            "avg_confidence": avg_confidence,
            "verdict": vote_share >= 50,
            "votes_for": sum(votes),
            "total": len(votes),
        },
        "language": _detect_language(text),
    }
