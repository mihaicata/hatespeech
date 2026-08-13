"""Hate speech detection for English and German comments, emoji-aware.

Each comment is embedded with a pretrained transformer (mean-pooled last
hidden state) and scored for hatefulness with that same model's
classification head, expressed as a 0-100 confidence.
"""
from dataclasses import dataclass
from typing import Optional

import emoji
import numpy as np
import torch
from langdetect import LangDetectException, detect
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# One model per supported language: language-specific models substantially
# outperform multilingual ones on this task, so we route by detected
# language instead of using a single shared model.
_MODELS = {
    "en": "facebook/roberta-hate-speech-dynabench-r4-target",
    "de": "Hate-speech-CNERG/dehatebert-mono-german",
}
_DEFAULT_LANG = "en"

_loaded: dict[str, tuple] = {}  # lang -> (tokenizer, model)


def _get_model(lang: str):
    lang = lang if lang in _MODELS else _DEFAULT_LANG
    if lang not in _loaded:
        name = _MODELS[lang]
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        model.eval()
        _loaded[lang] = (tokenizer, model)
    tokenizer, model = _loaded[lang]
    return lang, tokenizer, model


def _detect_language(text: str) -> str:
    try:
        code = detect(text)
    except LangDetectException:
        return _DEFAULT_LANG
    return "de" if code == "de" else _DEFAULT_LANG


def _demojize(text: str) -> str:
    # Turn emoji into readable words (e.g. "🤬" -> "face with symbols on
    # mouth") so the language model can pick up on emoji-carried sentiment
    # instead of dropping/mangling emoji tokens outright.
    described = emoji.demojize(text, delimiters=(" ", " "))
    return described.replace("_", " ").replace(":", "")


def _hate_label_index(id2label: dict) -> int:
    for i, label in id2label.items():
        normalized = label.lower()
        if "hate" in normalized and "non" not in normalized and "not" not in normalized:
            return i
    return 1  # binary hate-speech models conventionally use 1 = positive class


@dataclass
class HateSpeechResult:
    vector: np.ndarray
    confidence: float
    language: str


def analyze_comment(text: str, language: Optional[str] = None) -> HateSpeechResult:
    """Score a user comment for hateful content.

    Args:
        text: the raw comment, English or German, may contain emoji.
        language: force "en" or "de" instead of auto-detecting.

    Returns:
        HateSpeechResult with an embedding vector, a 0-100 confidence score,
        and the language that was used to score the comment.
    """
    if not text or not text.strip():
        return HateSpeechResult(vector=np.zeros(0), confidence=0.0, language=language or _DEFAULT_LANG)

    lang = language or _detect_language(text)
    lang, tokenizer, model = _get_model(lang)

    processed = _demojize(text)
    inputs = tokenizer(processed, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1][0]
    vector = last_hidden.mean(dim=0).numpy()

    hate_idx = _hate_label_index(model.config.id2label)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    confidence = round(probs[hate_idx].item() * 100, 2)

    return HateSpeechResult(vector=vector, confidence=confidence, language=lang)
