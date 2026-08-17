# Hate Speech Detector

Scores a user comment (English or German, emoji included) for hateful
content.

## Live demo

**[mihaicata.github.io/hatespeech](https://mihaicata.github.io/hatespeech/)**
— static page, runs entirely client-side via
[transformers.js](https://github.com/huggingface/transformers.js) and the
quantized ONNX build of
[Horbee/xlm-roberta-base-offensive-comment-classifier](https://huggingface.co/Horbee/xlm-roberta-base-offensive-comment-classifier)
(EN+DE, ~280MB, downloaded once and cached by the browser). Source in
[`docs/index.html`](docs/index.html). Single method (fine-tuned transformer).

## Local Python version — 5 methods + equal-vote ensemble

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8090

Pick which methods to run; each scores the comment independently (0-100
confidence + a hateful/not-hateful vote), shown as a matrix, with an
"Ensemble" row that's the share of selected methods that voted hateful —
every method counts equally regardless of how confident it is.

| Method | File | How it works |
| --- | --- | --- |
| Keyword Matching | `keywords.py`, `methods.py::keyword_method` | Normalizes text (leetspeak, repeated chars) and scans a curated EN+DE slur/insult/threat list. |
| TF-IDF + Logistic Regression | `methods.py::tfidf_method` | Char n-gram TF-IDF (language-agnostic) + linear classifier, trained on `tweet_eval` (EN) + GermEval 2018 (DE). |
| Word2Vec + Logistic Regression | `methods.py::word2vec_method` | Word2Vec trained from scratch on that same corpus; documents are mean-pooled word vectors fed into a classifier. |
| Sentence Embedding k-NN | `methods.py::embedding_knn_method` | Multilingual sentence-transformer embeds the comment; vote is a similarity-weighted k-NN against a curated reference set (`reference_examples.py`) — no classifier training. |
| Fine-tuned Transformer | `hate_speech.py::analyze_comment` | Language-routed transformer classifiers (`facebook/roberta-hate-speech-dynabench-r4-target` EN, `Hate-speech-CNERG/dehatebert-mono-german` DE). |

`training/train_models.py` is the one-off script that built the TF-IDF and
Word2Vec artifacts in `models/` (re-run it if you want to retrain on more
data). `app.py` exposes `GET /api/methods` and `POST /api/analyze`
(`{text, methods: [...]}`); `templates/index.html` is the UI.
