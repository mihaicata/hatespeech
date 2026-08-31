# Hate Speech Detector

Scores a user comment (English or German, emoji included) for hateful
content, using 5 independent methods plus an equal-vote ensemble.

## Live demo

**[mihaicata.github.io/hatespeech](https://mihaicata.github.io/hatespeech/)**
— fully static, runs 100% client-side (no server) via
[transformers.js](https://github.com/huggingface/transformers.js) and
hand-ported scikit-learn/gensim models. Source in
[`docs/index.html`](docs/index.html); browser-ready model weights in
[`docs/models/`](docs/models/).

## Local Python version

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8090. Same 5 methods, served from `app.py`
(`GET /api/methods`, `POST /api/analyze`) instead of running in-browser.

## The 5 methods

Pick which ones to run; each scores the comment independently (0-100
confidence + a hateful/not-hateful vote), shown as a matrix, with an
"Ensemble" row — the share of selected methods that voted hateful, every
method weighted equally regardless of how confident it individually is.

| Method | Local (Python) | Static demo (JS) | How it works |
| --- | --- | --- | --- |
| Keyword Matching | `keywords.py` | inlined in `docs/index.html` | Normalizes text (leetspeak, repeated chars) and scans a curated EN+DE slur/insult/threat list. |
| TF-IDF + Logistic Regression | `methods.py::tfidf_method` | `docs/models/tfidf_*` | Char n-gram TF-IDF (language-agnostic) + linear classifier, trained on `tweet_eval` (EN) + GermEval 2018 (DE). The JS port is a byte-exact reimplementation of sklearn's `char_wb` analyzer, verified against the Python output. |
| Word2Vec + Logistic Regression | `methods.py::word2vec_method` | `docs/models/word2vec_*` | Word2Vec trained from scratch on that same corpus; documents are mean-pooled word vectors fed into a classifier. |
| Sentence Embedding k-NN | `methods.py::embedding_knn_method` | same, via `Xenova/paraphrase-multilingual-MiniLM-L12-v2` (ONNX) | Multilingual sentence-transformer embeds the comment; vote is a similarity-weighted k-NN against a curated reference set (`reference_examples.py`) — no classifier training. |
| Fine-tuned Transformer | `hate_speech.py::analyze_comment` (2 language-routed models) | `Horbee/xlm-roberta-base-offensive-comment-classifier` (ONNX) | Fine-tuned transformer classifier. The local version routes EN/DE to two separate models; the static demo uses one multilingual model (lighter to ship as ONNX). |

`training/` has the scripts that produced everything: `fetch_data.sh` +
`train_models.py` train the TF-IDF and Word2Vec models and save them to
`models/`; `export_for_web.py` exports those same models to the compact
JSON + binary-`Float32Array` format `docs/index.html` loads directly (no
scikit-learn/gensim needed in the browser — just a hand-rolled dot product).

Note the static demo downloads real models on first use — the sentence
embedder (~118MB) and the transformer classifier (~280MB) are the big ones,
cached by the browser afterward. Keyword/TF-IDF/Word2Vec together add
under 10MB.

## Benchmarking on your own data

[`evaluate_methods.ipynb`](evaluate_methods.ipynb) runs all 5 methods (+
ensemble) over a DataFrame of `text`/`label` rows and reports
accuracy/precision/recall/F1, confusion matrices, and the disagreement
cases. Ships with a small built-in demo set; swap in `pd.read_csv(...)`
for your own, or flip on the notebook's `USE_REAL_TEST_SET` flag to
evaluate against the same held-out `tweet_eval` + GermEval 2018 split
`train_models.py` reports its own numbers on.

```bash
source .venv/bin/activate
jupyter lab   # open evaluate_methods.ipynb
```

The notebook's last section (optional, machine-local) generates a Word
report at `reports/evaluation_report.docx` — metrics table, per-method
confusion matrices, and ROC/precision-recall curves — reusing the actual
`classification_metrics()` / table / curve-plotting code from a separate
local project, `hateblocker-main`, rather than reimplementing it. It only
runs where that project is checked out; the code itself isn't vendored
into this repo.
