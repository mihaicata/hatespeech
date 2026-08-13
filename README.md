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
[`docs/index.html`](docs/index.html).

## Local Python version

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5060

- `hate_speech.py` — `analyze_comment(text)` detects the language, converts
  emoji to words so their sentiment isn't lost, runs the matching
  language-specific model (`facebook/roberta-hate-speech-dynabench-r4-target`
  for English, `Hate-speech-CNERG/dehatebert-mono-german` for German), and
  returns a mean-pooled embedding vector plus a 0-100 confidence score.
- `app.py` — small Flask API (`POST /api/analyze`) wrapping that function.
- `templates/index.html` — single-page UI to try it out.
