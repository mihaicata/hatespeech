# Hate Speech Detector

Scores a user comment (English or German, emoji included) for hateful
content. Each comment is embedded with a pretrained transformer and scored
0-100 by that model's classification head.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5050

## How it works

- `hate_speech.py` — `analyze_comment(text)` detects the language, converts
  emoji to words so their sentiment isn't lost, runs the matching
  language-specific model (`facebook/roberta-hate-speech-dynabench-r4-target`
  for English, `Hate-speech-CNERG/dehatebert-mono-german` for German), and
  returns a mean-pooled embedding vector plus a 0-100 confidence score.
- `app.py` — small Flask API (`POST /api/analyze`) wrapping that function.
- `templates/index.html` — single-page UI to try it out.
