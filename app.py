from flask import Flask, jsonify, render_template, request

from hate_speech import analyze_comment

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "text is required"}), 400

    result = analyze_comment(text)
    return jsonify(
        {
            "confidence": result.confidence,
            "language": result.language,
            "vector_dim": int(result.vector.shape[0]),
            "vector_preview": result.vector[:8].tolist(),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5060)
