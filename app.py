from flask import Flask, jsonify, render_template, request

from methods import METHOD_REGISTRY, run_methods

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/methods")
def methods_list():
    return jsonify([{"id": mid, "label": label} for mid, label, _ in METHOD_REGISTRY])


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    selected = data.get("methods") or [mid for mid, _, _ in METHOD_REGISTRY]

    if not text.strip():
        return jsonify({"error": "text is required"}), 400

    return jsonify(run_methods(text, selected))


if __name__ == "__main__":
    app.run(debug=True, port=8090)
