from flask import Flask, render_template, request, jsonify
from gan_model import generate_summary, tokenizer, discriminator, device
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt")
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    text = data["text"]
    length = int(data["length"])
    mode = data["mode"]
    summary, ids = generate_summary(text, length)
    sentences = sent_tokenize(text)
    bullets = sentences[:5]
    article_ids = tokenizer(text, return_tensors="pt", truncation=True)["input_ids"].to(device)
    summary_ids = tokenizer(summary, return_tensors="pt", truncation=True)["input_ids"].to(device)
    score = discriminator(article_ids, summary_ids).item()
    score = round(score * 100, 2)
    return jsonify({
        "summary": summary,
        "bullets": bullets,
        "score": score,
        "mode": mode
    })
if __name__ == "__main__":
    app.run(debug=True)