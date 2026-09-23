import json
import re
import random
import string

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# --- One-time NLTK setup ----------------------------------------------------
for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer()
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# --- 1. Load intents.json (same forgiving parser as the notebook) ----------
with open("intents.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

rows = []
response_bank = {}
for entry in raw_data["intents"]:
    intent_name = entry.get("intent")
    patterns = entry.get("text")
    responses = entry.get("responses", [])
    if not intent_name or not patterns:
        continue
    response_bank[intent_name] = responses
    for pattern in patterns:
        rows.append({"query": pattern, "intent": intent_name})

df = pd.DataFrame(rows)


# --- 2. Preprocessing -- identical logic to the notebook -------------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(lemmas)


df["clean_query"] = df["query"].apply(preprocess_text)


# --- 3. Feature extraction + model (trained on ALL data for deployment) ---
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
X = vectorizer.fit_transform(df["clean_query"])
y = df["intent"]

min_class_count = y.value_counts().min()
safe_cv = max(2, min(3, min_class_count))

base_svc = LinearSVC(random_state=RANDOM_STATE, max_iter=10000)
model = CalibratedClassifierCV(base_svc, cv=safe_cv)
model.fit(X, y)

CONFIDENCE_THRESHOLD = 0.35
FALLBACK_RESPONSE = (
    "I'm sorry, I'm not confident I understood that correctly. "
    "Could you rephrase your question, or contact the relevant office directly?"
)

print(f"Model ready: {len(model.classes_)} intents, {X.shape[1]} vocabulary terms.")


# --- 4. Inference ------------------------------------------------------------
def predict(message):
    if not message or not message.strip():
        return FALLBACK_RESPONSE, "none", 0.0

    cleaned = preprocess_text(message)
    features = vectorizer.transform([cleaned])
    probs = model.predict_proba(features)[0]
    top_idx = np.argmax(probs)
    confidence = float(probs[top_idx])
    intent = model.classes_[top_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return FALLBACK_RESPONSE, "fallback", confidence

    candidates = response_bank.get(intent) or [FALLBACK_RESPONSE]
    reply = random.choice(candidates)
    return reply, intent, confidence


# --- 5. Flask app --------------------------------------------------------
app = Flask(__name__)
CORS(app)  # allows the front-end (opened as a local file or local server) to call this API


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "intents_loaded": len(model.classes_)})


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    reply, intent, confidence = predict(message)
    return jsonify({"reply": reply, "intent": intent, "confidence": confidence})


if __name__ == "__main__":
    print("Starting AskCampus backend at http://127.0.0.1:5000")
    print("Test it with: curl -X POST http://127.0.0.1:5000/predict -H 'Content-Type: application/json' -d '{\"message\": \"how much are the fees\"}'")
    app.run(host="127.0.0.1", port=5000, debug=False)
