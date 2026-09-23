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


# --- 4. Multi-turn fee flow --------------------------------------------------
# Single-user, in-memory conversation state (fine for a local single-session
# demo; would need per-session keys for a multi-user deployment).
CONVO_STATE = {"awaiting_course_for_fees": False}

COURSE_FEES = {
    "computer science": {"tuition_per_unit": 1350, "laboratory": 4000, "misc": 3000, "classroom": 1000, "id_fee": 150},
    "information technology": {"tuition_per_unit": 1300, "laboratory": 3800, "misc": 3000, "classroom": 1000, "id_fee": 150},
    "information system": {"tuition_per_unit": 1250, "laboratory": 3200, "misc": 3000, "classroom": 1000, "id_fee": 150},
}
COURSE_ALIASES = {
    "computer science": "computer science", "cs": "computer science",
    "information technology": "information technology", "it": "information technology",
    "information system": "information system", "is": "information system",
}
STANDARD_LOAD_UNITS = 21


def match_course(message):
    text = message.lower().strip()
    words = text.split()

    # Full course names can match anywhere in the message -- unambiguous.
    for full_name in ("computer science", "information technology", "information system"):
        if re.search(rf"\b{re.escape(full_name)}\b", text):
            return COURSE_ALIASES[full_name]

    # Short codes ("cs", "it", "is") collide with common English words
    # ("it", "is"), so only trust them when the message is short -- i.e.
    # actually looks like a direct answer, not an unrelated sentence.
    if len(words) <= 3:
        for code in ("cs", "it", "is"):
            if re.search(rf"\b{re.escape(code)}\b", text):
                return COURSE_ALIASES[code]

    return None


def build_fee_breakdown(course):
    fees = COURSE_FEES[course]
    tuition_total = fees["tuition_per_unit"] * STANDARD_LOAD_UNITS
    grand_total = tuition_total + fees["laboratory"] + fees["misc"] + fees["classroom"] + fees["id_fee"]
    return (
        f"Here's a generic fee breakdown for {course.title()} (based on a standard {STANDARD_LOAD_UNITS}-unit load):\n"
        f"- Tuition: PHP {fees['tuition_per_unit']:,}/unit x {STANDARD_LOAD_UNITS} units = PHP {tuition_total:,}\n"
        f"- Laboratory Fee: PHP {fees['laboratory']:,}\n"
        f"- Miscellaneous Fee: PHP {fees['misc']:,}\n"
        f"- Classroom/Facilities Fee: PHP {fees['classroom']:,}\n"
        f"- ID Fee: PHP {fees['id_fee']:,}\n"
        f"Estimated Total: PHP {grand_total:,}\n\n"
        f"(These are generic sample figures for this demo -- actual fees are published each semester by the Finance Office.)"
    )


# --- 5. Inference ------------------------------------------------------------
def predict(message):
    if not message or not message.strip():
        return FALLBACK_RESPONSE, "none", 0.0

    # If we're mid-conversation waiting for a course name, try that first --
    # but don't get stuck: if the message clearly asks something else instead,
    # drop the pending question and answer normally.
    if CONVO_STATE["awaiting_course_for_fees"]:
        course = match_course(message)
        if course:
            CONVO_STATE["awaiting_course_for_fees"] = False
            return build_fee_breakdown(course), "fees_detail", 1.0

        cleaned = preprocess_text(message)
        features = vectorizer.transform([cleaned])
        probs = model.predict_proba(features)[0]
        top_idx = np.argmax(probs)
        confidence = float(probs[top_idx])
        intent = model.classes_[top_idx]
        if confidence >= CONFIDENCE_THRESHOLD and intent != "fees":
            CONVO_STATE["awaiting_course_for_fees"] = False
            # fall through to normal handling below
        else:
            return (
                "Sorry, I didn't catch the program -- please say Computer Science, "
                "Information Technology, or Information System.",
                "fees_clarify",
                confidence,
            )

    cleaned = preprocess_text(message)
    features = vectorizer.transform([cleaned])
    probs = model.predict_proba(features)[0]
    top_idx = np.argmax(probs)
    confidence = float(probs[top_idx])
    intent = model.classes_[top_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return FALLBACK_RESPONSE, "fallback", confidence

    if intent == "fees":
        CONVO_STATE["awaiting_course_for_fees"] = True

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