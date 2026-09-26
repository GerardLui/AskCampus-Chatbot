import json
import re
import random
import string
import difflib

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
#
# "Sticky" by design: once the conversation is about fees, it STAYS about
# fees across multiple turns (so "IT", then later "what about CS", then
# later "comsci" all keep working) until the user clearly asks about
# something else entirely.
LAST_TOPIC = {"topic": None}

COURSE_FEES = {
    "computer science": {"tuition_per_unit": 1350, "laboratory": 4000, "misc": 3000, "classroom": 1000, "id_fee": 150},
    "information technology": {"tuition_per_unit": 1300, "laboratory": 3800, "misc": 3000, "classroom": 1000, "id_fee": 150},
    "information system": {"tuition_per_unit": 1250, "laboratory": 3200, "misc": 3000, "classroom": 1000, "id_fee": 150},
}
COURSE_ALIASES = {
    "computer science": "computer science", "computerscience": "computer science",
    "compsci": "computer science", "comsci": "computer science", "cs": "computer science",
    "information technology": "information technology", "infotech": "information technology",
    "info tech": "information technology", "it": "information technology",
    "information system": "information system", "information systems": "information system",
    "info system": "information system", "infosystem": "information system", "is": "information system",
}
STANDARD_LOAD_UNITS = 21

# Keywords used only for typo-tolerant fuzzy matching (not exact matching).
COURSE_KEYWORDS = {
    "computer science": ["computer", "science"],
    "information technology": ["information", "technology"],
    "information system": ["information", "system"],
}


def fuzzy_match_course(message):
    """Catch minor typos (e.g. 'compute4r science') by comparing each word
    against known course keywords by similarity instead of exact match."""
    words = re.sub(r"[^a-z0-9\s]", " ", message.lower()).split()
    best_course, best_score = None, 0.0
    for course, keywords in COURSE_KEYWORDS.items():
        hits = sum(
            1 for kw in keywords
            if any(difflib.SequenceMatcher(None, w, kw).ratio() >= 0.8 for w in words)
        )
        score = hits / len(keywords)
        if score > best_score:
            best_score, best_course = score, course
    return best_course if best_score >= 0.5 else None


def match_course(message, relaxed=False):
    text = message.lower().strip()
    words = text.split()

    # Multi-word / unambiguous names and abbreviations can match anywhere.
    for alias in sorted(COURSE_ALIASES, key=len, reverse=True):
        if len(alias) > 2:  # everything except the bare "cs"/"it"/"is" codes
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return COURSE_ALIASES[alias]

    # Bare 2-letter codes ("cs", "it", "is") collide with common English
    # words ("it", "is"). When `relaxed=True` (only used when the message
    # was ALREADY confidently classified as being about fees), trust them
    # anywhere -- e.g. "what are the fees for IT" is unambiguous in that
    # context. Otherwise, only trust a code when it's the ENTIRE message --
    # a genuine direct answer like "IT" -- not just a short sentence, since
    # fees mode is sticky and can persist across many turns; a later
    # unrelated question like "is scholarship available" must not collide
    # with the course code "IS" just because it's short.
    if relaxed:
        for code in ("cs", "it", "is"):
            if re.search(rf"\b{re.escape(code)}\b", text):
                return COURSE_ALIASES[code]
    elif len(words) == 1 and words[0] in ("cs", "it", "is"):
        return COURSE_ALIASES[words[0]]

    # Last resort: typo-tolerant fuzzy match against full course names.
    return fuzzy_match_course(message)


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
        f"Want the breakdown for another program too? Just say Computer Science, IT, or IS.\n"
        f"(These are generic sample figures for this demo -- actual fees are published each semester by the Finance Office.)"
    )


FOLLOWUP_INTENTS = {"admission", "scholarship", "dormitory", "placement", "document"}
YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "opo", "oo"}
NO_SINGLE_WORDS = {"no", "nope", "nah", "wala", "hindi"}
NO_PHRASES = {"no thanks", "im good", "i'm good"}
AWAITING_FOLLOWUP = {"active": False}


def is_yes(text):
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) <= 3 and any(w in YES_WORDS for w in words)


def is_no(text):
    t = text.lower().strip()
    if t in NO_PHRASES:
        return True
    words = re.findall(r"[a-z']+", t)
    return len(words) <= 3 and any(w in NO_SINGLE_WORDS for w in words)


# --- 5. Inference ------------------------------------------------------------
def predict(message):
    if not message or not message.strip():
        return FALLBACK_RESPONSE, "none", 0.0

    # Handle a simple yes/no reply to our own "anything else?" invite, then
    # get out of the way -- this should never block a genuine new question.
    if AWAITING_FOLLOWUP["active"]:
        if is_yes(message):
            AWAITING_FOLLOWUP["active"] = False
            return (
                "Sure! What would you like to know next -- courses, fees, "
                "library, dormitories, admission, or scholarships?",
                "followup_yes",
                1.0,
            )
        if is_no(message):
            AWAITING_FOLLOWUP["active"] = False
            return ("Glad I could help! Feel free to ask anytime.", "followup_no", 1.0)
        AWAITING_FOLLOWUP["active"] = False  # treat anything else as a fresh question

    # If a course name/code appears in the message AND we're still in a fees
    # conversation (however many turns ago it started), answer directly and
    # stay in fees mode -- this is what makes "IT" then later "comsci" both
    # keep working without re-asking about fees from scratch.
    if LAST_TOPIC["topic"] == "fees":
        course = match_course(message)
        if course:
            AWAITING_FOLLOWUP["active"] = True
            return build_fee_breakdown(course), "fees_detail", 1.0

    cleaned = preprocess_text(message)
    features = vectorizer.transform([cleaned])
    probs = model.predict_proba(features)[0]
    top_idx = np.argmax(probs)
    confidence = float(probs[top_idx])
    intent = model.classes_[top_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        if LAST_TOPIC["topic"] == "fees":
            return (
                "Sorry, I didn't catch the program -- please say Computer Science, "
                "Information Technology, or Information System.",
                "fees_clarify",
                confidence,
            )
        return FALLBACK_RESPONSE, "fallback", confidence

    if intent == "fees":
        LAST_TOPIC["topic"] = "fees"
        # They may have already named their course in this same message
        # ("what are the fees for computer science?") -- don't make them
        # repeat it.
        course = match_course(message, relaxed=True)
        if course:
            AWAITING_FOLLOWUP["active"] = True
            return build_fee_breakdown(course), "fees_detail", confidence
    else:
        # A clearly different, confident topic ends the fees conversation.
        LAST_TOPIC["topic"] = intent

    candidates = response_bank.get(intent) or [FALLBACK_RESPONSE]
    reply = random.choice(candidates)

    if intent in FOLLOWUP_INTENTS:
        reply = reply + " Is there anything else you'd like to know?"
        AWAITING_FOLLOWUP["active"] = True

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
