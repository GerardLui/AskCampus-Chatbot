import os
import random
import threading
import time

import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import nltk

# --- One-time NLTK setup ----------------------------------------------------
for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# All conversation logic lives in askcampus_dialogue.py, the SAME file the
# notebook uses for run_cli_chatbot(), so the web app and the notebook demo
# always behave identically.
from askcampus_dialogue import AskCampusBot, ConversationState, load_response_bank

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

HERE = os.path.dirname(os.path.abspath(__file__))


# --- 1. Load the ACTUAL trained model + vectorizer from the notebook -------
# Produced by the notebook's "Exporting the Final Model for Deployment" cell.
# This is the same fitted object the notebook exported -- app.py never retrains.
try:
    vectorizer = joblib.load(os.path.join(HERE, "vectorizer.joblib"))
    model = joblib.load(os.path.join(HERE, "model.joblib"))
except FileNotFoundError as e:
    raise SystemExit(
        "Missing vectorizer.joblib / model.joblib. Run the notebook's export "
        "cell and copy both files into this backend/ folder first."
    ) from e

# Reply text comes from intents.json (replies only -- no training here).
response_bank = load_response_bank(os.path.join(HERE, "intents.json"))

bot = AskCampusBot(model, vectorizer, response_bank)
print(f"Model loaded: {len(model.classes_)} intents, {len(vectorizer.vocabulary_)} vocabulary terms.")


# --- 2. Per-user conversation state -----------------------------------------
# Each browser tab sends its own session_id, so every user gets a separate
# ConversationState instead of sharing one global. Sessions idle for longer
# than SESSION_TTL_SECONDS are discarded so memory doesn't grow forever.
# (Still in-memory: sessions reset when the server restarts.)
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
SESSION_TTL_SECONDS = 30 * 60


def get_session(session_id):
    now = time.time()
    with SESSIONS_LOCK:
        for sid in [s for s, st in SESSIONS.items() if now - st.last_seen > SESSION_TTL_SECONDS]:
            del SESSIONS[sid]
        if session_id not in SESSIONS:
            SESSIONS[session_id] = ConversationState()
        return SESSIONS[session_id]


# --- 3. Flask app -----------------------------------------------------------
app = Flask(__name__)
CORS(app)  # allows the front-end (opened as a local file or local server) to call this API


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "intents_loaded": len(model.classes_),
        "active_sessions": len(SESSIONS),
    })


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    # Old front-ends that don't send a session_id share one "default" session.
    session_id = str(data.get("session_id") or "default")[:100]
    state = get_session(session_id)
    result = bot.respond(message, state)
    result["session_id"] = session_id
    return jsonify(result)


if __name__ == "__main__":
    print("Starting AskCampus backend at http://127.0.0.1:5000")
    print("Test it with: curl -X POST http://127.0.0.1:5000/predict -H 'Content-Type: application/json' "
          "-d '{\"message\": \"how much are the fees\", \"session_id\": \"test\"}'")
    app.run(host="127.0.0.1", port=5000, debug=False)
