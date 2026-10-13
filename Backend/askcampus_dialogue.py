"""
askcampus_dialogue.py -- AskCampus conversation logic, shared by the
notebook (run_cli_chatbot) and the Flask backend (app.py).

The ML model (TF-IDF + calibrated LinearSVC) makes every intent decision.
This module adds what a single-message classifier cannot see on its own:

  * Course entity extraction -- aliases, safe handling of the 2-letter codes
    "cs"/"it"/"is", and difflib fuzzy matching for typos.
  * Sticky fees context -- after a fees question, naming a program returns
    that program's fee breakdown until the topic clearly changes.
  * Same-message handling -- "what are the fees for computer science?"
  * A yes/no follow-up after admission/scholarship/dorm/placement/document.

All conversation state lives in a ConversationState object, so every user
gets their own conversation (no module-level globals).
"""

import re
import random
import string
import difflib
from dataclasses import dataclass, field
import time

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_lemmatizer = WordNetLemmatizer()

CONFIDENCE_THRESHOLD = 0.35
FALLBACK_RESPONSE = (
    "I'm sorry, I'm not confident I understood that correctly. "
    "Could you rephrase your question, or contact the relevant office directly?"
)


# ---------------------------------------------------------------------------
# Preprocessing -- must stay identical to the notebook's training pipeline
# ---------------------------------------------------------------------------
def preprocess_text(text):
    """Lowercase, remove punctuation, tokenize, and lemmatize."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    return " ".join(_lemmatizer.lemmatize(tok) for tok in tokens)


# ---------------------------------------------------------------------------
# Course knowledge (generic sample figures for the demo)
# ---------------------------------------------------------------------------
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

FOLLOWUP_INTENTS = {"admission", "scholarship", "dormitory", "placement", "document", "document_request"}
YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "opo", "oo"}
NO_SINGLE_WORDS = {"no", "nope", "nah", "wala", "hindi"}
NO_PHRASES = {"no thanks", "im good", "i'm good"}


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
    """Return the canonical course name mentioned in `message`, or None.

    Bare 2-letter codes ("cs", "it", "is") collide with English words, so:
    relaxed=True  (message already classified as fees) -> trust them anywhere,
    relaxed=False -> only when the code is the ENTIRE message ("IT").
    """
    text = message.lower().strip()
    words = text.split()

    for alias in sorted(COURSE_ALIASES, key=len, reverse=True):
        if len(alias) > 2 and re.search(rf"\b{re.escape(alias)}\b", text):
            return COURSE_ALIASES[alias]

    if relaxed:
        for code in ("cs", "it", "is"):
            if re.search(rf"\b{re.escape(code)}\b", text):
                return COURSE_ALIASES[code]
    else:
        if len(words) == 1 and words[0].strip(string.punctuation) in ("cs", "it", "is"):
            return COURSE_ALIASES[words[0].strip(string.punctuation)]
        # Code at the END of a message, right after a cue word: "what about IT",
        # "how about cs?", "I'm taking IS". "is scholarship available" does not match.
        m = re.search(r"\b(?:about|for|in|of|taking|under|sa)\s+(cs|it|is)\s*[?.!]*$", text)
        if m:
            return COURSE_ALIASES[m.group(1)]

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


def is_yes(text):
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) <= 3 and any(w in YES_WORDS for w in words)


def is_no(text):
    t = text.lower().strip()
    if t in NO_PHRASES:
        return True
    words = re.findall(r"[a-z']+", t)
    return len(words) <= 3 and any(w in NO_SINGLE_WORDS for w in words)


# ---------------------------------------------------------------------------
# Conversation state + the bot
# ---------------------------------------------------------------------------
@dataclass
class ConversationState:
    """Everything the bot remembers about ONE user's conversation."""
    last_topic: str = None
    awaiting_followup: bool = False
    last_seen: float = field(default_factory=time.time)


class AskCampusBot:
    """Wraps the trained model + vectorizer + replies. Stateless itself:
    pass a ConversationState into respond() for each user."""

    def __init__(self, model, vectorizer, response_bank, threshold=CONFIDENCE_THRESHOLD):
        self.model = model
        self.vectorizer = vectorizer
        self.response_bank = response_bank
        self.threshold = threshold

    def classify(self, message):
        """Run the ML pipeline: preprocess -> TF-IDF -> calibrated LinearSVC."""
        features = self.vectorizer.transform([preprocess_text(message)])
        probs = self.model.predict_proba(features)[0]
        top_idx = int(np.argmax(probs))
        return self.model.classes_[top_idx], float(probs[top_idx])

    def respond(self, message, state):
        """Return a dict: reply, intent, confidence, handled_by."""
        state.last_seen = time.time()

        def out(reply, intent, confidence, handled_by):
            return {"reply": reply, "intent": intent,
                    "confidence": float(confidence), "handled_by": handled_by}

        if not message or not message.strip():
            return out(FALLBACK_RESPONSE, None, 0.0, "fallback")

        # 1) Short yes/no reply to our own "anything else?" invite.
        if state.awaiting_followup:
            state.awaiting_followup = False
            if is_yes(message):
                return out("Sure! What would you like to know next -- courses, fees, "
                           "library, dormitories, admission, or scholarships?",
                           "followup_yes", 1.0, "context (follow-up)")
            if is_no(message):
                return out("Glad I could help! Feel free to ask anytime.",
                           "followup_no", 1.0, "context (follow-up)")
            # anything else is a fresh question -> fall through

        # 2) Sticky fees context: a course named while we're still on fees.
        if state.last_topic == "fees":
            course = match_course(message)
            if course:
                state.awaiting_followup = True
                return out(build_fee_breakdown(course), "fees_detail", 1.0,
                           "context (fees) + course entity")

        # 3) The ML model.
        intent, confidence = self.classify(message)

        if confidence < self.threshold:
            if state.last_topic == "fees":
                return out("Sorry, I didn't catch the program -- please say Computer Science, "
                           "Information Technology, or Information System.",
                           "fees_clarify", confidence, "context (fees)")
            return out(FALLBACK_RESPONSE, None, confidence, "fallback (low confidence)")

        if intent == "fees":
            state.last_topic = "fees"
            course = match_course(message, relaxed=True)
            if course:  # "what are the fees for computer science?"
                state.awaiting_followup = True
                return out(build_fee_breakdown(course), "fees_detail", confidence,
                           "model + course entity")
        else:
            state.last_topic = intent  # a different, confident topic ends fees mode

        reply = random.choice(self.response_bank.get(intent) or [FALLBACK_RESPONSE])
        if intent in FOLLOWUP_INTENTS:
            reply += " Is there anything else you'd like to know?"
            state.awaiting_followup = True
        return out(reply, intent, confidence, "model")


def load_response_bank(intents_path):
    """Build {intent: [responses]} from intents.json, skipping malformed entries."""
    import json
    with open(intents_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    bank = {}
    for entry in raw["intents"]:
        if entry.get("intent") and entry.get("text"):
            bank[entry["intent"]] = entry.get("responses", [])
    return bank
