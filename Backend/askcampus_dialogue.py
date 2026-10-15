"""
askcampus_dialogue.py

Conversation logic for AskCampus. Both the notebook chat and the Flask
backend (app.py) import this file, so they always behave the same way.

The trained classifier (TF-IDF + calibrated LinearSVC) decides the intent of
every message. This file only adds the things a one-message-at-a-time
classifier cannot handle by itself:

  * Recognizing which program a student means: full names, nicknames like
    "comsci", the short codes CS / IT / IS, and small typos.
  * Remembering that we are talking about fees, so a follow-up like "IT"
    returns the IT fee breakdown instead of being treated as a new question.
  * Answering "what are the fees for computer science?" in one step.
  * Handling a simple yes/no after the bot asks "anything else?".

Each user gets their own ConversationState, so two people chatting at the
same time never share memory.
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

# Below this confidence the bot asks the student to rephrase instead of guessing.
CONFIDENCE_THRESHOLD = 0.35
FALLBACK_RESPONSE = (
    "I'm sorry, I'm not confident I understood that correctly. "
    "Could you rephrase your question, or contact the relevant office directly?"
)


# ---------------------------------------------------------------------------
# Text preprocessing
# This has to match the notebook's preprocess_text exactly. If the live bot
# cleaned text differently from training, it would see words the model never
# learned and accuracy would quietly drop.
# ---------------------------------------------------------------------------
def preprocess_text(text):
    """Lowercase, remove punctuation, tokenize, and lemmatize a message."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    return " ".join(_lemmatizer.lemmatize(tok) for tok in tokens)


# ---------------------------------------------------------------------------
# Program information
# The fee amounts are sample figures for the demo, not official fees.
# ---------------------------------------------------------------------------
COURSE_FEES = {
    "computer science": {"tuition_per_unit": 1350, "laboratory": 4000, "misc": 3000, "classroom": 1000, "id_fee": 150},
    "information technology": {"tuition_per_unit": 1300, "laboratory": 3800, "misc": 3000, "classroom": 1000, "id_fee": 150},
    "information system": {"tuition_per_unit": 1250, "laboratory": 3200, "misc": 3000, "classroom": 1000, "id_fee": 150},
}

# Every way a student might name a program, mapped to its official name.
COURSE_ALIASES = {
    "computer science": "computer science", "computerscience": "computer science",
    "compsci": "computer science", "comsci": "computer science", "cs": "computer science",
    "information technology": "information technology", "infotech": "information technology",
    "info tech": "information technology", "it": "information technology",
    "information system": "information system", "information systems": "information system",
    "info system": "information system", "infosystem": "information system", "is": "information system",
}
STANDARD_LOAD_UNITS = 21

# Words we compare against when the student makes a typo (used by fuzzy matching only).
COURSE_KEYWORDS = {
    "computer science": ["computer", "science"],
    "information technology": ["information", "technology"],
    "information system": ["information", "system"],
}

# After answering one of these topics, the bot asks "anything else?".
FOLLOWUP_INTENTS = {"admission", "scholarship", "dormitory", "placement", "document", "document_request"}

# Short yes/no replies, including common Filipino ones.
YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "opo", "oo"}
NO_SINGLE_WORDS = {"no", "nope", "nah", "wala", "hindi"}
NO_PHRASES = {"no thanks", "im good", "i'm good"}


def fuzzy_match_course(message):
    """Find a program name even when it is misspelled, e.g. "compute4r science".

    Each word in the message is compared with the program keywords. Two words
    count as a match when they are at least 80% similar.
    """
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
    """Return the program named in the message, or None if there isn't one.

    The short codes "is" and "it" are also everyday English words. In an early
    version, "where is the library" was read as the IS program. So the codes
    only count when:
      - the message is already about fees (relaxed=True), or
      - the code is the whole message, like "IT", or
      - it comes at the end after a word like "about" or "for", like "what about IT".
    """
    text = message.lower().strip()
    words = text.split()

    # Full names and nicknames are safe to match anywhere in the message.
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
        # Matches "what about IT" or "I'm taking IS", but not "is scholarship available".
        m = re.search(r"\b(?:about|for|in|of|taking|under|sa)\s+(cs|it|is)\s*[?.!]*$", text)
        if m:
            return COURSE_ALIASES[m.group(1)]

    # Last resort: allow for typos.
    return fuzzy_match_course(message)


def build_fee_breakdown(course):
    """Build the fee reply for one program, computing the totals from COURSE_FEES."""
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


# The yes/no checks look at whole words in short replies only. An early version
# searched inside words, so "technology" was read as a "no".
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
# Conversation memory and the bot itself
# ---------------------------------------------------------------------------
@dataclass
class ConversationState:
    """What the bot remembers about one user's conversation."""
    last_topic: str = None            # e.g. "fees" while we are talking about fees
    awaiting_followup: bool = False   # True right after the bot asks "anything else?"
    last_seen: float = field(default_factory=time.time)


class AskCampusBot:
    """Holds the trained model, the vectorizer and the replies.

    The bot keeps no memory of its own. Each call to respond() gets the
    ConversationState of the user who sent the message.
    """

    def __init__(self, model, vectorizer, response_bank, threshold=CONFIDENCE_THRESHOLD):
        self.model = model
        self.vectorizer = vectorizer
        self.response_bank = response_bank
        self.threshold = threshold

    def classify(self, message):
        """Clean the message, turn it into TF-IDF features, and ask the model.
        Returns the top intent and its probability."""
        features = self.vectorizer.transform([preprocess_text(message)])
        probs = self.model.predict_proba(features)[0]
        top_idx = int(np.argmax(probs))
        return self.model.classes_[top_idx], float(probs[top_idx])

    def respond(self, message, state):
        """Answer one message. Returns the reply, the intent, the confidence,
        and which part of the bot answered (shown under each reply)."""
        state.last_seen = time.time()

        def out(reply, intent, confidence, handled_by):
            return {"reply": reply, "intent": intent,
                    "confidence": float(confidence), "handled_by": handled_by}

        if not message or not message.strip():
            return out(FALLBACK_RESPONSE, None, 0.0, "fallback")

        # Step 1: a yes/no answer to the bot's own "anything else?" question.
        if state.awaiting_followup:
            state.awaiting_followup = False
            if is_yes(message):
                return out("Sure! What would you like to know next -- courses, fees, "
                           "library, dormitories, admission, or scholarships?",
                           "followup_yes", 1.0, "context (follow-up)")
            if is_no(message):
                return out("Glad I could help! Feel free to ask anytime.",
                           "followup_no", 1.0, "context (follow-up)")
            # Anything else is treated as a new question.

        # Step 2: we are still talking about fees and the student named a program.
        if state.last_topic == "fees":
            course = match_course(message)
            if course:
                state.awaiting_followup = True
                return out(build_fee_breakdown(course), "fees_detail", 1.0,
                           "context (fees) + course entity")

        # Step 3: everything else goes to the machine learning model.
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
            if course:  # the program was named in the same message
                state.awaiting_followup = True
                return out(build_fee_breakdown(course), "fees_detail", confidence,
                           "model + course entity")
        else:
            state.last_topic = intent  # a new topic ends the fees conversation

        reply = random.choice(self.response_bank.get(intent) or [FALLBACK_RESPONSE])
        if intent in FOLLOWUP_INTENTS:
            reply += " Is there anything else you'd like to know?"
            state.awaiting_followup = True
        return out(reply, intent, confidence, "model")


def load_response_bank(intents_path):
    """Read the replies for each intent from intents.json, skipping broken entries."""
    import json
    with open(intents_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    bank = {}
    for entry in raw["intents"]:
        if entry.get("intent") and entry.get("text"):
            bank[entry["intent"]] = entry.get("responses", [])
    return bank
