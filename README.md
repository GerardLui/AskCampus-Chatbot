# AskCampus — NLP-Based Student Information Chatbot

AskCampus is a machine-learning-powered chatbot that answers common university
student questions (fees, enrollment, library hours, hostel info, and more) using
natural language understanding instead of hard-coded if/else rules.

Built for our Programming Languages / CS Elective final project.

## Features
- Intent classification using TF-IDF + Linear SVM (scikit-learn)
- NLP preprocessing pipeline: tokenization, lowercasing, punctuation removal, lemmatization (NLTK)
- Confidence-threshold fallback so the bot says "I'm not sure" instead of guessing wildly
- Multi-turn context: fee questions stay "sticky" across turns (ask about IT, then later just say "CS" or "comsci" -- it keeps giving fee breakdowns without losing the thread)
- Light conversational follow-ups: after answering something substantial (admission, scholarship, dormitory, placement), the bot invites a follow-up and handles a simple yes/no reply naturally, without turning into open-ended chit-chat
- Full evaluation: accuracy, precision, recall, F1-score, confusion matrix, error analysis
- Live chat demo backed by a local Flask API serving the **actual trained model**
  (no reimplementation of the classifier in JavaScript — the browser talks to the
  same TF-IDF + LinearSVC pipeline used for evaluation)

## Project Structure
```
AskCampus-NLP-Chatbot/
├── data/
│   └── intents.json                       # training data: intents, patterns, responses
├── notebooks/
│   └── AskCampus_NLP_Chatbot.ipynb        # full pipeline: preprocessing → training → evaluation
├── backend/
│   ├── app.py                             # Flask API serving the real trained model
│   ├── requirements.txt
│   └── intents.json                       # same dataset, used to train the deployed model
├── webapp/
│   └── index.html                         # chat UI, calls the local backend
├── docs/
│   ├── nlp_pipeline_diagram.png
│   └── presentation.pdf
├── results/
│   ├── confusion_matrix.png
│   └── evaluation_summary.md
├── requirements.txt
└── README.md
```

## How to Run

### 1. Run the notebook (evaluation model)
```bash
pip install -r requirements.txt
```
Open `notebooks/AskCampus_NLP_Chatbot.ipynb` in Jupyter or Google Colab and run all
cells top to bottom. It will:
1. Load and parse `data/intents.json`
2. Preprocess and vectorize the text (TF-IDF)
3. Train and compare Naive Bayes vs. LinearSVC
4. Evaluate the best model and print metrics + confusion matrix

This notebook is where the reported performance numbers below come from.

### 2. Run the live chat demo (deployed model)
The chat UI talks to a small local Flask server running the same model
architecture, trained on the same dataset.

```bash
cd backend
pip install -r requirements.txt
python app.py
```
Wait for the terminal to print `Running on http://127.0.0.1:5000`, then open
`webapp/index.html` in your browser. The status indicator turns green once
it's connected, and you can chat with the real model directly.

> **Note:** the backend runs locally, so the chat demo only works on the
> machine that has `backend/app.py` running — this was a deliberate choice
> to avoid depending on external hosting during grading/presentation.

## Model Performance
*(from the notebook's held-out 80/20 test split, on the current 36-intent dataset)*

| Metric | Score |
|---|---|
| Accuracy | 77.3% |
| Precision | ~82% |
| Recall | 77.3% |
| F1-score | ~76% |

> Note: an earlier draft of this dataset scored higher (~88%) with more but
> lower-quality data (irrelevant course branches, broken placeholder links,
> a malformed entry). We deliberately trimmed to 36 clean, accurate intents
> over a larger noisy set -- a few thin intents (2-3 test examples each)
> account for most of the score movement, which is expected on a small
> held-out split.

## Architecture Note
There are two places this model "runs," and they're intentionally kept in
sync rather than being separate implementations:
- **Notebook**: trains and evaluates the model on an 80/20 split, producing
  the metrics above.
- **Backend (`backend/app.py`)**: trains the same pipeline on the full
  dataset (more coverage for real use) and exposes it via a `/predict`
  endpoint. The web UI calls this directly — there is no second,
  independently-trained model hiding in JavaScript.

## Dataset
`data/intents.json` contains 36 intents and ~374 sample student queries, each
paired with response templates, scoped to AskCampus's three actual programs
(Computer Science, Information Technology, Information System). Both the
notebook and the backend train directly on this file, so there's a single
source of truth for training data across evaluation and deployment.

Content notes:
- Fee questions are handled as a short back-and-forth: the bot asks which
  program you're in, then gives a generic PHP-based tuition/fee breakdown
  for that program. This context is "sticky" -- once you're talking about
  fees, later messages like "comsci" or "what about CS" still resolve
  correctly, until you clearly ask about something else.
- After answering admission, scholarship, dormitory, or placement questions,
  the bot invites a follow-up ("anything else you'd like to know?") and
  handles a plain yes/no reply -- a small conversational touch, kept
  deliberately narrow so it doesn't become open-ended chit-chat.
- "Hostel" was renamed to "Dormitory" to match AskCampus's actual housing
  setup (a separate building primarily for student-athletes and long-distance
  commuters).
- Several placeholder responses from the original template dataset (fake
  links, "XYZ" names, an irrelevant department, a broken external link) were
  replaced with real or clearly-marked-generic AskCampus content.
