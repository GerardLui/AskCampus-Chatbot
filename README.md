# AskCampus  — NLP-Based Student Information Chatbot
AskCampus is a machine-learning-powered chatbot that answers common university
student questions (fees, enrollment, library hours, hostel info, and more) using
natural language understanding instead of hard-coded if/else rules.

Built for our Programming Languages / CS Elective final project.

## Features
- Intent classification using TF-IDF + Linear SVM (scikit-learn)
- NLP preprocessing pipeline: tokenization, lowercasing, punctuation removal, lemmatization (NLTK)
- Confidence-threshold fallback so the bot says "I'm not sure" instead of guessing wildly
- Full evaluation: accuracy, precision, recall, F1-score, confusion matrix, error analysis
- Lightweight browser-based demo (no backend server required)

## Project Structure
AskCampus-NLP-Chatbot/
├── data/
│ └── intents.json # training data: intents, patterns, responses
├── notebooks/
│ └── AskCampus_NLP_Chatbot.ipynb # full pipeline: preprocessing → training → evaluation
├── webapp/
│ └── index.html # standalone browser demo
├── docs/
│ ├── nlp_pipeline_diagram.png
│ └── presentation.pdf
├── results/
│ ├── confusion_matrix.png
│ └── evaluation_summary.md
├── requirements.txt
└── README.md


## How to Run

### 1. Set up the environment
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
Open `notebooks/AskCampus_NLP_Chatbot.ipynb` in Jupyter or Google Colab and run all
cells top to bottom. It will:
1. Load and parse `data/intents.json`
2. Preprocess and vectorize the text (TF-IDF)
3. Train and compare Naive Bayes vs. LinearSVC
4. Evaluate the best model and print metrics + confusion matrix

### 3. Try the live demo
Open `webapp/index.html` directly in a browser — the trained model is embedded
in the page as JSON, so it runs entirely client-side.

**Live demo:** _(add your GitHub Pages link here once Step 7 is done)_

## Model Performance
| Metric | Score |
|---|---|
| Accuracy | 87.8% |
| Precision | 88.7% |
| Recall | 87.8% |
| F1-score | 86.4% |


## Dataset
`data/intents.json` contains 38 intents and ~408 sample student queries, each
paired with response templates. The notebook trains directly on this file (an
earlier draft trained on a disconnected placeholder dataset — that's been fixed).
