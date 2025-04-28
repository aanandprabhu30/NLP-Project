# 📝 NLP Project Task Breakdown – Updated (Full Pipeline Complete)

This markdown file tracks the step-by-step progress of the NLP classification project. Each task is designed to be small, focused, and Git-commit-worthy.

---

## 🚀 Project Initialization
- [x] Initialize GitHub repo
- [x] Create initial Jupyter notebook file
- [x] Add .gitignore file to exclude system/junk files
- [x] Create a README.md with project description and project structure

---

## 📄 Dataset Creation
- [x] Create full dataset of 105 research paper abstracts (35 per discipline)
- [x] Add discipline, subfield, and methodology labels (CS, IS, IT + 5 subfields each + 3 methodologies)
- [x] Save dataset as CSV and load into Jupyter

---

## ✨ Text Preprocessing
- [x] Load dataset into pandas
- [x] Clean text: lowercase, remove punctuation/numbers
- [x] Remove stopwords using NLTK
- [x] Lemmatize using spaCy (optional stage for future refinement)

---

## 🎯 Feature Extraction
- [x] Vectorize text using TF-IDF (max_features = 5000)
- [x] Encode labels using LabelEncoder
- [x] Inspect feature matrix and encoded targets

---

## 🧠 Discipline Classification
- [x] Train Logistic Regression classifier for discipline labels
- [x] Evaluate using classification report + confusion matrix
- [x] Save model and vectorizer using `joblib`
- [x] Evaluate model on external 9-paper test set

---

## 🧪 Subfield Classification
- [x] Define 5 subfields per discipline (CS, IS, IT)
- [x] Train Logistic Regression models for CS, IS, and IT separately
- [x] Generate confusion matrices and interpret results
- [x] Save all models and vectorizers to `/Artefacts/`
- [x] Log experiments in Notion with per-class scores and justification

---

## 📚 Methodology Classification
- [x] Define clear methodology labels (Qualitative, Quantitative, Mixed)
- [x] Create labeled dataset (expanded from 105 abstracts)
- [x] Train baseline Logistic Regression classifier (with class balancing)
- [x] Evaluate performance and confusion matrix
- [x] Save model/vectorizer and document results in Notion

---

## 🛠️ Final Polish & Documentation
- [x] Add inline comments and markdown explanations in all notebooks
- [x] Organize `.pkl` files into `/Artefacts/`
- [x] Finalize `README.md` with updated full structure and goals
- [x] Finalize `TASKS.md` to reflect project completion
- [x] Write final project summary and conclusions in Notion
- [x] Prepare for viva/presentation (script or slides)

---

# ✅ Current Status: Full NLP Pipeline Complete (Discipline → Subfield → Methodology)
