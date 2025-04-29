# 📝 NLP Project Task Breakdown – Updated (Cross-Validation Complete)

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

## 🌟 Feature Extraction
- [x] Vectorize text using TF-IDF (max_features = 5000)
- [x] Encode labels using LabelEncoder
- [x] Inspect feature matrix and encoded targets

---

## 🧐 Discipline Classification
- [x] Train Logistic Regression classifier for discipline labels
- [x] Evaluate using classification report + confusion matrix
- [x] Save model and vectorizer using `joblib`
- [x] Evaluate model on external 9-paper test set

---

## 🔮 Subfield Classification
- [x] Define 5 subfields per discipline (CS, IS, IT)
- [x] Train Logistic Regression models for CS, IS, and IT separately
- [x] Generate confusion matrices and interpret results
- [x] Save all models and vectorizers to `/Artefacts/`
- [x] Document subfield cross-validation results, interpretations, and justifications in Notion

---

## 📚 Methodology Classification
- [x] Define methodology labels (Qualitative, Quantitative, Mixed)
- [x] Train Logistic Regression model for Methodology classification
- [x] Perform 5-fold stratified cross-validation
- [x] Document Methodology cross-validation results, interpretations, and justifications in Notion

---

## 🛠️ Final Polish & Documentation
- [x] Add inline comments and markdown explanations in all notebooks
- [x] Organize `.pkl` files into `/Artefacts/`
- [x] Finalize `README.md` with updated structure, goals, and evaluation results
- [x] Update `TASKS.md` to reflect cross-validation phase completion
- [x] Write final comparative analysis between Discipline, Subfield, and Methodology classifiers in Notion
- [x] Save Notion documentation export (`N.zip`) and codebase archive (`NLP-Project.zip`)
- [x] Commit cross-validation results separately for Subfield and Methodology phases

---

# ✅ Current Status: Cross-Validation and Comparative Analysis Completed (April 29, 2025)

- Discipline classifier: Stable and high accuracy
- Subfield classifiers: Low accuracy, high variance (expected due to fine-grained labels and small dataset)
- Methodology classifier: Moderate accuracy, acceptable stability
- Comparative analysis between all tasks completed and documented

Ready for future phases: advanced modeling (e.g., BERT fine-tuning) or dataset expansion.

