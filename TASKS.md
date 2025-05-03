# 📝 NLP Project Task Breakdown – Updated (Final Models Complete)

This markdown file tracks the step-by-step progress of the NLP classification project. Each task is designed to be small, focused, and Git-commit-worthy.

---

## 🚀 Project Initialization
- [x] Initialize GitHub repo
- [x] Create initial Jupyter notebook file
- [x] Add .gitignore file to exclude system/junk files
- [x] Create a README.md with project description and structure

---

## 📄 Dataset Creation
- [x] Create full dataset of 105 research paper abstracts (35 per discipline)
- [x] Add discipline, subfield, and methodology labels (CS, IS, IT + 5 subfields each + 3 methodologies)
- [x] Save dataset as CSV and load into Jupyter
- [x] Create `/Data/` folder to store all .csv files
- [x] Add external evaluation set (9 entries) and prototype dataset (15 abstracts)
- [x] Create enriched dataset with title + abstract + all 3 labels

---

## ✨ Text Preprocessing
- [x] Load dataset into pandas
- [x] Clean text: lowercase, remove punctuation/numbers
- [x] Remove stopwords using NLTK
- [x] Lemmatize using spaCy (optional stage)

---

## 🌟 Feature Extraction
- [x] Vectorize text using TF-IDF (ngram_range = (1,2), min_df = 2, max_df = 0.95)
- [x] Encode labels using LabelEncoder
- [x] Inspect feature matrix and encoded targets

---

## 🧐 Discipline Classification
- [x] Train Logistic Regression classifier (v1.0)
- [x] Upgrade to SVM + Bigram TF-IDF + SMOTE (v1.2)
- [x] Evaluate on external 9-entry dataset
- [x] Save both models and vectorizers to `/Artefacts/`
- [x] Perform 5-fold cross-validation and log fold-wise accuracy
- [x] Add discipline comparison to Notion

---

## 🔮 Subfield Classification
- [x] Define 5 subfields per discipline (CS, IS, IT)
- [x] Train Logistic Regression baseline (v1.0)
- [x] Upgrade to SVM + SMOTE + Bigram TF-IDF (v1.2)
- [x] Save all subfield models and vectorizers (logreg and SVM versions)
- [x] Perform 5-fold CV for each discipline
- [x] Analyze F1, macro, weighted scores and confusion matrices
- [x] Write full comparative subfield analysis in Notion

---

## 📚 Methodology Classification
- [x] Define methodology labels (QLT, QNT, M)
- [x] Train Logistic Regression baseline (v1.0)
- [x] Upgrade to SVM + SMOTE + Bigram TF-IDF (v1.2)
- [x] Add Title + Abstract as input features (v2.0)
- [x] Save all models and vectorizers to `/Artefacts/`
- [x] Perform stratified train-test split and 5-fold CV
- [x] Document performance evolution and architecture reuse justification in Notion

---

## 📁 Project Organisation & Docs
- [x] Organize all `.pkl` files in `/Artefacts/`
- [x] Create and populate `/Data/` with all datasets
- [x] Finalize `README.md` with updated project state and architecture summary
- [x] Update `TASKS.md` to reflect full modeling completion
- [x] Document all concepts (e.g. make_pipeline, SMOTE, SVM, TF-IDF) in Notion
- [x] Version and label models v1.0 ➝ v1.2 ➝ v2.0 in GitHub/Notion
- [x] Save `N.zip` (Notion export) and `NLP-Project.zip` (codebase snapshot)

---

## ✅ Current Status: All Pipelines Finalized – May 3, 2025

- All classification tasks completed with final architecture: `SVM + SMOTE + Bigram TF-IDF`.
- v2.0 Methodology classifier using Title + Abstract achieves best QLT performance.
- All models and vectorizers versioned, saved, and documented.
- Results aligned across Jupyter, GitHub, and Notion.

**System is ready for:**
- Transformer-based experimentation (e.g. SciBERT)
- Dataset expansion
- Optional API deployment or research publication