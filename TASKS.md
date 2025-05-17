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
- [x] Implement MiniLM embeddings + Logistic Regression (Scaled) (v2.1.1)
- [x] Implement SciBERT + XGBoost (v2.2)
- [x] Apply SMOTE + SciBERT + XGBoost (v2.2.1)
- [x] Perform 5-fold CV on v2.2.1 and summarize fold-wise accuracy (v2.2.1-CV)
- [x] Identify v2.2.1 as best trained model; v2.2.1-CV as most stable evaluation

---

## 📊 Final Cross-Validation & Reporting
- [x] Implement 5-fold cross-validation for all finalized models (v1.1, v1.2, v2.0)
- [x] Log fold-wise scores, mean accuracy, and standard deviation per classifier
- [x] Create version comparison table (Previous vs Final accuracy and variance)
- [x] Add consolidated fold-wise table across all classifiers
- [x] Write final summary + interpretation + architecture justification in Notion
- [x] Update `README.md` and `TASKS.md` to reflect final experimental validation
- [x] Include v2.1.1 and v2.2.1-CV in CV tables
- [x] Correct standard deviations and scores using confirmed notebook outputs
- [x] Update version comparison table to show v2.0 → v2.1.1 → v2.2.1 performance changes
- [x] Update README to include new architecture, model artefacts, goals, and full CV results
---

## ✅ Current Status: May 3, 2025

## ✅ Current Status: May 6, 2025

- All classification tasks completed using final architectures:
  - `Discipline`: Logistic Regression + bigram TF-IDF
  - `Subfield`: SVM + SMOTE + bigram TF-IDF
  - `Methodology`: XGBoost + SMOTE + SciBERT (768-dim)
- Best Methodology performance:
  - **v2.2.1** → Accuracy: 76.19%, Macro F1: 0.54
  - **v2.2.1-CV** → Most stable: Accuracy 65.71%, Std Dev: 0.10
- All model artefacts, CV metrics, and tables updated in Notion, GitHub, and README
- Project is now clean, reproducible, and ready for:
  - Data augmentation
  - Transformer fine-tuning
  - API inference or paper write-up

---

## 🆕 Expanded Dataset & Final Discipline Classifier (May 17, 2025)
- [x] Scrape, curate, and manually label a full 1138-paper dataset (CS/IS/IT) for discipline classification
- [x] Deduplicate and validate all abstracts, links, and discipline labels
- [x] Extract SciBERT (768-dim) embeddings for all 1138 papers
- [x] Train and evaluate XGBoost classifier on SciBERT embeddings for discipline (v2.2)
- [x] Save discipline artefacts: classifier, embeddings, label encoder (`/Artefacts/`)
- [x] Document discipline v2.2 workflow, metrics, and insights in README and Notion
- [x] Update all tables and results to reflect discipline model’s expanded dataset and new architecture

---

## 🗂️ Data Collection Scripts Organization (May 17, 2025)

- [x] Create a new `/scripts/` folder in the project root for all data collection utilities
- [x] Move all scraping, harvesting, and data download scripts (total: 13) into the `/scripts/` folder
- [x] Ensure all scripts are clearly named and versioned for transparency and reproducibility
- [x] Add a README table entry for each script with a one-line description of its function
- [x] Document, for each script:
    - Source targeted (e.g., arXiv, AMCIS, Semantic Scholar)
    - Discipline or topic focus (CS, IS, IT, general)
    - Any manual review required (e.g., for IT relevance)
- [x] Update Notion and README to reference the `/scripts/` folder as the official source for data gathering code

---

## ✅ Current Status: All Pipelines Finalized and Validated – May 17, 2025

- Discipline classifier upgraded to SciBERT + XGBoost, trained and validated on full 1138-paper dataset
- Subfield and Methodology classifiers remain based on 105-paper labeled set (best models: SVM + SMOTE + bigram TF-IDF for Subfield, XGBoost + SMOTE + SciBERT for Methodology)
- All model artefacts, embeddings, and encoders saved and versioned
- Documentation in Notion, README, and TASKS.md fully updated
- Repository and dataset are reproducible, clean, and ready for final submission and review

---

## 2025-05-17 – Final polish and full project sync

- Added discipline v2.2 classifier: SciBERT + XGBoost trained on full 1138-paper dataset (`disc_scibert_xgboost_v2.2.pkl`), achieving 91% test-set accuracy.
- Performed ablation with Methodology v2.1.1 (LogReg-scaled on MiniLM embeddings) and documented results.
- Moved all 13 data collection scripts to `/Scripts/` and updated README with descriptions for each script.
- Synced the artefact table in README with the current `/Artefacts/` folder (31 files).
- Added and updated requirements.txt for full environment reproducibility.
- Checked for and removed `.DS_Store` files and other OS-specific droppings from the repo.
- Ensured TASKS.md, README.md, and Notion are now fully up to date with all final experiments, models, and artefacts.

---
