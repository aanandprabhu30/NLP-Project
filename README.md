# 🧠 NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research abstracts by:

- 🧐 **Discipline** – Computer Science (CS), Information Systems (IS), Information Technology (IT)
- 🧐 **Subfield** – AI, ML, CV, CYB, BSP, SEC, CLD, etc.
- 🧐 **Research Methodology** – Qualitative, Quantitative, Mixed Methods

---

## 🛠 Built With

- Python
- Jupyter Notebook
- scikit-learn
- pandas
- seaborn
- matplotlib
- joblib

---

## 📍 Current Phase (as of 17th May 2025)

✅ **Discipline classifier finalized for 1138-paper dataset:**  
**Subfield and Methodology classifiers finalized and cross-validated on 105-paper labeled subset.**

- **Discipline Model (1138 papers):**  
  - `XGBoost + SciBERT (768-dim)` trained on full, hand-validated dataset  
- **Subfield/Methodology Models (105 papers):**  
  - `SVM + SMOTE + Bigram TF-IDF` (Subfields, 105 papers)  
  - `XGBoost + SMOTE + SciBERT (768-dim)` (Methodology, 105 papers)  
- **Evaluation:**  
  - 5-fold stratified cross-validation and ablation studies conducted on the 105-paper labeled set (subfield/methodology tasks)  
  - Results include accuracy, standard deviation, fold-wise breakdown, and version comparison  
- **Artefacts:**  
  - All trained models, embeddings, vectorizers, and label encoders saved as `.joblib` in `/Artefacts`
- **Documentation:**  
  - Full project evaluation and experiment history tracked in Notion and key notebooks
- **Scripts:**
  - All scripts used for scraping data have been stored

> 🔁 **Final architectures:**  
> - `Discipline`: XGBoost + SciBERT (768-dim, 1138 papers)  
> - `Subfield`: SVM + SMOTE + bigram TF-IDF (105 papers)  
> - `Methodology`: XGBoost + SMOTE + SciBERT (Title + Abstract, 768-dim, 105 papers)

> 🔁 v1.1 was skipped in versioning to standardize upgrades directly from v1.0 ➝ v1.2 ➝ v2.0 ➝ v2.2.1
---

## 🚀 Next Phase (Future Work)

- 🔎 **Expand and annotate labeled data** for subfield and methodology (beyond the current 105 papers), especially targeting underrepresented classes like Mixed Methods and minority subfields.
- 🧪 **Re-run discipline, subfield, and methodology classifiers** on larger, more diverse datasets to evaluate generalizability and performance improvements.
- 🤖 **Experiment with fine-tuning transformer models** (e.g., SciBERT, BERT, RoBERTa via HuggingFace) on project-specific data to improve contextual understanding.
- 🧭 **Explore hierarchical and ensemble modeling**: chaining discipline ➝ subfield ➝ methodology, and combining predictions for improved robustness.
- 📦 **Package the final inference pipeline** into an API or deployable tool (e.g., Streamlit app) for easy use and demonstration.
- 🧠 **Systematize documentation**: summarize best practices for data cleaning, model versioning, artefact management, and reproducibility, ensuring clarity for future contributors or publication.

---

## 🗂️ Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `/Artefacts/` | Trained classifiers + vectorizers + evaluation visuals |
| `/Data/` | All labeled datasets used across classification tasks |
| `/Scripts/`| All Scripts used for scraping data|
| `README.md` | This file |
| `TASKS.md` | To-do log and milestones |
| `CrossValidation_AllModels (v1.0).ipynb` | Cross-validation for original pipeline |
| `CrossValidation_AllModels (v1.2 and v2.0).ipynb` | Full CV for Subfield v1.2 and Methodology v2.0 |
| `Evaluate_DisciplineClassifier (v1.0).ipynb` | Manual test set evaluation (9 entries) |
| `methodology_classifier_(v2.1)_bert.ipynb` | MiniLM BERT embeddings + SVM/LogReg |
| `methodology_classifier_(v2.2)_scibert.ipynb` | ✅ SciBERT + XGBoost (+SMOTE) |
| `NLP_Classifier_DisciplineOnly (v1.1).ipynb` | Logistic Regression on Discipline |
| `NLP_Classifier_SubfieldOnly_CS (v1.2).ipynb` | SVM + SMOTE for CS subfields |
| `NLP_Classifier_SubfieldOnly_IS (v1.2).ipynb` | SVM + SMOTE for IS subfields |
| `NLP_Classifier_SubfieldOnly_IT (v1.2).ipynb` | SVM + SMOTE for IT subfields |
| `NLP_Methodology_Classifier (v1.2).ipynb` | SVM + SMOTE for Methodology |
| `NLP_Methodology_Classifier (v2.0).ipynb` | Methodology with Title + Abstract (TF-IDF) |
| `NLP_Pipeline_Prototype_15_Abstracts.ipynb` | Early 15-entry pipeline prototype |
| `NLP_Classifier_DisciplineOnly_SciBERT_XGBoost_(v2.2).ipynb` | ✅ Final discipline classifier using SciBERT embeddings + XGBoost on 1138-paper dataset |

---

## 📊 Data Files (`/Data/`)

| File | Description |
|------|-------------|
| `Evaluation Dataset - 9 entries.csv` | Held-out test set for Discipline model |
| `NLP_Abstract_Dataset (Discipline).csv` | Initial prototype dataset (15 entries) |
| `NLP_Abstract_Dataset (Discipline)(105).csv` | Final labeled Discipline set |
| `NLP_Abstract_Dataset (Method)(105).csv` | Final labeled Methodology set |
| `NLP_Abstract_Dataset (Subfield)(105).csv` | Final labeled Subfield set |
| `NLP_Dataset_Title_Abstract_Discipline_Subfield_Methodology.csv` | Combined dataset for v2.0 |
| `Discipline (1138).csv` | Final deduplicated, hand-labeled discipline dataset (CS/IS/IT, 1138 papers) |

---

## 🧠 Model Artifacts (`/Artefacts/`)

| File | Description |
|------|-------------|
| `baseline_classifier_logreg.pkl` | Generic placeholder baseline |
| `discipline_classifier_logreg.pkl` | Logistic Regression for Discipline |
| `tfidf_vectorizer.pkl` | Shared TF-IDF for Discipline |
| `subfield_classifier_logreg_cs.pkl` | LogReg for CS subfields |
| `tfidf_vectorizer_cs.pkl` | TF-IDF for CS (LogReg) |
| `subfield_classifier_logreg_is.pkl` | LogReg for IS subfields |
| `tfidf_vectorizer_is.pkl` | TF-IDF for IS (LogReg) |
| `subfield_classifier_logreg_it.pkl` | LogReg for IT subfields |
| `tfidf_vectorizer_it.pkl` | TF-IDF for IT (LogReg) |
| `cs_subfield_classifier_svm_smote.pkl` | SVM + SMOTE for CS subfields |
| `cs_subfield_vectorizer_smote.pkl` | Bigram TF-IDF for CS (SVM) |
| `is_subfield_classifier_svm_smote.pkl` | SVM + SMOTE for IS subfields |
| `is_subfield_vectorizer_smote.pkl` | Bigram TF-IDF for IS (SVM) |
| `it_subfield_classifier_svm_smote.pkl` | SVM + SMOTE for IT subfields |
| `it_subfield_vectorizer_smote.pkl` | Bigram TF-IDF for IT (SVM) |
| `methodology_classifier_logreg.pkl` | Logistic Regression (v1.0) – Abstract only |
| `methodology_classifier_svm.pkl` | SVM + SMOTE (v1.2) – Abstract only |
| `methodology_classifier_v2_titleabstract.pkl` | SVM + SMOTE (v2.0) – Title + Abstract |
| `tfidf_vectorizer_methodology.pkl` | Used in v1.0 (LogReg only) |
| `tfidf_vectorizer_methodology_smote.pkl` | Used in v1.2 (SVM + SMOTE) |
| `tfidf_vectorizer_methodology_v2_titleabstract.pkl` | Used in v2.0 |
| `methodology_confusion_matrix_v2.png` | Confusion matrix for Methodology v2.0 |
| `methodology_scibert_xgb_v2.2_model.pkl` | SciBERT + XGBoost (v2.2) baseline |
| `methodology_scibert_xgb_v2.2.1_smote_model.pkl` | ✅ Best-performing model (v2.2.1) |
| `methodology_scibert_xgb_v2.2_scaler.pkl` | StandardScaler used in v2.2 & v2.2.1 |
| `methodology_scibert_xgb_v2.2_label_encoder.pkl` | LabelEncoder for BERT-based classifiers |
| `disc_scibert_xgboost_v2.2.pkl` | Final discipline classifier model (SciBERT + XGBoost, v2.2) |
| `scibert_embeddings_discipline_v2.2.pkl` | Discipline set SciBERT embeddings (768-dim) |
| `discipline_label_encoder_v2.2.pkl` | Label encoder for discipline classifier (v2.2) |
| `methodology_logreg_v2.1.1_scaled.pkl` | Logistic Regression classifier (Scaled, Methodology v2.1.1, MiniLM embeddings) |
| `methodology_scaler_v2.1.1.pkl`        | StandardScaler for scaling MiniLM embeddings (Methodology v2.1.1)              |
---
### Data Collection Scripts (`/Scripts/`)

| Script                                    | Description                                                                                                 |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| arxiv_cs.py                               | Scrape general Computer Science papers from arXiv.                                                          |
| arxiv_cs_se_scraper.py                    | Scrape Software Engineering papers from arXiv for IT/SE dataset (requires manual review for IT relevance).  |
| arxiv_csdc_scraper.py                     | Scrape Data Center topic papers from arXiv (IT infrastructure).                                             |
| arxiv_csni_scraper.py                     | Scrape Network Infrastructure papers from arXiv (for IT dataset).                                           |
| download_it_papers_no_pandas_v3.py        | Download and parse IT papers from multiple sources (cloud, edge, etc.); no pandas dependency.               |
| harvest_it_links.py                       | Collect/harvest links for IT papers prior to full metadata scraping.                                        |
| is_scraper.py                             | Scrape AMCIS Information Systems conference papers (IS dataset).                                            |
| is_scraper_ss.py                          | Scrape Information Systems papers from Semantic Scholar (supplements AMCIS collection).                     |
| it_core.py                                | Core IT domain paper collector (scraping, metadata extraction, discipline filtering).                       |
| it_v2.py                                  | Updated version of IT paper collector/validator (improved over it_core.py).                                 |
| IT.py                                     | General-purpose IT paper downloader and cleaner; aggregates outputs from multiple IT scripts.               |
| itc.py                                    | Custom/incremental IT collection script (handles specific IT subtopics or custom link batches).             |
| semantic_scholar_web_scraper_loose.py     | Looser Semantic Scholar scraper for additional computing papers (CS/IS/IT) for manual review.               |
---

## 🔁 Discipline Version Map

| Version   | Model                                    | Vectorizer / Embedding                | Dataset               | Notes                       |
|-----------|-------------------------------------------|---------------------------------------|-----------------------|-----------------------------|
| v1.0      | Logistic Regression                       | TF-IDF (unigram)                      | 105                   | Baseline, small dataset     |
| v1.1      | Logistic Regression                       | TF-IDF (bigram)                       | 105                   | Improved context, 80/20 CV  |
| v2.2      | XGBoost (`disc_scibert_xgboost_v2.2.joblib`) | SciBERT (768-dim, Title+Abstract)    | **1138**              | ✅ Final, full dataset      |

---

## 🔁 Methodology Version Map

| Version   | Model                                | Vectorizer / Embedding                          | Notes |
|-----------|---------------------------------------|--------------------------------------------------|-------|
| v1.0      | `methodology_classifier_logreg.pkl`   | `tfidf_vectorizer_methodology.pkl`              | Abstract-only baseline |
| v1.2      | `methodology_classifier_svm.pkl`      | `tfidf_vectorizer_methodology_smote.pkl`        | SVM + SMOTE |
| v2.0      | `methodology_classifier_v2_titleabstract.pkl` | `tfidf_vectorizer_methodology_v2_titleabstract.pkl` | Title + Abstract |
| v2.1.1    | `Logistic Regression (Scaled)`          | `MiniLM (384-dim) via sentence-transformers`      | CV only – no model saved |
| v2.2      | `methodology_scibert_xgb_v2.2_model.pkl` | `SciBERT (768-dim, Title + Abstract)`             | XGBoost baseline |
| v2.2.1    | `methodology_scibert_xgb_v2.2.1_smote_model.pkl` | `SciBERT + SMOTE (768-dim)`                  | ✅ Best performance |
| v2.2.1-CV | N/A                                   | N/A                                              | CV-only evaluation (5-fold) |

> ℹ️ Version 1.0 classifier notebooks were overwritten. Only `.pkl` artifacts retained for comparison and version history.

---

## 📊 Cross-Validation Summary (v1.1 / v1.2 / v2.0 / v2.1.1 / v2.2.1-CV)

### Fold-wise Cross-Validation Scores

| **Classifier**              | **Fold 1** | **Fold 2** | **Fold 3** | **Fold 4** | **Fold 5** | **Mean Accuracy** | **Std Dev** |
|-----------------------------|------------|------------|------------|------------|------------|-------------------|-------------|
| **Discipline (v1.1)**       | 0.7143     | 0.9048     | 0.7143     | 0.8571     | 0.6667     | **0.7429**        | **0.1151**  |
| **CS Subfield (v1.2)**      | 0.5714     | 0.4286     | 0.5714     | 0.2857     | 0.1429     | **0.4000**        | **0.1895**  |
| **IS Subfield (v1.2)**      | 0.4286     | 0.2857     | 0.5714     | 0.4286     | 0.2857     | **0.4571**        | **0.1069**  |
| **IT Subfield (v1.2)**      | 0.5714     | 0.5714     | 0.5714     | 0.5714     | 0.2857     | **0.5143**        | **0.1143**  |
| **Methodology (v2.0)**      | 0.5238     | 0.7143     | 0.7143     | 0.7143     | 0.6286     | **0.6381**        | **0.0883**  |
| **Methodology (v2.1.1)**    | 0.5238     | 0.5714     | 0.3333     | 0.2857     | 0.4762     | **0.4381**        | **0.1143**  |
| **Methodology (v2.2.1-CV)** | 0.7143     | 0.7619     | 0.4762     | 0.6190     | 0.7143     | **0.6571**        | **0.1017**  |

> ⚠️ Mixed Methods class remained unclassified across all versions. BERT-based models improved semantic depth but still struggled with extreme class imbalance.
---

### 🔁 Version Comparison Table


| **Task**     | **Version** | **Dataset Size** | **Eval Type**   | **Accuracy** | **Std Dev** | **Notes**                                |
|--------------|-------------|------------------|-----------------|--------------|-------------|------------------------------------------|
| Discipline   | v1.1        | 105              | 5-fold CV       | 0.7714       | 0.1151      | LogReg + bigram TF-IDF                   |
| Discipline   | v1.1        | 105              | Test split      | 0.9048       | —           | LogReg + bigram TF-IDF                   |
| Discipline   | v2.2        | 1138             | Test split      | 0.91         | —           | SciBERT + XGBoost                        |
| Subfield – CS| v1.2        | 105              | 5-fold CV       | 0.4000       | 0.1895      | SVM + SMOTE + bigram TF-IDF              |
| Subfield – IS| v1.2        | 105              | 5-fold CV       | 0.4571       | 0.1069      | SVM + SMOTE + bigram TF-IDF              |
| Subfield – IT| v1.2        | 105              | 5-fold CV       | 0.5143       | 0.1143      | SVM + SMOTE + bigram TF-IDF              |
| Methodology  | v2.0        | 105              | 5-fold CV       | 0.6381       | 0.0883      | SVM + SMOTE + TF-IDF (Title + Abstract)  |
| Methodology  | v2.1.1      | 105              | 5-fold CV       | 0.4381       | 0.1143      | MiniLM + LogReg (Scaled)                 |
| Methodology  | v2.2.1-CV   | 105              | 5-fold CV       | 0.6571       | 0.1017      | SciBERT + XGBoost + SMOTE                |
| Methodology  | v2.2.1      | 105              | Test split      | 0.7619       | —           | SciBERT + XGBoost + SMOTE                |

> **Notes:**  
> - “Test split” means a standard train/test split (often 80/20 or similar), not cross-validation.
> - v2.2 discipline classifier (SciBERT + XGBoost) is on the full 1138-paper dataset.
> - All subfield and methodology classifiers were evaluated on the smaller, manually labeled 105-paper dataset.
---

## 🎯 Project Goals

- ✅ Build a scalable, modular NLP pipeline with clear versioning and task separation  
- ✅ Handle small, imbalanced datasets using SMOTE, class weighting, and stratified cross-validation  
- ✅ Demonstrate baseline-to-semantic progression across TF-IDF and transformer-based features  
- ✅ Apply transformer embeddings (MiniLM and SciBERT) for contextual understanding of abstracts  
- ✅ Evaluate classical vs. semantic models on accuracy, macro F1, and classwise recall  
- 🧠 Establish generalizable architecture ready for future fine-tuning, domain adaptation, or ensemble modeling  

---

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Submitted as part of my BSc Final Year Project in Computer Science – University of London_