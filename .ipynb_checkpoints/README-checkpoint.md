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

## 📍 Current Phase

✅ All classifiers finalized:  
**Discipline ➜ Subfield ➜ Methodology**

- Used:
  - `Logistic Regression + Bigram TF-IDF` for Discipline  
  - `SVM + SMOTE + Bigram TF-IDF` for Subfields  
  - `XGBoost + SMOTE + SciBERT (768-dim)` for Methodology  
- Evaluated using **5-fold stratified cross-validation**  
- Results include **accuracy, standard deviation, fold-wise breakdown, and version comparison**  
- Saved all trained models and vectorizers in `/Artefacts`  
- Documented full evaluation in Notion and `CrossValidation_AllModels.ipynb`  

> 🔁 Final architecture:  
> - `Discipline`: Logistic Regression + bigram TF-IDF  
> - `Subfield`: SVM + SMOTE + bigram TF-IDF  
> - `Methodology`: XGBoost + SMOTE + SciBERT (Title + Abstract, 768-dim)

> 🔁 v1.1 was skipped in versioning to standardize upgrades directly from v1.0 ➝ v1.2 ➝ v2.0 ➝ v2.2.1
---

## 🚀 Next Phase (Future Work)

- 🔎 Scrape and annotate additional abstracts, especially for **Mixed Methods**, to improve class balance  
- 🧪 Re-run `v2.2.1` architecture on expanded dataset to evaluate generalizability gains  
- 🤖 Experiment with **SciBERT fine-tuning** and other transformer models (e.g., BERT, RoBERTa via Hugging Face)  
- 🧭 Explore **hierarchical modeling**: Discipline ➝ Subfield ➝ Methodology  
- 📦 Package the final pipeline into an **inference-ready API** or lightweight UI prototype (e.g., Streamlit)  
- 🧠 Document best practices for model versioning, resampling, and semantic feature integration  

---

## 🗂️ Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `/Artefacts/` | Trained classifiers + vectorizers + evaluation visuals |
| `/Data/` | All labeled datasets used across classification tasks |
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
---
## 🔁 Methodology Version Map

| Version   | Model                                | Vectorizer / Embedding                          | Notes |
|-----------|---------------------------------------|--------------------------------------------------|-------|
| v1.0      | `methodology_classifier_logreg.pkl`   | `tfidf_vectorizer_methodology.pkl`              | Abstract-only baseline |
| v1.2      | `methodology_classifier_svm.pkl`      | `tfidf_vectorizer_methodology_smote.pkl`        | SVM + SMOTE |
| v2.0      | `methodology_classifier_v2_titleabstract.pkl` | `tfidf_vectorizer_methodology_v2_titleabstract.pkl` | Title + Abstract |
| v2.1.1    | Logistic Regression (Scaled)          | MiniLM (384-dim) via sentence-transformers      | CV only – no model saved |
| v2.2      | `methodology_scibert_xgb_v2.2_model.pkl` | SciBERT (768-dim, Title + Abstract)             | XGBoost baseline |
| v2.2.1    | `methodology_scibert_xgb_v2.2.1_smote_model.pkl` | SciBERT + SMOTE (768-dim)                  | ✅ Best performance |
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

| **Task**         | **Version** | **Prev Accuracy** | **Final Accuracy** | **Δ Accuracy** | **Prev Std Dev** | **Final Std Dev** | **Δ Std Dev** |
|------------------|-------------|-------------------|---------------------|----------------|------------------|--------------------|----------------|
| Discipline       | v1.1        | 0.7714            | 0.7429              | ↓ -0.0285      | 0.0923           | 0.1151             | ↑ +0.0228      |
| Subfield – CS    | v1.2        | 0.2857            | 0.4000              | ↑ +0.1143      | 0.1807           | 0.1895             | ↑ +0.0088      |
| Subfield – IS    | v1.2        | 0.4000            | 0.4571              | ↑ +0.0571      | 0.1069           | 0.1069             | ➖ No change   |
| Subfield – IT    | v1.2        | 0.4286            | 0.5143              | ↑ +0.0857      | 0.1565           | 0.1143             | ↓ -0.0422      |
| Methodology      | v2.0 → v2.1.1 | 0.6381           | 0.4381              | ↓ -0.2000      | 0.0883           | 0.1143             | ↑ +0.0260      |
| Methodology      | v2.1.1 → v2.2.1-CV | 0.4381      | 0.6571              | ↑ +0.2190      | 0.1143           | 0.1017             | ↓ -0.0126      |
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