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

- Used `SVM + SMOTE + Bigram TF-IDF` architecture  
- Evaluated using accuracy, F1, precision, recall, confusion matrix  
- Saved all trained models and vectorizers in `/Artefacts`  
- Documented all experiments and rationales in Notion  

---

## 🚀 Next Phase (Future Work)

- Expand dataset size and class balance  
- Explore **transformers** (e.g., SciBERT, BERT, HuggingFace)  
- Build hierarchical end-to-end multi-label model  
- Package as an API or minimal UI prototype  

---

## 🗂️ Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `/Artefacts/` | Trained classifiers + vectorizers + evaluation visuals |
| `/Data/` | All labeled datasets used across classification tasks |
| `NLP_Classifier_DisciplineOnly (v1.1).ipynb` | Discipline classifier (CS, IS, IT) |
| `NLP_Classifier_SubfieldOnly_CS (v1.2).ipynb` | Subfield classifier for CS |
| `NLP_Classifier_SubfieldOnly_IS (v1.2).ipynb` | Subfield classifier for IS |
| `NLP_Classifier_SubfieldOnly_IT (v1.2).ipynb` | Subfield classifier for IT |
| `NLP_Methodology_Classifier (v1.2).ipynb` | SVM + SMOTE for Methodology |
| `NLP_Methodology_Classifier (v2.0).ipynb` | Methodology with Title + Abstract |
| `CrossValidation_AllModels.ipynb` | 5-fold cross-validation notebook |
| `Evaluate_DisciplineClassifier (v1.0).ipynb` | Discipline test split evaluation |
| `NLP_Pipeline_Prototype_15_Abstracts.ipynb` | Early prototype with 15-entry set |
| `README.md` | This file |
| `TASKS.md` | To-do log and milestones |

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

---
## 🔁 Methodology Version Map

| Version | Model | Vectorizer | Notes |
|---------|--------|------------|-------|
| v1.0 | `methodology_classifier_logreg.pkl` | `tfidf_vectorizer_methodology.pkl` | Abstract-only baseline |
| v1.2 | `methodology_classifier_svm.pkl` | `tfidf_vectorizer_methodology_smote.pkl` | SVM + SMOTE |
| v2.0 | `methodology_classifier_v2_titleabstract.pkl` | `tfidf_vectorizer_methodology_v2_titleabstract.pkl` | Title + Abstract |

---

## 🎯 Project Goals

- Build a scalable, modular NLP pipeline  
- Handle small, imbalanced datasets robustly  
- Lay groundwork for future semantic modeling with transformer architectures  

---

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Submitted as part of my BSc Final Year Project in Computer Science – University of London_