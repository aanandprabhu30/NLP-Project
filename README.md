# NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research papers into:

- 🧐 **Discipline** (e.g., Computer Science, Information Systems, Information Technology)
- 🧐 **Subfield** (e.g., AI, CV, CYB, etc.)
- 🧐 **Methodology** (Qualitative, Quantitative, Mixed Methods)

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

✅ Finalized full pipeline:  
**Discipline ➔ Subfield ➔ Methodology** classification.

- Upgraded all models to use `SVM + SMOTE + Bigram TF-IDF`.
- Logged evaluation metrics, confusion matrices, and heatmaps.
- All model versions (v1.0 ➔ v1.2) documented and stored.
- 5-fold cross-validation completed for each classification layer.
- Concepts and experimental justifications are stored in **Notion**.

---

## 🚀 Next Phase (Future Work)

- Expand to larger labeled datasets.
- Apply transformer-based models (e.g., **SciBERT**, **Hugging Face Transformers**).
- Integrate hierarchical multi-label classification for end-to-end prediction.
- Deploy as a minimal prototype or API.

---

## 🗂️ Repository Structure

| Folder/File | Description |
|:---|:---|
| `/Artefacts/` | All trained models and TF-IDF vectorizers (LogReg and SVM versions) |
| `/Data/` | CSV datasets used for training and evaluation |
| `NLP_Pipeline_Prototype_15_Abstracts.ipynb` | Early prototype for proof-of-concept |
| `NLP_Classifier_DisciplineOnly.ipynb` | Classifier for Discipline (CS, IS, IT) |
| `NLP_Classifier_SubfieldOnly_CS.ipynb` | Subfield classifier for CS (AI, ML, CV, CYB, PAST) |
| `NLP_Classifier_SubfieldOnly_IS.ipynb` | Subfield classifier for IS (BSP, DSA, ENT, GOV, IMP) |
| `NLP_Classifier_SubfieldOnly_IT.ipynb` | Subfield classifier for IT (CLD, SEC, IOTNET, OPS) |
| `NLP_Methodology_Classifier.ipynb` | Methodology classifier (QLT, QNT, M) |
| `CrossValidation_AllModels.ipynb` | 5-fold CV for all models |
| `Evaluate_DisciplineClassifier.ipynb` | Separate test evaluation on discipline model |
| `README.md` | Project overview (this file) |
| `TASKS.md` | Timeline and progress log |

---

## 📊 Data Files (`/Data/`)

| File | Description |
|:---|:---|
| `Evaluation Dataset - 9 entries.csv` | External evaluation set for Discipline classifier |
| `NLP_Abstract_Dataset (Discipline).csv` | 15-entry prototype dataset used in early pipeline testing |
| `NLP_Abstract_Dataset (Discipline)(105).csv` | Final Discipline-labeled dataset |
| `NLP_Abstract_Dataset (Subfield)(105).csv` | Subfield-labeled dataset for CS, IS, IT |
| `NLP_Abstract_Dataset (Methodology)(105).csv` | Methodology-labeled dataset |

---

## 🧠 Model Artifacts (`/Artefacts/`)

| File | Description |
|:---|:---|
| `discipline_classifier_logreg.pkl` | Bi-gram Logistic Regression classifier for Discipline |
| `tfidf_vectorizer.pkl` | Bigram TF-IDF vectorizer for Discipline model |
| `subfield_classifier_logreg_cs.pkl` | LogReg classifier for CS subfields |
| `tfidf_vectorizer_cs.pkl` | TF-IDF for CS subfields (Logreg) |
| `subfield_classifier_logreg_is.pkl` | LogReg classifier for IS subfields |
| `tfidf_vectorizer_is.pkl` | TF-IDF for IS subfields (Logreg) |
| `subfield_classifier_logreg_it.pkl` | LogReg classifier for IT subfields |
| `tfidf_vectorizer_it.pkl` | TF-IDF for IT subfields(Logreg) |
| `methodology_classifier_logreg.pkl` | LogReg classifier for Methodology |
| `tfidf_vectorizer_methodology.pkl` | Bigram TF-IDF for Methodology  |
| `cs_subfield_classifier_svm_smote.pkl` | SVM+SMOTE classifier for CS subfields |
| `cs_subfield_vectorizer_smote.pkl` | Bigram TF-IDF for CS subfields |
| `is_subfield_classifier_svm_smote.pkl` | SVM+SMOTE classifier for IS subfields |
| `is_subfield_vectorizer_smote.pkl` | Bigram TF-IDF for IS subfields |
| `it_subfield_classifier_svm_smote.pkl` | SVM+SMOTE classifier for IT subfields |
| `it_subfield_vectorizer_smote.pkl` | Bigram TF-IDF for IT subfields |
| `methodology_classifier_svm.pkl` | SVM classifier for Methodology (v1.2) |
| `baseline_classifier_logreg.pkl` | base reference LogReg model |

---

## 📝 Other Notes

- All model versioning (v1.0 to v1.2) and pipeline evolution is documented and reproducible.

---

## 🎯 Project Goals

- Build a scalable, interpretable NLP classification pipeline for academic text.
- Validate methodology across small, imbalanced datasets.
- Prepare for research extensions with transformer-based models.

---

## 👨‍💻 Author

Aanand Prabhu ([@aanandprabhu30](https://github.com/aanandprabhu30))

> _This project is submitted as part of my BSc Final Year Project in Computer Science._
