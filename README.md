# NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research papers into:

- 🧐 **Discipline** (e.g., Computer Science, Information Systems, Information Technology)
- 🧐 **Subfield** (e.g., AI, Security, EdTech, etc.)
- 🧐 **Methodology** (Qualitative, Quantitative, Mixed)

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

✅ Completed full pipeline:  
**Discipline ➔ Subfield ➔ Methodology** classification.

- Conducted detailed 5-fold cross-validation for Discipline, Subfield, and Methodology tasks.
- Documented fold-wise accuracies, average accuracies, standard deviations, and comparative analysis in Notion.

---

## 🚀 Next Phase (Future Work)

- Expand to larger datasets
- Experiment with advanced models like **BERT**, **Hugging Face Transformers**
- Explore hierarchical classification (e.g., Discipline ➔ Subfield ➔ Methodology in one model)

---

## 🗂️ Repository Structure

| Folder/File | Description |
|:---|:---|
| `/Artefacts/` | Saved Logistic Regression models and TF-IDF vectorizers for all classification phases |
| `NLP_Pipeline_Prototype_15_Abstracts.ipynb` | Early prototype testing on 15 abstracts |
| `NLP_Classifier_DisciplineOnly.ipynb` | Classifies abstracts into CS, IS, or IT disciplines |
| `Evaluate_DisciplineClassifier.ipynb` | Separate evaluation of the Discipline classifier |
| `NLP_Classifier_SubfieldOnly_CS.ipynb` | Classifies CS abstracts into AI, ML, CV, CYB, PAST |
| `NLP_Classifier_SubfieldOnly_IS.ipynb` | Classifies IS abstracts into BSP, DSA, ENT, GOV, IMP |
| `NLP_Classifier_SubfieldOnly_IT.ipynb` | Classifies IT abstracts into CLD, EMERGE, IOTNET, OPS, SEC |
| `NLP_Methodology_Classifier.ipynb` | Classifies abstracts into Research Methodologies: QLT, QNT, M |
| `CrossValidation_AllModels.ipynb` | 5-fold cross-validation results for all tasks |
| `Evaluation Dataset - 9 entries.csv` | Dataset used for evaluating discipline classifier externally |
| `NLP_Abstract_Dataset (Discipline)(105).csv` | Main 105 abstracts labeled by Discipline |
| `NLP_Abstract_Dataset (Subfield)(105).csv` | Main 105 abstracts labeled by Subfield |
| `NLP_Abstract_Dataset (Methodology)(105).csv` | Main 105 abstracts labeled by Methodology |

---

## 🗃️ Saved Models and Vectorizers (`/Artefacts/`)

| File | Description |
|:---|:---|
| `discipline_classifier_logreg.pkl` | Logistic Regression model for CS, IS, IT classification |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer fitted on 105-discipline dataset |
| `subfield_classifier_logreg_cs.pkl` | Logistic Regression model for CS subfields |
| `tfidf_vectorizer_cs.pkl` | TF-IDF vectorizer for CS subfield dataset |
| `subfield_classifier_logreg_is.pkl` | Logistic Regression model for IS subfields |
| `tfidf_vectorizer_is.pkl` | TF-IDF vectorizer for IS subfield dataset |
| `subfield_classifier_logreg_it.pkl` | Logistic Regression model for IT subfields |
| `tfidf_vectorizer_it.pkl` | TF-IDF vectorizer for IT subfield dataset |
| `methodology_classifier_logreg.pkl` | Logistic Regression model for Methodology classification |
| `tfidf_vectorizer_methodology.pkl` | TF-IDF vectorizer for Methodology classification |

---

## 📝 Other Files

| File | Purpose |
|:---|:---|
| `.gitignore` | Hides system/cache files from Git |
| `README.md` | Project overview (this file) |
| `TASKS.md` | Tracks task completion throughout the project |

---

## 🎯 Project Goals

- Build a clean, reproducible text classification pipeline.
- Focus on modularity, clarity, and scalability for future upgrades.
- Ensure beginner-friendliness for understanding basic NLP pipelines.
- Align with academic best practices for final year BSc projects.

---

## 👨‍💻 Author

Aanand Prabhu ([@aanandprabhu30](https://github.com/aanandprabhu30))

---

> _This project is part of my final year project BSc in Computer Science._
