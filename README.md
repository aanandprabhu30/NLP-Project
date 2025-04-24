# NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research papers into:

- 📚 **Discipline** (e.g., Computer Science, Information Systems)
- 🧠 **Subfield** (e.g., AI, Security, EdTech)
- 🧪 **Methodology** (Qualitative, Quantitative, Mixed)

### 🛠 Built With
- Python
- Jupyter Notebook
- scikit-learn
- spaCy
- NLTK

> 🔍 Current Phase: Completed discipline + subfield classification pipeline.  
> 📌 Next Phase: Methodology classification.

## 📁 Repository Structure
- [`NLP_Classifier_DisciplineOnly.ipynb`](./NLP_Classifier_DisciplineOnly.ipynb)  
  → Classifies abstracts into computing disciplines (CS, IS, IT) using Logistic Regression on TF-IDF features.

- [`NLP_Classifier_SubfieldOnly_CS.ipynb`](./NLP_Classifier_SubfieldOnly_CS.ipynb)  
  → Classifies CS abstracts into subfields: AI, ML, CV, CYB, PAST. Includes evaluation and heatmap.

- [`NLP_Classifier_SubfieldOnly_IS.ipynb`](./NLP_Classifier_SubfieldOnly_IS.ipynb)  
  → Classifies IS abstracts into subfields: BSP, DSA, ENT, GOV, IMP. Includes confusion matrix and saved model.

- [`NLP_Classifier_SubfieldOnly_IT.ipynb`](./NLP_Classifier_SubfieldOnly_IT.ipynb)  
  → Classifies IT abstracts into subfields: CLD, EMERGE, IOTNET, OPS, SEC. Evaluated using Logistic Regression.

- [`Evaluate_DisciplineClassifier.ipynb`](./Evaluate_DisciplineClassifier.ipynb)  
  → Uses a separate 9-paper dataset to evaluate discipline-level classifier performance and confusion matrix.

- [`NLP_Pipeline_Prototype_15_Abstracts.ipynb`](./NLP_Pipeline_Prototype_15_Abstracts.ipynb)  
  → Early prototype testing on 15 abstracts to validate end-to-end pipeline structure before scaling to 105.

### 📂 Artefacts/
Contains all saved models and vectorizers from the classification pipelines:

- `discipline_classifier_logreg.pkl`  
  → Final Logistic Regression model trained on 105 abstracts to classify into CS, IS, or IT.

- `tfidf_vectorizer.pkl`  
  → TF-IDF vectorizer fitted on the full 105-paper dataset for discipline classification.

- `subfield_classifier_logreg_cs.pkl`  
  → Logistic Regression model trained on 35 CS abstracts to classify into AI, ML, CV, CYB, or PAST.

- `tfidf_vectorizer_cs.pkl`  
  → TF-IDF vectorizer used for CS subfield classification.

- `subfield_classifier_logreg_is.pkl`  
  → Logistic Regression model trained on 35 IS abstracts to classify into BSP, DSA, ENT, GOV, or IMP.

- `tfidf_vectorizer_is.pkl`  
  → TF-IDF vectorizer used for IS subfield classification.

- `subfield_classifier_logreg_it.pkl`  
  → Logistic Regression model trained on 35 IT abstracts to classify into CLD, EMERGE, IOTNET, OPS, or SEC.

- `tfidf_vectorizer_it.pkl`  
  → TF-IDF vectorizer used for IT subfield classification.

### 📁 Other Files (non-notebooks)
- `.gitignore`: Hides unwanted system/cache files from Git
- `README.md`: This file!
- `TASKS.md` : To keep track of all my tasks as they are being completed 

### 📌 Project Goals
- Build a clean text classification pipeline
- Make it beginner-friendly and modular
- Ensure version control using GitHub

### 👨‍💻 Author
- Aanand Prabhu ([@aanandprabhu30](https://github.com/aanandprabhu30))

---

_This project is part of my final year project BSc in Computer Science.