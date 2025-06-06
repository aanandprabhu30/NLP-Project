# 🧠 NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research abstracts by:

- 🧐 **Discipline** – Computer Science (CS), Information Systems (IS), Information Technology (IT)
- 🧐 **Subfield** – AI, ML, CV, CYB, BSP, SEC, CLD, etc.
- 🧐 **Research Methodology** – Qualitative, Quantitative, Mixed 

---

## 🛠 Built With

- Python
- Jupyter Notebook / Google Colab
- scikit-learn
- XGBoost
- Hugging Face Transformers
- PEFT (LoRA)
- pandas, seaborn, matplotlib
- joblib

---

## 🧪 Environment Setup (nlp-bert kernel for local use)

Most of the project was executed locally using a dedicated virtual environment and Jupyter kernel named **`nlp-bert`**, created specifically for BERT and XGBoost model stability.

> 🔧 **Note:** The `xgboost` models repeatedly crashed under the default Anaconda kernel. A clean virtualenv-based kernel (`nlp-bert`) resolved this.

### ⚙ Python Version
- Python 3.11

---

### 📦 To Recreate the Environment Locally:

```bash
# Step 1: Create a virtual environment
python3 -m venv nlp-bert
source nlp-bert/bin/activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Register the kernel in Jupyter
pip install ipykernel
python -m ipykernel install --user --name=nlp-bert --display-name "Python 3 (nlp-bert)" 
```
---
## 📍 Current Phase (as of 5th June 2025)

✅ **Discipline classifier v3.1 (SciBERT + LoRA) trained on full 5402-paper dataset**  
🔬 `v3.0` (DeBERTa + LoRA) retained for documentation; underperformed (Macro F1 = 0.38)  
✅ **Subfield classifiers finalized for CS (1498 papers), IS (374 papers), and IT (504 papers)**  
✅ **Methodology classifier (2028-paper set) uses two-stage architecture with threshold tuning**

---

- **Discipline Model:**  
  - ✅ `v2.2`: SciBERT + XGBoost → Accuracy = 91%, Macro F1 = 0.89  (1138 papers)  
  - ❌ `v3.0`: DeBERTa + LoRA → Accuracy = 54%, Macro F1 = 0.38  (1138 papers)  
  - ✅ `v3.1`: SciBERT + LoRA → Accuracy = 82.05%, Macro F1 = 0.81  (5402 papers)  
    - Class F1s: CS = 0.85, IS = 0.82, IT = 0.76  

- **Subfield Models:**  
  - ✅ CS: `v2.3`: XGBoost + SPECTER (default) → Accuracy = 76%, Macro F1 = 0.75  (1498 papers)  
  - ✅ IS: `v2.4`: XGBoost (tuned) + SPECTER → Accuracy = 89%, Macro F1 = 0.90  (374 papers)  
  - ✅ IT: `v2.4`: XGBoost (tuned) + SPECTER → Accuracy = 83%, Macro F1 = 0.80  (504 papers)  

- **Methodology Models (2028 papers):**  
  - `v2.3`: SPECTER + XGBoost + SMOTE → Macro F1 = 0.66 (Mixed F1 = 0.35)  
  - `v2.5a`: SPECTER + XGBoost (manual class weights) → Macro F1 = 0.61  
  - ✅ `v2.6`: Two-stage XGBoost (Mixed vs Non-Mixed → Qual vs Quant)  
    - Mixed Threshold = 0.15 → Macro F1 = 0.66  
    - Class F1s: Qual = 0.91, Quant = 0.81, Mixed = 0.25  

- **Evaluation:**  
  - Full 80/20 stratified split used for final performance reporting  
  - 5-fold CV retained for early versions (v2.0–v2.2.1)  
  - Metrics include per-class F1, macro/weighted F1, and confusion heatmaps  

- **Artefacts:**  
  - All models, tokenizers, vectorizers saved as `.pkl` files under `/Artefacts`  
  - **All `.pkl` artefacts are tracked using Git LFS** to ensure scalable versioning  
  - Two-stage methodology components (v2.6) stored as separate classifiers + threshold  

- **Documentation:**  
  - Project versions and training logs tracked in Notion  
  - Final results and artefact names synced in README and TASKS.md  

- **Scripts:**  
  - Modular scripts for data scraping, embedding generation, classifier training, and evaluation in `Scripts/`

> 🔁 **Final architectures:**  
> - `Discipline`: ✅ v3.1 (SciBERT + LoRA)  
> - `Subfield`: ✅ v2.3/v2.4 (SPECTER + XGBoost tuned)  
> - `Methodology`:  
>   - `v2.3`: Single-stage (SMOTE + XGBoost)  
>   - ✅ `v2.6`: Two-stage XGBoost with threshold tuning (Mixed threshold = 0.15)

> ℹ️ Version Notes:  
> - `v1.x`: TF-IDF + classical models (LogReg, SVM)  
> - `v2.x`: Contextual embeddings (SciBERT, SPECTER)  
> - `v3.0`: DeBERTa + LoRA (underperformed)  
> - ✅ `v3.1`: SciBERT + LoRA (best contextual + PEFT performance so far)
> -  `v2.2` predictions used to generate trust scores for all 5,402 papers → retained 4,838 high-confidence samples (trust ≥ 0.8) for training `v4.0`


---
## 🚀 Next Phase (Future Work – Post v3.1)

Following the completion of the full v2.x series and the new SciBERT + LoRA model for Discipline (`v3.1`), the next phase focuses on improving robustness, generalizability, and deployment readiness — starting with `v4.0` trained on high-confidence labels.

- 🎯 **Train `discipline_classifier_scilora_v4.0` on trusted subset (4,838 samples)**  
  – Use curriculum learning, class weighting (based on effective sample size), and/or focal loss to improve IT F1 and push toward 95% macro accuracy.

- ⚖️ **Rebalance Methodology classifier with cost-sensitive strategies**  
  – Improve Mixed-class F1 by using `scale_pos_weight` or focal loss in the binary Mixed-vs-NonMixed stage.

- 🤖 **Fine-tune SPECTER or SciBERT on the full corpus**  
  – Apply parameter-efficient fine-tuning (LoRA, adapters) or continued pretraining on the 5402-paper set for better domain alignment.

- 🧪 **Augment and expand the Methodology dataset**  
  – Label 50–100 additional Mixed-method abstracts to balance classes; apply back-translation or NLPAug for synthetic variation.

- 🧮 **Try multi-label formulation for Methodology**  
  – Predict Qualitative and Quantitative flags independently; infer Mixed if both are active to improve hybrid classification flexibility.

- 🔄 **Dynamic ensemble fallback for Methodology**  
  – Switch between `v2.3` and `v2.6` using prediction entropy or confidence thresholds to recover Mixed recall.

- 🧩 **Explore ensemble/meta-classifiers**  
  – Combine outputs from TF-IDF+SVM, BERT+LR, and SPECTER+XGB via meta-learning (e.g., Logistic Regression or XGBoost stacking).

- 🧭 **Integrate full hierarchical inference pipeline**  
  – Run classifiers in sequence: Discipline → Subfield → Methodology; invoke AI/ML disambiguator if needed.

- 🧠 **Build an inference interface**  
  – Create a lightweight web UI using Streamlit or FastAPI to classify abstracts and visualize model explanations (e.g., SHAP).

- 📊 **Extend error analysis and ablation studies**  
  – Analyze failure patterns across `v2.3`, `v2.6`, and `v3.1`; investigate class-specific confusion and drift to guide `v3.2`.

- 🔄 **Upgrade evaluation pipeline**  
  – Refactor `evaluate_methodology.py` to support threshold sweeps, exportable results, and consistent version logging.

- 📝 **Standardize tracking and version control**  
  – Maintain updated Notion and README entries for each version; hash and verify `.pkl` artefacts (tracked with Git LFS) for reproducibility.


---

## 🗂️ Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `/Artefacts/` | Trained classifiers + vectorizers + evaluation visuals |
| `/Data/` | All labeled datasets used across classification tasks |
| `/Scripts/`| All Scripts used for scraping data|
| `README.md` | This file |
| `TASKS.md` | To-do log and milestones |
| `/Notebooks/` | All experiment notebooks across v1.x, v2.x, and v3.x for Discipline, Subfield, Methodology |

---
## 📓 Notebooks (`/Notebooks/`)

| Notebook | Description |
|----------|-------------|
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
| `NLP_Classifier_DisciplineOnly_SciBERT_XGBoost_(v2.2).ipynb` | ✅ Final discipline classifier using SciBERT + XGBoost (1138-paper set) |
| `cs_subfield_classifier_specter_xgboost (v2.3 and v2.4).ipynb` | ✅ Final CS subfield classifiers (1498 papers, default + tuned) |
| `it_subfield_classifier_specter_xgboost (v2.3 and v2.4).ipynb` | ✅ Final IT subfield classifiers (504 papers) |
| `is_subfield_classifier_specter_xgboost (v2.3 and v2.4).ipynb` | ✅ Final IS subfield classifiers (374 papers) |
| `ai_vs_ml_disambiguator.ipynb` | 🧩 Fallback classifier to separate AI/ML in CS |
| `methodology_classifier_specter_xgboost_(v2.3,_v2.4_and_v2.5).ipynb` | SPECTER + XGBoost Methodology pipeline (default → tuned → weighted) |
| `methodology_classifier_specter_xgboost_v2.6.ipynb` | ✅ Two-stage Methodology Classifier (Mixed ➝ Qual/Quant) |
| `discipline_classifier_deberta_lora_v3.0.ipynb` | 🧪 DeBERTa + LoRA experiment (F1 = 0.38, not selected) |
| `lora_discipline_classifier(v3.1).ipynb` | ✅ Final v3.1 discipline classifier using LoRA-tuned SciBERT (F1 = 0.81) |
| `discipline_trust_score_filtering_v0.1.ipynb` | ✅ Used v2.2 predictions to generate trust scores and filter 4,838 high-confidence samples |

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
| `CS_subfields.csv` | Final CS subfield dataset (1498 papers) collected via arXiv API for training v2.3 and v2.4 |
| `IS_subfields.csv`      | Final IS subfield dataset (374 papers, hand-labeled, multi-source) for v2.3 and v2.4 |
| `IT_subfields.csv`      | Final IT subfield dataset (504 papers, hand-labeled, multi-source) for v2.3 and v2.4 |
| `methodology.csv` | Final methodology dataset (2,028 papers, hand-labeled, arXiv and Semantic scholar) for v2.3-2.5a|
| `Expanded Discipline Dataset.csv` | Combined 5,402-paper dataset (CS, IS, IT) used in `v3.1` SciBERT + LoRA discipline classifier |
| `expanded_discipline_with_preds.csv` | All 5,402 papers with v2.2 predicted labels and trust scores (used for filtering) |
| `trusted_discipline_dataset.csv` | High-confidence subset (4,838 papers, trust ≥ 0.8) used for training v4.0 |
---

## 🧠 Model Artifacts (`/Artefacts/`)

> 💾 All `.pkl` artefacts below are tracked using **Git LFS** for efficient versioning and storage.


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
| `cs_subfield_xgb_model_v2.3.pkl`         | XGBoost (default) model for CS subfield classification (1498-paper dataset, SPECTER embeddings) |
| `cs_subfield_label_encoder_v2.3.pkl`     | Label encoder for CS subfield classifier v2.3 |
| `cs_subfield_xgb_model_v2.4_tuned.pkl`   | Tuned XGBoost model (GridSearchCV) for CS subfield classification (v2.4, SPECTER) |
| `ai_ml_disambiguator_logreg_v1.pkl`      | Logistic Regression model trained to disambiguate AI vs ML using SPECTER embeddings |
| `ai_ml_label_encoder.pkl`                | Label encoder for the AI vs ML disambiguator |
| `is_subfield_xgb_model_v2.3.pkl`         | XGBoost (default) model for IS subfield classification (SPECTER, 374-paper dataset) |
| `is_subfield_xgb_model_v2.4_tuned.pkl`   | XGBoost (GridSearchCV-tuned) model for IS subfield classification (SPECTER, 374-paper dataset) |
| `is_subfield_label_encoder_v2.3.pkl`     | Label encoder for IS subfield classifier (v2.3/v2.4)                               |
| `it_subfield_xgb_model_v2.3.pkl`         | XGBoost (default) model for IT subfield classification (SPECTER, 504-paper dataset) |
| `it_subfield_xgb_model_v2.4_tuned.pkl`   | XGBoost (GridSearchCV-tuned) model for IT subfield classification (SPECTER, 504-paper dataset) |
| `it_subfield_label_encoder_v2.3.pkl`     | Label encoder for IT subfield classifier (v2.3/v2.4)                               |
| `methodology_xgb_v2.3.pkl`                   | Trained methodology classifier model (v2.3: SPECTER + XGBoost default)         |
| `methodology_label_encoder_v2.3.pkl`         | Label encoder for methodology classifier (v2.3)                                |                   | `methodology_xgb_model_v2.4_tuned.pkl`       | Tuned methodology classifier model (v2.4: SPECTER + XGBoost + SMOTE)           |
| `methodology_xgb_manual_weights_v2.5a.pkl`   | Methodology classifier model with manual class weights (v2.5a: Mixed=2, Qualitative=1, Quantitative=1) |
| `methodology_xgb_class_weighted_v2.5.pkl`    | Methodology classifier model with optimized class weights (v2.5: grid-tuned ratios) |
| `methodology_binary_mixed_model_v2.6.pkl`     | Binary XGBoost classifier for detecting Mixed vs Non-Mixed (Stage 1 of v2.6 two-stage model) |
| `methodology_qual_quant_model_v2.6.pkl`       | XGBoost classifier for classifying Qualitative vs Quantitative (Stage 2)                     |
| `methodology_mixed_threshold_v2.6.pkl`        | Threshold value (float = 0.15) used in Stage 1 classification                                |
| `methodology_specter_embeddings_v2.6.pkl`     | 768-dim SPECTER embeddings generated from Title + Abstract for all 2028 methodology samples  |
| `discipline_classifier_deberta_lora_v3.0.pkl` | 🧪 LoRA-tuned DeBERTa discipline classifier (v3.0); strong CS recall but failed IT prediction (F1 = 0.38); not selected |
| `tokenizer_deberta_lora_v3.0.pkl`             | Tokenizer used for v3.0 DeBERTa classifier (HuggingFace `microsoft/deberta-base`) |
| `label2id_deberta_lora_v3.0.pkl`              | Manual mapping: {'CS': 0, 'IS': 1, 'IT': 2} for v3.0 LoRA classifier |
| `lora_model_v3.1.pkl`                      | ✅ Final v3.1 discipline classifier (LoRA-tuned SciBERT); Accuracy = 82.05%, Macro F1 = 0.81 |
| `tokenizer_v3.1.pkl`                       | Tokenizer used for v3.1 SciBERT + LoRA model (`allenai/scibert_scivocab_uncased`) |
| `label2id_v3.1.pkl`                        | Label mapping: {'CS': 0, 'IS': 1, 'IT': 2} used for v3.1 |
| `id2label_v3.1.pkl`                        | Reverse mapping for prediction decoding |
| `model_info_v3.1.pkl`                      | Metadata JSON: accuracy, F1 scores, timestamp, and configuration for v3.1 |

---
### Data Collection Scripts (`/Scripts/`)

| Script                             | Description                                                                                               |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `arxiv_cs.py`                        | Scrape general Computer Science papers from arXiv.                                                        |
| `arxiv_cs_se_scraper.py`              | Scrape Software Engineering papers from arXiv for IT/SE dataset (requires manual review for IT relevance).|
| `arxiv_csdc_scraper.py`               | Scrape Data Center topic papers from arXiv (IT infrastructure).                                           |
| `arxiv_csni_scraper.py`               | Scrape Network Infrastructure papers from arXiv (for IT dataset).                                         |
| `download_it_papers_no_pandas_v3.py`  | Download and parse IT papers from multiple sources (cloud, edge, etc.); no pandas dependency.             |
| `harvest_it_links.py`                 | Collect/harvest links for IT papers prior to full metadata scraping.                                      |
| `is_scraper.py`                       | Scrape AMCIS Information Systems conference papers (IS dataset).                                          |
| `is_scraper_ss.py`                    | Scrape Information Systems papers from Semantic Scholar (supplements AMCIS collection).                   |
| `it_core.py`                          | Core IT domain paper collector (scraping, metadata extraction, discipline filtering).                     |
| `it_v2.py`                            | Updated version of IT paper collector/validator (improved over it_core.py).                               |
| `IT.py`                               | General-purpose IT paper downloader and cleaner; aggregates outputs from multiple IT scripts.             |
| `itc.py`                              | Custom/incremental IT collection script (handles specific IT subtopics or custom link batches).           |
| `semantic_scholar_web_scraper_loose.py`| Looser Semantic Scholar scraper for additional computing papers (CS/IS/IT) for manual review.            |
| `fetch_arxiv_cs_subfields_balanced.py` | Collect up to 300 recent arXiv papers per CS subfield (AI, ML, CV, CYB, PAST) to create a balanced 1498-paper dataset for training v2.3 and v2.4 classifiers |
| `is_bsp.py`                           | Scrape IS papers for BSP (Blockchain, Security & Privacy) subfield from Semantic Scholar                 |
| `is_dsa.py`                           | Scrape IS papers for DSA (Decision Support & Analytics) subfield from Semantic Scholar                   |
| `is_ent.py`                           | Scrape IS papers for ENT (Enterprise Systems) subfield from Semantic Scholar                             |
| `is_gov.py`                           | Scrape IS papers for GOV (e-Governance & Public Systems) subfield from Semantic Scholar                  |
| `is_imp.py`                           | Scrape IS papers for IMP (Tech Adoption & Impact) subfield from Semantic Scholar                         |
| `extra_papers.py`                     | Supplement and validate additional IT papers (OPS and IOTNET) from Semantic Scholar                      |
| `it_ss.py`                            | Scrape IT subfield papers (IoT, Edge, Cloud, etc.) from Semantic Scholar                                 |
| `it_arxiv.py `                        | Scrape IT-specific papers (Cloud, Edge, Infrastructure, etc.) from arXiv                                 |
| `reclassify_discipline.py`    | Auto-relabel or remove abstracts based on updated discipline definitions and context (discipline reclassification)|
| `methodology_ss.py`          | Scrape methodology-related papers from Semantic Scholar to expand the methodology dataset                        |
| `methodology_checker_v3.py`  | Advanced methodology label validation (v3) for detecting and correcting methodology annotation errors            |
| `discipline_auditor.py`       | Audit and report on discipline label consistency across all datasets                                             |
| `methodology_checker.py`      | Initial methodology label checker for basic validation of methodology categories                                 |
| `arxiv_methodology.py`        | Fetch and filter arXiv abstracts using methodology-focused keywords to build the methodology corpus              |
---

## 🔁 Discipline Version Map

| Version   | Model                                    | Vectorizer / Embedding                | Dataset               | Notes                       |
|-----------|-------------------------------------------|---------------------------------------|-----------------------|-----------------------------|
| v1.0      | Logistic Regression                       | TF-IDF (unigram)                      | 105                   | Baseline, small dataset     |
| v1.1      | Logistic Regression                       | TF-IDF (bigram)                       | 105                   | Improved context, 80/20 CV  |
| v2.2      | XGBoost (`disc_scibert_xgboost_v2.2.pkl`) | SciBERT (768-dim, Title+Abstract)    | **1138**              | ✅ Final, full dataset      |
| v3.0      | XGBoost + LoRA-tuned DeBERTa (`discipline_classifier_deberta_lora_v3.0.pkl`) | DeBERTa (768-dim, PEFT via LoRA) | **1138** | Experimental; strong CS recall (F1 = 0.67), but failed IT class (F1 = 0.00); retained for documentation only |
| v3.1 | XGBoost + LoRA-wrapped SciBERT (`lora_model_v3.1.pkl`) | SciBERT (768-dim, PEFT via LoRA) | **5402** | ✅ Best generalization; CS F1 = 0.85, IS F1 = 0.82, IT F1 = 0.76 |

---
## 🧠 Subfield Version Map

| Version | Discipline | Model                     | Vectorizer / Embedding          | Dataset                          | Split                  | Accuracy | Macro F1 | Key Class F1 Scores                            | Notes |
|---------|------------|----------------------------|----------------------------------|----------------------------------|------------------------|----------|----------|------------------------------------------------|-------|
| v1.0    | CS         | Logistic Regression        | Unigram TF-IDF                   | 35 abstracts                     | 70/30 stratified       | 0.18     | 0.20     | CYB: 0.33, others: 0.00                         | Pipeline validated; severe CYB bias |
| v1.2    | CS         | SVM + SMOTE (k=1)          | Bigram TF-IDF (min_df=2)         | 35 abstracts                     | 80/20 stratified       | 0.50     | 0.43     | CV: 1.00, CYB: 0.67, AI: 0.50                   | Improved recall of minority subfields |
| v2.3    | CS         | XGBoost (default)          | SPECTER (768-dim)                | 1498-paper CS dataset            | 80/20 stratified       | 0.76     | 0.75     | AI: 0.58, CV: 0.85, CYB: 0.83, ML: 0.66, PAST: 0.86 | Strong baseline; slight AI/ML confusion |
| v2.4    | CS         | XGBoost (tuned)            | SPECTER (768-dim)                | 1498-paper CS dataset            | 80/20 stratified       | 0.75     | 0.74     | AI: 0.55, CV: 0.84, CYB: 0.85, ML: 0.62, PAST: 0.87 | Regularized model with better generalization |
| —       | CS (AI/ML) | Logistic Regression        | SPECTER (768-dim)                | 600-paper AI/ML subset           | 80/20 stratified       | 0.68     | 0.67     | AI: 0.67, ML: 0.68                              | Binary disambiguator activated when CS v2.4 predicts AI or ML |
| v1.0    | IS         | Logistic Regression        | Unigram TF-IDF                   | 35 abstracts                     | 70/30 stratified       | 0.27     | 0.20     | ENT: 0.33, others: 0.00                         | ENT-dominant predictions; low diversity |
| v1.2    | IS         | SVM + SMOTE (k=1)          | Bigram TF-IDF (min_df=2)         | 35 abstracts                     | 80/20 stratified       | 0.29     | 0.21     | DSA: 0.67, IMP: 0.40                            | Slightly improved diversity |
| v2.3    | IS | XGBoost (default) | SPECTER (768-dim) | 374-paper IS dataset | 80/20 stratified | 0.88 | 0.88 | BSP: 0.92, DSA: 0.86, ENT: 0.81, GOV: 0.88, IMP: 0.94 | Huge jump from v1.2; SPECTER features |
| v2.4    | IS | XGBoost (tuned)   | SPECTER (768-dim) | 374-paper IS dataset | 80/20 stratified | 0.89 | 0.90 | BSP: 0.92, DSA: 0.83, ENT: 0.91, GOV: 0.87, IMP: 0.94 | Best overall, extremely stable        |
| v1.0    | IT         | Logistic Regression        | Unigram TF-IDF                   | 35 abstracts                     | 70/30 stratified       | 0.27     | 0.27     | CLD: 0.46, SEC: 0.50                            | CLD-dominant |
| v1.2    | IT         | SVM + SMOTE (k=1)          | Bigram TF-IDF (min_df=2)         | 35 abstracts                     | 80/20 stratified       | 0.71     | 0.71     | CLD: 1.00, OPS: 0.67, SEC: 0.67, IOTNET: 0.50   | - |
| v2.3    | IT | XGBoost (default) | SPECTER (768-dim) | 504-paper IT dataset | 80/20 stratified | 0.83 | 0.80 | CLD: 0.87, IOTNET: 0.83, OPS: 0.61, SEC: 0.89 | Major boost over v1.2; SPECTER embeddings |
| v2.4    | IT | XGBoost (tuned)   | SPECTER (768-dim) | 504-paper IT dataset | 80/20 stratified | 0.83 | 0.80 | CLD: 0.86, IOTNET: 0.84, OPS: 0.64, SEC: 0.87 | Marginal improvement, very stable          |

---

## 🔁 Methodology Version Map

| Version   | Model                                | Vectorizer / Embedding                          | Notes |
|-----------|---------------------------------------|--------------------------------------------------|-------|
| v1.0      | `methodology_classifier_logreg.pkl`   | `tfidf_vectorizer_methodology.pkl`              | Abstract-only baseline |
| v1.2      | `methodology_classifier_svm.pkl`      | `tfidf_vectorizer_methodology_smote.pkl`        | SVM + SMOTE |
| v2.0      | `methodology_classifier_v2_titleabstract.pkl` | `tfidf_vectorizer_methodology_v2_titleabstract.pkl` | Title + Abstract |
| v2.1.1    | `Logistic Regression (Scaled)`          | `MiniLM (384-dim) via sentence-transformers`      | CV only – no model saved |
| v2.2      | `methodology_scibert_xgb_v2.2_model.pkl` | `SciBERT (768-dim, Title + Abstract)`             | XGBoost baseline |
| v2.2.1    | `methodology_scibert_xgb_v2.2.1_smote_model.pkl` | `SciBERT + SMOTE (768-dim)`                  | ✅ Best performance on small data |
| v2.2.1-CV | N/A                                   | N/A                                              | CV-only evaluation (5-fold) |
| v2.3   | `methodology_xgb_v2.3.pkl`                   | `SPECTER (768-dim)`                     | XGBoost default parameters on SPECTER embeddings                  |
| v2.4   | `methodology_xgb_model_v2.4_tuned.pkl`       | `SPECTER (768-dim)`                     | XGBoost tuned via GridSearchCV on SPECTER embeddings              |
| v2.5   | `methodology_xgb_class_weighted_v2.5.pkl`    | `SPECTER (768-dim)`                     | Balanced weights via `compute_class_weight(class_weight="balanced")` |
| v2.5a  | `methodology_xgb_manual_weights_v2.5a.pkl`   | `SPECTER (768-dim)`                     | Manual weights (Mixed=2, Qualitative=1, Quantitative=1); Mixed F1 ↑0.11→0.19 |
| v2.6   | `methodology_binary_mixed_model_v2.6.pkl` + `methodology_qual_quant_model_v2.6.pkl` | `SPECTER (768-dim)` | Two-stage model: Mixed vs Non-Mixed → Qual vs Quant. Threshold = 0.15 (saved separately). |

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
| Discipline   | v3.0        | 1138             | Test split      | 0.54         | —           | DeBERTa (PEFT via LoRA); strong CS recall (F1 = 0.67), but failed IT class (F1 = 0.00); not selected |
| Discipline   | v3.1        | 5402             | Test split      | 0.8205       | —           | SciBERT (LoRA via PEFT); strong generalization across classes (CS F1 = 0.85, IS F1 = 0.82, IT F1 = 0.76) |

| Subfield – CS| v2.3        | 1498             | Test split      | 0.76         | –           | XGBoost (default) + SPECTER (768-dim); strong CV/CYB/PAST performance |
| Subfield – CS| v2.4        | 1498             | Test split      | 0.75         | –           | XGBoost (tuned) + SPECTER (768-dim); regularized, improved PAST/CYB   |
| Subfield – IS| v2.3        | 374               | Test split      | 0.88         | –           | XGBoost (default) + SPECTER (768-dim); huge jump over v1.2             |
| Subfield – IS| v2.4        | 374               | Test split      | 0.89         | –           | XGBoost (tuned) + SPECTER (768-dim); best overall, extremely stable    |
| Subfield – IT| v2.3        | 504              | Test split      | 0.83         | –           | XGBoost (default) + SPECTER (768-dim); major boost over v1.2           |
| Subfield – IT| v2.4        | 504              | Test split      | 0.83         | –           | XGBoost (tuned) + SPECTER (768-dim); marginal improvement, very stable |
| Methodology  | v2.0        | 105              | 5-fold CV       | 0.6381       | 0.0883      | SVM + SMOTE + TF-IDF (Title + Abstract)  |
| Methodology  | v2.1.1      | 105              | 5-fold CV       | 0.4381       | 0.1143      | MiniLM + LogReg (Scaled)                 |
| Methodology  | v2.2.1-CV   | 105              | 5-fold CV       | 0.6571       | 0.1017      | SciBERT + XGBoost + SMOTE                |
| Methodology  | v2.2.1      | 105              | Test split      | 0.7619       | —           | SciBERT + XGBoost + SMOTE                |
| Methodology | v2.3  | 2,028 | Test split | 0.75 | – | Major boost in Mixed F1 to 0.35; strong Qual (0.83) & Quant (0.81) |
| Methodology | v2.4  | 2,028 | Test split | 0.73 | – | Marginal drop in Mixed F1 to 0.11; Qual stable (0.83), Quant slight dip (0.79) |
| Methodology | v2.5  | 2,028 | Test split | 0.74 | – | Balanced weights improved Mixed F1 to 0.20; maintained Qual (0.83) & Quant (0.79) |
| Methodology | v2.5a | 2,028 | Test split | 0.74 | – | Manual weights maintained Mixed F1 ≈ 0.19; robust Qual (0.82) & Quant (0.80) |
| Methodology | v2.6  | 2,028 | Test split | 0.77 | – | Two-stage XGBoost + SPECTER; Mixed vs Non-Mixed → Qual vs Quant; threshold = 0.15; Mixed F1 = 0.25, Qual F1 = 0.91, Quant F1 = 0.81 |

> **Notes:**  
> - “Test split” means a standard train/test split (often 80/20 or similar), not cross-validation.  
> - v2.2 discipline classifier (SciBERT + XGBoost) is on the full 1,138-paper dataset.  
> - All methodology classifiers up through v2.2.1 were trained & evaluated on the smaller, manually labeled 105-paper subset.  
> - From v2.3 onward, methodology uses the expanded 2,028-paper dataset (arXiv + Semantic Scholar) with test-split evaluation.  
> - v2.3 methodology (SPECTER + XGBoost default) gave a major boost in Mixed F1 (0.35) while retaining strong Qual (0.83) & Quant (0.81).  
> - v2.4 methodology (GridSearchCV-tuned XGBoost on SPECTER) saw a drop in Mixed F1 (0.11) but maintained Qual (0.83) & Quant (0.79).  
> - v2.5 methodology (balanced class weights via `compute_class_weight`) improved Mixed F1 to 0.20 and kept Qual (0.83) & Quant (0.79).  
> - v2.5a methodology (manual weights Mixed=2, Qualitative=1, Quantitative=1) achieved Mixed F1 ≈ 0.19, Qual F1 ≈ 0.82, Quant F1 ≈ 0.80.
> - v2.6 methodology (two-stage XGBoost with threshold = 0.15) improved Qual F1 to 0.91 and Quant F1 to 0.81, but Mixed F1 dropped to 0.25. Represents tradeoff between Mixed recall and overall balance.
> - v2.3 and v2.4 CS subfield classifiers are trained on a 1,498-paper arXiv dataset with SPECTER embeddings.  
> - v2.3 and v2.4 IS subfield classifiers are trained on a 374-paper Semantic Scholar dataset with SPECTER embeddings.  
> - v2.3 and v2.4 IT subfield classifiers are trained on a 504-paper arXiv + Semantic Scholar dataset with SPECTER embeddings.  
> - v2.4 subfield pipeline includes a second-stage AI/ML disambiguator (Logistic Regression on SPECTER) that can be invoked when the main CS classifier predicts AI or ML.
> - v3.0 discipline classifier (DeBERTa + LoRA) was trained on the same 1,138-paper dataset but failed to generalize; strong CS recall (F1 = 0.67), but IT recall = 0.00; accuracy = 54%, macro F1 = 0.38 — not selected.  
> - v3.1 discipline classifier (SciBERT + LoRA via PEFT) was trained on a significantly larger 5,402-paper dataset and showed strong generalization across all classes — CS F1 = 0.85, IS F1 = 0.82, IT F1 = 0.76; overall Accuracy = 82.05%, Macro F1 = 0.81.

---

## 🎯 Project Goals

✅ Build a scalable, modular NLP pipeline for automated classification of computing research abstracts across Discipline, Subfield, and Methodology levels  
✅ Handle both small-scale (105 manually labeled) and large-scale (1138+1498 abstracts) datasets using consistent, version-controlled workflows  
✅ Apply SMOTE and stratified k-fold cross-validation to improve recall for minority classes and evaluate model robustness  
✅ Demonstrate the evolution from classical models (Logistic Regression + TF-IDF) to semantic models (XGBoost + SPECTER/SciBERT)  
✅ Integrate pretrained transformer embeddings (MiniLM, SciBERT, SPECTER) for contextual understanding without fine-tuning  
✅ Use GridSearchCV to tune hyperparameters and optimize generalization in embedding-based models  
✅ Design fallback logic (e.g., AI vs ML disambiguator) to refine predictions in semantically overlapping subfields  
✅ Save all artefacts (models, vectorizers, encoders) using `joblib`, ensuring full reproducibility  
✅ Structure and document the full pipeline for extension to deployment (API or app) and publication-readiness  
✅ Create a clear foundation for future tasks like hierarchical inference, joint modeling, or end-to-end classification

---

## 🛠️ Reproducibility

To recreate the exact Python environment:

- With pip:
  ```bash
  pip install -r requirements.txt

---

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Submitted as part of my BSc Final Year Project in Computer Science – University of London_