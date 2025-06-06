# 📝 NLP Project Task Breakdown – Updated (Final Models Complete)

This markdown file tracks the step-by-step progress of the NLP classification project. Each task is designed to be small, focused, and Git-commit-worthy.

---

## 🗂️ ✅ Completed Setup & Milestone Tracker (v0 → v2.2.1)

> Archive of all setup, preprocessing, dataset creation, and baseline experiments.  
> This section covers TF-IDF models, cross-validation, and v1.0 to v2.2.1 classifier training.  
> All progress from v2.3 onward is tracked under version-specific entries above.

<details>
<summary>Click to expand full milestone list</summary>

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
</details>

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
- [x] Update all tables and results to reflect discipline model's expanded dataset and new architecture

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
## 🧠 2025–05–20 – Final CS Subfield Classifier + AI/ML Disambiguator Integration

- Added `cs_subfield_classifier_specter_xgboost (v2.3 and v2.4).ipynb` notebook containing:
    - v2.3: XGBoost (default) on 1498-paper CS dataset using SPECTER embeddings
    - v2.4: GridSearchCV-tuned XGBoost with improved generalization
- Saved artefacts for both models and label encoder in `/Artefacts/`:
    - `cs_subfield_xgb_model_v2.3.pkl`, `cs_subfield_xgb_model_v2.4_tuned.pkl`
    - `cs_subfield_label_encoder_v2.3.pkl`
- Collected CS dataset via arXiv API using new script: `fetch_arxiv_cs_subfields_balanced.py`
    - Stored as `CS_subfields.csv` in `/Data/`
    - Logged script in README under `/Scripts/` with description

- Created second-stage AI/ML disambiguator:
    - `ai_vs_ml_disambiguator.ipynb` (Logistic Regression + SPECTER, 600-paper filtered set)
    - Evaluation: 68% Accuracy, Macro F1: 0.67 (balanced F1 for AI and ML)
    - Saved model and encoder in `/Artefacts/` as:
        - `ai_ml_disambiguator_logreg_v1.pkl`
        - `ai_ml_label_encoder.pkl`
    - Logic integrated into v2.4 inference as fallback when prediction is AI or ML

- Updated:
    - `README.md`: added new artefacts, scripts, data, notebooks, version tables, and fallback description
    - `TASKS.md`: current entry
    - Notion: subfield pipeline complete and reproducible, fallback logic documented
---
## 🧠 2025–05–27 – Final IS & IT Subfield Classifiers (SPECTER + XGBoost v2.3/v2.4)

- Added `is_subfield_classifier_specter_xgboost (v2.3 and v2.4).ipynb` notebook:
    - v2.3: XGBoost (default) on 374-paper IS dataset using SPECTER embeddings
    - v2.4: GridSearchCV-tuned XGBoost for IS subfields (best Macro F1: 0.90)
- Added `it_subfield_classifier_specter_xgboost (v2.3 and v2.4).ipynb` notebook:
    - v2.3: XGBoost (default) on 504-paper IT dataset using SPECTER embeddings
    - v2.4: GridSearchCV-tuned XGBoost for IT subfields (best Macro F1: 0.80)
- Saved all new model and label encoder artefacts in `/Artefacts/`:
    - `is_subfield_xgb_model_v2.3.pkl`, `is_subfield_xgb_model_v2.4_tuned.pkl`, `is_subfield_label_encoder_v2.3.pkl`
    - `it_subfield_xgb_model_v2.3.pkl`, `it_subfield_xgb_model_v2.4_tuned.pkl`, `it_subfield_label_encoder_v2.3.pkl`
- Collected and finalized IS and IT subfield datasets:
    - `IS_subfields.csv` (374 papers, multi-source, hand-labeled)
    - `IT_subfields.csv` (504 papers, multi-source, hand-labeled)
    - Added both files to `/Data/` and README tables
- Added and documented all new scripts used for IS/IT scraping (Semantic Scholar, ArXiv) in `/Scripts/` and README
- Updated:
    - `README.md`: All version tables, dataset/artefact/script documentation, and project summary
    - `TASKS.md`: Current entry
    - Notion: Progress checkpointed; subfield classification for all three disciplines is now fully complete and reproducible

---
## 🧠 2025–05–29 – Methodology Classifier Enhancements (SPECTER + XGBoost v2.3 → v2.5a)

- Added `notebooks/methodology_classifier_specter_xgboost (v2.3, v2.4 and v2.5).ipynb` notebook:
    - v2.3: XGBoost default on 2,028-paper methodology dataset using SPECTER embeddings (Mixed F1=0.35, Qual F1=0.83, Quant F1=0.81)
    - v2.4: GridSearchCV-tuned XGBoost on SPECTER embeddings (Mixed F1=0.11, Qual F1=0.83, Quant F1=0.79)
    - v2.5: Balanced weights via `compute_class_weight(class_weight="balanced")` (Mixed F1=0.20, Qual F1=0.83, Quant F1=0.79)
    - v2.5a: Manual class weights (Mixed=2, Qualitative=1, Quantitative=1) (Mixed F1≈0.19, Qual F1≈0.82, Quant F1≈0.80)
- Saved all new model and encoder artefacts in `/Artefacts/`:
    - `methodology_xgb_v2.3.pkl`
    - `methodology_label_encoder_v2.3.pkl`
    - `methodology_xgb_model_v2.4_tuned.pkl`
    - `methodology_xgb_class_weighted_v2.5.pkl`
    - `methodology_xgb_manual_weights_v2.5a.pkl`
- Updated the `methodology.csv` dataset to 2,028 hand-labeled papers and added to `/Data/` and README tables
- Added supporting config and script files:
    - `configs/methodology_v2.5a_weights.yaml`
    - New scraping and validation scripts: `arxiv_methodology.py`, `methodology_ss.py`, `methodology_checker.py`, `methodology_checker_v3.py`, `discipline_auditor.py`, `reclassify_discipline.py`
- Updated:
    - `README.md`: Added new version maps, artefact paths, dataset entries, version comparison and notes
    - `TASKS.md`: Refreshed "Upcoming Phase" with v2.5b weight-tuning and next steps
    - Notion: Logged v2.5a results and checkpointed methodology pipeline progress
    - Implement SPECTER + XGBoost (v2.3) on 2028-paper dataset
    - Tune XGBoost using GridSearchCV (v2.4) and add weighted variant (v2.5)

----
## 🗓 2025–05–30 – Methodology Classifier v2.6 (Two-Stage + Threshold Tuning)

- Added `methodology_classifier_specter_xgboost_v2.6.ipynb` notebook:
    - Introduced a two-stage classification pipeline using SPECTER + XGBoost
    - Stage 1: Binary classifier (Mixed vs Non-Mixed)
    - Stage 2: Qual vs Quant classifier (only for Non-Mixed abstracts)
    - Conducted threshold tuning (0.10 to 0.55); selected threshold = 0.15 for balanced macro F1

- Final evaluation (test split):
    - Accuracy = 0.77
    - Macro F1 = 0.66
    - Mixed F1 = 0.25 | Qual F1 = 0.91 | Quant F1 = 0.81

- Saved the following artefacts in `/Artefacts/`:
    - `methodology_binary_mixed_model_v2.6.pkl`
    - `methodology_qual_quant_model_v2.6.pkl`
    - `methodology_mixed_threshold_v2.6.pkl`
    - `methodology_specter_embeddings_v2.6.pkl`

- Generated final confusion matrix + per-class classification report (label-aligned)
- Updated:
    - `README.md`: new version entry, artefacts, version map, comparison table, and next phase
    - `Version Comparison Table`: added v2.6 results for methodology
    - `Methodology Version Map`: documented two-stage setup and threshold logic
    - `Model Artifacts`: added all four v2.6 artefacts
----
## 🧠 2025–06–04 – Discipline Classifier v3.0 (DeBERTa + LoRA)

- Added `discipline_classifier_deberta_lora_v3.0.ipynb` notebook:
  - Implemented LoRA (Low-Rank Adaptation) on `microsoft/deberta-base` for 3-class discipline classification
  - Used Hugging Face `peft`, `transformers`, and `datasets` libraries in a modular pipeline
  - Trained on the full 1138-paper dataset; batch size 8, 5 epochs, learning rate = 2e-5

- **Final Evaluation (Test Split):**
  - Accuracy = 0.54
  - Macro F1 = 0.38
  - Per-class F1 scores:  
    - CS = 0.67  
    - IS = 0.47  
    - IT = 0.00 (unclassified)

- **Artefacts Saved to `/Artefacts/`:**
  - `discipline_classifier_deberta_lora_v3.0.pkl`
  - `tokenizer_deberta_lora_v3.0.pkl`
  - `label2id_deberta_lora_v3.0.pkl`

- **Project Updated:**
  - `README.md`: All references updated — version map, current phase, comparison table, model artefacts, footnotes
  - `.gitignore`: Added `*.pkl` to exclude large binaries from future commits
  - `requirements.txt`: Appended `transformers`, `peft`, `datasets`, `evaluate`, `bitsandbytes`
  - `LOGS.md`: This entry
  - `Notion`: v3.0 logged as experimental classifier (retained but not selected)

- **Remarks:**  
  While CS recall was strong (0.95), model failed to predict IT entirely. v2.2 remains the deployed version for discipline classification.
----
## 🧠 2025–06–05 – Discipline Classifier v3.1 (SciBERT + LoRA)

- Added `lora_discipline_classifier(v3.1).ipynb` notebook:
  - Applied LoRA (Low-Rank Adaptation) using Hugging Face `peft` on `allenai/scibert_scivocab_uncased` for 3-class discipline classification
  - Switched from Hugging Face `Trainer` to pure PyTorch training loop to ensure compatibility across `transformers`, `peft`, and `accelerate`
  - Trained on **expanded 5,402-paper dataset** (CS/IS/IT balanced with cleaning and validation)
  - LoRA config: `r=8`, `alpha=16`, `dropout=0.1`, `bias="none"`, 3 epochs, batch size 8, learning rate = 2e-4

- **Final Evaluation (Test Split, 20%):**
  - Accuracy = **82.05%**
  - Macro F1 = **0.81**
  - Per-class F1 scores:  
    - CS = **0.85**  
    - IS = **0.82**  
    - IT = **0.76**

- **Artefacts Saved to `/Artefacts/`:**
  - `lora_model_v3.1.pkl`
  - `tokenizer_v3.1.pkl`
  - `label2id_v3.1.pkl`
  - `id2label_v3.1.pkl`
  - `model_info_v3.1.pkl` (metadata)

- **Project Updated:**
  - `README.md`: Added to current phase, version map, model artefacts, comparison table, and notes section
  - `LOGS.md`: This entry
  - `Notion`: v3.1 logged as best-performing discipline classifier
  - `.gitattributes`: Ensured `.pkl` files under `/Artefacts` tracked via Git LFS
  - `.gitignore`: Cleaned to avoid accidental large binary commits

- **Remarks:**  
  v3.1 shows strong class balance and generalization on a 5× larger dataset. While macro F1 (0.81) is slightly lower than v2.2 (0.89), v3.1 offers better scale, consistent IT recall (F1 = 0.76), and improved robustness — making it the preferred model for broader deployment.
----
## 🧠 2025–06–06 – Trust-Based Filtering of Expanded Discipline Dataset

- Added `discipline_trust_score_filtering_v0.1.ipynb` notebook to compute trust scores on the 5,402-paper discipline dataset using the `v2.2` SciBERT + XGBoost classifier.
    - Embedded all abstracts using `allenai/scibert_scivocab_uncased` (title + abstract).
    - Predicted class probabilities with `disc_scibert_xgboost_v2.2.pkl`.
    - Computed `trust_score` as the max predicted probability across classes.
    - Saved predictions and scores to `expanded_discipline_with_preds.csv`.
    - Filtered dataset to retain only entries with `trust_score ≥ 0.8` → **4,838 high-confidence samples** (≈89.6%).

- **Artefacts Saved:**
    - `expanded_discipline_with_preds.csv`
    - `trusted_discipline_dataset.csv` (used for v4.0 training)
    - `scibert_embeddings_5402_v2.2.npy` (cached embeddings for reproducibility)

- **Project Updated:**
    - `README.md`: Added dataset filtering explanation and artefact references.
    - `LOGS.md`: This entry.
    - `Notion`: Sprint 1.0 marked complete (label cleanup and trust filtering).
    - `.gitattributes`: Ensured Git LFS is tracking `.npy` and large `.csv` artefacts.
    - `TASKS.md`: Updated under expanded dataset preparation.

- **Remarks:**
    - This concludes Sprint 1.0 – label quality enhancement via trust filtering.
    - The `trusted_discipline_dataset.csv` will be used to train `discipline_classifier_scilora_v4.0`.
    - Ensures cleaner supervision and curriculum learning during fine-tuning.
----

## 🚀 Post-v3.1 Phase – Modular Inference, Generalization & Deployment

Following the completion of all v2.x series and the experimental DeBERTa + LoRA classifier (v3.0), this phase focuses on **label quality**, **model robustness**, and **preparation for large-scale deployment**.

- ✅ **Close v3.0 (DeBERTa + LoRA) experiment**  
  - Accuracy = 54%, Macro F1 = 0.38  
  - Strong CS recall (F1 = 0.67), but failed IT entirely (F1 = 0.00)  
  - **Not selected** for deployment; retained for documentation

- ✅ **Add and evaluate v3.1 (SciBERT + LoRA) for Discipline**  
  - Trained on full 5,402-paper corpus  
  - Accuracy = 82.05%, Macro F1 = 0.81  
  - Per-class F1s: CS = 0.85, IS = 0.82, IT = 0.76  
  - **More generalizable** than v2.2, with consistent IT recall and better scaling

- ✅ **Perform trust-based label filtering using v2.2**  
  - Used `disc_scibert_xgboost_v2.2.pkl` to assign predicted labels and trust scores  
  - Saved `expanded_discipline_with_preds.csv` with model predictions and `trust_score`  
  - Retained 4,838 abstracts with `trust_score ≥ 0.8` for high-quality fine-tuning  
  - Output dataset: `trusted_discipline_dataset.csv` to be used in v4.0 fine-tuning

- [ ] Begin `v4.0` – Curriculum-based SciBERT + LoRA training  
  - Use trusted 4,838-paper subset with blended loss and label smoothing  
  - Target: ≥95% Accuracy and Macro F1

- [ ] Rebalance Mixed-class performance post-v2.6  
  - Explore focal loss, `scale_pos_weight`, or cost-sensitive XGBoost to recover Mixed F1 without harming Qual

- [ ] Expand and augment methodology dataset  
  - Collect & annotate +50–100 new Mixed-method abstracts  
  - Apply back-translation or NLPAug for underrepresented classes

- [ ] Integrate macro-F1 + per-class F1 into CI/CD  
  - Add evaluation summary and threshold-sweep tracking into `scripts/evaluate_methodology.py`

- [ ] Fine-tune SciBERT or SPECTER on domain corpus  
  - Try continued pretraining or adapter tuning on 2,028-paper corpus  
  - Improve subfield and methodology classifier alignment

- [ ] Implement full hierarchical inference pipeline  
  - Route predictions: Discipline → Subfield → Methodology  
  - Dynamically invoke fallback disambiguators (e.g., AI vs ML)

- [ ] Package as an API or lightweight app  
  - Build Streamlit or FastAPI demo with full pipeline + SHAP-style confidence explanations

- [ ] Standardize experiment tracking & artefact versioning  
  - Log configs, metrics, and hashes per version in Notion and README

- [ ] Explore model ensembling or meta-learning  
  - Stack outputs of TF-IDF+SVM, BERT+LR, SPECTER+XGB via meta-classifier (e.g., XGBoost)

- [ ] Conduct structured ablation & error analysis  
  - Compare confusion matrices across v2.3, v2.6, and v3.1  
  - Focus on Mixed boundary errors and edge-case drift

 

---
