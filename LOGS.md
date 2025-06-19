# 📝 NLP Project Development Log

## 📋 Project Overview

This markdown file tracks the step-by-step progress of the NLP classification project, from initial setup to the latest v6.0 release. Each entry is dated and includes detailed metrics, artifacts, and implementation notes.

### 🎯 Project Goals

- Develop accurate classifiers for academic paper categorization
- Create a robust pipeline for discipline, subfield, and methodology classification
- Establish best practices for academic text classification
- Build a foundation for future research in academic paper analysis

### 🏗️ System Architecture

The project follows a modular architecture with three main components:

1. **Data Pipeline**
   - Text preprocessing and normalization
   - Feature extraction (TF-IDF, embeddings)
   - Dataset management and versioning

2. **Model Pipeline**
   - Transformer-based models (SciBERT, SPECTER)
   - Classical ML models (XGBoost, SVM)
   - Ensemble methods and threshold tuning

3. **Evaluation Pipeline**
   - Cross-validation framework
   - Performance metrics tracking
   - Error analysis and model debugging

## 📊 Project Status (June 19, 2025)

### ✅ Discipline Classifier: PRODUCTION READY (v6.0)

- Accuracy: 94.77% (XGBoost, Macro F1: 0.9483, 0.23% from 95% target)
- Per-class accuracy: CS = 92.60%, IS = 94.86%, IT = 97.27%
- Features: 11,546 (11,500 TF-IDF + 46 domain-specific)
- Dataset: 8,128 (5,402 original + 2,726 augmented)
- Status: PRODUCTION READY with complete deployment pipeline

### ✅ Methodology Classifier: PRODUCTION READY (v6.0)

- Accuracy: 91.87% (ensemble, Macro F1: 0.9180, 3.13% from 95% target)
- Features: 11,527 (11,500 TF-IDF + 27 domain-specific)
- Dataset: 4,675 (3,288 original + 1,387 augmented)
- Status: PRODUCTION READY with complete deployment pipeline

**Per-Class Metrics (Methodology v6.0):**

| Methodology   | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| Mixed        | 0.91      | 0.89   | 0.90     |   294   |
| Qualitative  | 0.94      | 0.91   | 0.92     |   294   |
| Quantitative | 0.91      | 0.95   | 0.93     |   347   |

### 🔬 Remaining Components (Functional but Not Production-Ready)

#### Subfield Classifiers (v2.4)

- [ ] Apply v6.0 advanced feature engineering techniques
- [ ] Implement sophisticated data augmentation strategies  
- [ ] Upgrade to production-ready pipeline architecture
- [ ] Current performance: CS=75%, IS=89%, IT=83%

### 🌟 Future Research Directions (Optional)

1. **System Integration**
   - Combine all three classifiers into unified pipeline
   - Implement confidence-based routing
   - Create comprehensive evaluation framework
2. **Advanced Techniques**
   - Apply SPECTER2 or SciNCL embeddings to all components
   - Implement multi-task learning across all classification tasks
   - Explore few-shot learning for emerging subfields
3. **Production Deployment**
   - Deploy discipline and methodology classifiers v6.0 as API services
   - Create web interface for abstract classification
   - Implement monitoring and performance tracking

---

## 📅 Development Timeline

### 🧠 June 19, 2025 – Methodology Classifier v6.0 (FINAL PRODUCTION VERSION)

#### Implementation (Methodology v6.0)

- Architecture: Matches discipline classifier v6.0 (multi-config TF-IDF, 27 domain features, targeted augmentation, 7-model ensemble, production-ready pipeline)
- Dataset: 4,675 samples (3,288 original + 1,387 augmented)
- Features: 11,527 (11,500 TF-IDF + 27 domain-specific)
- Data Augmentation: Targeted for class balance, especially Mixed
- Hyperparameter Optimization: Best params selected for XGBoost
- Production Pipeline: Complete MethodologyClassifierV6 class with artifact management

#### Results (Methodology v6.0)

- Best Model: Ensemble (7-model)
- Accuracy: 91.87%
- Macro F1: 0.9180
- Per-class metrics:

  | Methodology   | Precision | Recall | F1-score | Support |
  |--------------|-----------|--------|----------|---------|
  | Mixed        | 0.91      | 0.89   | 0.90     |   294   |
  | Qualitative  | 0.94      | 0.91   | 0.92     |   294   |
  | Quantitative | 0.91      | 0.95   | 0.93     |   347   |

- Gap to Target: 3.13% from 95% goal
- Status: PRODUCTION READY with complete deployment pipeline

#### Artifacts (Methodology v6.0)

- `methodology_classifier_v6.0_pipeline.pkl` – Complete production pipeline (91.87% accuracy)
- `xgb_final_model_v6.0.pkl` – Best single XGBoost model
- `tfidf_vectorizers_v6.0.pkl` – 4 TF-IDF configurations
- `feature_extractor_v6.0.pkl` – Domain-specific feature extractor (27 features)
- `ensemble_models_v6.0.pkl` – 7-model ensemble for optimal performance
- `label_encoder_v6.0.pkl` – Label encoder for methodology classes
- `best_params_v6.0.pkl` – Best hyperparameters for the final model
- `results_summary_v6.0.json` – Performance metrics and analysis

---

### 🧠 June 11, 2025 – Discipline Classifier v6.0 (FINAL PRODUCTION VERSION)

#### Implementation (Discipline v6.0)

- Advanced Feature Engineering: 46 domain-specific features targeting CS/IS/IT classification
- Multi-TF-IDF Pipeline: 4 configurations (standard, extended n-grams, character-level, technical terms)
- Sophisticated Augmentation: Target ratio 0.85, sentence shuffling, keyword injection, text combination
- Enhanced Dataset: 8,128 samples (5,402 original + 2,726 augmented)
- Optimized XGBoost: Proven parameters from v5.0 with 500 estimators, depth=6, lr=0.1
- Ensemble Exploration: Tested 7-model ensemble and stacking (94.59-94.71% range)
- Production Pipeline: Complete DisciplineClassifierV6 class with artifact management

#### Results (Discipline v6.0)

- Single XGBoost Model (SELECTED)
  - Accuracy: 94.77%
  - Macro F1: 0.9483
  - Per-class accuracy: CS = 92.60%, IS = 94.86%, IT = 97.27%
  - Feature matrix: 11,546 features (11,500 TF-IDF + 46 domain)
- Ensemble Attempts (NOT SELECTED)
  - Equal weight ensemble: 94.59%
  - Optimized weight ensemble: 94.71%
  - Stacking classifier: 94.59%
- Final Achievement: +2.08% over v5.0, gap to 95% target: 0.23%

#### Artifacts (Discipline v6.0)

- `discipline_classifier_v6.0_pipeline.pkl` – Complete production pipeline (94.77% accuracy)
- `xgb_final_model_v6.0.pkl` – Best single XGBoost model
- `tfidf_vectorizers_v6.0.pkl` – 4 TF-IDF configurations
- `feature_extractor_v6.0.pkl` – Domain-specific feature extractor (46 features)
- `ensemble_models_v6.0.pkl` – 7-model ensemble for optimal performance
- `label_encoder_v6.0.pkl` – Label encoder for discipline classes
- `best_params_v6.0.pkl` – Best hyperparameters for the final model
- `complete_checkpoint_v6.0.pkl` – Full training state for reproducibility
- `results_summary_v6.0.json` – Performance metrics and analysis

---

## 🗂️ Legacy Models and Experiments

### 🧠 June 10, 2025 – Discipline Classifier v5.0

#### Implementation (Discipline v5.0)

- **Ensemble Approach**: Optimized weighting (30% SciBERT + 70% XGBoost)
- **Dependency Optimization**: Removed bitsandbytes for Colab Pro compatibility
- **Enhanced Augmentation**: nlpaug with WordNet synonym replacement (30% rate)
- **Memory Optimization**: Tesla T4 GPU optimized training pipeline
- **Trust Score Integration**: Used v2.2 high-confidence predictions for training
- **Target Balancing**: 80% of max class size (2,429 samples per minority class)

#### Results (Discipline v5.0)

- Ensemble Model: Accuracy: 92.69%, Macro F1: 92.63%
- Per-class F1: CS = 0.93, IS = 0.93, IT = 0.92
- Component Performance: SciBERT + LoRA + Focal Loss: 88.96% accuracy; XGBoost (TF-IDF): 92.41% accuracy
- Dataset: 6,187 augmented papers (from 5,402 base)

#### Artifacts (Discipline v5.0)

- `discipline_classifier_v5_0_colab_pro.ipynb` (training notebook)
- `classifier_final_20250610_110300/` (model directory)
  - `transformer/` (SciBERT + LoRA model)
  - `xgboost.pkl` (XGBoost model)
  - `tfidf.pkl` (TF-IDF vectorizer)
  - `config.json` (ensemble configuration)

---

### 🧠 June 10, 2025 – Discipline Classifier v4.0

#### Implementation (Discipline v4.0)

- SciBERT + LoRA + Focal Loss + Augmentation
- Targeted data augmentation for minority classes
- Focal loss for class imbalance
- LoRA fine-tuning (1.5M trainable parameters)
- XGBoost ensemble on TF-IDF features

#### Results (Discipline v4.0)

- Transformer Model: Accuracy: 89.7%, Macro F1: 0.89, Per-class F1: CS = 0.92, IS = 0.89, IT = 0.87
- XGBoost Ensemble: Accuracy: 92.76%, Macro F1: 0.93

#### Artifacts (Discipline v4.0)

- `adapter_model_v4.0.safetensors` (1.2 MB)
- `adapter_config_v4.0.json`
- `tokenizer_v4.0.json`, `tokenizer_config_v4.0.json`
- `vocab_v4.0.txt`, `special_tokens_map_v4.0.json`
- `xgb_model_v4.0.pkl` (ensemble model)
- `tfidf_vectorizer_v4.0.pkl`

---

### 🧠 May 30, 2025 – Methodology Classifier v2.6 (Superseded by v6.0)

#### Implementation (Methodology v2.6)

- Two-stage classification pipeline: Stage 1 (Binary: Mixed vs Non-Mixed), Stage 2 (Qual vs Quant)
- Threshold tuning (selected: 0.15)

#### Results (Methodology v2.6)

- Accuracy: 77%, Macro F1: 0.66
- Per-class F1: Mixed = 0.25, Qual = 0.91, Quant = 0.81

#### Artifacts (Methodology v2.6)

- `methodology_binary_mixed_model_v2.6.pkl`
- `methodology_qual_quant_model_v2.6.pkl`
- `methodology_mixed_threshold_v2.6.pkl`
- `methodology_specter_embeddings_v2.6.pkl`

---

### 🧠 May 29, 2025 – Methodology Classifier v2.3-2.5a

#### Implementation (Methodology v2.3-2.5a)

- SPECTER + XGBoost variants on 2,028-paper dataset
- v2.3: Default XGBoost; v2.4: GridSearchCV-tuned XGBoost; v2.5: Balanced class weights; v2.5a: Manual class weights (Mixed=2, Qualitative=1, Quantitative=1)

#### Results (Methodology v2.3-2.5a)

- v2.3: Mixed F1=0.35, Qual F1=0.83, Quant F1=0.81
- v2.4: Mixed F1=0.11, Qual F1=0.83, Quant F1=0.79
- v2.5: Mixed F1=0.20, Qual F1=0.83, Quant F1=0.79
- v2.5a: Mixed F1≈0.19, Qual F1≈0.82, Quant F1≈0.80

#### Artifacts (Methodology v2.3-2.5a)

- `methodology_xgb_v2.3.pkl`
- `methodology_label_encoder_v2.3.pkl`
- `methodology_xgb_model_v2.4_tuned.pkl`
- `methodology_xgb_class_weighted_v2.5.pkl`
- `methodology_xgb_manual_weights_v2.5a.pkl`

---

### 🧠 May 27, 2025 – IS & IT Subfield Classifiers

#### Implementation (IS & IT)

- SPECTER + XGBoost (v2.3/v2.4) for both IS and IT
- IS dataset: 374 papers (multi-source, hand-labeled)
- IT dataset: 504 papers (multi-source, hand-labeled)

#### Results (IS & IT)

- IS Classifier: v2.3: Default XGBoost; v2.4: GridSearchCV-tuned (Macro F1: 0.90)
- IT Classifier: v2.3: Default XGBoost; v2.4: GridSearchCV-tuned (Macro F1: 0.80)

#### Artifacts (IS & IT)

- `is_subfield_xgb_model_v2.3.pkl`
- `is_subfield_xgb_model_v2.4_tuned.pkl`
- `is_subfield_label_encoder_v2.3.pkl`
- `it_subfield_xgb_model_v2.3.pkl`
- `it_subfield_xgb_model_v2.4_tuned.pkl`
- `it_subfield_label_encoder_v2.3.pkl`

---

### 🧠 May 20, 2025 – CS Subfield Classifier

#### Implementation (CS)

- SPECTER + XGBoost (v2.3/v2.4) on 1,498-paper CS dataset
- Added AI/ML disambiguator as fallback
- Dataset collected via arXiv API

#### Results (CS)

- Main Classifier: v2.3: Default XGBoost; v2.4: GridSearchCV-tuned
- AI/ML Disambiguator: Accuracy: 68%, Macro F1: 0.67 (balanced for AI and ML)

#### Artifacts (CS)

- `cs_subfield_xgb_model_v2.3.pkl`
- `cs_subfield_xgb_model_v2.4_tuned.pkl`
- `cs_subfield_label_encoder_v2.3.pkl`
- `ai_ml_disambiguator_logreg_v1.pkl`
- `ai_ml_label_encoder.pkl`

---

## 📚 Early Development (v0 → v2.2.1)

> This section documents the foundational work that established the project's baseline performance.  
> Key milestones include:  
>
> - Initial dataset creation and preprocessing pipeline  
> - TF-IDF + classical ML models (v1.0)  
> - Cross-validation framework and evaluation metrics  
> - First transformer-based models (v2.0-2.2.1)  
> All subsequent improvements (v2.3+) built upon these foundations.

### 🚀 Project Initialization (v0)

- [x] Initialize GitHub repo with proper structure
- [x] Create initial Jupyter notebook for experimentation
- [x] Add .gitignore for Python/Jupyter files
- [x] Create README.md with project overview
- [x] Set up virtual environment (Python 3.11)
- [x] Install initial dependencies (scikit-learn, pandas, etc.)

### 📄 Dataset Creation (v1.0)

- [x] Create 105-paper dataset (35 per discipline)
  - CS: AI, ML, CV, CYB, PAST
  - IS: BSP, DSA, ENT, GOV, IMP
  - IT: CLD, EDG, IOT, NET, OPS
- [x] Add discipline, subfield, and methodology labels
- [x] Create enriched dataset with title + abstract
- [x] Add external evaluation set (9 entries)
- [x] Create prototype dataset (15 abstracts)

### ✨ Text Preprocessing Pipeline

- [x] Clean text (lowercase, remove punctuation)
- [x] Remove stopwords using NLTK
- [x] Lemmatize using spaCy
- [x] Handle special characters and formatting
- [x] Implement consistent text normalization

### 🌟 Feature Extraction (v1.0-1.2)

- [x] TF-IDF vectorization
  - ngram_range = (1,2)
  - min_df = 2
  - max_df = 0.95
- [x] Label encoding for all classification tasks
- [x] Feature matrix inspection and validation
- [x] Implement SMOTE for class balancing

### 🧐 Model Training & Evaluation

#### v1.0 (Baseline)

- [x] Train Logistic Regression classifiers
  - Discipline: 90.48% accuracy
  - Subfield: 65-70% accuracy per discipline
  - Methodology: 63.81% accuracy

#### v1.2 (Enhanced)

- [x] Upgrade to SVM + Bigram TF-IDF
- [x] Implement SMOTE for minority classes
- [x] Add Title + Abstract features
- [x] Improve Methodology classification

#### v2.0-2.2.1 (Transformer Era)

- [x] Implement MiniLM embeddings + Logistic Regression
- [x] Add SciBERT + XGBoost pipeline
- [x] Apply SMOTE to transformer embeddings
- [x] Achieve 76.19% accuracy on Methodology

### 📊 Cross-Validation Framework

- [x] Implement 5-fold stratified CV
- [x] Log fold-wise scores and metrics
- [x] Calculate mean accuracy and std dev
- [x] Generate confusion matrices
- [x] Document performance evolution

### 🗂️ Artifact Management

- [x] Save all models and vectorizers
- [x] Document model configurations
- [x] Track version dependencies
- [x] Maintain evaluation metrics
- [x] Update documentation

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Submitted as part of my BSc Final Year Project in Computer Science – University of London_
>
> Project Status: Discipline (v6.0, 94.77% accuracy, June 11, 2025) and Methodology (v6.0, 91.87% accuracy, June 19, 2025) components completed at production-ready level.
