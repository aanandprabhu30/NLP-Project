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

## 📊 Current Status (June 11, 2025)

### 🎯 Key Achievements

#### Discipline Classifier v6.0 (PRODUCTION READY)

- Single Model Accuracy: 94.77% (XGBoost with advanced feature engineering)
- Feature Engineering: 11,546 total features (11,500 TF-IDF + 46 domain-specific)
- Dataset: 8,128 papers (augmented from 5,402 base with sophisticated strategies)
- Performance: CS=92.60%, IS=94.86%, IT=97.27% per-class accuracy
- Gap to Target: Only 0.23% short of 95% goal
- Status: PRODUCTION READY with complete deployment pipeline

#### Discipline Classifier v5.0

- Ensemble Accuracy: 92.69% (30% SciBERT + 70% XGBoost)
- Individual Components: SciBERT = 88.96%, XGBoost = 92.41%
- Optimized for Colab Pro (no bitsandbytes dependency)
- Enhanced nlpaug synonym-based augmentation
- Dataset: 6,187 papers (augmented from 5,402 base)

#### Discipline Classifier v4.0

- Accuracy: 89.7% (transformer), 92.76% (XGBoost standalone)
- IT F1 improved from 0.68 → 0.87
- Balanced dataset: 7,037 papers (CS=3037, IS=2000, IT=2000)

#### Subfield Classifiers (v2.4) - FUNCTIONAL

- CS: 75% accuracy (1498 papers)
- IS: 89% accuracy (374 papers)
- IT: 83% accuracy (504 papers)

#### Methodology Classifier (v2.6) - FUNCTIONAL

- Two-stage architecture with threshold tuning
- Accuracy: 77%, Macro F1: 0.66
- Mixed F1: 0.25, Qual F1: 0.91, Quant F1: 0.81

## 📅 Development Timeline

### 🧠 June 11, 2025 – Discipline Classifier v6.0 (FINAL PRODUCTION VERSION)

#### Implementation (v6.0)

- Advanced Feature Engineering: 46 domain-specific features targeting CS/IS/IT classification
- Multi-TF-IDF Pipeline: 4 configurations (standard, extended n-grams, character-level, technical terms)
- Sophisticated Augmentation: Target ratio 0.85, sentence shuffling, keyword injection, text combination
- Enhanced Dataset: 8,128 samples (5,402 original + 2,726 augmented)
- Optimized XGBoost: Proven parameters from v5.0 with 500 estimators, depth=6, lr=0.1
- Ensemble Exploration: Tested 7-model ensemble and stacking (94.59-94.71% range)
- Production Pipeline: Complete DisciplineClassifierV6 class with artifact management

#### Results (v6.0)

- Single XGBoost Model (SELECTED)
  - Accuracy: 94.77%
  - Macro F1: 94.83%
  - Per-class accuracy: CS = 92.60%, IS = 94.86%, IT = 97.27%
  - Feature matrix: 11,546 features (11,500 TF-IDF + 46 domain)
- Ensemble Attempts (NOT SELECTED)
  - Equal weight ensemble: 94.59%
  - Optimized weight ensemble: 94.71%
  - Stacking classifier: 94.59%
- Final Achievement: +2.08% over v5.0, gap to 95% target: 0.23%

#### Artifacts (v6.0)

- `discipline_classifier_v6.0_pipeline.pkl` (complete production pipeline)
- `xgb_final_model_v6.0.pkl` (best single model)
- `tfidf_vectorizers_v6.0.pkl` (4 TF-IDF configurations)
- `feature_extractor_v6.0.pkl` (domain feature extractor)
- `ensemble_models_v6.0.pkl` (7 ensemble models for comparison)
- `complete_checkpoint_v6.0.pkl` (full training state)
- `results_summary_v6.0.json` (performance metrics)

#### Key Technical Innovations

- Domain Feature Extraction: CS/IS/IT keyword ratios, technical pattern detection, research methodology indicators
- Advanced Augmentation Strategy: Class-balanced targeting with 85% of max class size (2,581 samples per class)
- Feature Importance Analysis: IT keyword count (0.0164), partial matches (0.0095), IS ratio (0.0089) as top features
- SHAP Interpretability: Model explanation with feature importance visualization
- Production Architecture: Modular design with preprocessing, feature extraction, and prediction pipelines

#### Project Completion Decision

- Strategic Completion: 94.77% accuracy represents optimal stopping point
- Diminishing Returns: Further improvements would require disproportionate effort for 0.23% gain
- Production Ready: Complete pipeline with comprehensive artifact management
- Status: FINAL VERSION for discipline classification component

### 🧠 June 10, 2025 – Discipline Classifier v5.0

#### Implementation (v5.0)

- **Ensemble Approach**: Optimized weighting (30% SciBERT + 70% XGBoost)
- **Dependency Optimization**: Removed bitsandbytes for Colab Pro compatibility
- **Enhanced Augmentation**: nlpaug with WordNet synonym replacement (30% rate)
- **Memory Optimization**: Tesla T4 GPU optimized training pipeline
- **Trust Score Integration**: Used v2.2 high-confidence predictions for training
- **Target Balancing**: 80% of max class size (2,429 samples per minority class)

#### Results (v5.0)

- **Ensemble Model**
  - Accuracy: 92.69%
  - Macro F1: 92.63%
  - Per-class F1: CS = 0.93, IS = 0.93, IT = 0.92
- **Component Performance**
  - SciBERT + LoRA + Focal Loss: 88.96% accuracy
  - XGBoost (TF-IDF): 92.41% accuracy
- **Dataset**: 6,187 augmented papers (from 5,402 base)

#### Artifacts (v5.0)

- `discipline_classifier_v5_0_colab_pro.ipynb` (training notebook)
- `classifier_final_20250610_110300/` (model directory)
  - `transformer/` (SciBERT + LoRA model)
  - `xgboost.pkl` (XGBoost model)
  - `tfidf.pkl` (TF-IDF vectorizer)
  - `config.json` (ensemble configuration)

#### Key Improvements over v4.0

- **Deployment Ready**: Simplified dependencies for production use
- **Comparable Performance**: 92.69% vs 92.76% (only 0.07% difference)
- **Efficient Training**: Optimized for cloud environments
- **Enhanced Augmentation**: Better synonym-based text variation

### 🧠 June 10, 2025 – Discipline Classifier v4.0

#### Implementation (v4.0)

- SciBERT + LoRA + Focal Loss + Augmentation
- Targeted data augmentation for minority classes
- Focal loss for class imbalance
- LoRA fine-tuning (1.5M trainable parameters)
- XGBoost ensemble on TF-IDF features

#### Results (v4.0)

- **Transformer Model**
  - Accuracy: 89.7%
  - Macro F1: 0.89
  - Per-class F1: CS = 0.92, IS = 0.89, IT = 0.87
- **XGBoost Ensemble**
  - Accuracy: 92.76%
  - Macro F1: 0.93

#### Artifacts (v4.0)

- `adapter_model_v4.0.safetensors` (1.2 MB)
- `adapter_config_v4.0.json`
- `tokenizer_v4.0.json`, `tokenizer_config_v4.0.json`
- `vocab_v4.0.txt`, `special_tokens_map_v4.0.json`
- `xgb_model_v4.0.pkl` (ensemble model)
- `tfidf_vectorizer_v4.0.pkl`

### 🧠 June 6, 2025 – Trust-Based Filtering

#### Implementation

- Filtered 5,402-paper dataset using v2.2 predictions
- Computed trust scores based on prediction probabilities
- Retained 4,838 high-confidence samples (trust_score ≥ 0.8)

#### Output

- `expanded_discipline_with_preds.csv`
- `trusted_discipline_dataset.csv` (for v4.0 training)
- `scibert_embeddings_5402_v2.2.npy`

### 🧠 June 5, 2025 – Discipline Classifier v3.1

#### Implementation (v3.1)

- SciBERT + LoRA on 5,402-paper dataset
- LoRA config: r=8, alpha=16, dropout=0.1
- 3 epochs, batch size 8, learning rate = 2e-4

#### Results (v3.1)

- Accuracy: 82.05%
- Macro F1: 0.81
- Per-class F1: CS = 0.85, IS = 0.82, IT = 0.76

#### Artifacts (v3.1)

- `lora_model_v3.1.pkl`
- `tokenizer_v3.1.pkl`
- `label2id_v3.1.pkl`
- `id2label_v3.1.pkl`
- `model_info_v3.1.pkl`

### 🧠 June 4, 2025 – Discipline Classifier v3.0

#### Implementation (v3.0)

- DeBERTa + LoRA experiment
- Trained on 1,138-paper dataset
- Batch size 8, 5 epochs, learning rate = 2e-5

#### Results (v3.0)

- Accuracy: 54%
- Macro F1: 0.38
- Per-class F1: CS = 0.67, IS = 0.47, IT = 0.00
- **Not selected** for deployment

#### Artifacts (v3.0)

- `discipline_classifier_deberta_lora_v3.0.pkl`
- `tokenizer_deberta_lora_v3.0.pkl`
- `label2id_deberta_lora_v3.0.pkl`

### 🧠 May 30, 2025 – Methodology Classifier v2.6

#### Implementation (v2.6)

- Two-stage classification pipeline
- Stage 1: Binary classifier (Mixed vs Non-Mixed)
- Stage 2: Qual vs Quant classifier
- Threshold tuning (selected: 0.15)

#### Results (v2.6)

- Accuracy: 77%
- Macro F1: 0.66
- Per-class F1: Mixed = 0.25, Qual = 0.91, Quant = 0.81

#### Artifacts (v2.6)

- `methodology_binary_mixed_model_v2.6.pkl`
- `methodology_qual_quant_model_v2.6.pkl`
- `methodology_mixed_threshold_v2.6.pkl`
- `methodology_specter_embeddings_v2.6.pkl`

### 🧠 May 29, 2025 – Methodology Classifier v2.3-2.5a

#### Implementation (v2.3-2.5a)

- SPECTER + XGBoost variants on 2,028-paper dataset
- v2.3: Default XGBoost
- v2.4: GridSearchCV-tuned XGBoost
- v2.5: Balanced class weights
- v2.5a: Manual class weights (Mixed=2, Qualitative=1, Quantitative=1)

#### Results (v2.3-2.5a)

- **v2.3**: Mixed F1=0.35, Qual F1=0.83, Quant F1=0.81
- **v2.4**: Mixed F1=0.11, Qual F1=0.83, Quant F1=0.79
- **v2.5**: Mixed F1=0.20, Qual F1=0.83, Quant F1=0.79
- **v2.5a**: Mixed F1≈0.19, Qual F1≈0.82, Quant F1≈0.80

#### Artifacts (v2.3-2.5a)

- `methodology_xgb_v2.3.pkl`
- `methodology_label_encoder_v2.3.pkl`
- `methodology_xgb_model_v2.4_tuned.pkl`
- `methodology_xgb_class_weighted_v2.5.pkl`
- `methodology_xgb_manual_weights_v2.5a.pkl`

### 🧠 May 27, 2025 – IS & IT Subfield Classifiers

#### Implementation (IS & IT)

- SPECTER + XGBoost (v2.3/v2.4) for both IS and IT
- IS dataset: 374 papers (multi-source, hand-labeled)
- IT dataset: 504 papers (multi-source, hand-labeled)

#### Results (IS & IT)

- **IS Classifier**
  - v2.3: Default XGBoost
  - v2.4: GridSearchCV-tuned (Macro F1: 0.90)
- **IT Classifier**
  - v2.3: Default XGBoost
  - v2.4: GridSearchCV-tuned (Macro F1: 0.80)

#### Artifacts (IS & IT)

- `is_subfield_xgb_model_v2.3.pkl`
- `is_subfield_xgb_model_v2.4_tuned.pkl`
- `is_subfield_label_encoder_v2.3.pkl`
- `it_subfield_xgb_model_v2.3.pkl`
- `it_subfield_xgb_model_v2.4_tuned.pkl`
- `it_subfield_label_encoder_v2.3.pkl`

### 🧠 May 20, 2025 – CS Subfield Classifier

#### Implementation (CS)

- SPECTER + XGBoost (v2.3/v2.4) on 1,498-paper CS dataset
- Added AI/ML disambiguator as fallback
- Dataset collected via arXiv API

#### Results (CS)

- **Main Classifier**
  - v2.3: Default XGBoost
  - v2.4: GridSearchCV-tuned
- **AI/ML Disambiguator**
  - Accuracy: 68%
  - Macro F1: 0.67 (balanced for AI and ML)

#### Artifacts (CS)

- `cs_subfield_xgb_model_v2.3.pkl`
- `cs_subfield_xgb_model_v2.4_tuned.pkl`
- `cs_subfield_label_encoder_v2.3.pkl`
- `ai_ml_disambiguator_logreg_v1.pkl`
- `ai_ml_label_encoder.pkl`

## 🚀 Next Steps

## 🚀 Project Status: DISCIPLINE CLASSIFICATION COMPLETE

### ✅ Discipline Classifier: PRODUCTION READY (v6.0)

- **FINAL STATUS**: 94.77% accuracy achieved (0.23% from 95% target)
- **DECISION**: Project complete due to diminishing returns
- **READY FOR**: Production deployment and real-world application

### 🔬 Remaining Components (Functional but Not Production-Ready)

#### Subfield Classifiers (v2.4)

- [ ] Apply v6.0 advanced feature engineering techniques
- [ ] Implement sophisticated data augmentation strategies  
- [ ] Upgrade to production-ready pipeline architecture
- [ ] Current performance: CS=75%, IS=89%, IT=83%

#### Methodology Classifier (v2.6)

- [ ] Apply v6.0 feature engineering approach
- [ ] Enhance data augmentation for Mixed class detection
- [ ] Implement production pipeline architecture
- [ ] Current performance: 77% accuracy

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
   - Deploy discipline classifier v6.0 as API service
   - Create web interface for abstract classification
   - Implement monitoring and performance tracking

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
> Project Status: Discipline Classification component completed at production-ready 94.77% accuracy (June 2025)
