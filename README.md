# 🧠 NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research abstracts by:

- 🧐 **Discipline** – Computer Science (CS), Information Systems (IS), Information Technology (IT)
- 🧐 **Subfield** – AI, ML, CV, CYB, BSP, SEC, CLD, etc.
- 🧐 **Research Methodology** – Qualitative, Quantitative, Mixed

---

## 📊 Current Status (as of July 19th, 2025)

✅ **Discipline classifier v6.0 achieved 94.77% accuracy (single XGBoost model)**  
✅ **Methodology classifier v6.0 achieved 91.87% accuracy (ensemble model)**  
✅ **Subfield classifier v6.0 achieved 89.61% (CS), 83.39% (IS), 88.48% (IT) accuracy**  
✅ **Unified Classification Pipeline - Complete hierarchical system with smart post-processing**  
✅ **Intelligent misclassification correction with confidence-based rules**  
✅ **Production-ready API wrapper for real-world deployment**  
✅ **Comprehensive analysis and recommendation system**  
✅ **Batch processing capabilities for large-scale classification**  
✅ **Significant improvements: Discipline +2.08% over v5.0, Methodology +14.87% over v2.6, Subfield +14.61% (CS), -5.61% (IS), +5.48% (IT) over v2.4**  
✅ **Enhanced feature engineering with 46 domain-specific features (discipline), 27 features (methodology), and 50-55 features (subfield)**  
✅ **Advanced TF-IDF pipeline with 11,500 features across 4 configurations**  
✅ **Targeted data augmentation: 8,128 total samples (discipline), 4,675 total samples (methodology), 39,153 total samples (subfield)**  
✅ **Excellent per-class performance across all classifiers**  
✅ **Production-ready pipelines with complete artifact management**

## 🚀 Key Improvements in v6.0

### Discipline Classification (94.77% accuracy)

- **Advanced Feature Engineering**: 46 domain-specific features including keyword ratios, technical patterns, and research type indicators
- **Multi-Configuration TF-IDF**: 4 different TF-IDF setups (standard, extended n-grams, character-level, technical terms)
- **Sophisticated Data Augmentation**: Sentence shuffling, keyword injection, and text combination strategies
- **Optimal Single Model**: XGBoost with proven parameters achieving 94.77% accuracy
- **Comprehensive Ensemble Testing**: Evaluated 7-model ensemble and stacking approaches
- **Production Pipeline**: Complete DisciplineClassifierV6 class with save/load functionality
- **Enhanced Error Analysis**: Detailed misclassification pattern analysis and SHAP interpretability

### Methodology Classification (91.87% accuracy)

- **Production-Ready Architecture**: Following discipline classifier v6.0 structure for consistency
- **Advanced Feature Engineering**: 27 domain-specific features for methodology detection
- **Multi-Configuration TF-IDF**: 4 TF-IDF configurations optimized for methodology classification
- **Sophisticated Data Augmentation**: 1,387 augmented samples for balanced training
- **Ensemble Optimization**: 7-model ensemble achieving 91.87% accuracy
- **Complete Pipeline**: MethodologyClassifierV6 class with full artifact management
- **Target Achievement**: 3.13% gap remaining to 95% target

### Subfield Classification (v6.0 - Production Ready)

- **CS Subfield Classifier**: 89.61% accuracy on 22,571 samples (AI/ML, CLOUDCS, CV, NLP, SE, SEC)
- **IS Subfield Classifier**: 83.39% accuracy on 14,416 samples (BPM, DT, GOV, HIS, KM)
- **IT Subfield Classifier**: 88.48% accuracy on 2,166 samples (CLOUDIT, DEVOPS, EMERGING, RISK)
- **Advanced Feature Engineering**: 50-55 domain-specific features per discipline targeting subfield classification
- **Multi-Configuration TF-IDF**: 4 TF-IDF configurations optimized for subfield detection
- **Sophisticated Data Augmentation**: 24,243 augmented samples for balanced training across all subfields
- **Production Pipeline**: Complete SubfieldClassifierV6 class with save/load functionality for each discipline
- **Enhanced Error Analysis**: Detailed misclassification pattern analysis and SHAP interpretability
- **LLM-Assisted Validation**: Multi-LLM classifier with OpenAI GPT-4o-mini for error paper reprocessing

### Unified Classification Pipeline (Complete Integration)

- **Hierarchical Classification System**: Complete `discipline → subfield → methodology` workflow in single pipeline
- **Smart Post-Processing**: Confidence-based correction system for misclassifications with transparent reasoning
- **Intelligent Error Correction**: Rule-based overrides for obvious misclassifications (e.g., deep learning papers correctly classified as CS)
- **Production-Ready API**: `AcademicPaperClassifier` class with batch processing and model management
- **Comprehensive Analysis**: Detailed confidence assessment and actionable recommendations for each classification
- **Transparent Operations**: Clear indication when corrections are applied with reasoning explanations
- **Batch Processing**: Efficient classification of multiple papers with progress tracking
- **Export Functionality**: JSON export and summary report generation for integration workflows
- **Confidence Thresholding**: Configurable confidence levels for quality control (default: 60%)
- **Multi-Modal Features**: Enhanced feature extractors tailored for each classification task

## 🚀 Next Steps

## Discipline Classification: COMPLETE at v6.0 (94.77% accuracy)

## Methodology Classification: COMPLETE at v6.0 (91.87% accuracy)

## Subfield Classification: COMPLETE at v6.0 (89.61% CS, 83.39% IS, 88.48% IT accuracy)

### ✅ Completed Extensions

- **Integrated Pipeline**: ✅ Complete discipline → subfield → methodology workflow implemented
- **Production Deployment**: ✅ Production-ready API with batch processing and error correction
- **Smart Post-Processing**: ✅ Intelligent misclassification correction with transparent reasoning
- **Comprehensive Analysis**: ✅ Detailed confidence assessment and actionable recommendations

### 🎯 Future Work (Optional Extensions)

- **Real-time Web API**: Deploy unified pipeline as REST API service with endpoints for single/batch classification
- **Web Interface**: Create user-friendly web application for non-technical users with drag-and-drop functionality
- **Model Monitoring**: Implement performance tracking and confidence distribution monitoring over time
- **Domain Expansion**: Extend classification to other academic disciplines (Engineering, Medicine, Social Sciences)
- **Multi-language Support**: Support for non-English academic papers and abstracts
- **Automated Retraining**: Pipeline for incorporating new labeled data and model updates
- **Research Publication**: Document methodology, unified pipeline architecture, and smart post-processing results
- **Performance Optimization**: Further tune IS subfield classifier (currently 83.39% vs 89% v2.4 baseline)
- **Confidence Calibration**: Fine-tune confidence thresholds based on real-world usage patterns
- **Integration Plugins**: Develop plugins for reference managers (Zotero, Mendeley) and academic databases

### 💡 All Classifiers: Project Complete

- **Discipline**: 94.77% accuracy achieved (0.23% from 95% target)
- **Methodology**: 91.87% accuracy achieved (3.13% from 95% target)
- **Subfield**: 89.61% (CS), 83.39% (IS), 88.48% (IT) accuracy achieved
- Production-ready pipelines with complete artifact management
- Advanced feature engineering with 11,546 features (discipline), 11,527 features (methodology), and 11,545-11,555 features (subfield)
- Robust augmentation strategies and comprehensive evaluation
- **Decision**: All classifiers achieve excellent performance with diminishing returns on further optimization

## 🛠 Built With

- Python
- Jupyter Notebook / Google Colab
- scikit-learn
- XGBoost
- Hugging Face Transformers
- PEFT (LoRA)
- pandas, seaborn, matplotlib
- joblib

## 🧪 Environment Setup

Most of the project was executed locally using a dedicated virtual environment and Jupyter kernel named **`nlp-bert`**, created specifically for BERT and XGBoost model stability.

> 🔧 **Note:** The `xgboost` models repeatedly crashed under the default Anaconda kernel. A clean virtualenv-based kernel (`nlp-bert`) resolved this.

### Python Version

- Python 3.11

### 📦 To Recreate the Environment Locally

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

## 🚀 Quick Start - Unified Classification Pipeline

### Single Paper Classification

```python
# Load the unified pipeline (from Notebooks/unified_classification_pipeline.ipynb)
# Run the notebook cells or copy the functions to your script

# Classify a paper
title = "Deep Learning for Medical Image Segmentation"
abstract = """This paper presents a novel deep learning architecture for segmenting 
medical images using convolutional neural networks. We employ a U-Net based approach 
with attention mechanisms to improve segmentation accuracy on CT scans."""

# Get classification results
results = classify_paper(title, abstract)

# Get detailed analysis and recommendations  
analyze_classification_confidence(results, title, abstract)
```

### Production API Usage

```python
# Initialize the production API
api_classifier = AcademicPaperClassifier(models_path='../Artefacts/')

# Single classification
results = api_classifier.classify(title, abstract)

# Batch classification
papers = [
    {'title': 'Title 1', 'abstract': 'Abstract 1'},
    {'title': 'Title 2', 'abstract': 'Abstract 2'}
]
batch_results = api_classifier.classify_batch(papers)
```

### Key Features

- **Smart Error Correction**: Automatically fixes obvious misclassifications (CS papers classified as IS)
- **Confidence Analysis**: Detailed breakdown of prediction reliability with actionable recommendations
- **Transparent Operations**: Shows when and why corrections are applied
- **Configurable Thresholds**: Adjust confidence levels for quality control
- **Batch Processing**: Efficiently process multiple papers
- **Export Capabilities**: Save results to JSON and generate summary reports

## 🔁 Model Architecture Summary

> **All Classifications: COMPLETE**
>
> - `Discipline`: ✅ **v6.0 (XGBoost + Advanced Features) - 94.77% accuracy - FINAL**
> - `Methodology`: ✅ **v6.0 (Ensemble + Advanced Features) - 91.87% accuracy - FINAL**
> - `Subfield`: ✅ **v6.0 (XGBoost + Advanced Features) - CS: 89.61%, IS: 83.39%, IT: 88.48% - FINAL**

## 🗂️ Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `/Artefacts/` | Trained classifiers + vectorizers + evaluation visuals |
| `/Data/` | All labeled datasets used across classification tasks |
| `/Scripts/`| All Scripts used for scraping data|
| `README.md` | This file |
| `/Notebooks/` | All experiment notebooks across v1.x, v2.x, v3.x, v4.x, v5.x, and v6.x |
| `/Notebooks/unified_classification_pipeline.ipynb` | **Complete integrated system with smart post-processing** |

## 📦 Key Model Files (v6.0 - Final)

### Unified Classification Pipeline (Production Ready)

- `/Notebooks/unified_classification_pipeline.ipynb` - Complete integrated system with all classifiers
- **Smart Classification Function**: `classify_paper(title, abstract, confidence_threshold=0.6)`
- **Confidence Analysis**: `analyze_classification_confidence(results, title, abstract)`
- **Production API**: `AcademicPaperClassifier` class for real-world deployment
- **Batch Processing**: `classify_papers_batch()` for efficient multi-paper classification
- **Post-Processing Rules**: Intelligent correction for CS/AI papers, cloud infrastructure, etc.
- **Transparent Corrections**: Shows original vs corrected predictions with reasoning
- **Export Functions**: JSON export and summary reporting capabilities

### Discipline Classifier v6.0 (Production Ready)

- `discipline_classifier_v6.0_pipeline.pkl` - Complete production pipeline (94.77% accuracy)
- `xgb_final_model_v6.0.pkl` - Best single XGBoost model
- `tfidf_vectorizers_v6.0.pkl` - 4 TF-IDF configurations
- `feature_extractor_v6.0.pkl` - Domain-specific feature extractor (46 features)
- `ensemble_models_v6.0.pkl` - 7-model ensemble for optimal performance
- `label_encoder_v6.0.pkl` - Label encoder for discipline classes
- `best_params_v6.0.pkl` - Best hyperparameters for the final model
- `complete_checkpoint_v6.0.pkl` - Full training state for reproducibility
- `results_summary_v6.0.json` - Performance metrics and analysis

### Methodology Classifier v6.0 (Production Ready)

- `methodology_classifier_v6.0_pipeline.pkl` - Complete production pipeline (91.87% accuracy)
- `xgb_final_model_v6.0.pkl` - Best single XGBoost model
- `tfidf_vectorizers_v6.0.pkl` - 4 TF-IDF configurations
- `feature_extractor_v6.0.pkl` - Domain-specific feature extractor (27 features)
- `ensemble_models_v6.0.pkl` - 7-model ensemble for optimal performance
- `label_encoder_v6.0.pkl` - Label encoder for methodology classes
- `best_params_v6.0.pkl` - Best hyperparameters for the final model
- `results_summary_v6.0.json` - Performance metrics and analysis

### Subfield Classifier v6.0 (Production Ready)

#### CS Subfield Classifier (89.61% accuracy)

- `cs_subfield_classifier_v6.0_pipeline.pkl` - Complete production pipeline
- `xgb_model_v6.0.pkl` - Best XGBoost model
- `tfidf_vectorizers_v6.0.pkl` - 4 TF-IDF configurations
- `feature_extractor_v6.0.pkl` - Domain-specific feature extractor (55 features)
- `label_encoder_v6.0.pkl` - Label encoder for CS subfield classes (AI/ML, CLOUDCS, CV, NLP, SE, SEC)

#### IS Subfield Classifier (83.39% accuracy)

- `is_subfield_classifier_v6.0_pipeline.pkl` - Complete production pipeline
- `xgb_model_v6.0.pkl` - Best XGBoost model
- `tfidf_vectorizers_v6.0.pkl` - 4 TF-IDF configurations
- `feature_extractor_v6.0.pkl` - Domain-specific feature extractor (50 features)
- `label_encoder_v6.0.pkl` - Label encoder for IS subfield classes (BPM, DT, GOV, HIS, KM)

#### IT Subfield Classifier (88.48% accuracy)

- `it_subfield_classifier_v6.0_pipeline.pkl` - Complete production pipeline
- `xgb_model_v6.0.pkl` - Best XGBoost model
- `tfidf_vectorizers_v6.0.pkl` - 4 TF-IDF configurations
- `feature_extractor_v6.0.pkl` - Domain-specific feature extractor (45 features)
- `label_encoder_v6.0.pkl` - Label encoder for IT subfield classes (CLOUDIT, DEVOPS, EMERGING, RISK)

### Legacy Models (Functional)

- Previous versions (v4.0, v5.0) available for comparison
- Subfield classifiers (v2.4) for CS/IS/IT classification  
- Previous methodology classifier (v2.6) for comparison

## 📊 Version Comparison

| **Task**     | **Version** | **Dataset Size** | **Accuracy** | **Notes**                                |
|--------------|-------------|------------------|--------------|------------------------------------------|
| Discipline   | **v6.0**    | **8128 (augmented)** | **0.9477** | **Single XGBoost model, advanced feature engineering** |
| Methodology  | **v6.0**    | **4675 (augmented)** | **0.9187** | **Ensemble model, advanced feature engineering** |
| Subfield – CS| **v6.0**    | **22571 (augmented)** | **0.8961** | **XGBoost model, advanced feature engineering** |
| Subfield – IS| **v6.0**    | **14416 (augmented)** | **0.8339** | **XGBoost model, advanced feature engineering** |
| Subfield – IT| **v6.0**    | **2166 (augmented)** | **0.8848** | **XGBoost model, advanced feature engineering** |
| Discipline   | v5.0    | 6187 (augmented) | 0.9269 | Ensemble: SciBERT (30%) + XGBoost (70%), nlpaug augmentation |
| Methodology  | v2.6        | 2028             | 0.77         | Two-stage XGBoost + SPECTER |
| Discipline   | v4.0-XGB    | 7037 (augmented) | 0.9276       | XGBoost standalone (TF-IDF features, best single model) |
| Discipline   | v4.0-BERT   | 7037 (augmented) | 0.8970       | SciBERT + LoRA + Focal Loss standalone |
| Discipline   | v3.1        | 5402             | 0.8205       | SciBERT (LoRA via PEFT); strong generalization |
| Subfield – CS| v2.4        | 1498             | 0.75         | XGBoost (tuned) + SPECTER (768-dim) |
| Subfield – IS| v2.4        | 374              | 0.89         | XGBoost (tuned) + SPECTER (768-dim) |
| Subfield – IT| v2.4        | 504              | 0.83         | XGBoost (tuned) + SPECTER (768-dim) |

> **Notes:**  
>
> - "Test split" means a standard train/test split (often 80/20 or similar), not cross-validation.  
> - v4.0 discipline classifier was trained on 7,037 augmented papers (balanced via targeted augmentation for IT and IS classes); achieved SciBERT accuracy = 89.7%, Macro F1 = 0.89 (CS F1 = 0.92, IS F1 = 0.89, IT F1 = 0.87). XGBoost standalone achieved 92.76% accuracy.
> - v5.0 discipline classifier trained on 6,187 augmented papers with nlpaug synonym-based augmentation; achieved ensemble accuracy = 92.69% using optimized weighting (30% SciBERT + 70% XGBoost). Improved dependency management by removing bitsandbytes requirement for Colab Pro compatibility.
> - **v6.0 discipline classifier** trained on 8,128 augmented papers with sophisticated augmentation strategies (sentence shuffling, keyword injection, text combination); achieved 94.77% accuracy using single XGBoost model with advanced feature engineering (11,546 features: 11,500 TF-IDF + 46 domain-specific). Represents final iteration for discipline classification.
> - **v6.0 methodology classifier** trained on 4,675 augmented papers (3,288 original + 1,387 augmented) with advanced feature engineering (11,527 features: 11,500 TF-IDF + 27 domain-specific); achieved 91.87% accuracy using 7-model ensemble. Represents final iteration for methodology classification.
> - **v6.0 subfield classifiers** trained on 39,153 total augmented papers (26,944 original + 12,209 augmented) with advanced feature engineering (11,545-11,555 features: 11,500 TF-IDF + 45-55 domain-specific); achieved CS = 89.61%, IS = 83.39%, IT = 88.48% accuracy using XGBoost models. Represents final iteration for subfield classification with LLM-assisted validation for error reprocessing.

## 🎯 Project Goals

### ✅ Core Classification Pipeline (Three-Tier System)

#### Discipline Classification (v6.0) - PRODUCTION READY

- 94.77% accuracy - Distinguishes CS/IS/IT disciplines
- Advanced feature engineering with 11,546 features (11,500 TF-IDF + 46 domain-specific)
- Sophisticated data augmentation strategies for 8,128 samples
- Production-ready DisciplineClassifierV6 pipeline with complete artifact management
- Evolution from 90% → 94.77% across six major versions

#### Methodology Classification (v6.0) - PRODUCTION READY

- 91.87% accuracy - Distinguishes Qualitative/Quantitative/Mixed methodologies
- Advanced feature engineering with 11,527 features (11,500 TF-IDF + 27 domain-specific)
- Sophisticated data augmentation strategies for 4,675 samples
- Production-ready MethodologyClassifierV6 pipeline with complete artifact management
- 7-model ensemble optimization for optimal performance
- Evolution from 77% → 91.87% across major versions

#### Subfield Classification (v6.0) - PRODUCTION READY

- CS Subfield Classifier: 89.61% accuracy on 22,571 papers (AI/ML, CLOUDCS, CV, NLP, SE, SEC)
- IS Subfield Classifier: 83.39% accuracy on 14,416 papers (BPM, DT, GOV, HIS, KM)  
- IT Subfield Classifier: 88.48% accuracy on 2,166 papers (CLOUDIT, DEVOPS, EMERGING, RISK)
- Advanced feature engineering + XGBoost architecture with LLM-assisted validation

#### System-Wide Technical Achievements

- Scalable, modular NLP pipeline for automated classification
- Handle large-scale datasets with class imbalance across all tiers
- Comprehensive data augmentation and preprocessing pipelines
- Evolution from classical models to advanced feature engineering
- Complete artifact management and version control
- Hierarchical pipeline: Discipline → Subfield → Methodology classification

### 🎯 Project Status Summary

- **Discipline classifier**: PRODUCTION READY at 94.77% accuracy
- **Methodology classifier**: PRODUCTION READY at 91.87% accuracy
- **Subfield classifiers**: PRODUCTION READY at 89.61% (CS), 83.39% (IS), 88.48% (IT) accuracy
- **Unified Classification Pipeline**: COMPLETE with smart post-processing and error correction
- **Production API**: Ready for real-world deployment with batch processing capabilities
- **Intelligent Error Correction**: Automatic fixes for obvious misclassifications with transparent reasoning
- Complete three-tier system demonstrates full research paper classification capability
- **Primary focus achieved**: High-performance discipline, methodology, and subfield classification with production deployment and intelligent post-processing

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Submitted as part of my BSc Final Year Project in Computer Science – University of London_
