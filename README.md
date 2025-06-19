# 🧠 NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research abstracts by:

- 🧐 **Discipline** – Computer Science (CS), Information Systems (IS), Information Technology (IT)
- 🧐 **Subfield** – AI, ML, CV, CYB, BSP, SEC, CLD, etc.
- 🧐 **Research Methodology** – Qualitative, Quantitative, Mixed

---

## 📊 Current Status (as of 11th June 2025)

✅ **Discipline classifier v6.0 achieved 94.77% accuracy (single XGBoost model)**  
✅ **Methodology classifier v6.0 achieved 91.87% accuracy (ensemble model)**  
✅ **Significant improvements: Discipline +2.08% over v5.0, Methodology +14.87% over v2.6**  
✅ **Enhanced feature engineering with 46 domain-specific features (discipline) and 27 features (methodology)**  
✅ **Advanced TF-IDF pipeline with 11,500 features across 4 configurations**  
✅ **Targeted data augmentation: 8,128 total samples (discipline), 4,675 total samples (methodology)**  
✅ **Excellent per-class performance across both classifiers**  
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

## 🚀 Next Steps

## Discipline Classification: COMPLETE at v6.0 (94.77% accuracy)

## Methodology Classification: COMPLETE at v6.0 (91.87% accuracy)

### 🎯 Future Work (Optional Extensions)

- **Subfield Classifier Enhancement**: Update CS/IS/IT subfield classifiers with v6.0 techniques
- **Integrated Pipeline**: Combine discipline → subfield → methodology in single workflow
- **Production Deployment**: Deploy both classifiers v6.0 as standalone services
- **Research Publication**: Document methodology and results for academic contribution

### 💡 Both Classifiers: Project Complete

- **Discipline**: 94.77% accuracy achieved (0.23% from 95% target)
- **Methodology**: 91.87% accuracy achieved (3.13% from 95% target)
- Production-ready pipelines with complete artifact management
- Advanced feature engineering with 11,546 features (discipline) and 11,527 features (methodology)
- Robust augmentation strategies and comprehensive evaluation
- **Decision**: Both classifiers achieve excellent performance with diminishing returns on further optimization

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

## 🔁 Model Architecture Summary

> **Discipline Classification: COMPLETE**
>
> - `Discipline`: ✅ **v6.0 (XGBoost + Advanced Features) - 94.77% accuracy - FINAL**
> - `Methodology`: ✅ **v6.0 (Ensemble + Advanced Features) - 91.87% accuracy - FINAL**
> **Other Classifiers: Functional but not updated to v6.0 standards**
>
> - `Subfield`: ✅ v2.4 (SPECTER + XGBoost tuned) - CS: 75%, IS: 89%, IT: 83%

## 🗂️ Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `/Artefacts/` | Trained classifiers + vectorizers + evaluation visuals |
| `/Data/` | All labeled datasets used across classification tasks |
| `/Scripts/`| All Scripts used for scraping data|
| `README.md` | This file |
| `/Notebooks/` | All experiment notebooks across v1.x, v2.x, v3.x and v4.x |

## 📦 Key Model Files (v6.0 - Final)

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

### Legacy Models (Functional)

- Previous versions (v4.0, v5.0) available for comparison
- Subfield classifiers (v2.4) for CS/IS/IT classification  
- Previous methodology classifier (v2.6) for comparison

## 📊 Version Comparison

| **Task**     | **Version** | **Dataset Size** | **Accuracy** | **Notes**                                |
|--------------|-------------|------------------|--------------|------------------------------------------|
| Discipline   | **v6.0**    | **8128 (augmented)** | **0.9477** | **Single XGBoost model, advanced feature engineering** |
| Methodology  | **v6.0**    | **4675 (augmented)** | **0.9187** | **Ensemble model, advanced feature engineering** |
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
> - **Subfield classifiers (v2.4)** achieve strong performance: CS = 75%, IS = 89%, IT = 83% using SPECTER embeddings + tuned XGBoost on respective domain-specific datasets.

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

#### Subfield Classification (v2.4) - FUNCTIONAL

- CS Subfield Classifier: 75% accuracy on 1,498 papers (AI, ML, CV, CYB, etc.)
- IS Subfield Classifier: 89% accuracy on 374 papers (BSP, DSA, ENT, GOV, etc.)  
- IT Subfield Classifier: 83% accuracy on 504 papers (CLD, EDG, IOT, NET, etc.)
- SPECTER embeddings + tuned XGBoost architecture across all domains

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
- **Subfield classifiers**: FUNCTIONAL with room for enhancement
- Complete three-tier system demonstrates full research paper classification capability
- **Primary focus achieved**: High-performance discipline and methodology classification with production deployment

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Submitted as part of my BSc Final Year Project in Computer Science – University of London_
