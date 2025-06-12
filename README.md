# 🧠 NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research abstracts by:

- 🧐 **Discipline** – Computer Science (CS), Information Systems (IS), Information Technology (IT)
- 🧐 **Subfield** – AI, ML, CV, CYB, BSP, SEC, CLD, etc.
- 🧐 **Research Methodology** – Qualitative, Quantitative, Mixed

---

## 📊 Current Status (as of 11th June 2025)

✅ **Discipline classifier v6.0 achieved 94.77% accuracy (single XGBoost model)**  
✅ **Significant improvement: +2.08% over v5.0 (92.69%)**  
✅ **Enhanced feature engineering with 46 domain-specific features**  
✅ **Advanced TF-IDF pipeline with 11,500 features across 4 configurations**  
✅ **Targeted data augmentation: 8,128 total samples (2,726 augmented)**  
✅ **Excellent per-class performance: CS=92.60%, IS=94.86%, IT=97.27%**  
✅ **Gap to 95% target reduced to just 0.23%**  
✅ **Production-ready pipeline with complete artifact management**

## 🚀 Key Improvements in v6.0

- **Advanced Feature Engineering**: 46 domain-specific features including keyword ratios, technical patterns, and research type indicators
- **Multi-Configuration TF-IDF**: 4 different TF-IDF setups (standard, extended n-grams, character-level, technical terms)
- **Sophisticated Data Augmentation**: Sentence shuffling, keyword injection, and text combination strategies
- **Optimal Single Model**: XGBoost with proven parameters achieving 94.77% accuracy
- **Comprehensive Ensemble Testing**: Evaluated 7-model ensemble and stacking approaches
- **Production Pipeline**: Complete DisciplineClassifierV6 class with save/load functionality
- **Enhanced Error Analysis**: Detailed misclassification pattern analysis and SHAP interpretability

## 🚀 Next Steps

## Discipline Classification: COMPLETE at v6.0 (94.77% accuracy)

### 🎯 Future Work (Optional Extensions)

- **Subfield Classifier Enhancement**: Update CS/IS/IT subfield classifiers with v6.0 techniques
- **Methodology Classifier Improvement**: Apply advanced feature engineering to methodology classification
- **Integrated Pipeline**: Combine discipline → subfield → methodology in single workflow
- **Production Deployment**: Deploy discipline classifier v6.0 as standalone service
- **Research Publication**: Document methodology and results for academic contribution

### 💡 Discipline Classifier: Project Complete

- 94.77% accuracy achieved (0.23% from 95% target)
- Production-ready pipeline with complete artifact management
- Advanced feature engineering with 11,546 features
- Robust augmentation strategy and comprehensive evaluation
- **Decision**: Stopping here due to diminishing returns on further optimization

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
> **Other Classifiers: Functional but not updated to v6.0 standards**
>
> - `Subfield`: ✅ v2.4 (SPECTER + XGBoost tuned) - CS: 75%, IS: 89%, IT: 83%
> - `Methodology`: ✅ v2.6 (Two-stage XGBoost with threshold tuning) - 77% accuracy

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
- `complete_checkpoint_v6.0.pkl` - Full training state for reproducibility
- `results_summary_v6.0.json` - Performance metrics and analysis

### Legacy Models (Functional)

- Previous versions (v4.0, v5.0) available for comparison
- Subfield classifiers (v2.4) for CS/IS/IT classification  
- Methodology classifier (v2.6) for Qual/Quant/Mixed classification

## 📊 Version Comparison

| **Task**     | **Version** | **Dataset Size** | **Accuracy** | **Notes**                                |
|--------------|-------------|------------------|--------------|------------------------------------------|
| Discipline   | **v6.0**    | **8128 (augmented)** | **0.9477** | **Single XGBoost model, advanced feature engineering** |
| Discipline   | v5.0    | 6187 (augmented) | 0.9269 | Ensemble: SciBERT (30%) + XGBoost (70%), nlpaug augmentation |
| Discipline   | v4.0-XGB    | 7037 (augmented) | 0.9276       | XGBoost standalone (TF-IDF features, best single model) |
| Discipline   | v4.0-BERT   | 7037 (augmented) | 0.8970       | SciBERT + LoRA + Focal Loss standalone |
| Discipline   | v3.1        | 5402             | 0.8205       | SciBERT (LoRA via PEFT); strong generalization |
| Subfield – CS| v2.4        | 1498             | 0.75         | XGBoost (tuned) + SPECTER (768-dim) |
| Subfield – IS| v2.4        | 374              | 0.89         | XGBoost (tuned) + SPECTER (768-dim) |
| Subfield – IT| v2.4        | 504              | 0.83         | XGBoost (tuned) + SPECTER (768-dim) |
| Methodology  | v2.6        | 2028             | 0.77         | Two-stage XGBoost + SPECTER |

> **Notes:**  
>
> - "Test split" means a standard train/test split (often 80/20 or similar), not cross-validation.  
> - v4.0 discipline classifier was trained on 7,037 augmented papers (balanced via targeted augmentation for IT and IS classes); achieved SciBERT accuracy = 89.7%, Macro F1 = 0.89 (CS F1 = 0.92, IS F1 = 0.89, IT F1 = 0.87). XGBoost standalone achieved 92.76% accuracy.
> - v5.0 discipline classifier trained on 6,187 augmented papers with nlpaug synonym-based augmentation; achieved ensemble accuracy = 92.69% using optimized weighting (30% SciBERT + 70% XGBoost). Improved dependency management by removing bitsandbytes requirement for Colab Pro compatibility.
> - **v6.0 discipline classifier** trained on 8,128 augmented papers with sophisticated augmentation strategies (sentence shuffling, keyword injection, text combination); achieved 94.77% accuracy using single XGBoost model with advanced feature engineering (11,546 features: 11,500 TF-IDF + 46 domain-specific). Represents final iteration for discipline classification.
> - **Subfield classifiers (v2.4)** achieve strong performance: CS = 75%, IS = 89%, IT = 83% using SPECTER embeddings + tuned XGBoost on respective domain-specific datasets.
> - **Methodology classifier (v2.6)** uses two-stage architecture with threshold tuning, achieving 77% accuracy on qualitative/quantitative/mixed classification.

## 🎯 Project Goals

### ✅ Core Classification Pipeline (Three-Tier System)

#### Discipline Classification (v6.0) - PRODUCTION READY

- 94.77% accuracy - Distinguishes CS/IS/IT disciplines
- Advanced feature engineering with 11,546 features (11,500 TF-IDF + 46 domain-specific)
- Sophisticated data augmentation strategies for 8,128 samples
- Production-ready DisciplineClassifierV6 pipeline with complete artifact management
- Evolution from 90% → 94.77% across six major versions

#### Subfield Classification (v2.4) - FUNCTIONAL

- CS Subfield Classifier: 75% accuracy on 1,498 papers (AI, ML, CV, CYB, etc.)
- IS Subfield Classifier: 89% accuracy on 374 papers (BSP, DSA, ENT, GOV, etc.)  
- IT Subfield Classifier: 83% accuracy on 504 papers (CLD, EDG, IOT, NET, etc.)
- SPECTER embeddings + tuned XGBoost architecture across all domains

#### Methodology Classification (v2.6) - FUNCTIONAL

- Two-stage classification pipeline (Binary → Ternary)
- Threshold tuning optimization (Mixed threshold = 0.15)
- 77% accuracy on 2,028-paper dataset
- Handles challenging Mixed methodology detection

#### System-Wide Technical Achievements

- Scalable, modular NLP pipeline for automated classification
- Handle large-scale datasets with class imbalance across all tiers
- Comprehensive data augmentation and preprocessing pipelines
- Evolution from classical models to advanced feature engineering
- Complete artifact management and version control
- Hierarchical pipeline: Discipline → Subfield → Methodology classification

### 🎯 Project Status Summary

- Discipline classifier: PRODUCTION READY at 94.77% accuracy
- Subfield and methodology classifiers: FUNCTIONAL with room for enhancement
- Complete three-tier system demonstrates full research paper classification capability
- Primary focus achieved: High-performance discipline classification with production deployment

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Submitted as part of my BSc Final Year Project in Computer Science – University of London_
