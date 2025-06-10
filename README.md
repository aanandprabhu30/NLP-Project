# 🧠 NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research abstracts by:

- 🧐 **Discipline** – Computer Science (CS), Information Systems (IS), Information Technology (IT)
- 🧐 **Subfield** – AI, ML, CV, CYB, BSP, SEC, CLD, etc.
- 🧐 **Research Methodology** – Qualitative, Quantitative, Mixed

---

## 📊 Current Status (as of 10th June 2025)

✅ **Discipline classifier v5.0 achieved 92.69% accuracy (ensemble approach)**  
✅ **SciBERT + LoRA + Focal Loss reached 88.96% accuracy**  
✅ **XGBoost standalone achieved 92.41% accuracy on TF-IDF features**  
✅ **IT class F1 improved dramatically from 0.68 → 0.87 through targeted augmentation**  
🔬 **v5.0 optimized without bitsandbytes dependency for Colab Pro compatibility**
✅ **Enhanced data augmentation using nlpaug with synonym-based techniques**  
✅ **Subfield classifiers finalized for CS (1498 papers), IS (374 papers), and IT (504 papers)**  
✅ **Methodology classifier (2028-paper set) uses two-stage architecture with threshold tuning**

## 🚀 Key Improvements in v5.0

- **Dependency Optimization**: Removed bitsandbytes dependency for better Colab Pro compatibility
- **Enhanced Data Augmentation**: Implemented nlpaug with WordNet synonym augmentation (30% replacement rate)
- **Memory Efficiency**: Optimized for Tesla T4 GPU environment in Colab Pro
- **Trust Score Integration**: Leveraged high-confidence v2.2 predictions with trust score weighting
- **Advanced Augmentation Strategy**: Target count balancing (80% of max class size = 2,429 samples)
- **Ensemble Optimization**: Found optimal weights (Transformer: 30%, XGBoost: 70%)
- **Focal Loss Implementation**: Applied Focal Loss with gamma=2.0 for class imbalance handling

## 🚀 Next Steps

- Investigate techniques to bridge remaining 2.31% gap to reach 95% target
- Experiment with advanced ensemble methods (stacking, voting classifiers)
- Explore newer transformer models (SPECTER2, SciNCL) for potential improvements
- Implement production web interface for abstract classification
- Document deployment pipeline and inference optimization
- Consider additional data sources to expand training set beyond 6,187 samples

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

## 🔁 Final Architectures

> - `Discipline`: ✅ **v5.0 (Ensemble: SciBERT + XGBoost) - 92.69% accuracy**
> - `Discipline`: ✅ v4.0 (SciBERT + LoRA + Focal Loss) with XGBoost ensemble - 92.76%
> - `Subfield`: ✅ v2.3/v2.4 (SPECTER + XGBoost tuned)  
> - `Methodology`:  
>   - `v2.3`: Single-stage (SMOTE + XGBoost)  
>   - ✅ `v2.6`: Two-stage XGBoost with threshold tuning (Mixed threshold = 0.15)

## 🗂️ Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `/Artefacts/` | Trained classifiers + vectorizers + evaluation visuals |
| `/Data/` | All labeled datasets used across classification tasks |
| `/Scripts/`| All Scripts used for scraping data|
| `README.md` | This file |
| `/Notebooks/` | All experiment notebooks across v1.x, v2.x, v3.x and v4.x |

## 📦 Key Model Files as of v4.0

- `adapter_model_v4.0.safetensors` - LoRA weights (1.2 MB)
- `xgb_model_v4.0.pkl` - XGBoost ensemble (92.76% accuracy)
- `tokenizer_v4.0.json` - Fast tokenizer

## 📊 Version Comparison

| **Task**     | **Version** | **Dataset Size** | **Accuracy** | **Notes**                                |
|--------------|-------------|------------------|--------------|------------------------------------------|
| Discipline   | **v5.0**    | **6187 (augmented)** | **0.9269** | **Ensemble: SciBERT (30%) + XGBoost (70%), nlpaug augmentation** |
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

## 🎯 Project Goals

✅ Build a scalable, modular NLP pipeline for automated classification of computing research abstracts  
✅ Achieve strong performance: v4.0 = 92.76%, v5.0 = 92.69% (approaching 95% goal)  
✅ Handle large-scale datasets with class imbalance (v4.0: 7,037 samples, v5.0: 6,187 samples)  
✅ Successfully implement data augmentation to improve minority class performance (IT F1: 0.68 → 0.87)  
✅ Demonstrate evolution from classical models (TF-IDF) to state-of-the-art transformers (SciBERT + LoRA)  
✅ Integrate parameter-efficient fine-tuning (PEFT) with LoRA for efficient training  
✅ Implement advanced loss functions (Focal Loss) for handling class imbalance  
✅ Create ensemble approaches: v4.0 XGBoost standalone, v5.0 optimized transformer-XGBoost ensemble  
✅ Design hierarchical pipeline: Discipline → Subfield → Methodology classification  
✅ Save all artifacts with Git LFS for version control and reproducibility  
✅ Document complete pipeline with clear versioning (v1.0 through v5.0)  
✅ Optimize for production deployment: v5.0 removes complex dependencies for better compatibility  
✅ Establish robust foundation with consistent 92%+ performance across versions

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Submitted as part of my BSc Final Year Project in Computer Science – University of London_
