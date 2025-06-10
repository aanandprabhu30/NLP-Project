# 🧠 NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research abstracts by:

- 🧐 **Discipline** – Computer Science (CS), Information Systems (IS), Information Technology (IT)
- 🧐 **Subfield** – AI, ML, CV, CYB, BSP, SEC, CLD, etc.
- 🧐 **Research Methodology** – Qualitative, Quantitative, Mixed

---

## 📊 Current Status (as of 10th June 2025)

✅ **Discipline classifier v4.0 achieved 89.7% accuracy (SciBERT + LoRA + Focal Loss)**  
✅ **XGBoost ensemble reached 92.76% accuracy on augmented balanced dataset**  
✅ **IT class F1 improved dramatically from 0.68 → 0.87 through targeted augmentation**  
🔬 **v3.1 (SciBERT + LoRA) retained as baseline; v4.0 shows significant improvement** 
✅ **Subfield classifiers finalized for CS (1498 papers), IS (374 papers), and IT (504 papers)**  
✅ **Methodology classifier (2028-paper set) uses two-stage architecture with threshold tuning**

## 🚀 Key Improvements in v4.0

- **Data Augmentation**: Balanced dataset through targeted augmentation (IT: 721→2000, IS: 1644→2000)
- **Focal Loss**: Addressed class imbalance, particularly improving IT classification
- **LoRA Fine-tuning**: Efficient parameter updates (only 1.5M trainable params vs 110M)
- **Ensemble Approach**: XGBoost on TF-IDF features achieved 92.76% accuracy
- **Trust Score Filtering**: Used high-confidence samples from v2.2 predictions

## 🎯 Quick Results

- **Discipline Classification**: 92.76% accuracy (XGBoost ensemble)
- **IT Class Performance**: F1 improved from 0.68 → 0.87
- **Target Achievement**: Near 95% goal (92.76% vs 95% target)

## 🚀 Next Steps

- Deploy ensemble model for production use
- Experiment with SciNCL/SPECTER2 for 95%+ accuracy
- Build web interface for abstract classification

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

> - `Discipline`: ✅ v4.0 (SciBERT + LoRA + Focal Loss) with XGBoost ensemble
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
| Discipline   | v3.1        | 5402             | 0.8205       | SciBERT (LoRA via PEFT); strong generalization |
| Discipline   | v4.0        | 7037 (augmented) | 0.897        | SciBERT + LoRA + Focal Loss; XGBoost ensemble = 0.9276 |
| Subfield – CS| v2.4        | 1498             | 0.75         | XGBoost (tuned) + SPECTER (768-dim) |
| Subfield – IS| v2.4        | 374              | 0.89         | XGBoost (tuned) + SPECTER (768-dim) |
| Subfield – IT| v2.4        | 504              | 0.83         | XGBoost (tuned) + SPECTER (768-dim) |
| Methodology  | v2.6        | 2028             | 0.77         | Two-stage XGBoost + SPECTER |

> **Notes:**  
>
> - "Test split" means a standard train/test split (often 80/20 or similar), not cross-validation.  
> - v4.0 discipline classifier was trained on 7,037 augmented papers (balanced via targeted augmentation for IT and IS classes); achieved significant improvements — CS F1 = 0.92, IS F1 = 0.89, IT F1 = 0.87; overall Accuracy = 89.7%, Macro F1 = 0.89. XGBoost ensemble on TF-IDF features achieved 92.76% accuracy.

## 🎯 Project Goals

✅ Build a scalable, modular NLP pipeline for automated classification of computing research abstracts  
✅ Achieve near-target accuracy of 92.76% (vs 95% goal) for discipline classification  
✅ Handle large-scale datasets (7,037 augmented abstracts) with class imbalance  
✅ Successfully implement data augmentation to improve minority class performance (IT F1: 0.68 → 0.87)  
✅ Demonstrate evolution from classical models (TF-IDF) to state-of-the-art transformers (SciBERT + LoRA)  
✅ Integrate parameter-efficient fine-tuning (PEFT) with LoRA for efficient training  
✅ Implement advanced loss functions (Focal Loss) for handling class imbalance  
✅ Create ensemble approach combining transformer and XGBoost for optimal performance  
✅ Design hierarchical pipeline: Discipline → Subfield → Methodology classification  
✅ Save all artifacts with Git LFS for version control and reproducibility  
✅ Document complete pipeline with clear versioning (v1.0 through v4.0)  
✅ Establish foundation for production deployment and future improvements

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Submitted as part of my BSc Final Year Project in Computer Science – University of London_
