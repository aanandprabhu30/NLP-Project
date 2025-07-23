# 🧠 NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research abstracts by:

- 🧐 **Discipline** – Computer Science (CS), Information Systems (IS), Information Technology (IT)
- 🧐 **Subfield** – AI, ML, CV, CYB, BSP, SEC, CLD, etc.
- 🧐 **Research Methodology** – Qualitative, Quantitative, Mixed

---

## 📊 Current Status (as of July 23rd, 2025)

✅ **Complete three-tier classification system with two approaches:**

**v6.0 (High Test Scores):** Advanced features but with data leakage issues

- Discipline: 94.77% | Methodology: 91.87% | Subfield: 82-90%

**v7.0 (Methodologically Sound):** Proper validation and real-world reliability  

- Discipline: 92.32% | Methodology: 86.92% | Subfield: 80-85%

✅ **Production-ready unified pipeline with smart post-processing**  
✅ **Batch processing and confidence analysis capabilities**  
✅ **Critical learning: Proper methodology is essential for real-world performance**

---

## 🔍 Key Difference: v6.0 vs v7.0

### v6.0: High Test Scores (Not Recommended)

- **Issues**: Data leakage, overfitting, poor real-world performance
- **Causes**: Augmented before splitting, fitted vectorizers on test data, excessive features (11,500+)
- **Result**: Impressive test scores but required manual corrections in practice

### v7.0: Methodologically Sound (Recommended)

- **Fixes**: Proper train/test splitting, conservative augmentation, reduced features (3,000)
- **Benefits**: Reliable real-world performance, proper validation methodology
- **Usage**: Recommended for all production deployments

---

## 🚀 Quick Start

### Load v7.0 Production Models

```python
from joblib import load
import pandas as pd

# Load v7.0 production models (recommended)
discipline_pipeline = load('Artefacts/current/discipline_classifier_v7/discipline_pipeline_v7.pkl')
methodology_pipeline = load('Artefacts/current/methodology_classifier_v7/methodology_pipeline_v7.pkl')

# Load production data
df = pd.read_csv('Data/Master.csv')  # 26,944 papers with all labels

# Classify a paper
title = "Deep Learning for Medical Image Segmentation"
abstract = "This paper presents a novel CNN architecture..."

discipline = discipline_pipeline.predict([abstract])
methodology = methodology_pipeline.predict([abstract])
```

### Unified Pipeline (v6.0 Based)

```python
# Load the unified pipeline (from Notebooks/unified/unified_classification_pipeline.ipynb)
results = classify_paper(title, abstract)
analyze_classification_confidence(results, title, abstract)
```

---

## 📁 Repository Structure

| Folder | Description |
|--------|-------------|
| `/Artefacts/` | **current/**: v7.0 production models \| **v6.0_educational/**: Educational examples \| **legacy/**: Historical development \| **shared/**: Common files |
| `/Data/` | **Master.csv**: Production dataset (37MB) - 26,944 papers with title, abstract, discipline, subfield, methodology |
| `/Scripts/` | Data scraping scripts |
| `/Notebooks/` | **current/**: v7.0 methodology \| **v6.0_educational/**: Shows ML pitfalls \| **unified/**: Complete system \| **legacy/**: Historical development |
| `/requirements/` | **requirements.txt**: Standard use \| **requirements-minimal.txt**: Production only \| **requirements-dev.txt**: Full development |

### Key Model Files

**v7.0 (Production Ready):**

- `current/discipline_classifier_v7/discipline_pipeline_v7.pkl`
- `current/methodology_classifier_v7/methodology_pipeline_v7.pkl`
- `current/cs_subfield_classifier_v7/cs_subfield_pipeline_v7.pkl`
- `current/is_subfield_classifier_v7/is_subfield_pipeline_v7.pkl`
- `current/it_subfield_classifier_v7/it_subfield_pipeline_v7.pkl`

**v6.0 (Educational Examples):**

- `unified_classification_pipeline.ipynb` - Complete system with post-processing
- `v6.0_educational/discipline_classifier_v6.0/` - Shows data leakage issues
- `v6.0_educational/methodology_classifier_v6.0/` - Shows overfitting problems

**Shared Files:**

- `shared/split_indices_v7.pkl` - Consistent data splits for v7.0

---

## 📊 Performance Summary

| **Task** | **v7.0 (Sound)** | **v6.0 (Flawed)** | **Recommendation** |
|----------|-------------------|--------------------|--------------------|
| Discipline | 92.32% | 94.77% | Use v7.0 for reliability |
| Methodology | 86.92% | 91.87% | Use v7.0 for real-world use |
| CS Subfield | 82.92% | 89.61% | Use v7.0 for production |
| IS Subfield | 80.33% | 83.39% | Use v7.0 for production |
| IT Subfield | 85.39% | 88.48% | Use v7.0 for production |

> **Key Learning**: v6.0's higher scores were due to data leakage. v7.0's scores represent true performance.

---

## 🛠 Environment Setup

```bash
# Create virtual environment
python3 -m venv nlp-bert
source nlp-bert/bin/activate

# Install dependencies (choose one):
# For v7.0 production only (minimal):
pip install -r requirements/requirements-minimal.txt

# For standard use (includes visualization & Jupyter):
pip install -r requirements/requirements.txt

# For full development (includes v6.0 educational examples):
pip install -r requirements/requirements-dev.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=nlp-bert
```

---

## 🎯 Future Work

- **Apply v7.0 methodology to v6.0 features** for optimal performance
- **Web API deployment** with Flask/FastAPI
- **Real-time monitoring** and confidence calibration
- **Domain expansion** to other academic fields

---

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _BSc Final Year Project in Computer Science – University of London_
