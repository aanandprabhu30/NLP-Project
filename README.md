# 🧠 NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research abstracts by:

- 🧐 **Discipline** – Computer Science (CS), Information Systems (IS), Information Technology (IT)
- 🧐 **Subfield** – AI, ML, CV, CYB, BSP, SEC, CLD, etc.
- 🧐 **Research Methodology** – Qualitative, Quantitative, Mixed

---

## 📊 Current Status (as of July 25th, 2025)

✅ **Complete three-tier classification system with two approaches:**

**v6.0 (High Test Scores):** Advanced features but with data leakage issues

- Discipline: 94.77% | Methodology: 91.87% | Subfield: 82-90%

**v7.0 (Methodologically Sound):** Proper validation and real-world reliability  

- Discipline: 92.32% | Methodology: 86.92% | Subfield: 80-85%

**v8.0 (Educational):** Demonstrates false premise research pitfalls

- Methodology: 82.76% (standalone) | MIXED F1: 0.564 vs v7.0's 0.68 | Educational use only

✅ **Production-ready unified pipeline with smart post-processing**  
✅ **Batch processing and confidence analysis capabilities**  
✅ **Critical learning: Proper methodology is essential for real-world performance**  
🎓 **v8.0 educational approach:** Case study in false premise research methodology

---

## 🔍 Key Differences: v6.0 vs v7.0 vs v8.0

### v6.0: High Test Scores (Not Recommended)

- **Issues**: Data leakage, overfitting, poor real-world performance
- **Causes**: Augmented before splitting, fitted vectorizers on test data, excessive features (11,500+)
- **Result**: Impressive test scores but required manual corrections in practice

### v7.0: Methodologically Sound (Production Standard)

- **Fixes**: Proper train/test splitting, conservative augmentation, reduced features (3,000)
- **Benefits**: Reliable real-world performance, proper validation methodology
- **Usage**: Recommended for general production deployments

### v8.0: Educational False Premise Case Study

- **Issue**: Built on false assumption that v7.0 was weak at Mixed detection (F1≈0.35 vs actual 0.68)
- **Reality**: v8.0 performs worse than v7.0 (0.564 vs 0.68 Mixed F1)
- **Value**: Educational example of how incorrect baselines invalidate "improvement" claims
- **Usage**: Educational study only; demonstrates importance of baseline verification

---

## 🚀 Quick Start

### Load v7.0 Production Model

```python
from joblib import load
import pandas as pd

# Load v7.0 methodology model (production recommended)
methodology_pipeline = load('Artefacts/current/methodology_classifier_v7/methodology_pipeline_v7.pkl')

# Load production data
df = pd.read_csv('Data/Master.csv')  # 26,944 papers with all labels

# Classify with v7.0 methodology approach
title = "A Mixed Methods Study of User Experience in Mobile Apps"
abstract = "This research combines quantitative analytics with qualitative interviews..."

methodology = methodology_pipeline.predict([abstract])  # Production approach
```

### Load v7.0 Production Models

```python
# Load v7.0 production models (general use)
discipline_pipeline = load('Artefacts/current/discipline_classifier_v7/discipline_pipeline_v7.pkl')
methodology_pipeline = load('Artefacts/current/methodology_classifier_v7/methodology_pipeline_v7.pkl')

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
| `/Artefacts/` | **current/**: v7.0 production models \| **v6.0_educational/**: Data leakage examples \| **v8.0_educational/**: False premise examples \| **legacy/**: Historical development \| **shared/**: Common files |
| `/Data/` | **Master.csv**: Production dataset (37MB) - 26,944 papers with title, abstract, discipline, subfield, methodology |
| `/Documentation/` | **LOGS.md**: Complete development log \| Project documentation and insights |
| `/Scripts/` | Data scraping scripts |
| `/Notebooks/` | **current/**: v7.0 production notebooks \| **v6.0_educational/**: Data leakage pitfalls \| **v8.0_educational/**: False premise pitfalls \| **unified/**: Complete system \| **legacy/**: Historical development |
| `/requirements/` | **requirements.txt**: Standard use \| **requirements-minimal.txt**: Production only \| **requirements-dev.txt**: Full development |

### Key Model Files

**v7.0 (Production Ready):**

- `current/discipline_classifier_v7/discipline_pipeline_v7.pkl`
- `current/methodology_classifier_v7/methodology_pipeline_v7.pkl`
- `current/cs_subfield_classifier_v7/cs_subfield_pipeline_v7.pkl`
- `current/is_subfield_classifier_v7/is_subfield_pipeline_v7.pkl`
- `current/it_subfield_classifier_v7/it_subfield_pipeline_v7.pkl`

**Educational Examples:**

- **v6.0**: `v6.0_educational/` - Data leakage and overfitting issues
- **v8.0**: `v8.0_educational/` - False premise and baseline verification issues
- `unified_classification_pipeline.ipynb` - Complete system with post-processing

**Shared Files:**

- `shared/split_indices_v7.pkl` - Consistent data splits for v7.0

---

## 📊 Performance Summary

| **Task** | **v7.0 (Production)** | **v6.0 (Educational)** | **v8.0 (Educational)** | **Recommendation** |
|----------|----------------------|------------------------|------------------------|-------------------|
| Discipline | **92.32%** | 94.77% (data leakage) | N/A | **Use v7.0 for production** |
| Methodology | **86.92%** | 91.87% (overfitting) | 82.76% (false premise) | **Use v7.0 for production** |
| Methodology (MIXED F1) | **0.68** | N/A | 0.564 (regression) | **v7.0 optimal for MIXED** |
| CS Subfield | **82.92%** | 89.61% (flawed validation) | N/A | **Use v7.0 for production** |
| IS Subfield | **80.33%** | 83.39% (flawed validation) | N/A | **Use v7.0 for production** |
| IT Subfield | **85.39%** | 88.48% (flawed validation) | N/A | **Use v7.0 for production** |

> **Key Learning**: v7.0 provides optimal, methodologically sound performance. v6.0 and v8.0 serve as educational examples of common ML research pitfalls.

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

- **Apply v7.0 methodology to advanced features** for optimal performance
- **Web API deployment** with Flask/FastAPI  
- **Real-time monitoring** and confidence calibration
- **Domain expansion** to other academic fields
- **Educational framework** for ML methodology best practices

---

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _BSc Final Year Project in Computer Science – University of London_
