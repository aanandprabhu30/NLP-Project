# 🧠 NLP Project – Identifying Research Methodologies in Computing

This project classifies computing research abstracts by:

- 🧐 **Discipline** – Computer Science (CS), Information Systems (IS), Information Technology (IT)
- 🧐 **Subfield** – AI, ML, CV, CYB, BSP, SEC, CLD, etc.
- 🧐 **Research Methodology** – Qualitative, Quantitative, Mixed

---

## 📊 Project Status: COMPLETE ✅ (August 2nd, 2025)

🎯 **Complete three-tier classification system with comprehensive educational framework:**

**v7.0 (Production Standard):** Methodologically sound and production-ready

- Discipline: 92.32% | Methodology: 86.92% | Subfield: 80-85% | **✅ Active for production use**

**v6.0 (Educational - Data Leakage):** High test scores but methodological flaws

- Discipline: 94.77% | Methodology: 91.87% | Subfield: 82-90% | **📚 Educational example of ML pitfalls**

**v8.0 (Educational - False Premise):** Demonstrates baseline verification importance

- Methodology: 82.76% | MIXED F1: 0.564 vs v7.0's 0.68 | **📚 Educational case study**

**v9.0 (Educational - Strategic Decision-Making):** Advanced techniques, discontinued approach

- Methodology: 78.42% (detailed 8-category classification) | **❌ Discontinued due to definitional overlap**

✅ **Production-ready unified pipeline with smart post-processing**  
✅ **Comprehensive documentation with educational examples**  
✅ **Simplified user-friendly classifier with sample data and layman explanations**  
✅ **Strategic lesson: Simple approaches often outperform complex ones**  
🎓 **Multiple educational frameworks:** Data leakage, false premise, and strategic decision-making case studies

---

## 🏆 **PROJECT COMPLETION SUMMARY**

**🎯 Primary Objectives Achieved:**

- ✅ **Multi-tier classification system** (Discipline → Subfield → Methodology)
- ✅ **Production-ready models** with methodologically sound validation (v7.0)
- ✅ **Comprehensive educational framework** showcasing ML best practices and common pitfalls
- ✅ **User-friendly tools** making complex systems accessible to broader audiences

**📚 Educational Value Delivered:**

- ✅ **Data leakage case study** (v6.0) - demonstrates methodological pitfalls
- ✅ **False premise research** (v8.0) - shows importance of baseline verification  
- ✅ **Strategic decision-making** (v9.0) - illustrates when to abandon complex approaches
- ✅ **User-centered design** (simplified classifier) - bridges technical capability and accessibility

**🚀 Ready for Deployment:**

- ✅ **Production models** available in `Artefacts/current/`
- ✅ **Simplified classifier** ready for immediate use with sample data
- ✅ **Complete documentation** for all approaches and methodologies
- ✅ **Single production dataset** (Master.csv) with proven 3-level methodology classification
- ✅ **Future-ready foundation** for web deployment and API development

---

## 🔍 Key Differences: v6.0 vs v7.0 vs v8.0 vs v9.0

### v7.0: Production Standard (✅ Recommended)

- **Approach**: Methodologically sound ML practices
- **Benefits**: Reliable real-world performance, proper validation methodology
- **Usage**: **Production deployments and general research**
- **Classification**: 3-level methodology (QUAL/QUANT/MIXED)

### v6.0: Data Leakage Example (📚 Educational)

- **Issues**: Data leakage, overfitting, poor real-world performance
- **Causes**: Augmented before splitting, fitted vectorizers on test data, excessive features (11,500+)
- **Value**: **Learn about common ML mistakes and methodological pitfalls**

### v8.0: False Premise Case Study (📚 Educational)

- **Issue**: Built on false assumption that v7.0 was weak at Mixed detection (F1≈0.35 vs actual 0.68)
- **Reality**: v8.0 performs worse than v7.0 (0.564 vs 0.68 Mixed F1)
- **Value**: **Learn about baseline verification and research methodology rigor**

### v9.0: Strategic Decision-Making Example (📚 Educational - Discontinued)

- **Approach**: Advanced data augmentation with detailed 8-category methodology classification
- **Technical Quality**: Solid implementation (78.42% accuracy)
- **Issue**: Too much definitional overlap and classification ambiguity between categories
- **Decision**: **Abandoned complex approach for simpler, more practical 3-level classification**
- **Value**: **Learn when to abandon technically successful but impractical approaches**

---

## 🚀 Quick Start

### For New Users: Simplified Classification (⭐ **RECOMMENDED**)

Ready-to-use notebook with sample data and plain English explanations:

```bash
# Open the simplified classifier (works immediately with sample data)
jupyter notebook Notebooks/academic_paper_classifier.ipynb

# Sample papers included:
# - CS/SEC/QUANT: "Machine Learning Approaches for Cybersecurity Threat Detection in IoT Networks"  
# - IT/EMERGING/QUAL: "Blockchain-based Supply Chain Management: A Systematic Review"
```

**Features:**

- ✅ **20 cells** (streamlined from 37-cell unified pipeline)
- ✅ **Sample data included** - works out of the box
- ✅ **Layman explanations** - plain English descriptions of results
- ✅ **No setup issues** - auto working directory detection
- ✅ **Variable-based interaction** - no input() function problems

### Load v7.0 Production Model

```python
from joblib import load
import pandas as pd

# Load v7.0 methodology model (production recommended)
methodology_pipeline = load('Artefacts/current/methodology_classifier_v7/methodology_pipeline_v7.pkl')

# Load production data (RECOMMENDED)
df = pd.read_csv('Data/Master.csv')  # 26,944 papers with 3-level methodology (QUAL/QUANT/MIXED)

# NOTE: MASTER_DETAILED_METHODOLOGY.csv is NOT used for production
# The 8-category detailed methodology was discontinued due to:
# - Definitional overlap between categories (ALGO vs SYST, CASE vs EXPT)
# - Classification ambiguity and reduced practical value
# Use Master.csv for all production and research work

# Classify with v7.0 methodology approach
title = "A Mixed Methods Study of User Experience in Mobile Apps"
abstract = "This research combines quantitative analytics with qualitative interviews..."

methodology = methodology_pipeline.predict([abstract])  # 3-level: QUAL/QUANT/MIXED
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
| `/Artefacts/` | **current/**: v7.0 production models \| **v6.0_educational/**: Data leakage examples \| **v8.0_educational/**: False premise examples \| **v9.0_educational/**: Strategic decision-making examples \| **legacy/**: Historical development \| **shared/**: Common files |
| `/Data/` | **Master.csv**: ✅ **Production dataset** (37MB, 26,944 papers, 3-level methodology) \| **MASTER_DETAILED_METHODOLOGY.csv**: ❌ **Discontinued** 8-category classification (educational use only) \| Comprehensive documentation |
| `/Documentation/` | **LOGS.md**: Complete development log \| Project documentation and insights |
| `/Scripts/` | Data scraping scripts |
| `/Notebooks/` | **academic_paper_classifier.ipynb**: Simplified user-friendly classifier with sample data ⭐ \| **current/**: v7.0 production notebooks \| **v6.0_educational/**: Data leakage examples \| **v8.0_educational/**: False premise examples \| **v9.0_educational/**: Strategic decision-making examples \| **unified/**: Complete systems \| **legacy/**: Historical development \| **Comprehensive READMEs for all folders** |
| `/requirements/` | **requirements.txt**: Standard use \| **requirements-minimal.txt**: Production only \| **requirements-dev.txt**: Full development |

### Key Model Files

**v7.0 (Production Ready):**

- `current/discipline_classifier_v7/discipline_pipeline_v7.pkl`
- `current/methodology_classifier_v7/methodology_pipeline_v7.pkl`
- `current/cs_subfield_classifier_v7/cs_subfield_pipeline_v7.pkl`
- `current/is_subfield_classifier_v7/is_subfield_pipeline_v7.pkl`
- `current/it_subfield_classifier_v7/it_subfield_pipeline_v7.pkl`

**User-Friendly Tool:**

- **`academic_paper_classifier.ipynb`**: Simplified 20-cell classifier with sample data and layman explanations ⭐

**Educational Examples:**

- **v6.0**: `v6.0_educational/` - Data leakage and overfitting issues (94.77% discipline, 91.87% methodology)
- **v8.0**: `v8.0_educational/` - False premise and baseline verification issues (82.76% methodology)  
- **v9.0**: `v9.0_educational/` - Strategic decision-making and discontinued 8-category classification (78.42% detailed methodology)
- `unified_classification_pipeline.ipynb` - Complete system with post-processing

**Educational Artifacts Available:**

- Complete model pipelines for all educational approaches
- Comprehensive performance metrics and analysis
- Strategic decision-making documentation and lessons learned

**Shared Files:**

- `shared/split_indices_v7.pkl` - Consistent data splits for v7.0

---

## 📊 Performance Summary

| **Task** | **v7.0 (Production)** | **v6.0 (Educational)** | **v8.0 (Educational)** | **v9.0 (Educational)** | **Recommendation** |
|----------|----------------------|------------------------|------------------------|------------------------|-------------------|
| Discipline | **92.32%** ✅ | 94.77% (data leakage) | N/A | N/A | **Use v7.0 for production** |
| Methodology (3-level) | **86.92%** ✅ | 91.87% (overfitting) | 82.76% (false premise) | N/A | **Use v7.0 for production** |
| Methodology (MIXED F1) | **0.68** ✅ | N/A | 0.564 (regression) | N/A | **v7.0 optimal for MIXED** |
| Detailed Methodology (8-level) | N/A | N/A | N/A | 78.42% (discontinued) | **Abandoned - use 3-level** |
| CS Subfield | **82.92%** ✅ | 89.61% (flawed validation) | N/A | N/A | **Use v7.0 for production** |
| IS Subfield | **80.33%** ✅ | 83.39% (flawed validation) | N/A | N/A | **Use v7.0 for production** |
| IT Subfield | **85.39%** ✅ | 88.48% (flawed validation) | N/A | N/A | **Use v7.0 for production** |

> **Key Learnings**:
>
> - **v7.0** provides optimal, methodologically sound performance for production use
> - **v6.0, v8.0, v9.0** serve as comprehensive educational examples of ML research challenges
> - **Strategic insight**: Simple approaches often outperform complex ones in practical applications
> - **User-centered design**: Simplified tools (20 cells vs 37) with sample data make complex systems accessible to broader audiences
> - **Documentation**: All approaches now have comprehensive READMEs for educational reference

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

## 📚 Documentation & Learning Resources

✅ **Comprehensive Documentation Available**

### **Notebooks Documentation**

- **Notebooks/README.md**: Complete guide to all notebook versions and their purposes
- **Notebooks/academic_paper_classifier.ipynb**: Simplified 20-cell classifier with sample data and plain English explanations ⭐
- **Notebooks/current/README.md**: v7.0 production notebooks (92.32% discipline, 86.92% methodology)
- **Notebooks/v6.0_educational/README.md**: Data leakage examples and methodological pitfalls
- **Notebooks/v8.0_educational/README.md**: False premise research case study
- **Notebooks/v9.0_educational/README.md**: Strategic decision-making and discontinued approaches
- **Notebooks/unified/README.md**: Complete integrated classification systems
- **Notebooks/legacy/README.md**: Historical development evolution (v1.x-v5.x)

### **Artifacts Documentation**

- **Artefacts/README.md**: Complete guide to all model versions and educational examples
- **Artefacts/v9.0_educational/README.md**: Strategic decision-making case study with discontinued 8-category classification

### **Data Documentation**

- **Data/README.md**: Production and experimental datasets documentation

### **Scripts Documentation**

- **Scripts/detailed_methodology_classifier_v9/README_classifier.md**: Automated 8-category classification tool (discontinued approach)

## 🎯 Future Work

- **Apply v7.0 methodology to advanced features** for optimal performance
- **Web API deployment** based on simplified classifier architecture with Flask/FastAPI  
- **Real-time monitoring** and confidence calibration
- **Domain expansion** to other academic fields
- **User interface development**: Build on simplified classifier foundation for web-based tool
- **Educational framework utilization**: Leverage comprehensive case studies (v6.0, v8.0, v9.0) for ML methodology training
- **Strategic decision-making frameworks**: Develop guidelines for when to abandon complex approaches in favor of simpler solutions

---

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _BSc Final Year Project in Computer Science – University of London_
>
> **Comprehensive Educational Framework**: All major approaches (v6.0-v9.0) fully documented with complete artifacts, performance analysis, and strategic decision-making case studies. Includes production-ready v7.0 models, simplified user-friendly classifier tool, and educational examples covering data leakage, false premise research, and complex classification design challenges.
