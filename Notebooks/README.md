# 📓 Notebooks Organization

This folder contains all Jupyter notebooks from the NLP project development, organized by version and purpose.

## 📁 Folder Structure

### `current/` - v7.0 (🎯 **PRODUCTION METHODOLOGY**)

Methodologically corrected notebooks that demonstrate proper ML practices

- `discipline_v7.ipynb` - 92.32% accuracy (properly validated)
- `methodology_classifier_v7.ipynb` - 86.92% accuracy (robust methodology)
- `cs_subfield_classifier_v7.ipynb` - 82.92% accuracy (methodologically correct)
- `is_subfield_classifier_v7.ipynb` - 80.33% accuracy (properly validated)
- `it_subfield_classifier_v7.ipynb` - 85.39% accuracy (production-ready)

**Use these for:** Learning proper ML methodology, production implementations, reliable approaches

### `v6.0_educational/` - v6.0 (⚠️ **EDUCATIONAL - SHOWS ML PITFALLS**)

High test scores but methodological flaws - valuable for learning about common mistakes

- `discipline_classifier_v6_0.ipynb` - 94.77% accuracy (shows data leakage issues)
- `methodology_classifier_v6_0.ipynb` - 91.87% accuracy (shows overfitting problems)
- `subfield_classifier_v6_0.ipynb` - 83-90% accuracy (shows validation issues)

**Use these for:** Understanding common ML mistakes, studying methodological pitfalls, educational purposes

### `v8.0_educational/` - v8.0 (⚠️ **EDUCATIONAL - FALSE PREMISE CASE STUDY**)

False premise research example - valuable for learning about baseline verification

- `methodology_classifier_v8.ipynb` - 82.76% accuracy, 0.564 Mixed F1 (demonstrates false premise issues)

**Built on false assumption:** v7.0 was weak at Mixed detection (F1≈0.35) when actual F1=0.68

**Use this for:** Learning about baseline verification, research methodology rigor, assumption validation

### `unified/`

Complete integrated classification systems

- `unified_classification_pipeline.ipynb` - Hierarchical discipline → subfield → methodology workflow with smart post-processing (v6.0 based)
- `unified_pipeline_v7.ipynb` - Updated unified pipeline with v7.0 methodological corrections and improved reliability

**Use these for:** End-to-end classification system, production deployment examples, comprehensive analysis

### `legacy/`

Historical development notebooks (v1.x - v5.x)

Contains all development iterations including:

- **v1.x**: Initial experiments with basic ML models
- **v2.x**: SPECTER embeddings + XGBoost experiments
- **v3.x**: SciBERT + LoRA implementations
- **v4.x**: Advanced feature engineering attempts
- **v5.x**: Ensemble approaches and dependency optimization

**Use these for:** Understanding project evolution, reproducing earlier experiments, historical reference

## 🎯 Quick Start Guide

### For v8.0 Alternative Approach

```bash
# Use v8.0 experimental two-stage approach
jupyter notebook current/methodology_classifier_v8.ipynb
```

### For Learning Proper ML Methodology

```bash
# Study v7.0 notebooks (recommended approach)
jupyter notebook current/discipline_v7.ipynb
```

### For Understanding Common ML Mistakes

```bash
# Study v6.0 notebooks (educational examples)
jupyter notebook v6.0_educational/discipline_classifier_v6_0.ipynb
```

### For Complete Classification System

```bash
# Use v7.0 methodologically corrected pipeline (recommended)
jupyter notebook unified/unified_pipeline_v7.ipynb

# Or use original unified pipeline (v6.0 based)
jupyter notebook unified/unified_classification_pipeline.ipynb
```

## 🔬 Key Differences Between Versions

### v7.0 vs v6.0 Methodology Comparison

| **Aspect** | **v7.0 (Correct)** | **v6.0 (Flawed)** |
|------------|--------------------|--------------------|
| **Data Splitting** | Split first, then augment | Augment first, then split |
| **TF-IDF Fitting** | Training data only | Entire dataset |
| **Features** | 3,000 (prevents overfitting) | 11,500+ (prone to overfitting) |
| **Validation** | Proper cross-validation | No validation framework |
| **Real Performance** | 92.32% (reliable) | 94.77% (inflated) |

## 📚 Learning Path Recommendations

### 1. Start with v7.0 (Production Standard)

- Learn methodologically sound ML practices
- Understand proper validation frameworks
- See optimal Mixed detection approach (0.68 F1)

### 2. Study v6.0 Educational Examples (Data Leakage)

- Identify methodological issues  
- Learn correct data splitting techniques
- Understand why test scores were inflated

### 3. Study v8.0 Educational Examples (False Premise)

- Learn about baseline verification importance
- Understand how false assumptions invalidate conclusions
- See why simple approaches often beat complex ones

### 4. Study Evolution (Legacy)

- See how the project developed over time
- Understand different approaches tried
- Learn from iterative improvements

## 🔄 Migration Notes

**If updating existing references:**

- Replace `../Notebooks/discipline_v7.ipynb` with `../Notebooks/current/discipline_v7.ipynb`
- Replace `../Notebooks/discipline_classifier_v6_0.ipynb` with `../Notebooks/v6.0_educational/discipline_classifier_v6_0.ipynb`
- Replace `../Notebooks/current/methodology_classifier_v8.ipynb` with `../Notebooks/v8.0_educational/methodology_classifier_v8.ipynb`

## 📊 Execution Recommendations

| **Purpose** | **Use This** | **Environment** |
|-------------|--------------|-----------------|
| **False Premise Study** | `v8.0_educational/methodology_classifier_v8.ipynb` | Local nlp-bert kernel |
| Production Code | `current/` | Local nlp-bert kernel |
| Learning ML Methodology | `current/` vs `v6.0_educational/` | Any Python 3.11+ |
| Research/Comparison | `legacy/` | Match original environment |
| Complete System (Reliable) | `unified/unified_pipeline_v7.ipynb` | Local nlp-bert kernel |
| Complete System (Educational) | `unified/unified_classification_pipeline.ipynb` | Local nlp-bert kernel |

## 🎓 Key Learning Insight

The most valuable outcome from this notebook collection is the comparison between v6.0 and v7.0, which demonstrates that **proper methodology is more important than complex feature engineering**. v7.0's slightly lower but methodologically sound results are far more valuable than v6.0's inflated scores.
