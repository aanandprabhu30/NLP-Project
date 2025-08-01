# ⚠️ v9.0 Educational Artifacts – Discontinued Detailed Methodology Classification

## 🎓 **Educational Purpose Only**

These artifacts represent a **technically successful but strategically discontinued approach** to methodology classification. They serve as an educational example of when to abandon complex solutions in favor of simpler, more practical alternatives.

---

## 📊 **Performance Summary**

| **Metric** | **Score** | **Details** |
|------------|-----------|-------------|
| **Test Accuracy** | **78.42%** | Final evaluation on held-out test set |
| **Cross-Validation** | **77.13% ± 1.31%** | 5-fold stratified CV (no data leakage) |
| **Classification Approach** | **8-category detailed methodology** | ALGO, CASE, SYST, REVW, EXPT, PERF, ANAL, UNKNOWN |
| **Status** | **❌ Discontinued** | Abandoned due to definitional overlap |

### **Per-Category Performance (Test Set)**

| **Category** | **Precision** | **Recall** | **F1-Score** | **Support** | **Definition Issues** |
|-------------|---------------|------------|--------------|-------------|----------------------|
| **ALGO** | 0.78 | 0.88 | **0.82** | 1,682 | Overlap with SYST |
| **CASE** | 0.78 | 0.89 | **0.83** | 1,482 | Overlap with EXPT |
| **REVW** | 0.88 | 0.85 | **0.87** | 775 | Clear definition |
| **SYST** | 0.76 | 0.74 | **0.75** | 1,141 | Overlap with ALGO |
| **PERF** | 0.74 | 0.58 | **0.65** | 482 | Overlap with EXPT |
| **EXPT** | 0.76 | 0.52 | **0.61** | 502 | Overlap with CASE, PERF |
| **ANAL** | 0.71 | 0.50 | **0.59** | 342 | Ambiguous boundaries |
| **UNKNOWN** | 0.93 | 0.89 | **0.91** | 147 | Catch-all category |

---

## 📁 **Artifact Files**

| **File** | **Size** | **Description** |
|----------|----------|-----------------|
| `methodology_pipeline_v9.pkl` | 7.1MB | **Complete trained pipeline** - XGBoost with TF-IDF features |
| `artifacts_v9.pkl` | 7.1MB | **Raw model artifacts** - Classifier, vectorizer, label encoder |
| `metrics_v9.txt` | 1KB | **Performance metrics** - Detailed classification report |

### **Pipeline Components**

```python
# Load the complete v9.0 pipeline
pipeline = joblib.load('methodology_pipeline_v9.pkl')

# Pipeline includes:
# - Data augmentation (50% augmentation ratio)
# - Dual TF-IDF vectorizers (3,000 total features)
# - XGBoost classifier with early stopping
# - Label encoding for 8 categories
```

---

## ⚠️ **Why Discontinued?**

### **Technical Success**

- ✅ **Solid Implementation**: Proper data augmentation, no leakage
- ✅ **Reasonable Performance**: 78.42% accuracy with good methodology
- ✅ **Production Pipeline**: Complete, deployable system

### **Strategic Issues**

- ❌ **Definitional Overlap**: Categories not mutually exclusive
- ❌ **Classification Ambiguity**: Difficult to assign papers consistently
- ❌ **Reduced Practical Value**: 8 categories too granular for most use cases
- ❌ **Annotation Challenges**: Human annotators struggled with boundaries

### **Specific Category Conflicts**

| **Category Pair** | **Overlap Issue** |
|------------------|-------------------|
| **ALGO vs SYST** | Algorithm papers often describe system implementations |
| **CASE vs EXPT** | Case studies frequently include experimental elements |
| **EXPT vs PERF** | Performance studies are inherently experimental |
| **ANAL vs others** | Analysis methods appear across all research types |

---

## 🎯 **Educational Value**

### **What This Demonstrates**

1. **Strategic Decision-Making**: When to abandon technically sound approaches
2. **Classification Design**: Importance of mutually exclusive categories
3. **Practical Considerations**: Complex ≠ Better in real-world applications
4. **Data Augmentation**: Advanced techniques for text classification
5. **Mature Engineering**: Knowing when to pivot from complex to simple

### **Key Lessons**

```python
# Technical Excellence ≠ Practical Success
if classification_accuracy > 0.75 and methodology_sound:
    if categories_have_overlap or annotation_difficult:
        # Strategic decision: Abandon complex approach
        return "Use simpler 3-level classification (QUAL/QUANT/MIXED)"
    else:
        return "Deploy complex system"
```

---

## 🔄 **Migration to v7.0 Production**

### **Why v7.0 is Superior for Production**

| **Aspect** | **v9.0 (Discontinued)** | **v7.0 (Production)** | **Winner** |
|------------|-------------------------|------------------------|------------|
| **Accuracy** | 78.42% (8-category) | 86.92% (3-category) | **v7.0** |
| **Category Clarity** | Ambiguous boundaries | Clear definitions | **v7.0** |
| **Annotation Ease** | Difficult, subjective | Straightforward | **v7.0** |
| **Practical Value** | Too granular | Right level of detail | **v7.0** |
| **MIXED Detection** | N/A (different scheme) | F1: 0.68 (excellent) | **v7.0** |

### **Recommended Migration**

```python
# BEFORE (v9.0 - Discontinued)
detailed_prediction = v9_pipeline.predict(abstract)
# Returns: ALGO, CASE, SYST, EXPT, PERF, REVW, ANAL, UNKNOWN

# AFTER (v7.0 - Production)
methodology_prediction = v7_pipeline.predict(abstract)  
# Returns: Qualitative, Quantitative, Mixed

# The 3-level approach provides better:
# - Inter-annotator agreement
# - Practical applicability  
# - Clear category boundaries
# - Higher accuracy (86.92% vs 78.42%)
```

---

## 💡 **Usage Recommendations**

### **For Educational Study** 📚

```python
import joblib

# Load v9.0 artifacts for educational analysis
pipeline = joblib.load('Artefacts/v9.0_educational/methodology_pipeline_v9.pkl')
artifacts = joblib.load('Artefacts/v9.0_educational/artifacts_v9.pkl')

# Study the approach but don't use for production
detailed_prediction = pipeline.predict([abstract])
print(f"8-category prediction: {detailed_prediction}")
print("Note: This approach was discontinued")
```

### **For Production Use** ✅

```python
# Use v7.0 instead
pipeline = joblib.load('Artefacts/current/methodology_classifier_v7/methodology_pipeline_v7.pkl')
production_prediction = pipeline.predict([abstract])
print(f"Production prediction: {production_prediction}")  # QUAL/QUANT/MIXED
```

---

## 🔗 **Related Educational Resources**

- **Notebook**: `Notebooks/v9.0_educational/methodology_classifier_v9.ipynb`
- **Dataset**: `Data/MASTER_DETAILED_METHODOLOGY.csv` (experimental 8-category labels)
- **Classifier Tool**: `Scripts/detailed_methodology_classifier_v9/` (automated classification system)

---

## 🎓 **Strategic Insight**

> **The most valuable lesson from v9.0**: Sometimes the best engineering decision is knowing when to abandon a technically successful approach in favor of a simpler, more practical solution.

**v9.0 → v7.0 represents mature strategic thinking in ML engineering.**

---

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Educational artifacts demonstrating strategic decision-making in ML classification design – BSc Final Year Project_
