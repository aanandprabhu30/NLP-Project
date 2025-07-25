# 📦 Artefacts Organization

This folder contains all trained models and artifacts from the NLP project, organized by version and purpose.

## 📁 Folder Structure

### `current/` - v7.0 (🎯 **PRODUCTION READY**)

Methodologically sound models recommended for all real-world use

- `discipline_classifier_v7/` - 92.32% accuracy (properly validated)
- `methodology_classifier_v7/` - 86.92% accuracy (robust single model)
- `cs_subfield_classifier_v7/` - 82.92% accuracy (methodologically correct)
- `is_subfield_classifier_v7/` - 80.33% accuracy (properly validated)
- `it_subfield_classifier_v7/` - 85.39% accuracy (production-ready)

**Use these for:** Production deployments, real-world applications, reliable classification

### `v6.0_educational/` - v6.0 (⚠️ **EDUCATIONAL USE ONLY**)

High test scores but methodological flaws - valuable for learning about ML pitfalls

- `discipline_classifier_v6.0/` - 94.77% accuracy (inflated by data leakage)
- `methodology_classifier_v6.0/` - 91.87% accuracy (overfitted ensemble)
- `subfield_classifier_v6.0/` - 83-90% accuracy (validation issues)

**Use these for:** Educational examples, research on ML methodology, case studies of common pitfalls

### `v8.0_educational/` - v8.0 (⚠️ **EDUCATIONAL USE ONLY**)

False premise research case study - valuable for learning about baseline verification

- `methodology_classifier_v8/` - 82.76% accuracy, 0.564 Mixed F1 (worse than v7.0's 0.68)
- `methodology_ensemble_v8/` - 84.57% accuracy, 0.644 Mixed F1 (still worse than v7.0)

**Built on false premise:** Assumed v7.0 was weak at Mixed detection (F1≈0.35) when actual F1=0.68

**Use these for:** Learning about baseline verification, research methodology rigor, assumption validation

### `shared/`

Files used across multiple versions

- `split_indices_v7.pkl` - Consistent train/validation/test splits for v7.0 models

### `legacy/`

Historical development artifacts (v2.x - v5.x)

Contains all previous experiments, baseline models, and development iterations including:

- v2.x: SPECTER + XGBoost experiments
- v3.x: SciBERT + LoRA implementations  
- v4.x: Advanced feature engineering attempts
- v5.x: Ensemble approaches

**Use these for:** Historical reference, understanding project evolution, reproducing earlier experiments

## 🎯 Quick Start

### For Latest Development

```python
# Load v8.0 models (cutting-edge development)
from joblib import load

# Recommended for production
methodology_model_v7 = load('current/methodology_classifier_v7/methodology_pipeline_v7.pkl')

# Alternative experimental approaches
# methodology_model_v8 = load('v8.0_educational/methodology_classifier_v8/artifacts_v8.pkl')  # Educational only
# ensemble_model_v8 = load('v8.0_educational/methodology_ensemble_v8/ensemble_artifacts.pkl')    # Educational only
```

### For Production Use

```python
# Load v7.0 models (recommended)
from joblib import load

discipline_model = load('current/discipline_classifier_v7/discipline_pipeline_v7.pkl')
methodology_model = load('current/methodology_classifier_v7/methodology_pipeline_v7.pkl')
```

### For Educational/Research

```python
# Load v6.0 models (educational examples of data leakage issues)
discipline_model_v6 = load('v6.0_educational/discipline_classifier_v6.0/discipline_classifier_v6.0_pipeline.pkl')

# Load v8.0 models (educational examples of false premise issues)
methodology_model_v8 = load('v8.0_educational/methodology_classifier_v8/artifacts_v8.pkl')
```

## 🔄 Migration Notes

If updating existing code:

- Replace `../Artefacts/discipline_classifier_v7/` with `../Artefacts/current/discipline_classifier_v7/`
- Replace `../Artefacts/discipline_classifier_v6.0/` with `../Artefacts/v6.0_educational/discipline_classifier_v6.0/`
- Replace `../Artefacts/current/methodology_classifier_v8/` with `../Artefacts/v8.0_educational/methodology_classifier_v8/`

## 📊 Version Comparison Summary

| **Version** | **Status** | **Accuracy** | **Recommendation** |
|-------------|------------|--------------|-------------------|
| **v7.0** | ✅ Production | 92.32% (0.68 Mixed F1) | Use for all deployments |
| **v8.0** | ⚠️ Educational | 82.76% (0.564 Mixed F1) | Study false premise issues |
| **v6.0** | ⚠️ Educational | 94.77% (inflated) | Study data leakage issues |
| **Legacy** | 📚 Historical | Various | Reference only |

## 🎓 Key Learning

The most important insight from this project: **v6.0's higher accuracy was due to data leakage and overfitting**. v7.0's slightly lower but methodologically sound performance represents true model capability and is far more valuable for real-world applications.
