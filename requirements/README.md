# 📦 Requirements

This folder contains different requirement files for various use cases of the NLP project.

## 📁 Available Files

### `requirements.txt` - **Standard Use** (Recommended)

For most users who want to explore the project with visualization and Jupyter support

```bash
pip install -r requirements/requirements.txt
```

**Includes:**

- Core v7.0 production packages (joblib, scikit-learn, xgboost, etc.)
- Visualization tools (matplotlib, seaborn)
- Jupyter notebook support
- Text processing tools (nltk)
- Progress bars and basic development tools

**Use this if you want to:**

- Run v7.0 production models
- Explore notebooks with visualizations
- Develop new features on the v7.0 foundation

---

### `requirements-minimal.txt` - **Production Only**

For deployment environments that only need to run v7.0 models

```bash
pip install -r requirements/requirements-minimal.txt
```

**Includes:**

- Only essential packages: joblib, numpy, pandas, scikit-learn, xgboost
- No visualization or development tools
- Smallest possible footprint

**Use this if you want to:**

- Deploy v7.0 models in production
- Minimize dependencies and installation time
- Create lightweight Docker containers

---

### `requirements-dev.txt` - **Full Development**

For developers who want to explore all versions including v6.0 educational examples

```bash
pip install -r requirements/requirements-dev.txt
```

**Includes:**

- Everything from `requirements.txt`
- Advanced ML packages (transformers, BERT, LoRA, SPECTER)
- Model interpretation tools (SHAP)
- All packages needed for v6.0 educational notebooks and legacy experiments

**Use this if you want to:**

- Run v6.0 educational examples to study ML pitfalls
- Explore legacy notebooks (v2.x-v5.x)
- Conduct research comparing different approaches
- Contribute to the project development

## 🚀 Quick Start Guide

### First-time Users

```bash
# Most users should start here
pip install -r requirements/requirements.txt
```

### Production Deployment

```bash
# For production environments
pip install -r requirements/requirements-minimal.txt
```

### Researchers/Developers

```bash
# For full project exploration
pip install -r requirements/requirements-dev.txt
```

## 🔄 Migration from Root Requirements

If you previously used requirements files in the project root:

- `requirements.txt` → `requirements/requirements.txt`
- `requirements-minimal.txt` → `requirements/requirements-minimal.txt`
- `requirements-dev.txt` → `requirements/requirements-dev.txt`

## 🎯 Version Compatibility

| **File** | **v7.0 & v8.0 Models** | **v6.0 Educational** | **Legacy (v2-v5)** | **Jupyter** | **Visualization** |
|----------|:----------------------:|:--------------------:|:------------------:|:------------:|:----------------:|
| `requirements.txt` | ✅ | ❌ | ❌ | ✅ | ✅ |
| `requirements-minimal.txt` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `requirements-dev.txt` | ✅ | ✅ | ✅ | ✅ | ✅ |

## 💡 Tips

- **Start with `requirements.txt`** unless you have specific needs
- **Use `requirements-minimal.txt`** for production deployments to reduce attack surface and dependencies
- **Only use `requirements-dev.txt`** if you need to run v6.0 educational examples or legacy notebooks
- **Virtual environments are recommended** to avoid conflicts with other projects
