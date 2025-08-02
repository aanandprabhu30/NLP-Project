# 📊 Data - Production Dataset

This folder contains the final, production-ready dataset for the NLP classification project.

## 🎯 **Ultra-Minimal Structure**

| File | Size | Date | Description |
|------|------|------|-------------|
| **`Master.csv`** | 37MB | July 23, 2025 | **Production dataset** - Discipline, subfield, methodology (3-level) |
| `MASTER_DETAILED_METHODOLOGY.csv` | 37MB | August 1, 2025 | **Experimental dataset** - Includes detailed 8-category methodology (discontinued) |
| `README.md` | 5.8KB | Documentation | This file |

## 📋 **Dataset Columns**

### **Master.csv** (Production)

```bash
title,abstract,discipline,subfield,methodology
```

- **title**: Paper title
- **abstract**: Paper abstract text  
- **discipline**: CS (Computer Science) | IS (Information Systems) | IT (Information Technology)
- **subfield**: Specialized area within discipline (AI/ML, CV, NLP, SE, SEC, BPM, DT, GOV, etc.)
- **methodology**: Qualitative | Quantitative | Mixed

### **MASTER_DETAILED_METHODOLOGY.csv** (⚠️ Experimental - Discontinued)

```bash
title,abstract,discipline,subfield,methodology,detailed_methodology
```

- **Same as above** plus:
- **detailed_methodology**: ALGO | CASE | SYST | REVW | EXPT | PERF | ANAL | UNKNOWN

**Note**: The detailed 8-category classification was abandoned due to definitional overlap and ambiguity. Use **Master.csv** for production work.

## 🚀 **Quick Start**

### Load Production Data

```python
import pandas as pd

# Load the complete production dataset
df = pd.read_csv('Data/Master.csv')

print(f"Total papers: {len(df)}")
print(f"Disciplines: {df['discipline'].unique()}")
print(f"Methodologies: {df['methodology'].unique()}")
```

### Filter by Task

```python
# Get discipline-specific data
cs_papers = df[df['discipline'] == 'CS']
is_papers = df[df['discipline'] == 'IS'] 
it_papers = df[df['discipline'] == 'IT']

# Get methodology-specific data
quant_papers = df[df['methodology'] == 'Quantitative']
qual_papers = df[df['methodology'] == 'Qualitative']
mixed_papers = df[df['methodology'] == 'Mixed']

# Filter by subfield or methodology
cs_ai_papers = df[(df['discipline'] == 'CS') & (df['subfield'] == 'AI/ML')]
```

## 📈 **Dataset Statistics**

### Production Dataset (Master.csv)

- **Total papers**: 26,944 (excluding header)
- **Date**: July 23, 2025
- **Quality**: Production-ready, LLM-validated
- **Completeness**: All three classification tasks included
- **Status**: ✅ **Active for production use**

### Experimental Dataset (MASTER_DETAILED_METHODOLOGY.csv)

- **Total papers**: 26,944 (same papers as Master.csv)
- **Date**: August 1, 2025
- **Purpose**: Experimental detailed methodology classification
- **Status**: ❌ **Discontinued due to classification ambiguity**
- **Use case**: Educational study of complex classification schemes

### Coverage

- **Disciplines**: 3 (CS, IS, IT)
- **Subfields**: 15+ specialized areas
- **Methodologies**: 3 (Qualitative, Quantitative, Mixed)
- **Quality**: Production-ready, clean labeled data

## 🎯 **Why This Structure?**

### **Dual-Purpose Design**

- **Production clarity** - Master.csv is the single source of truth for production
- **Educational value** - MASTER_DETAILED_METHODOLOGY.csv demonstrates complex classification challenges
- **Lesson preservation** - Keep experimental work for learning purposes

### **Production Focus (Master.csv)**

- **Simplicity** - 3-level methodology classification (QUAL/QUANT/MIXED)
- **Proven approach** - v7.0 methodology with 86.92% production accuracy
- **Clear definitions** - No ambiguity in category boundaries
- **LLM-validated** - Higher quality, reliable labels

### **Educational Value (MASTER_DETAILED_METHODOLOGY.csv)**

- **Complex classification study** - 8-category detailed methodology attempt
- **Strategic decision-making** - Shows when to abandon technically sound but impractical approaches
- **Definitional overlap example** - Demonstrates classification design challenges

## 💡 **Usage Recommendations**

### **For Production Use** (✅ Recommended)

```python
# Standard production loading
df = pd.read_csv('Data/Master.csv')
```

### **For Educational Study** (📚 Learning Only)

```python
# Study detailed classification challenges
df_detailed = pd.read_csv('Data/MASTER_DETAILED_METHODOLOGY.csv')

# Compare methodology approaches
print(df_detailed['methodology'].value_counts())  # 3-level
print(df_detailed['detailed_methodology'].value_counts())  # 8-level (discontinued)
```

### **For Model Training**

```python
# Split data for training
from sklearn.model_selection import train_test_split

X = df[['title', 'abstract']]
y_discipline = df['discipline'] 
y_subfield = df['subfield']
y_methodology = df['methodology']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_discipline, test_size=0.2, random_state=42)
```

### **For Analysis**

```python
# Analyze distribution
print(df['discipline'].value_counts())
print(df['methodology'].value_counts())
print(df.groupby('discipline')['subfield'].value_counts())
```

## 🔄 **Which Dataset to Use?**

### **For Production/Research Work**

```python
# ✅ USE THIS - Master.csv (3-level methodology)
df = pd.read_csv('Data/Master.csv')
methodology_data = df[['title', 'abstract', 'methodology']]  # QUAL/QUANT/MIXED
```

### **For Educational Study Only**

```python
# 📚 EDUCATIONAL ONLY - MASTER_DETAILED_METHODOLOGY.csv
df_detailed = pd.read_csv('Data/MASTER_DETAILED_METHODOLOGY.csv')
detailed_data = df_detailed[['title', 'abstract', 'detailed_methodology']]  # 8 categories

# WARNING: detailed_methodology was discontinued due to:
# - Definitional overlap between categories
# - Classification ambiguity 
# - Reduced practical value
```

### **Migration from Old Structure**

```python
# OLD (multiple files):
# discipline_df = pd.read_csv('Data/task_specific/discipline/trusted_discipline_dataset.csv')
# methodology_df = pd.read_csv('Data/task_specific/methodology/methodology_final.csv')

# NEW (single file):
df = pd.read_csv('Data/Master.csv')
discipline_data = df[['title', 'abstract', 'discipline']]
methodology_data = df[['title', 'abstract', 'methodology']]
subfield_data = df[['title', 'abstract', 'discipline', 'subfield']]
```

---

## 👨‍💻 Author

Aanand Prabhu  
[GitHub → @aanandprabhu30](https://github.com/aanandprabhu30)

> _Streamlined for production use - BSc Final Year Project in Computer Science – University of London_
