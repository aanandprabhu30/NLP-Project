# 📊 Data - Production Dataset

This folder contains the final, production-ready dataset for the NLP classification project.

## 🎯 **Ultra-Minimal Structure**

| File | Size | Date | Description |
|------|------|------|-------------|
| **`Master.csv`** | 37MB | July 22, 2025 | **Complete production dataset** - All papers with discipline, subfield, methodology labels |
| `README.md` | 5.8KB | Documentation | This file |

## 📋 **Master.csv Columns**

``` bash
title,abstract,discipline,subfield,methodology,confidence
```

- **title**: Paper title
- **abstract**: Paper abstract text  
- **discipline**: CS (Computer Science) | IS (Information Systems) | IT (Information Technology)
- **subfield**: Specialized area within discipline (AI/ML, CV, NLP, SE, SEC, BPM, DT, GOV, etc.)
- **methodology**: Qualitative | Quantitative | Mixed
- **confidence**: Model confidence score

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

# Filter by confidence threshold
high_confidence = df[df['confidence'] > 0.8]
```

## 📈 **Dataset Statistics**

### Production Dataset (Master.csv)

- **Total papers**: ~26,944
- **Date**: July 22, 2025 (latest)
- **Quality**: Production-ready, LLM-validated
- **Completeness**: All three classification tasks included

### Coverage

- **Disciplines**: 3 (CS, IS, IT)
- **Subfields**: 15+ specialized areas
- **Methodologies**: 3 (Qualitative, Quantitative, Mixed)
- **Confidence**: Included for quality control

## 🎯 **Why This Structure?**

### **Simplicity**

- **Single source of truth** - no confusion about which dataset to use
- **Production-ready** - consolidated, validated, complete
- **Minimal storage** - 77% reduction from 158MB to 37MB

### **Completeness**

- **All three tasks** in one file (discipline, subfield, methodology)
- **All required fields** (title, abstract, labels, confidence)
- **Latest data** (July 2025 - most recent validation)

### **Production Focus**

- **v7.0 methodology** - methodologically sound data practices
- **LLM-validated** - higher quality labels
- **Ready for deployment** - no preprocessing needed

## 💡 **Usage Recommendations**

### **For Production Use**

```python
# Standard production loading
df = pd.read_csv('Data/Master.csv')
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

## 🔄 **Migration from Old Structure**

If updating from previous data organization:

```python
# OLD (multiple files):
# discipline_df = pd.read_csv('Data/task_specific/discipline/trusted_discipline_dataset.csv')
# methodology_df = pd.read_csv('Data/task_specific/methodology/methodology_final.csv')
# subfield_df = pd.read_csv('Data/task_specific/subfield/CS_subfields.csv')

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
