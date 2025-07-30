# 📚 Legacy Development Notebooks – Historical Evolution (v1.x - v5.x)

## 🏛️ **Purpose**

This collection contains **all historical development notebooks** from the project's evolution, documenting the journey from initial experiments to production-ready models. These notebooks showcase different approaches, methodologies, and lessons learned during the development process.

**📊 Performance Note**: All accuracy values marked with ✓ are **verified from actual notebook outputs**. Values without ✓ may be estimates or incomplete results.

---

## 🗂️ **Version Timeline**

### **v1.x Era (2021-2022): Foundation & Early Experiments**

- Basic TF-IDF + Logistic Regression approaches
- Initial class balancing with SMOTE
- Simple train-test splits
- Proof-of-concept implementations

### **v2.x Era (2022-2023): Advanced Features & Deep Learning**

- Introduction of SPECTER embeddings
- SciBERT and transformer experiments
- XGBoost ensemble methods
- Title + Abstract feature combinations

### **v3.x Era (2023): Transformer Fine-tuning**

- DeBERTa with LoRA adaptation
- Advanced transformer architectures
- Fine-tuning for domain specificity

### **v4.x Era (2023-2024): Feature Engineering**

- Advanced feature engineering attempts
- Complex preprocessing pipelines
- Dependency optimization

### **v5.x Era (2024): Ensemble & Optimization**

- Ensemble approaches
- Performance optimization
- Preparation for v6/v7 methodological corrections

---

## 📄 **Notebook Inventory**

### **🏗️ v1.x Foundation Era**

#### `CrossValidation_AllModels (v1.0).ipynb`

- **Size**: 15KB, 405 lines
- **Purpose**: Initial cross-validation framework for all models
- **Models**: Logistic Regression across all classifiers
- **Approach**: Basic 5-fold CV evaluation
- **Legacy Status**: Foundational validation approach

#### `NLP_Classifier_DisciplineOnly (v1.1).ipynb`

- **Size**: 6.4KB, 191 lines
- **Purpose**: Pure discipline classification (CS/IS/IT)
- **Architecture**: Logistic Regression + TF-IDF
- **Features**: Basic unigram/bigram features
- **Historical Value**: Shows initial classification approach

#### `NLP_Classifier_SubfieldOnly_CS (v1.2).ipynb`

#### `NLP_Classifier_SubfieldOnly_IS (v1.2).ipynb`

#### `NLP_Classifier_SubfieldOnly_IT (v1.2).ipynb`

- **Size**: ~8-9KB each, ~250-270 lines
- **Purpose**: Domain-specific subfield classification
- **Architecture**: SVM + SMOTE + TF-IDF
- **Approach**: Separate models for each discipline
- **Legacy Insight**: Early specialization strategy

#### `NLP_Methodology_Classifier (v1.2).ipynb`

- **Size**: 11KB, 363 lines
- **Purpose**: Research methodology classification
- **Architecture**: LinearSVC + SMOTE + Bigram TF-IDF
- **Classes**: QLT (Qualitative), QNT (Quantitative), M (Mixed)
- **Performance**: Foundation for methodology detection

### **🚀 v2.x Advanced Era**

#### `CrossValidation_AllModels (v1.2 and v2.0).ipynb`

- **Size**: 14KB, 378 lines
- **Purpose**: Comprehensive cross-validation for v1.2 & v2.0 models
- **Models Evaluated**:
  - Discipline Classifier (Logistic Regression + Bigram TF-IDF v1.1)
  - Subfield Classifiers (SVM + SMOTE v1.2)
  - Methodology Classifier (SVM + SMOTE + Title+Abstract v2.0)
- **Results**: 5-fold CV stability analysis

#### `NLP_Methodology_Classifier (v2.0).ipynb`

- **Size**: 11KB, 338 lines
- **Purpose**: Enhanced methodology classification
- **Innovation**: **Title + Abstract** input (vs Abstract-only)
- **Architecture**: SVM + SMOTE + TF-IDF bigrams
- **Performance**: 71% accuracy, F1: 0.68 weighted
- **Insight**: Title information improves qualitative detection

#### `methodology_classifier_(v2.1)_bert.ipynb`

- **Size**: 10KB, 351 lines
- **Purpose**: First transformer experiment for methodology
- **Architecture**: Sentence-BERT (all-MiniLM-L6-v2) + SVM/LogReg
- **Experiments**:
  - Unscaled embeddings: 43.8% accuracy
  - Scaled embeddings: 52.4% accuracy
- **Lesson**: TF-IDF still outperformed BERT on small datasets

#### `methodology_classifier_(v2.2)_scibert.ipynb`

- **Size**: 12KB, 456 lines
- **Purpose**: Domain-specific BERT for methodology
- **Architecture**: SciBERT embeddings + XGBoost
- **Performance**: 65-76% accuracy range
- **Cross-validation**: Mean 65.7% ± 10.2%
- **Insight**: Domain BERT helped but still below TF-IDF

#### `cs_subfield_classifier_specter_xgboost (v2.3 and v2.4).ipynb`

#### `is_subfield_classifier_specter_xgboost (v2.3 and v2.4).ipynb`

#### `it_subfield_classifier_specter_xgboost (v2.3 and v2.4).ipynb`

- **Size**: ~16KB each, ~535-543 lines
- **Purpose**: SPECTER embeddings + XGBoost for subfields
- **Architecture**: 768-dim SPECTER embeddings + XGBoost
- **Performance**:
  - CS: 75-76% accuracy, F1: 0.74-0.75
  - IS: 88-89% accuracy, F1: 0.88-0.90
  - IT: Similar performance range
- **Innovation**: Scientific document embeddings

#### `methodology_classifier_specter_xgboost (v2.3, v2.4 and v2.5).ipynb`

- **Size**: 21KB, 688 lines
- **Purpose**: SPECTER + XGBoost for methodology
- **Versions**: Progressive improvements v2.3 → v2.4 → v2.5
- **Architecture**: SPECTER embeddings + XGBoost with hyperparameter tuning
- **Innovation**: Scientific paper embeddings for methodology detection

#### `methodology_classifier_specter_xgboost_v2.6.ipynb`

- **Size**: 17KB, 478 lines
- **Purpose**: Final v2.x methodology classifier
- **Architecture**: Advanced SPECTER + XGBoost implementation
- **Features**: Optimized preprocessing and feature engineering

#### `NLP_Classifier_DisciplineOnly_SciBERT_XGBoost_(v2.2).ipynb`

- **Size**: 10KB, 386 lines
- **Purpose**: Domain BERT for discipline classification
- **Architecture**: SciBERT + XGBoost ensemble
- **Performance**: 91% test accuracy
- **Classes**: Excellent CS/IS performance, lower IT recall
- **Innovation**: First successful transformer-based discipline classifier

### **🧬 v3.x Transformer Era**

#### `discipline_classifier_deberta_lora_v3.0.ipynb`

- **Size**: 9.9KB, 325 lines
- **Purpose**: Advanced transformer fine-tuning
- **Architecture**: DeBERTa + LoRA (Low-Rank Adaptation)
- **Innovation**: Parameter-efficient fine-tuning
- **Approach**: Domain-specific transformer adaptation

#### `lora_discipline_classifier(v3.1).ipynb`

- **Size**: 18KB, 593 lines
- **Purpose**: Enhanced LoRA implementation
- **Architecture**: Advanced LoRA configuration
- **Features**:
  - Mixed precision training
  - Custom evaluation metrics
  - Compute accuracy and F1 scoring
- **Technical**: Modern transformer training practices

### **🔧 v4.x Engineering Era**

#### `discipline_classifier_v4_0.ipynb`

- **Size**: 24KB, 787 lines
- **Purpose**: Advanced feature engineering attempts
- **Approach**: Complex preprocessing pipelines
- **Focus**: Feature optimization and engineering

### **🏭 v5.x Optimization Era**

#### `Discipline_classifier_v5_0_colab_pro.ipynb`

- **Size**: 22KB, 749 lines
- **Purpose**: Production optimization experiments
- **Platform**: Google Colab Pro optimization
- **Focus**: Performance and scalability improvements

### **🔍 Special Purpose Notebooks**

#### `ai_vs_ml_disambiguator.ipynb`

- **Size**: 6.2KB, 202 lines
- **Purpose**: AI vs ML paper disambiguation
- **Performance**: 68% accuracy, F1: 0.67-0.68
- **Application**: Fine-grained CS subfield distinction

#### `discipline_trust_score_filtering_v0.1.ipynb`

- **Size**: 7.0KB, 212 lines
- **Purpose**: Confidence-based filtering experiments
- **Innovation**: Trust score implementation
- **Application**: Quality control for predictions

#### `NLP_Pipeline_Prototype_15_Abstracts.ipynb`

- **Size**: 8.8KB, 301 lines
- **Purpose**: Initial pipeline prototype
- **Dataset**: 15 paper proof-of-concept
- **Value**: Early system integration attempt

#### `Evaluate_DisciplineClassifier (v1.0).ipynb`

- **Size**: 5.0KB, 185 lines
- **Purpose**: Standalone evaluation framework
- **Focus**: Model assessment and validation

---

## 📊 **Performance Evolution**

### **Methodology Classification Progress**

| **Version** | **Architecture** | **Accuracy** | **Key Innovation** |
|-------------|------------------|--------------|-------------------|
| v1.2 | SVM + TF-IDF (Abstract) | ~65% | SMOTE balancing |
| v2.0 | SVM + TF-IDF (Title+Abstract) | 71% | Title inclusion |
| v2.1 | BERT + SVM | 44-52% | First transformer attempt |
| v2.2 | SciBERT + XGBoost | 66-76% | Domain-specific BERT |
| v2.3-2.6 | SPECTER + XGBoost | 70-80% | Scientific embeddings |

### **Discipline Classification Evolution**

| **Version** | **Architecture** | **Accuracy** | **Innovation** |
|-------------|------------------|--------------|----------------|
| v1.1 | LogReg + TF-IDF | ~80% | Basic approach |
| v2.2 | SciBERT + XGBoost | 91% | Transformer breakthrough |
| v3.0-3.1 | DeBERTa + LoRA | 90%+ | Fine-tuning efficiency |
| v4.0-5.0 | Advanced Engineering | Variable | Feature optimization |

### **Subfield Classification Journey**

| **Discipline** | **v1.2 (SVM)** | **v2.3-2.4 (SPECTER)** | **Key Learning** |
|-------------|----------------|------------------------|------------------|
| **CS** | ~70% | 75-76% | SPECTER helps AI/ML detection |
| **IS** | ~75% | 88-89% | Business context well-captured |
| **IT** | ~70% | 75-80% | Infrastructure keywords effective |

---

## 🧠 **Key Lessons Learned**

### **Architecture Insights**

1. **TF-IDF Resilience**: Traditional TF-IDF consistently outperformed early BERT on small datasets
2. **Domain Embeddings**: SPECTER and SciBERT provided meaningful improvements over generic embeddings
3. **Ensemble Benefits**: XGBoost consistently outperformed single classifiers
4. **Title Value**: Including titles improved methodology detection

### **Technical Discoveries**

1. **Feature Scaling**: Critical for transformer embeddings
2. **Class Imbalance**: SMOTE effective for small academic datasets  
3. **Domain Specificity**: Academic-trained models outperformed generic ones
4. **Hyperparameter Sensitivity**: XGBoost required careful tuning

### **Methodological Evolution**

1. **Cross-Validation**: Early adoption of proper validation practices
2. **Multiple Metrics**: F1-score more informative than accuracy alone
3. **Confidence Scoring**: Trust scores valuable for production deployment
4. **Pipeline Integration**: Hierarchical classification more effective

---

## 🔬 **Research Contributions**

### **Novel Approaches Tested**

- **SPECTER Embeddings**: First application to academic paper classification
- **LoRA Fine-tuning**: Parameter-efficient domain adaptation
- **Hierarchical Classification**: Discipline → Subfield → Methodology workflow
- **Smart Post-processing**: Rule-based confidence corrections

### **Methodological Innovations**

- **Title + Abstract**: Comprehensive text feature utilization
- **Domain Feature Engineering**: Keyword-based academic features
- **Cross-validation Frameworks**: Systematic model evaluation
- **Trust Score Filtering**: Confidence-based quality control

---

## 📈 **Performance Benchmarks**

### **Best Historical Results**

| **Classifier** | **Best Version** | **Peak Performance** | **Architecture** |
|-------------|------------------|---------------------|------------------|
| **Discipline** | v2.2 | 91% accuracy ✓ | SciBERT + XGBoost |
| **Methodology** | v2.6 | 77% accuracy ✓ | SPECTER + XGBoost |
| **CS Subfield** | v2.4 | 76% accuracy ✓ | SPECTER + XGBoost |
| **IS Subfield** | v2.4 | 89% accuracy ✓ | SPECTER + XGBoost |
| **IT Subfield** | v2.4 | 83% accuracy ✓ | SPECTER + XGBoost |

### **Stability Analysis**

- **Cross-validation**: Most models showed ±5-10% standard deviation
- **Methodology**: Highest variance due to class imbalance
- **Discipline**: Most stable classifier across versions
- **Subfields**: IS most consistent, CS most challenging

---

## 🛠️ **Technical Artifacts**

### **Model Types Explored**

- **Traditional ML**: Logistic Regression, SVM, Random Forest
- **Gradient Boosting**: XGBoost with extensive hyperparameter tuning
- **Transformers**: BERT, SciBERT, DeBERTa with LoRA
- **Embeddings**: SPECTER, Sentence-BERT, domain-specific models

### **Feature Engineering Techniques**

- **Text Features**: Unigrams, bigrams, TF-IDF, embeddings
- **Domain Features**: Academic keyword matching, citation patterns
- **Preprocessing**: Stop-word removal, stemming, domain-specific cleaning
- **Balancing**: SMOTE, class weighting, threshold tuning

---

## 🎓 **Educational Value**

### **For ML Learning**

1. **Evolution Study**: See how approaches evolved over time
2. **Comparative Analysis**: Direct comparison of different architectures
3. **Failure Analysis**: Understanding why certain approaches didn't work
4. **Incremental Improvement**: How small changes compound over time

### **For Research Methods**

1. **Hypothesis Testing**: Systematic approach to architecture selection
2. **Ablation Studies**: Component-wise performance analysis
3. **Validation Practices**: Cross-validation methodology development
4. **Documentation**: Comprehensive experiment tracking

### **For Production Systems**

1. **Prototyping**: Rapid experimentation and iteration
2. **Benchmarking**: Establishing performance baselines
3. **Risk Assessment**: Understanding model limitations
4. **Deployment Preparation**: Path from research to production

---

## ⚡ **Quick Navigation**

### **By Research Interest**

```bash
# Traditional ML approaches
jupyter notebook legacy/NLP_Classifier_DisciplineOnly\ \(v1.1\).ipynb

# Transformer experiments  
jupyter notebook legacy/methodology_classifier_\(v2.2\)_scibert.ipynb

# SPECTER embeddings
jupyter notebook legacy/cs_subfield_classifier_specter_xgboost\ \(v2.3\ and\ v2.4\).ipynb

# Advanced fine-tuning
jupyter notebook legacy/lora_discipline_classifier\(v3.1\).ipynb
```

### **By Performance Era**

- **Foundation (v1.x)**: Highly variable, 28-90% depending on task
- **Breakthrough (v2.x)**: Transformer adoption, 80-91% accuracy
- **Optimization (v3.x-5.x)**: Fine-tuning and engineering

---

## 🔗 **Related Artifacts**

Most legacy notebooks have corresponding model artifacts in:

- `../Artefacts/legacy/` - Trained models and vectorizers
- Performance metrics embedded in notebook outputs
- Cross-validation results for stability assessment

---

## 🏆 **Historical Significance**

These notebooks represent **6+ months of systematic research and development**, showcasing:

1. **Scientific Method**: Hypothesis-driven experimentation
2. **Iterative Improvement**: Building on previous results  
3. **Comparative Analysis**: Fair evaluation across approaches
4. **Documentation**: Complete experimental record
5. **Open Research**: Transparent methodology sharing

**This collection serves as a complete case study in applied machine learning research, from initial concepts to production-ready systems.**
