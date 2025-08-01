# Research Methodology Classifier for Computing Disciplines

## Overview

This tool classifies research papers in computing disciplines based on their research methodology using GPT-4o-mini. It analyzes paper abstracts and assigns one of seven methodology categories commonly used in computing research.

## Usage

```bash
python methodology_classifier.py input.csv
```

## Methodology Categories

### 1. EXPT (Experimental Research Study)

**What it covers:**

- Controlled experiments with hypotheses and variables
- A/B testing and randomized controlled trials
- User studies with control and treatment groups
- Statistical significance testing
- Empirical validation of theories

**Keywords:** experiment, hypothesis, control group, treatment, statistical significance, p-value, participants, subjects

### 2. PERF (Performance Benchmarking Evaluation)

**What it covers:**

- Performance measurement and evaluation
- Comparative benchmarking of systems/algorithms
- Scalability and efficiency studies
- Resource utilization analysis
- Load testing and stress testing

**Keywords:** benchmark, performance, evaluation, metrics, throughput, latency, scalability, efficiency

### 3. SYST (System Tool Development)

**What it covers:**

- Development of complete software systems
- Tool and framework implementation
- Platform and infrastructure development
- End-to-end application development
- Prototype system implementation

**Keywords:** system architecture, tool, framework, platform, prototype, implementation, application

### 4. ALGO (Algorithm Method Development)

**What it covers:**

- Novel algorithm design and development
- New computational methods and techniques
- Optimization algorithms
- Machine learning model architectures
- Data structures and algorithmic approaches

**Keywords:** algorithm, method, technique, approach, novel, proposed, mathematical formulation

### 5. CASE (Case Study Analysis)

**What it covers:**

- In-depth analysis of real-world implementations
- Organizational technology adoption studies
- Industry deployment experiences
- Lessons learned from practical applications
- Field studies and action research

**Keywords:** case study, organization, deployment, field study, lessons learned, real-world, industry

### 6. REVW (Literature Review Survey)

**What it covers:**

- Systematic literature reviews
- Research surveys and meta-analyses
- State-of-the-art reviews
- Bibliometric analyses
- Research trend identification

**Keywords:** systematic review, literature review, survey, meta-analysis, state-of-the-art, research trends

### 7. ANAL (Analytical Theoretical Study)

**What it covers:**

- Mathematical proofs and theorems
- Formal methods and verification
- Theoretical frameworks and models
- Complexity analysis
- Abstract mathematical analysis

**Keywords:** theorem, proof, formal analysis, mathematical model, complexity, theoretical framework

## Features

- **Multi-threaded processing** with up to 50 concurrent API calls
- **API key rotation** across 3 keys with rate limiting (500 RPM per key)
- **Checkpoint system** saves progress every 100 papers
- **Automatic retry** with exponential backoff for failed requests
- **Progress bar** with ETA and processing statistics
- **Multiple encoding support** (UTF-8, Latin-1, CP1252, ISO-8859-1)

## Output Files

1. `input_classified.csv` - Original CSV with `new_methodology` column filled with 4-letter codes
2. `methodology_classifier.log` - Detailed processing log
3. `classification_statistics.json` - Summary statistics with methodology distribution
4. `checkpoint.json` - Progress checkpoint (automatically removed on completion)

## Special Classifications

- **UNKNOWN**: Abstract is too short (<50 words) or unclear
- **NO_MATCH**: No clear methodology is indicated in the abstract
- **ERROR**: Classification failed due to technical issues

## Requirements

```bash
pip install openai
```

## Resume from Checkpoint

If processing is interrupted, simply run the same command again. The script will automatically resume from the last checkpoint, preserving all previously classified papers.

## Example Output

After processing, the script provides:

- Methodology distribution statistics
- Processing rate and time
- Count of papers per methodology
- Low-confidence classifications logged for review
