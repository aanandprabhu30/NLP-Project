#!/usr/bin/env python3
"""Refined Methodology Checker v3 (no pandas)

This version uses only Python's built‑in libraries (`csv`, `re`, `argparse`, `pathlib`)
so you can run it anywhere without installing extra packages.

USAGE
-----
python methodology_checker_v3.py --input your.csv [--output out.csv]

The script expects at least these columns (case‑sensitive):
    Title, Abstract, Link, Discipline, Methodology

It appends two new columns:
    Predicted_Methodology, Needs_Review  (True/False)

and writes the result to <input>_checked_v3.csv unless --output is given.
"""
import re
import csv
import argparse
from pathlib import Path
from typing import List

# ------------------------ Keyword dictionaries ------------------------ #

QUAL_KWS = [
    "interview", "semi-structured interview", "unstructured interview",
    "focus group", "case study", "ethnograph", "participant observation",
    "field note", "grounded theory", "content analysis", "thematic analysis",
    "narrative inquiry", "phenomenolog", "autoethnograph", "discourse analysis",
    "qualitative"
]

QUANT_KWS = [
    "experiment", "controlled experiment", "randomized controlled trial",
    "survey", "questionnaire", "likert", "regression", "logistic regression",
    "statistical", "empirical", "dataset", "numerical simulation",
    "benchmark", "anova", "analysis of variance", "chi-square", "χ²",
    "t-test", "f-test", "p-value", "confidence interval", "measurement",
    "ablation study", "machine learning model",
    "quantitative"
]

MIXED_KWS = [
    "mixed method", "mixed-method", "mixed methodology",
    "multi-method", "multi method", "qualitative and quantitative",
    "quantitative and qualitative", "both qualitative and quantitative",
    "convergent parallel", "exploratory sequential", "explanatory sequential",
    "triangulation", "integrated approach", "combined methods",
    "concurrent mixed", "sequential mixed", "hybrid approach",
    "integrative analysis", "multi-strategy", "multi strategy",
    # Additions:
    "combination of methods", "both approaches", "integrated methodology",
    "integration of qualitative and quantitative", "integration of methods",
    "complementary methods", "combining approaches", "combining methods",
    "using both methods", "using both approaches", "parallel mixed methods",
    "sequential mixed methods", "joint analysis", "synthesizing methods"
]

QUAL_CONTEXT = [
    "explore", "understand", "perception", "experience", "meaning", "interpret", "perspective",
    "narrative", "theme", "subjective", "insight", "motivation", "attitude", "belief", "viewpoint"
]

QUANT_CONTEXT = [
    "measure", "test", "effect", "relationship", "association", "statistically significant",
    "variable", "correlation", "predict", "analyze", "frequency", "distribution", "sample size",
    "randomized", "controlled", "experiment", "quantify", "statistical analysis"
]

MIXED_CONTEXT = [
    "combine", "integrate", "both", "converge", "merge", "complementary", "triangulate",
    "concurrent", "sequential", "hybrid", "integrative", "multi-strategy", "multi strategy",
    # Additions:
    "combination", "integration", "synthesis", "joint", "together", "blend", "synthesizing",
    "using both", "using combination", "using integration", "using synthesis"
]

# ------------------------ Helper functions --------------------------- #

def _contains(text: str, keywords: List[str]) -> bool:
    """Case-insensitive substring match for any keyword."""
    text = text.lower()
    return any(kw.lower() in text for kw in keywords)

def score_keywords(text, kws, weight=2):
    score = 0
    for kw in kws:
        if ' ' in kw and kw.lower() in text:
            score += 3
        elif kw.lower() in text:
            score += weight
    return score

def predict_methodology(abstract: str) -> str:
    text = abstract.lower()
    # Methodology keyword scores
    qual_score = score_keywords(text, QUAL_KWS, 2)
    quant_score = score_keywords(text, QUANT_KWS, 2)
    mixed_score = score_keywords(text, MIXED_KWS, 2)
    # Context keyword scores
    qual_context = score_keywords(text, QUAL_CONTEXT, 1)
    quant_context = score_keywords(text, QUANT_CONTEXT, 1)
    mixed_context = score_keywords(text, MIXED_CONTEXT, 1)
    # Bonus for co-occurrence
    if qual_score and qual_context:
        qual_score += 2
    if quant_score and quant_context:
        quant_score += 2
    if mixed_score and mixed_context:
        mixed_score += 2
    scores = {"Qualitative": qual_score, "Quantitative": quant_score, "Mixed": mixed_score}
    max_score = max(scores.values())
    # New logic: If both qual and quant scores are present, classify as Mixed (unless mixed is already highest)
    if qual_score > 0 and quant_score > 0 and mixed_score == 0:
        return "Mixed"
    if max_score == 0:
        return "Unclear"
    top = [k for k, v in scores.items() if v == max_score]
    if len(top) == 1:
        return top[0]
    return "Unclear"

def clean_whitespace(text: str) -> str:
    return re.sub(r"\\s+", " ", str(text)).strip()

# ----------------------------- Main ---------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Keyword-based methodology checker v3 (no pandas).")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", help="Path to output CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_checked_v3.csv")

    with input_path.open(newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        required_cols = {"Abstract", "Methodology"}
        if not required_cols.issubset(fieldnames):
            raise ValueError(f"CSV must have columns: {', '.join(required_cols)}")

        # Prepare writer with new columns appended
        new_fieldnames = fieldnames + ["Predicted_Methodology", "Needs_Review"]
        rows = []
        mismatches = 0
        total = 0

        for row in reader:
            total += 1
            abstract_clean = clean_whitespace(row["Abstract"])
            predicted = predict_methodology(abstract_clean)
            needs_review = predicted != row["Methodology"]

            row["Abstract"] = abstract_clean
            row["Predicted_Methodology"] = predicted
            row["Needs_Review"] = str(needs_review)  # CSV as string

            if needs_review:
                mismatches += 1
            rows.append(row)

    with output_path.open("w", newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mismatch_rate = mismatches / total * 100 if total else 0
    print(f"✓ Saved to {output_path} | {mismatches}/{total} rows flagged ({mismatch_rate:.1f}%)")

if __name__ == "__main__":
    main()
