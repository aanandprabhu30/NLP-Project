#!/usr/bin/env python3
"""Refined Methodology Checker (no pandas)

Usage:
    python methodology_checker.py --input path/to/your.csv [--output corrected.csv]

The script:
    1. reads the CSV (expects columns: Title, Abstract, Link, Discipline, Methodology)
    2. cleans whitespace in Abstract
    3. predicts Methodology label (Qualitative, Quantitative, Mixed, Unclear) using refined keyword rules
    4. adds two new columns:
        - Predicted_Methodology
        - Needs_Review (True if Predicted != Methodology)
    5. writes to the output CSV (default: <input_basename>_checked.csv)
"""

import re
import argparse
import csv
from pathlib import Path
from typing import List

# Keyword lists (expand as needed)
QUAL_KWS = [
    "interview", "focus group", "case study", "ethnograph", "observation",
    "content analysis", "grounded theory", "delphi", "narrative", "phenomenolog",
    "qualitative"
]

QUANT_KWS = [
    "experiment", "survey", "regression", "statistical", "empirical",
    "dataset", "quantitative", "measurement", "randomized", "simulation",
    "anova", "t-test", "logistic regression", "analysis of variance"
]

MIXED_KWS = [
    "mixed method", "multi-method", "qualitative and quantitative",
    "quantitative and qualitative", "both qualitative and quantitative",
    "mixed approach"
]

def _contains(text: str, keywords: List[str]) -> bool:
    text = text.lower()
    for kw in keywords:
        if kw in text:
            return True
    return False

def predict_methodology(abstract: str) -> str:
    text = abstract.lower()
    is_mixed_kw = _contains(text, MIXED_KWS)
    has_qual = _contains(text, QUAL_KWS)
    has_quant = _contains(text, QUANT_KWS)

    if is_mixed_kw or (has_qual and has_quant):
        return "Mixed"
    if has_qual:
        return "Qualitative"
    if has_quant:
        return "Quantitative"
    return "Unclear"

def clean_whitespace(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    parser = argparse.ArgumentParser(description="Keyword-based methodology checker (no pandas).")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=False, help="Path to output CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_checked.csv")

    with open(input_path, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        required_cols = {"Abstract", "Methodology"}
        if not fieldnames or not required_cols.issubset(set(fieldnames)):
            raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}. Found: {fieldnames}")

        # Prepare output columns
        output_fieldnames = list(fieldnames) + ["Predicted_Methodology", "Needs_Review"]
        rows = []
        total = 0
        for row in reader:
            total += 1
            # Clean whitespace in all fields
            for k in row:
                row[k] = clean_whitespace(row[k])
            # Predict methodology
            pred = predict_methodology(row["Abstract"])
            row["Predicted_Methodology"] = pred
            row["Needs_Review"] = str(pred != row["Methodology"])
            rows.append(row)
            if total % 500 == 0:
                print(f"Processed {total} rows...")

    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Saved checked file to {output_path} ({total} rows)")

if __name__ == "__main__":
    main()
