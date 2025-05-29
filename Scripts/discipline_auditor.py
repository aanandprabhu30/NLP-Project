#!/usr/bin/env python3
"""Discipline Auditor

Checks whether the Discipline label (CS / IS / IT) in a CSV is consistent
with cues in the Title+Abstract.

Usage:
    python discipline_auditor.py --input your.csv [--output audit.csv]

Adds two columns:
    Predicted_Discipline  – CS / IS / IT / Unknown
    Discipline_Mismatch   – True if Predicted ≠ Discipline
"""

import re, csv, argparse
from pathlib import Path

CS_KWS = [
    "algorithm", "machine learning", "deep learning", "neural network",
    "computer vision", "natural language", "software", "compiler", "database",
    "cryptography", "operating system", "distributed system", "programming",
    "hardware", "robotic", "cloud computing", "edge computing", "gpu",
    "artificial intelligence", "reinforcement learning", "data mining",
    "pattern recognition", "bioinformatics", "theoretical computer science",
    "automata", "complexity", "data structure", "graph theory",
    "computer graphics", "virtual reality", "augmented reality",
    "human-computer interaction", "information retrieval", "semantic web",
    "blockchain", "quantum computing", "software engineering",
    "mobile computing", "parallel computing", "computer architecture"
]

IS_KWS = [
    "information system", "erp", "enterprise resource planning",
    "e-commerce", "e-business", "blockchain adoption", "decision support",
    "technology acceptance", "user adoption", "digital transformation",
    "it governance", "business intelligence", "analytics", "it capability",
    "management information", "socio-technical", "organizational",
    "knowledge management", "information management", "business process",
    "customer relationship management", "crm", "supply chain management",
    "scm", "information security management", "it strategy",
    "change management", "project management", "information policy",
    "digital business", "enterprise architecture", "data governance",
    "information ethics", "it alignment", "information quality"
]

IT_KWS = [
    "network architecture", "data center", "infrastructure", "edge computing",
    "cloud infrastructure", "virtual machine", "storage system",
    "it operations", "devops", "system administration", "cybersecurity",
    "iot", "internet of things", "sdn", "software-defined network",
    "virtualization", "container", "kubernetes",
    "network security", "firewall", "load balancing", "backup", "disaster recovery",
    "system integration", "serverless", "microservices", "api management",
    "identity management", "access control", "endpoint security",
    "mobile device management", "network monitoring", "dns", "dhcp",
    "cloud migration", "cloud security", "zero trust", "vpn", "wireless network",
    "lan", "wan", "network protocol", "network management"
]

CS_CONTEXT = [
    "theory", "algorithmic", "computational complexity", "formal methods",
    "proof", "simulation", "implementation", "architecture", "protocol"
]

IS_CONTEXT = [
    "adoption", "business", "organization", "management", "strategy",
    "process", "policy", "governance", "decision", "stakeholder", "user",
    "acceptance", "impact", "practice", "socio-technical", "change"
]

IT_CONTEXT = [
    "deployment", "infrastructure", "operations", "maintenance", "integration",
    "security", "network", "system administration", "configuration",
    "performance", "monitoring", "backup", "disaster recovery"
]

def find_kw(text, kws):
    return any(re.search(rf"\b{re.escape(kw)}\b", text, flags=re.I) for kw in kws)

def score_keywords(text, kws, weight=2):
    score = 0
    for kw in kws:
        # Phrase match (multi-word, higher weight)
        if ' ' in kw and kw.lower() in text:
            score += 3
        # Substring match (case-insensitive)
        elif kw.lower() in text:
            score += weight
    return score

def predict(text):
    text = text.lower()
    cs_score = score_keywords(text, CS_KWS, 2) + score_keywords(text, CS_CONTEXT, 1)
    is_score = score_keywords(text, IS_KWS, 2) + score_keywords(text, IS_CONTEXT, 1)
    it_score = score_keywords(text, IT_KWS, 2) + score_keywords(text, IT_CONTEXT, 1)
    scores = {"CS": cs_score, "IS": is_score, "IT": it_score}
    max_score = max(scores.values())
    if max_score == 0:
        return "Unknown"
    # Check for ties
    top_disciplines = [k for k, v in scores.items() if v == max_score]
    if len(top_disciplines) == 1:
        return top_disciplines[0]
    return "Unknown"

def main():
    ap = argparse.ArgumentParser(description="Discipline auditor")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output) if args.output else inp.with_name(inp.stem + "_discipline_audit.csv")

    with inp.open(newline='', encoding='utf-8') as f_in, outp.open("w", newline='', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        if "Discipline" not in reader.fieldnames or "Abstract" not in reader.fieldnames:
            raise ValueError("CSV must have Discipline and Abstract columns")
        fieldnames = reader.fieldnames + ["Predicted_Discipline", "Discipline_Mismatch"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        mismatches = 0
        total = 0
        for row in reader:
            total += 1
            text = f"{row.get('Title','')} {row.get('Abstract','')}"
            pred = predict(text)
            mismatch = pred != row["Discipline"]
            row["Predicted_Discipline"] = pred
            row["Discipline_Mismatch"] = str(mismatch)
            if mismatch:
                mismatches += 1
            writer.writerow(row)

    rate = mismatches / total * 100 if total else 0
    print(f"Audit complete → {outp} | {mismatches}/{total} mismatches ({rate:.1f}%)")

if __name__ == "__main__":
    main()
