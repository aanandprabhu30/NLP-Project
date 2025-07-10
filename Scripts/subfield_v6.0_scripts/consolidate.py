#!/usr/bin/env python3
"""
Consolidate 74 GPT-4o subfields into 14 target subfields for training v6.0 classifiers.
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os
from datetime import datetime

# Define the consolidation mapping based on your specifications
# Each consolidated category groups related subfields for better classification balance
SUBFIELD_CONSOLIDATION = {
    # ===== CS Subfields (5 consolidated categories) =====
    
    # CS-AI/ML: Core AI/ML research including robotics and data science
    # Merges AI/ML (4026), DS (215), ROB (237), ROBOT (1) = ~4,479 samples
    # These share common ML techniques, algorithms, and research approaches
    'CS-AI/ML': ['AI/ML', 'DS', 'ROB', 'ROBOT'],
    
    # CS-SEC: Security remains separate due to distinct domain
    # SEC (2674) = 2,674 samples - cryptography, cybersecurity, vulnerability analysis
    'CS-SEC': ['SEC'],
    
    # CS-CV: Computer Vision and Graphics/Visualization
    # Merges CV (2266), GFX (58) = ~2,324 samples
    # Both deal with visual computing, image processing, and rendering
    'CS-CV': ['CV', 'GFX'],
    
    # CS-NLP: Natural Language Processing and Information Retrieval
    # Merges NLP (1074), IR (22) = ~1,096 samples
    # IR uses NLP techniques for search/indexing, natural grouping
    'CS-NLP': ['NLP', 'IR'],
    
    # CS-SE: Software Engineering and all other CS subfields
    # This becomes the general CS category for systems, theory, and applied CS
    # Includes: SE (919), NET (804), CLOUD (491), ALGO (317), DB (131), and others
    # Total: ~4,374 samples - covers software development, systems, and CS theory
    'CS-SE': ['SE', 'DB', 'ALGO', 'THEORY', 'HCI', 'BIO', 'ARCH', 'MOBILE', 'CLOUD', 
          'IOT', 'QUANTUM', 'GAME', 'EDU', 'HPC', 'EMBED', 'OS', 'NET'],
    
    # ===== IS Subfields (5 consolidated categories) =====
    
    # IS-HIS: Health Information Systems
    # HIS (1600), HEALTH (1) = ~1,601 samples
    # Focused on healthcare IT, EHR, medical informatics
    'IS-HIS': ['HIS', 'HEALTH'],
    
    # IS-DT: Digital Transformation and Innovation
    # Merges DT (1444), INNOV (816), EDTECH (578), SMM (287), ECOMM (193), MOBILE (25), SOCIAL (92)
    # Total: ~3,435 samples - all about digital business transformation
    'IS-DT': ['DT', 'INNOV', 'EDTECH', 'SMM', 'ECOMM', 'MOBILE', 'SOCIAL'],
    
    # IS-GOV: Government Information Systems
    # GOV (1133) = 1,133 samples - e-government, public sector IT
    'IS-GOV': ['GOV'],
    
    # IS-BPM: Business Process and Enterprise Systems
    # Merges BPM (582), ERP (296), CRM (136), SCM (493), PM (123), EA (67), ITSM (27), EPM (2)
    # Total: ~1,726 samples - enterprise software and business processes
    'IS-BPM': ['BPM', 'ERP', 'CRM', 'SCM', 'EPM', 'PM', 'EA', 'ITSM'],
    
    # IS-KM: Knowledge Management and Decision Support
    # Merges KM (392), DSS (477), BI (307), RESEARCH (401), ISSEC (367), STRAT (255), 
    # ISM (117), PRIV (118), ETHICS (94)
    # Total: ~2,528 samples - organizational knowledge, analytics, and IS management
    'IS-KM': ['KM', 'DSS', 'BI', 'RESEARCH', 'ISSEC', 'STRAT', 'ISM', 'PRIV', 'ETHICS'],
    
    # ===== IT Subfields (4 consolidated categories) =====
    
    # IT-CLOUD: Cloud and Infrastructure Management
    # Merges CLOUD (391), INFRA (58), DC (17), NETADMIN (41), SYSADMIN (6)
    # Total: ~513 samples - cloud services, data centers, infrastructure admin
    'IT-CLOUD': ['CLOUD', 'INFRA', 'DC', 'NETADMIN', 'SYSADMIN'],
    
    # IT-DEVOPS: Development Operations and Automation
    # Merges DEVOPS (514), AUTO (75), WEBDEV (18), APPDEV (2), SERVICE (4), MONITOR (38)
    # Total: ~651 samples - CI/CD, automation, development operations
    'IT-DEVOPS': ['DEVOPS', 'AUTO', 'WEBDEV', 'APPDEV', 'SERVICE', 'MONITOR'],
    
    # IT-EMERGING: Emerging Tech and Support Services
    # Merges EMERGING (131), TELECOM (62), WORKPLACE (1), TRAINING (25), DOCS (2), STANDARDS (9)
    # Total: ~230 samples - new technologies, training, standards
    'IT-EMERGING': ['EMERGING', 'TELECOM', 'WORKPLACE', 'TRAINING', 'DOCS', 'STANDARDS'],
    
    # IT-RISK: Risk, Security Operations, and Compliance
    # Merges RISK (84), Risk Management (1), FORENSICS (43), COMPLIANCE (28), DRBC (23), ASSET (1)
    # Total: ~180 samples - IT risk management, forensics, compliance
    'IT-RISK': ['RISK', 'Risk Management', 'FORENSICS', 'COMPLIANCE', 'DRBC', 'ASSET'],
}

# Create reverse mapping for easy lookup
def create_reverse_mapping():
    """Create reverse mapping from original subfield to consolidated subfield"""
    reverse_map = {}
    for consolidated, originals in SUBFIELD_CONSOLIDATION.items():
        for original in originals:
            reverse_map[original.upper()] = consolidated
    return reverse_map

def analyze_unmapped_subfields(df, reverse_map):
    """Find any subfields that don't have a mapping"""
    all_subfields = df['Subfield'].unique()
    unmapped = []
    
    for subfield in all_subfields:
        if subfield.upper() not in reverse_map:
            count = len(df[df['Subfield'] == subfield])
            unmapped.append((subfield, count))
    
    return sorted(unmapped, key=lambda x: x[1], reverse=True)

def consolidate_dataset(input_file, output_file):
    """Main consolidation function"""
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} papers")
    
    # Create reverse mapping
    reverse_map = create_reverse_mapping()
    
    # Check for unmapped subfields
    unmapped = analyze_unmapped_subfields(df, reverse_map)
    if unmapped:
        print("\n⚠️  Warning: Found unmapped subfields:")
        for subfield, count in unmapped:
            print(f"  {subfield}: {count} papers")
        print("\nThese will be mapped to default categories based on discipline.")
    
    # Apply consolidation
    def consolidate_subfield(row):
        discipline = row['Discipline']
        subfield = row['Subfield'].upper()
        
        # Try to find in mapping
        if subfield in reverse_map:
            return reverse_map[subfield]
        
        # Default mappings for unmapped subfields based on discipline characteristics
        if discipline == 'CS':
            # Default unmapped CS papers to Software Engineering (most general CS category)
            return 'CS-SE'
        elif discipline == 'IS':
            # Default unmapped IS papers to Knowledge Management (covers IS research/management)
            return 'IS-KM'
        elif discipline == 'IT':
            # Default unmapped IT papers to Emerging Tech (covers new/uncategorized IT)
            return 'IT-EMERGING'
        else:
            return 'UNKNOWN'
    
    # Apply consolidation
    df['Consolidated_Subfield'] = df.apply(consolidate_subfield, axis=1)
    
    # Extract clean subfield name (remove discipline prefix)
    df['Subfield_Clean'] = df['Consolidated_Subfield'].str.replace(r'^(CS|IS|IT)-', '', regex=True)
    
    # Print consolidation summary
    print("\n📊 Consolidation Summary:")
    print("="*50)
    
    # Overall distribution
    consolidated_counts = df['Consolidated_Subfield'].value_counts()
    print("\nConsolidated Subfield Distribution:")
    for subfield, count in consolidated_counts.items():
        print(f"  {subfield}: {count:,} papers")
    
    # By discipline with detailed analysis
    for discipline in ['CS', 'IS', 'IT']:
        disc_df = df[df['Discipline'] == discipline]
        print(f"\n{discipline} Subfields ({len(disc_df):,} total):")
        disc_counts = disc_df['Subfield_Clean'].value_counts()
        for subfield, count in disc_counts.items():
            pct = (count / len(disc_df)) * 100
            # Mark class balance status
            if count < 200:
                status = "⚠️ SEVERE"  # Needs heavy augmentation
            elif count < 500:
                status = "⚡ LOW"     # Needs moderate augmentation
            else:
                status = "✅"         # Well balanced
            print(f"  {status} {subfield}: {count:,} papers ({pct:.1f}%)")
    
    # Identify severely imbalanced classes
    print("\n⚠️  Classes with <200 samples (need heavy augmentation):")
    severe_imbalance = []
    for subfield, count in consolidated_counts.items():
        if count < 200:
            severe_imbalance.append(subfield)
            print(f"  {subfield}: {count} papers")
    
    print("\n⚡ Classes with 200-500 samples (need moderate augmentation):")
    moderate_imbalance = []
    for subfield, count in consolidated_counts.items():
        if 200 <= count < 500:
            moderate_imbalance.append(subfield)
            print(f"  {subfield}: {count} papers")
    
    # Print consolidation logic summary
    print("\n📋 Consolidation Logic Applied:")
    print("="*50)
    print("CS Consolidations:")
    print("  - AI/ML: AI/ML + Data Science + Robotics (ML-based research)")
    print("  - SEC: Security (kept separate due to distinct domain)")
    print("  - CV: Computer Vision + Graphics (visual computing)")
    print("  - NLP: NLP + Information Retrieval (text processing)")
    print("  - SE: Software Engineering + all other CS (systems/theory)")
    print("\nIS Consolidations:")
    print("  - HIS: Health Information Systems (healthcare IT)")
    print("  - DT: Digital Transformation + Innovation + EdTech + Social (digital business)")
    print("  - GOV: Government IS (kept separate - public sector)")
    print("  - BPM: Business Process + ERP/CRM/SCM (enterprise systems)")
    print("  - KM: Knowledge Management + BI/DSS + Research (organizational knowledge)")
    print("\nIT Consolidations:")
    print("  - CLOUD: Cloud + Infrastructure + Data Centers (infrastructure management)")
    print("  - DEVOPS: DevOps + Automation + Development (operations)")
    print("  - EMERGING: Emerging Tech + Telecom + Training (new technologies)")
    print("  - RISK: Risk Management + Forensics + Compliance (security operations)")
    
    # Save consolidated dataset
    print(f"\nSaving consolidated dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Create separate files for each discipline
    output_dir = os.path.dirname(output_file) or '.'
    
    for discipline in ['CS', 'IS', 'IT']:
        disc_df = df[df['Discipline'] == discipline].copy()
        disc_file = os.path.join(output_dir, f'{discipline.lower()}_subfields_consolidated.csv')
        disc_df.to_csv(disc_file, index=False)
        print(f"Saved {discipline} dataset: {disc_file} ({len(disc_df):,} papers)")
    
    # Save consolidation report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_papers': len(df),
        'original_subfields': 74,
        'consolidated_subfields': 14,
        'distribution': consolidated_counts.to_dict(),
        'unmapped_subfields': unmapped,
        'class_balance': {
            'severe_imbalance': severe_imbalance,
            'moderate_imbalance': moderate_imbalance,
            'well_balanced': [s for s, c in consolidated_counts.items() 
                            if c >= 500 and s not in severe_imbalance and s not in moderate_imbalance]
        },
        'consolidation_logic': {
            'CS-AI/ML': 'AI/ML research, data science, robotics',
            'CS-SEC': 'Security, cryptography, cybersecurity',
            'CS-CV': 'Computer vision, graphics, visualization',
            'CS-NLP': 'Natural language processing, information retrieval',
            'CS-SE': 'Software engineering, systems, theory, other CS',
            'IS-HIS': 'Health information systems, medical informatics',
            'IS-DT': 'Digital transformation, innovation, e-commerce',
            'IS-GOV': 'Government information systems',
            'IS-BPM': 'Business processes, ERP, CRM, SCM',
            'IS-KM': 'Knowledge management, BI, decision support',
            'IT-CLOUD': 'Cloud services, infrastructure, data centers',
            'IT-DEVOPS': 'DevOps, automation, development operations',
            'IT-EMERGING': 'Emerging technologies, training, standards',
            'IT-RISK': 'Risk management, forensics, compliance'
        }
    }
    
    report_file = os.path.join(output_dir, 'consolidation_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved consolidation report: {report_file}")
    
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Consolidate GPT-4o subfields to 14 categories")
    parser.add_argument('--input', default='final_classified_papers.csv', 
                       help='Input CSV with 74 subfields')
    parser.add_argument('--output', default='consolidated_papers.csv',
                       help='Output CSV with 14 subfields')
    
    args = parser.parse_args()
    
    # Perform consolidation
    df = consolidate_dataset(args.input, args.output)
    
    print("\n✅ Consolidation complete!")
    print("\nNext steps:")
    print("1. Review the consolidation report")
    print("2. Implement data augmentation for imbalanced classes")
    print("3. Apply v6.0 architecture to train subfield classifiers")

if __name__ == "__main__":
    main()