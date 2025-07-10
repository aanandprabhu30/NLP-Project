#!/usr/bin/env python3
"""
Extract error papers from GPT-4o classification results.
This script identifies papers that failed classification and prepares them for re-processing.
"""

import pandas as pd
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

def load_classification_results(file_path: str) -> pd.DataFrame:
    """Load classification results from CSV or JSON file"""
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        return pd.read_csv(file_path)

def identify_error_papers(df: pd.DataFrame) -> pd.DataFrame:
    """Identify papers with ERROR discipline or failed classification"""
    # Multiple ways to identify errors
    error_conditions = (
        (df['Discipline'] == 'ERROR') |
        (df['Discipline'].isna()) |
        (df['Discipline'] == 'NONE') |
        (df['Subfield'].str.contains('ERROR', na=False)) |
        (df['Subfield'].str.contains('Max retries exceeded', na=False)) |
        (df['Subfield'].str.contains('rate limit', na=False, case=False)) |
        (df['Subfield'].str.contains('PARSE_ERROR', na=False))
    )
    
    return df[error_conditions].copy()

def analyze_errors(error_df: pd.DataFrame) -> Dict:
    """Analyze error types and patterns"""
    analysis = {
        'total_errors': len(error_df),
        'error_types': {},
        'missing_data': {
            'no_title': 0,
            'no_abstract': 0,
            'no_content': 0
        }
    }
    
    # Count error types
    if 'Subfield' in error_df.columns:
        error_types = error_df['Subfield'].value_counts()
        analysis['error_types'] = error_types.to_dict()
    
    # Check for missing data
    if 'Title' in error_df.columns:
        analysis['missing_data']['no_title'] = error_df['Title'].isna().sum()
    if 'Abstract' in error_df.columns:
        analysis['missing_data']['no_abstract'] = error_df['Abstract'].isna().sum()
        analysis['missing_data']['no_content'] = (
            (error_df['Title'].isna() | (error_df['Title'] == '')) & 
            (error_df['Abstract'].isna() | (error_df['Abstract'] == ''))
        ).sum()
    
    return analysis

def create_error_report(error_df: pd.DataFrame, analysis: Dict, output_dir: str):
    """Create detailed error report"""
    report_path = os.path.join(output_dir, 'error_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("GPT-4O CLASSIFICATION ERROR REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write(f"Total Error Papers: {analysis['total_errors']}\n\n")
        
        f.write("Error Type Distribution:\n")
        f.write("-"*30 + "\n")
        for error_type, count in analysis['error_types'].items():
            f.write(f"  {error_type}: {count}\n")
        
        f.write("\nMissing Data Analysis:\n")
        f.write("-"*30 + "\n")
        f.write(f"  Papers without title: {analysis['missing_data']['no_title']}\n")
        f.write(f"  Papers without abstract: {analysis['missing_data']['no_abstract']}\n")
        f.write(f"  Papers without any content: {analysis['missing_data']['no_content']}\n")
        
        # Sample errors
        f.write("\nSample Error Papers:\n")
        f.write("-"*30 + "\n")
        for idx, row in error_df.head(5).iterrows():
            f.write(f"\nPaper {idx}:\n")
            f.write(f"  Title: {row.get('Title', 'N/A')[:100]}...\n")
            f.write(f"  Abstract: {row.get('Abstract', 'N/A')[:200]}...\n")
            f.write(f"  Error: {row.get('Subfield', 'Unknown')}\n")
    
    print(f"Error report saved to: {report_path}")

def save_error_papers(error_df: pd.DataFrame, original_df: pd.DataFrame, 
                     output_dir: str) -> Tuple[str, str]:
    """Save error papers in format ready for re-classification"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save error papers with original indices
    error_papers_path = os.path.join(output_dir, 'error_papers.csv')
    error_df.to_csv(error_papers_path, index=True, index_label='original_index')
    
    # Save in the same format as the original input (for the classifier script)
    reprocess_path = os.path.join(output_dir, 'papers_to_reprocess.csv')
    
    # Extract just Title and Abstract columns for re-processing
    reprocess_df = error_df[['Title', 'Abstract']].copy()
    
    # Clean up any potential issues
    reprocess_df['Title'] = reprocess_df['Title'].fillna('')
    reprocess_df['Abstract'] = reprocess_df['Abstract'].fillna('')
    
    # Remove truly empty papers
    reprocess_df = reprocess_df[
        (reprocess_df['Title'] != '') | (reprocess_df['Abstract'] != '')
    ]
    
    reprocess_df.to_csv(reprocess_path, index=False)
    
    print(f"Error papers saved to: {error_papers_path}")
    print(f"Papers ready for reprocessing: {reprocess_path}")
    print(f"Total papers to reprocess: {len(reprocess_df)}")
    
    return error_papers_path, reprocess_path

def merge_reprocessed_results(original_results_path: str, 
                            reprocessed_results_path: str,
                            error_indices_path: str,
                            output_path: str):
    """Merge reprocessed results back into the original dataset"""
    # Load all data
    original_df = load_classification_results(original_results_path)
    reprocessed_df = load_classification_results(reprocessed_results_path)
    error_indices = pd.read_csv(error_indices_path)['original_index'].tolist()
    
    # Create a copy of original results
    merged_df = original_df.copy()
    
    # Update the error rows with reprocessed results
    for i, idx in enumerate(error_indices):
        if i < len(reprocessed_df):
            merged_df.loc[idx] = reprocessed_df.iloc[i]
    
    # Save merged results
    if output_path.endswith('.json'):
        merged_df.to_json(output_path, orient='records', indent=2)
    else:
        merged_df.to_csv(output_path, index=False)
    
    print(f"Merged results saved to: {output_path}")
    
    # Calculate improvement statistics
    original_errors = (original_df['Discipline'] == 'ERROR').sum()
    remaining_errors = (merged_df['Discipline'] == 'ERROR').sum()
    print(f"\nImprovement Statistics:")
    print(f"  Original errors: {original_errors}")
    print(f"  Remaining errors: {remaining_errors}")
    print(f"  Resolved: {original_errors - remaining_errors}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract and analyze error papers from GPT-4o classification")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract error papers')
    extract_parser.add_argument('--input', required=True, help='Input classification results (CSV or JSON)')
    extract_parser.add_argument('--output-dir', default='error_papers', help='Output directory')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge reprocessed results')
    merge_parser.add_argument('--original', required=True, help='Original classification results')
    merge_parser.add_argument('--reprocessed', required=True, help='Reprocessed results')
    merge_parser.add_argument('--indices', required=True, help='Error indices file')
    merge_parser.add_argument('--output', required=True, help='Output merged file')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        # Load classification results
        print(f"Loading classification results from: {args.input}")
        df = load_classification_results(args.input)
        print(f"Total papers loaded: {len(df)}")
        
        # Identify error papers
        error_df = identify_error_papers(df)
        print(f"Error papers found: {len(error_df)}")
        
        if len(error_df) == 0:
            print("No error papers found!")
            return
        
        # Analyze errors
        analysis = analyze_errors(error_df)
        
        # Create error report
        create_error_report(error_df, analysis, args.output_dir)
        
        # Save error papers
        error_path, reprocess_path = save_error_papers(error_df, df, args.output_dir)
        
        print("\nNext steps:")
        print(f"1. Review the error report: {os.path.join(args.output_dir, 'error_analysis_report.txt')}")
        print(f"2. Re-run classification on: {reprocess_path}")
        print("3. After reprocessing, use the 'merge' command to combine results")
        
    elif args.command == 'merge':
        merge_reprocessed_results(args.original, args.reprocessed, args.indices, args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()