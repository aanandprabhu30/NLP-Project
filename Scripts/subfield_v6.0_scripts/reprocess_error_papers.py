#!/usr/bin/env python3
"""
Reprocess error papers with enhanced error handling and multiple AI providers.
This script is optimized for handling difficult classification cases.
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the classifier components from the main script
# Assuming classifier_multi_llm_improved.py is in the same directory
try:
    from classifier_multi_llm_improved import (
        OpenAIClassifier, RateLimiter, RateLimitConfig, 
        CostTracker, SUBFIELD_DESCRIPTIONS, VALID_SUBFIELDS
    )
except ImportError:
    logger.error("Please ensure classifier_multi_llm_improved.py is in the same directory")
    raise

def create_enhanced_prompt() -> str:
    """Create an enhanced prompt specifically for error cases"""
    # Create formatted subfield descriptions
    cs_subfields = '\n'.join([f"  - {code}: {desc}" for code, desc in SUBFIELD_DESCRIPTIONS['CS'].items()])
    is_subfields = '\n'.join([f"  - {code}: {desc}" for code, desc in SUBFIELD_DESCRIPTIONS['IS'].items()])
    it_subfields = '\n'.join([f"  - {code}: {desc}" for code, desc in SUBFIELD_DESCRIPTIONS['IT'].items()])
    
    return f"""You are an expert classifier for computing research papers. This paper previously failed classification, so analyze it extra carefully.

Title: {{title}}
Abstract: {{abstract}}

IMPORTANT INSTRUCTIONS:
1. If the title or abstract is very short or unclear, make your best educated guess based on any available keywords
2. Look for ANY technical terms, methodologies, or domain indicators
3. Default to CS if truly ambiguous between disciplines
4. Always provide a valid subfield code from the lists below

DISCIPLINES (choose one):
- CS: Computer Science (algorithms, AI/ML, software development, theoretical computing)
- IS: Information Systems (business IT, enterprise systems, organizational technology)
- IT: Information Technology (infrastructure, operations, practical implementation)

CS SUBFIELDS:
{cs_subfields}

IS SUBFIELDS:
{is_subfields}

IT SUBFIELDS:
{it_subfields}

OUTPUT FORMAT (exactly):
DISCIPLINE|SUBFIELD_CODE|DISC_CONFIDENCE|SUB_CONFIDENCE

Example: CS|AI/ML|85|75

Note: Use lower confidence scores (50-70) if uncertain, but always make a classification."""

def create_fallback_classifier(paper: Dict) -> Dict:
    """Simple rule-based fallback classifier for extreme cases"""
    title = str(paper.get('Title', '')).lower()
    abstract = str(paper.get('Abstract', '')).lower()
    text = f"{title} {abstract}"
    
    # Discipline keywords
    cs_keywords = ['algorithm', 'neural', 'machine learning', 'artificial intelligence', 
                   'computer vision', 'nlp', 'software', 'programming', 'code', 'data structure']
    is_keywords = ['business', 'enterprise', 'management', 'organization', 'erp', 'crm',
                   'information system', 'decision support', 'e-commerce', 'digital transformation']
    it_keywords = ['infrastructure', 'network admin', 'server', 'cloud deploy', 'devops',
                   'it support', 'help desk', 'system admin', 'backup', 'security operations']
    
    # Count keyword matches
    cs_score = sum(1 for kw in cs_keywords if kw in text)
    is_score = sum(1 for kw in is_keywords if kw in text)
    it_score = sum(1 for kw in it_keywords if kw in text)
    
    # Determine discipline
    if cs_score >= is_score and cs_score >= it_score:
        discipline = 'CS'
        subfield = 'SE'  # Default to Software Engineering
    elif is_score >= it_score:
        discipline = 'IS'
        subfield = 'ISM'  # Default to IS Management
    else:
        discipline = 'IT'
        subfield = 'INFRA'  # Default to Infrastructure
    
    # Look for specific subfield indicators
    if discipline == 'CS':
        if any(kw in text for kw in ['neural', 'deep learning', 'machine learning']):
            subfield = 'AI/ML'
        elif any(kw in text for kw in ['image', 'vision', 'visual']):
            subfield = 'CV'
        elif any(kw in text for kw in ['security', 'crypto', 'vulnerab']):
            subfield = 'SEC'
    
    return {
        'Title': paper.get('Title', ''),
        'Abstract': paper.get('Abstract', ''),
        'Discipline': discipline,
        'Subfield': subfield,
        'Discipline_Confidence': 60,
        'Subfield_Confidence': 50,
        'Classifier': 'fallback'
    }

class EnhancedErrorProcessor:
    """Process error papers with multiple strategies"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        # Setup with more conservative limits for error cases
        self.rate_config = RateLimitConfig(
            requests_per_minute=200,  # Even more conservative
            tokens_per_minute=100000,
            requests_per_day=5000
        )
        self.rate_limiter = RateLimiter(self.rate_config, "openai-errors")
        self.cost_tracker = CostTracker()
        
        # Setup classifier
        config = {"providers": {"openai": {"api_key": self.api_key, "model": "gpt-4o-mini"}}}
        self.classifier = OpenAIClassifier(self.rate_limiter, self.cost_tracker, config)
        
    def process_with_strategies(self, paper: Dict, idx: int) -> Dict:
        """Process paper with multiple strategies"""
        strategies = [
            ("enhanced_prompt", self._try_enhanced_prompt),
            ("simplified_prompt", self._try_simplified_prompt),
            ("fallback_rules", self._try_fallback)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Trying {strategy_name} for paper {idx}")
                result = strategy_func(paper, idx)
                
                if result['Discipline'] != 'ERROR':
                    result['Strategy'] = strategy_name
                    return result
                    
            except Exception as e:
                logger.warning(f"{strategy_name} failed for paper {idx}: {e}")
                continue
        
        # If all strategies fail, return a fallback classification
        logger.warning(f"All strategies failed for paper {idx}, using rule-based fallback")
        return create_fallback_classifier(paper)
    
    def _try_enhanced_prompt(self, paper: Dict, idx: int) -> Dict:
        """Try with enhanced prompt"""
        prompt = create_enhanced_prompt()
        return self.classifier.classify_paper(paper, idx, prompt)
    
    def _try_simplified_prompt(self, paper: Dict, idx: int) -> Dict:
        """Try with simplified prompt"""
        simple_prompt = """Classify this paper:
Title: {title}
Abstract: {abstract}

Choose discipline: CS (computer science), IS (information systems), or IT (information technology)
Choose appropriate subfield code.

Output: DISCIPLINE|SUBFIELD|CONFIDENCE1|CONFIDENCE2

Example: CS|AI/ML|80|75"""
        
        return self.classifier.classify_paper(paper, idx, simple_prompt)
    
    def _try_fallback(self, paper: Dict, idx: int) -> Dict:
        """Use rule-based fallback"""
        return create_fallback_classifier(paper)

def process_error_batch(papers: List[Dict], output_file: str, 
                       batch_size: int = 10, use_fallback: bool = True):
    """Process a batch of error papers with enhanced handling"""
    processor = EnhancedErrorProcessor()
    results = []
    
    print(f"Processing {len(papers)} error papers...")
    print(f"Batch size: {batch_size}")
    print(f"Fallback enabled: {use_fallback}")
    
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]
        batch_results = []
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(papers)-1)//batch_size + 1}")
        
        for j, paper in enumerate(batch):
            idx = i + j
            
            # Skip truly empty papers
            if not paper.get('Title') and not paper.get('Abstract'):
                logger.warning(f"Skipping empty paper at index {idx}")
                continue
            
            try:
                result = processor.process_with_strategies(paper, idx)
                batch_results.append(result)
                
                print(f"  Paper {idx}: {result['Discipline']}|{result['Subfield']} "
                      f"(strategy: {result.get('Strategy', 'unknown')})")
                
            except Exception as e:
                logger.error(f"Failed to process paper {idx}: {e}")
                if use_fallback:
                    result = create_fallback_classifier(paper)
                    result['Strategy'] = 'emergency_fallback'
                    batch_results.append(result)
        
        results.extend(batch_results)
        
        # Save intermediate results
        if i > 0 and i % 50 == 0:
            save_intermediate_results(results, output_file.replace('.csv', '_intermediate.csv'))
            print(f"Saved intermediate results ({len(results)} papers)")
        
        # Brief pause between batches
        if i + batch_size < len(papers):
            time.sleep(2)
    
    # Save final results
    save_results(results, output_file)
    
    # Print summary
    print_processing_summary(results, processor.cost_tracker)
    
    return results

def save_intermediate_results(results: List[Dict], filename: str):
    """Save intermediate results during processing"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys(), extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

def save_results(results: List[Dict], output_file: str):
    """Save final results"""
    if output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = ['Title', 'Abstract', 'Discipline', 'Subfield', 
                             'Discipline_Confidence', 'Subfield_Confidence', 
                             'Classifier', 'Strategy']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")

def print_processing_summary(results: List[Dict], cost_tracker: CostTracker):
    """Print processing summary"""
    print("\n" + "="*60)
    print("REPROCESSING SUMMARY")
    print("="*60)
    
    # Count by discipline
    disciplines = {}
    strategies = {}
    errors = 0
    
    for result in results:
        disc = result.get('Discipline', 'UNKNOWN')
        if disc == 'ERROR':
            errors += 1
        else:
            disciplines[disc] = disciplines.get(disc, 0) + 1
        
        strategy = result.get('Strategy', 'unknown')
        strategies[strategy] = strategies.get(strategy, 0) + 1
    
    print("\nClassification Results:")
    for disc, count in sorted(disciplines.items()):
        print(f"  {disc}: {count}")
    print(f"  Errors: {errors}")
    
    print("\nStrategies Used:")
    for strategy, count in sorted(strategies.items()):
        print(f"  {strategy}: {count}")
    
    print(f"\nTotal Cost: ${cost_tracker.get_cost():.4f}")
    print(f"Average Cost per Paper: ${cost_tracker.get_cost()/max(1, len(results)):.4f}")
    print("="*60)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reprocess error papers with enhanced strategies")
    parser.add_argument('--input', required=True, help='Input CSV with error papers')
    parser.add_argument('--output', default='reprocessed_papers.csv', help='Output file')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--no-fallback', action='store_true', help='Disable fallback classification')
    parser.add_argument('--test', type=int, help='Test with first N papers')
    
    args = parser.parse_args()
    
    # Load error papers
    papers = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append(row)
            if args.test and len(papers) >= args.test:
                break
    
    print(f"Loaded {len(papers)} error papers")
    
    # Process papers
    results = process_error_batch(
        papers, 
        args.output, 
        batch_size=args.batch_size,
        use_fallback=not args.no_fallback
    )
    
    print("\nReprocessing complete!")

if __name__ == "__main__":
    main()