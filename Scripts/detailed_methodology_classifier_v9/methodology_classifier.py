#!/usr/bin/env python3
"""
Research Methodology Classifier using GPT-4o-mini
Processes CSV files and classifies research methodologies based on abstracts.
"""

import csv
import json
import time
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from typing import Dict, List, Tuple
import openai
from collections import Counter

# Configuration
API_KEYS = [
    "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",
    "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",
    "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"
]

# Rate limiting configuration
MAX_RPM_PER_KEY = 500  # Tier 2 limit
MAX_CONCURRENT = 50    # Maximum concurrent requests
BATCH_SIZE = 50       # Process papers in batches
CHECKPOINT_INTERVAL = 100  # Save progress every N papers

# Methodology definitions
METHODOLOGY_PROMPT = """
Classify the research methodology of this paper based on its abstract. 

METHODOLOGY CATEGORIES (Choose the MOST appropriate):
1. EXPT (Experimental Research Study): Controlled experiments with hypotheses, treatments, statistical analysis. Keywords: experiment, hypothesis, control group, treatment, statistical significance, p-value, experimental setup, participants/subjects.
2. PERF (Performance Benchmarking Evaluation): Evaluating performance, comparing systems/algorithms. Keywords: benchmark, performance evaluation, comparison, metrics, throughput, latency, scalability, efficiency, dataset evaluation.
3. SYST (System Tool Development): Building complete systems, tools, frameworks, platforms. Keywords: "we implement", "system architecture", "tool", "framework", "platform", "prototype system", "application".
4. ALGO (Algorithm Method Development): Proposing new algorithms, techniques, methods. Keywords: "proposed algorithm", "novel method", "new technique", "approach", "we propose", mathematical formulation.
5. CASE (Case Study Analysis): In-depth analysis of specific real-world implementations. Keywords: case study, organization, deployment, field study, lessons learned, real-world implementation, industry.
6. REVW (Literature Review Survey): Systematic reviews or surveys of existing research. Keywords: systematic review, literature review, survey, meta-analysis, "we review", state-of-the-art, research trends.
7. ANAL (Analytical Theoretical Study): Mathematical proofs, formal analysis, theoretical models. Keywords: theorem, proof, formal analysis, mathematical model, complexity analysis, theoretical framework.

Abstract: {abstract}

IMPORTANT CLASSIFICATION RULES:
- EXPT: Must have controlled experiments with clear methodology
- PERF: Focus on measuring and comparing performance
- SYST vs ALGO: SYST=complete implementation, ALGO=core technique
- CASE: Real-world deployment or organizational study
- REVW: Primary purpose is reviewing other papers
- When paper does multiple things, choose the PRIMARY contribution
- Use the 4-letter code (EXPT, PERF, SYST, ALGO, CASE, REVW, ANAL) in your response

Respond with JSON only:
{{"methodology": "EXPT|PERF|SYST|ALGO|CASE|REVW|ANAL", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

If abstract is too short (<50 words) or unclear, use methodology: "UNKNOWN"
If no methodology is indicated, use methodology: "NO_MATCH"
"""

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('methodology_classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIKeyRotator:
    """Manages API key rotation and rate limiting"""
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_index = 0
        self.lock = threading.Lock()
        self.request_counts = {key: 0 for key in api_keys}
        self.request_times = {key: [] for key in api_keys}
        
    def get_next_key(self) -> str:
        """Get the next available API key respecting rate limits"""
        with self.lock:
            # Clean old request times (older than 1 minute)
            current_time = time.time()
            for key in self.api_keys:
                self.request_times[key] = [
                    t for t in self.request_times[key] 
                    if current_time - t < 60
                ]
            
            # Find key with lowest recent usage
            min_requests = float('inf')
            best_key = None
            
            for key in self.api_keys:
                recent_requests = len(self.request_times[key])
                if recent_requests < MAX_RPM_PER_KEY and recent_requests < min_requests:
                    min_requests = recent_requests
                    best_key = key
            
            if best_key:
                self.request_times[best_key].append(current_time)
                return best_key
            
            # All keys are at limit, wait a bit
            time.sleep(1)
            return self.get_next_key()

class ProgressTracker:
    """Tracks processing progress and statistics"""
    def __init__(self):
        self.processed = 0
        self.failed = 0
        self.start_time = time.time()
        self.methodology_counts = Counter()
        self.lock = threading.Lock()
        
    def update(self, success: bool, methodology: str = None):
        """Update progress statistics"""
        with self.lock:
            if success:
                self.processed += 1
                if methodology:
                    self.methodology_counts[methodology] += 1
            else:
                self.failed += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        with self.lock:
            elapsed = time.time() - self.start_time
            rate = self.processed / elapsed if elapsed > 0 else 0
            
            return {
                'processed': self.processed,
                'failed': self.failed,
                'elapsed_time': elapsed,
                'processing_rate': rate,
                'methodology_counts': dict(self.methodology_counts)
            }
    
    def print_progress(self, total: int):
        """Print progress bar"""
        stats = self.get_stats()
        processed = stats['processed']
        failed = stats['failed']
        rate = stats['processing_rate']
        
        progress = processed / total if total > 0 else 0
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '█' * filled + '-' * (bar_length - filled)
        
        eta = (total - processed) / rate if rate > 0 else 0
        eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
        
        print(f"\rProgress: [{bar}] {progress*100:.1f}% | "
              f"Processed: {processed}/{total} | Failed: {failed} | "
              f"Rate: {rate:.1f}/s | ETA: {eta_str}", end='', flush=True)

class CheckpointManager:
    """Manages saving and loading progress checkpoints"""
    def __init__(self, checkpoint_file: str = 'checkpoint.json'):
        self.checkpoint_file = checkpoint_file
        self.lock = threading.Lock()
        
    def save_checkpoint(self, processed_rows: List[Dict], last_index: int):
        """Save progress to checkpoint file"""
        with self.lock:
            temp_file = f"{self.checkpoint_file}.tmp"
            checkpoint_data = {
                'processed': processed_rows,
                'last_index': last_index,
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
                # Atomic rename
                os.replace(temp_file, self.checkpoint_file)
                logger.info(f"Checkpoint saved: {last_index} papers processed")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def load_checkpoint(self) -> Tuple[List[Dict], int]:
        """Load checkpoint if exists"""
        if not os.path.exists(self.checkpoint_file):
            return [], 0
            
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Checkpoint loaded: {data['last_index']} papers already processed")
            return data['processed'], data['last_index']
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return [], 0

def classify_abstract(abstract: str, api_key: str, retry_count: int = 3) -> Dict:
    """Classify a single abstract using GPT-4o-mini"""
    if not abstract or len(abstract.split()) < 50:
        return {
            'methodology': 'UNKNOWN',
            'confidence': 0.0,
            'reasoning': 'Abstract too short or missing'
        }
    
    client = openai.OpenAI(api_key=api_key)
    
    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a research methodology classifier. Respond only with valid JSON."},
                    {"role": "user", "content": METHODOLOGY_PROMPT.format(abstract=abstract)}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate response
            if 'methodology' not in result:
                result['methodology'] = 'UNKNOWN'
            if 'confidence' not in result:
                result['confidence'] = 0.0
            if 'reasoning' not in result:
                result['reasoning'] = 'No reasoning provided'
                
            return result
            
        except openai.RateLimitError:
            wait_time = 2 ** attempt
            logger.warning(f"Rate limit hit, waiting {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Classification error (attempt {attempt + 1}): {e}")
            if attempt == retry_count - 1:
                return {
                    'methodology': 'ERROR',
                    'confidence': 0.0,
                    'reasoning': str(e)
                }
            time.sleep(2 ** attempt)

def process_paper(row: Dict, index: int, api_key_rotator: APIKeyRotator) -> Tuple[int, Dict]:
    """Process a single paper"""
    api_key = api_key_rotator.get_next_key()
    
    result = classify_abstract(row.get('abstract', ''), api_key)
    
    # Update row with classification
    row['new_methodology'] = result['methodology']
    
    # Log low confidence classifications
    if result['confidence'] < 0.7 and result['methodology'] not in ['UNKNOWN', 'ERROR', 'NO_MATCH']:
        logger.warning(f"Low confidence classification for row {index}: "
                      f"{result['methodology']} ({result['confidence']:.2f}) - {result['reasoning']}")
    
    return index, row

def read_csv_with_encoding(file_path: str) -> Tuple[List[Dict], str]:
    """Read CSV file trying different encodings"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                logger.info(f"Successfully read CSV with {encoding} encoding")
                return rows, encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading CSV with {encoding}: {e}")
    
    raise ValueError(f"Could not read CSV file with any of the encodings: {encodings}")

def write_csv_with_encoding(file_path: str, rows: List[Dict], fieldnames: List[str], encoding: str):
    """Write CSV file with specified encoding"""
    with open(file_path, 'w', encoding=encoding, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    """Main processing function"""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python methodology_classifier.py <input_csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.replace('.csv', '_classified.csv')
    
    # Initialize components
    api_key_rotator = APIKeyRotator(API_KEYS)
    progress_tracker = ProgressTracker()
    checkpoint_manager = CheckpointManager()
    
    # Read CSV file
    logger.info(f"Reading CSV file: {input_file}")
    try:
        rows, encoding = read_csv_with_encoding(input_file)
        total_rows = len(rows)
        logger.info(f"Found {total_rows} rows to process")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)
    
    # Load checkpoint if exists
    processed_rows, start_index = checkpoint_manager.load_checkpoint()
    
    # If we have processed rows, update them in our rows list
    if processed_rows:
        for proc_row in processed_rows:
            # Find matching row by title (assuming title is unique)
            for i, row in enumerate(rows):
                if row.get('title') == proc_row.get('title'):
                    rows[i] = proc_row
                    break
    
    # Prepare rows to process
    rows_to_process = [(i, row) for i, row in enumerate(rows) if i >= start_index]
    
    # Process in batches with parallel execution
    semaphore = threading.Semaphore(MAX_CONCURRENT)
    processed_count = start_index
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        # Submit tasks in batches
        for batch_start in range(0, len(rows_to_process), BATCH_SIZE):
            batch = rows_to_process[batch_start:batch_start + BATCH_SIZE]
            
            futures = {}
            for index, row in batch:
                with semaphore:
                    future = executor.submit(process_paper, row, index, api_key_rotator)
                    futures[future] = index
            
            # Process completed futures
            for future in as_completed(futures):
                try:
                    index, processed_row = future.result()
                    rows[index] = processed_row
                    
                    progress_tracker.update(True, processed_row['new_methodology'])
                    processed_count += 1
                    
                    # Save checkpoint periodically
                    if processed_count % CHECKPOINT_INTERVAL == 0:
                        checkpoint_manager.save_checkpoint(
                            [row for row in rows if row.get('new_methodology')],
                            processed_count
                        )
                    
                    # Update progress bar
                    progress_tracker.print_progress(total_rows)
                    
                except Exception as e:
                    logger.error(f"Failed to process row {futures[future]}: {e}")
                    progress_tracker.update(False)
                finally:
                    semaphore.release()
    
    print("\n")  # New line after progress bar
    
    # Final checkpoint save
    checkpoint_manager.save_checkpoint(
        [row for row in rows if row.get('new_methodology')],
        processed_count
    )
    
    # Write output CSV
    logger.info(f"Writing results to: {output_file}")
    fieldnames = list(rows[0].keys())
    if 'new_methodology' not in fieldnames:
        fieldnames.append('new_methodology')
    
    write_csv_with_encoding(output_file, rows, fieldnames, encoding)
    
    # Generate statistics report
    stats = progress_tracker.get_stats()
    
    print("\n=== Classification Statistics ===")
    print(f"Total processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Processing time: {stats['elapsed_time']:.2f} seconds")
    print(f"Average rate: {stats['processing_rate']:.2f} papers/second")
    print("\nMethodology distribution:")
    
    for methodology, count in sorted(stats['methodology_counts'].items(), 
                                   key=lambda x: x[1], reverse=True):
        percentage = (count / stats['processed'] * 100) if stats['processed'] > 0 else 0
        print(f"  {methodology}: {count} ({percentage:.1f}%)")
    
    # Write statistics to file
    with open('classification_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Classification complete!")

if __name__ == "__main__":
    main()