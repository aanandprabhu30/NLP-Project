#!/usr/bin/env python3
"""
OpenAI GPT-4o-mini paper classification system with checkpoint support.
Optimized for tier 1 usage with conservative rate limits.
"""

import os
import time
import csv
import json
import logging
import threading
import signal
import sys
import hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass

# Check and install required packages
def install_package(package_name, import_name=None):
    """Install package if not present"""
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package_name}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Install required packages
install_package("openai")
install_package("tqdm")
install_package("python-dotenv", "dotenv")
install_package("tiktoken")

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openai_classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiter configuration
@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int
    tokens_per_minute: Optional[int] = None
    requests_per_day: Optional[int] = None
    
class RateLimiter:
    """Thread-safe rate limiter with exponential backoff"""
    
    def __init__(self, config: RateLimitConfig, name: str):
        self.config = config
        self.name = name
        self.request_times = deque()
        self.token_usage = deque()
        self.daily_requests = 0
        self.last_reset_day = datetime.now().day
        self.lock = threading.Lock()
        self.backoff_until = None
        
    def wait_if_needed(self, estimated_tokens: int = 100) -> float:
        """Wait if rate limit would be exceeded, returns wait time"""
        with self.lock:
            now = datetime.now()
            wait_time = 0.0
            
            # Check if we're in backoff period
            if self.backoff_until and now < self.backoff_until:
                wait_time = (self.backoff_until - now).total_seconds()
                time.sleep(wait_time)
                return wait_time
            
            # Reset daily counters if needed
            if now.day != self.last_reset_day:
                self.daily_requests = 0
                self.last_reset_day = now.day
            
            # Clean old request times
            minute_ago = now - timedelta(minutes=1)
            while self.request_times and self.request_times[0] < minute_ago:
                self.request_times.popleft()
            
            # Check requests per minute
            if len(self.request_times) >= self.config.requests_per_minute:
                wait_time = 60 - (now - self.request_times[0]).total_seconds() + 0.1  # Minimal buffer
                if wait_time > 0:
                    time.sleep(wait_time)
                    return wait_time
            
            # Check tokens per minute if applicable
            if self.config.tokens_per_minute:
                while self.token_usage and self.token_usage[0][0] < minute_ago:
                    self.token_usage.popleft()
                
                current_tokens = sum(tokens for _, tokens in self.token_usage)
                if current_tokens + estimated_tokens > self.config.tokens_per_minute:
                    wait_time = 60 - (now - self.token_usage[0][0]).total_seconds() + 0.1
                    if wait_time > 0:
                        time.sleep(wait_time)
                        return wait_time
            
            # Check daily requests if applicable
            if self.config.requests_per_day and self.daily_requests >= self.config.requests_per_day:
                # Wait until next day
                tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                wait_time = (tomorrow - now).total_seconds()
                logger.warning(f"{self.name}: Daily limit reached, waiting until tomorrow")
                time.sleep(min(wait_time, 300))  # Max 5 minute wait
                return wait_time
            
            # Record the request
            self.request_times.append(now)
            self.daily_requests += 1
            if self.config.tokens_per_minute:
                self.token_usage.append((now, estimated_tokens))
            
            return wait_time
    
    def set_backoff(self, seconds: float):
        """Set exponential backoff period"""
        with self.lock:
            self.backoff_until = datetime.now() + timedelta(seconds=seconds)
            logger.warning(f"{self.name}: Backing off for {seconds:.1f} seconds")

# Cost tracking
@dataclass
class CostTracker:
    """Track costs for OpenAI"""
    provider: str = "openai"
    input_cost_per_1k: float = 0.00015  # GPT-4o-mini pricing
    output_cost_per_1k: float = 0.0006
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    errors: int = 0
    
    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.requests += 1
    
    def get_cost(self) -> float:
        """Calculate total cost"""
        input_cost = (self.input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (self.output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        return {
            "provider": self.provider,
            "requests": self.requests,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": f"${self.get_cost():.4f}",
            "avg_cost_per_request": f"${self.get_cost() / max(1, self.requests):.4f}"
        }

def estimate_openai_tokens(prompt: str, model: str = "gpt-4o-mini") -> int:
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(prompt))
    except Exception:
        return len(prompt.split()) * 2

# Subfield code mappings
VALID_SUBFIELDS = {
    'CS': [
        'AI/ML', 'CV', 'NLP', 'ROB', 'DS', 'DB', 'NET', 'SEC', 'SE', 'HCI', 
        'GFX', 'OS', 'ALGO', 'THEORY', 'BIO', 'ARCH', 'MOBILE', 'CLOUD', 
        'IOT', 'QUANTUM', 'GAME', 'EDU', 'IR', 'HPC', 'EMBED'
    ],
    'IS': [
        'ERP', 'BI', 'ECOMM', 'KM', 'DSS', 'ISM', 'DT', 'GOV', 'BPM', 'SMM', 
        'HIS', 'EDTECH', 'SCM', 'CRM', 'EA', 'ITSM', 'INNOV', 'ISSEC', 'PRIV', 
        'MOBILE', 'SOCIAL', 'PM', 'STRAT', 'RESEARCH', 'ETHICS'
    ],
    'IT': [
        'INFRA', 'NETADMIN', 'SYSADMIN', 'SUPPORT', 'WEBDEV', 'APPDEV', 'DEVOPS', 
        'RISK', 'DC', 'CLOUD', 'ASSET', 'SERVICE', 'FORENSICS', 'COMPLIANCE', 
        'TELECOM', 'TRAINING', 'CONSULT', 'EMERGING', 'MONITOR', 'DRBC', 'VENDOR', 
        'WORKPLACE', 'AUTO', 'DOCS', 'STANDARDS'
    ]
}

# Detailed subfield descriptions
SUBFIELD_DESCRIPTIONS = {
    'CS': {
        'AI/ML': 'Artificial Intelligence and Machine Learning (neural networks, deep learning, reinforcement learning)',
        'CV': 'Computer Vision (image processing, object detection, visual recognition)',
        'NLP': 'Natural Language Processing (text analysis, language models, speech processing)',
        'ROB': 'Robotics (autonomous systems, robot control, sensor integration)',
        'DS': 'Data Science (data analysis, statistical computing, data mining)',
        'DB': 'Databases (data management, query optimization, NoSQL, distributed databases)',
        'NET': 'Networking (protocols, network architecture, wireless networks)',
        'SEC': 'Security (cryptography, cybersecurity, security protocols, vulnerability analysis)',
        'SE': 'Software Engineering (development methodologies, testing, design patterns)',
        'HCI': 'Human-Computer Interaction (user interfaces, usability, interaction design)',
        'GFX': 'Graphics (rendering, visualization, GPU programming, computer graphics)',
        'OS': 'Operating Systems (kernel design, scheduling, memory management)',
        'ALGO': 'Algorithms (algorithm design, complexity analysis, optimization)',
        'THEORY': 'Theoretical Computer Science (computability, formal methods, logic)',
        'BIO': 'Bioinformatics (computational biology, genomics, protein analysis)',
        'ARCH': 'Computer Architecture (processor design, memory systems, hardware)',
        'MOBILE': 'Mobile Computing (mobile apps, mobile systems, mobile networking)',
        'CLOUD': 'Cloud Computing (distributed systems, virtualization, cloud architecture)',
        'IOT': 'Internet of Things (sensor networks, edge computing, IoT protocols)',
        'QUANTUM': 'Quantum Computing (quantum algorithms, quantum hardware, quantum theory)',
        'GAME': 'Game Development (game engines, game AI, graphics programming)',
        'EDU': 'Computer Science Education (CS pedagogy, educational tools, curriculum)',
        'IR': 'Information Retrieval (search engines, indexing, ranking algorithms)',
        'HPC': 'High Performance Computing (parallel computing, supercomputing, GPU computing)',
        'EMBED': 'Embedded Systems (microcontrollers, real-time systems, firmware)'
    },
    'IS': {
        'ERP': 'Enterprise Resource Planning (SAP, Oracle, business integration)',
        'BI': 'Business Intelligence (analytics, dashboards, data warehousing)',
        'ECOMM': 'E-Commerce (online business, digital marketplaces, payment systems)',
        'KM': 'Knowledge Management (organizational knowledge, collaboration systems)',
        'DSS': 'Decision Support Systems (business decision making, expert systems)',
        'ISM': 'Information Systems Management (IS governance, strategy, leadership)',
        'DT': 'Digital Transformation (digitalization, business model innovation)',
        'GOV': 'Government Information Systems (e-government, public sector IT)',
        'BPM': 'Business Process Management (workflow, process automation, optimization)',
        'SMM': 'Social Media Management (social platforms, digital marketing, analytics)',
        'HIS': 'Health Information Systems (EHR, medical informatics, healthcare IT)',
        'EDTECH': 'Educational Technology (e-learning, LMS, educational systems)',
        'SCM': 'Supply Chain Management (logistics, inventory, supply chain systems)',
        'CRM': 'Customer Relationship Management (customer data, sales systems)',
        'EA': 'Enterprise Architecture (IT-business alignment, architecture frameworks)',
        'ITSM': 'IT Service Management (ITIL, service delivery, IT operations)',
        'INNOV': 'Innovation Management (technology adoption, digital innovation)',
        'ISSEC': 'Information Systems Security (access control, security policies)',
        'PRIV': 'Privacy and Data Protection (GDPR, data governance, compliance)',
        'MOBILE': 'Mobile Information Systems (mobile business apps, m-commerce)',
        'SOCIAL': 'Social Information Systems (social networks, collaborative systems)',
        'PM': 'Project Management (IT projects, agile, project methodologies)',
        'STRAT': 'Strategic Information Systems (competitive advantage, IT strategy)',
        'RESEARCH': 'Information Systems Research Methods (design science, case studies)',
        'ETHICS': 'Information Systems Ethics (IT ethics, responsible computing)'
    },
    'IT': {
        'INFRA': 'IT Infrastructure (servers, storage, network infrastructure)',
        'NETADMIN': 'Network Administration (routers, switches, network management)',
        'SYSADMIN': 'System Administration (OS management, user administration)',
        'SUPPORT': 'IT Support and Help Desk (troubleshooting, user assistance)',
        'WEBDEV': 'Web Development (websites, web applications, frontend/backend)',
        'APPDEV': 'Application Development (software development, coding, deployment)',
        'DEVOPS': 'DevOps and CI/CD (automation, continuous integration, deployment)',
        'RISK': 'IT Risk Management (risk assessment, mitigation strategies)',
        'DC': 'Data Center Management (facility management, cooling, power)',
        'CLOUD': 'Cloud Services and Administration (AWS, Azure, cloud deployment)',
        'ASSET': 'IT Asset Management (hardware/software inventory, lifecycle)',
        'SERVICE': 'IT Service Management (service desk, incident management)',
        'FORENSICS': 'Digital Forensics (investigation, evidence collection, analysis)',
        'COMPLIANCE': 'IT Compliance and Governance (regulations, auditing, policies)',
        'TELECOM': 'Telecommunications (phone systems, VoIP, communication infrastructure)',
        'TRAINING': 'IT Training and Education (user training, certification programs)',
        'CONSULT': 'IT Consulting (advisory services, implementation, solutions)',
        'EMERGING': 'Emerging Technologies (new tech evaluation, adoption strategies)',
        'MONITOR': 'IT Monitoring and Performance (system monitoring, metrics, alerts)',
        'DRBC': 'Disaster Recovery and Business Continuity (backup, recovery planning)',
        'VENDOR': 'Vendor Management (procurement, contracts, vendor relationships)',
        'WORKPLACE': 'Workplace Technology (collaboration tools, productivity software)',
        'AUTO': 'IT Automation (scripting, process automation, RPA)',
        'DOCS': 'IT Documentation (technical writing, procedures, knowledge base)',
        'STANDARDS': 'IT Standards and Best Practices (ISO, frameworks, methodologies)'
    }
}

# OpenAI classifier
class OpenAIClassifier:
    def __init__(self, rate_limiter: RateLimiter, cost_tracker: CostTracker, config: Dict = None):
        self.name = "openai"
        self.rate_limiter = rate_limiter
        self.cost_tracker = cost_tracker
        self.config = config or {}
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        # Try to get API key from config first, then environment
        api_key = self.config.get("providers", {}).get("openai", {}).get("api_key")
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in config or OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        
        # Get model from config or use default
        openai_config = self.config.get("providers", {}).get("openai", {})
        self.model = openai_config.get("model", "gpt-4o-mini")
    
    def _make_request(self, prompt: str) -> Tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
            timeout=30.0
        )
        content = response.choices[0].message.content
        input_tokens = estimate_openai_tokens(prompt, self.model)
        output_tokens = response.usage.completion_tokens
        return content.strip(), input_tokens, output_tokens
    
    def classify_paper(self, paper: Dict, idx: int, prompt: str) -> Dict:
        title = str(paper.get('Title', ''))[:500]
        abstract = str(paper.get('Abstract', ''))[:2000]
        
        if not title and not abstract:
            return self._empty_result(title, abstract)
        
        full_prompt = prompt.format(title=title, abstract=abstract)
        
        max_retries = 3
        backoff_base = 2
        
        for retry in range(max_retries):
            try:
                estimated_tokens = estimate_openai_tokens(full_prompt, self.model)
                self.rate_limiter.wait_if_needed(estimated_tokens)
                
                response, input_tokens, output_tokens = self._make_request(full_prompt)
                
                self.cost_tracker.add_usage(input_tokens, output_tokens)
                
                result = self._parse_response(response, title, abstract)
                if result['Discipline'] != 'ERROR' and self._validate_response(result):
                    result['Timestamp'] = datetime.now().isoformat()
                    return result
                elif result['Discipline'] != 'ERROR':
                    logger.debug(f"Validation failed for paper {idx}: {result}")
                
            except Exception as e:
                self.cost_tracker.errors += 1
                error_msg = str(e)
                
                if "rate" in error_msg.lower() or "429" in error_msg:
                    backoff_time = min(backoff_base ** retry * 1.5, 30)  # Faster retry
                    self.rate_limiter.set_backoff(backoff_time)
                    if retry < max_retries - 1:
                        continue
                
                logger.error(f"Error on paper {idx}: {error_msg}")
                
                if retry == max_retries - 1:
                    return self._error_result(title, abstract, error_msg[:50])
        
        return self._error_result(title, abstract, "Max retries exceeded")
    
    def _parse_response(self, content: str, title: str, abstract: str) -> Dict:
        """Parse the LLM response"""
        lines = content.strip().split('\n')
        first_line = lines[0].strip()
        
        parts = first_line.split('|')
        
        if len(parts) >= 4:
            disc = parts[0].strip().upper()
            subfield_code = parts[1].strip()
            
            if ',' in subfield_code:
                subfield_code = subfield_code.split(',')[0].strip()
            
            if disc not in ['CS', 'IS', 'IT']:
                disc = 'CS'  # Default
            
            try:
                disc_conf = int(parts[2].strip())
                sub_conf = int(parts[3].strip())
            except:
                disc_conf = 80
                sub_conf = 75
            
            return {
                'Title': title,
                'Abstract': abstract,
                'Discipline': disc,
                'Subfield': subfield_code,
                'Discipline_Confidence': disc_conf,
                'Subfield_Confidence': sub_conf,
                'Classifier': self.name
            }
        
        return self._error_result(title, abstract, "PARSE_ERROR")
    
    def _empty_result(self, title: str, abstract: str) -> Dict:
        """Return empty result"""
        return {
            'Title': title,
            'Abstract': abstract,
            'Discipline': 'NONE',
            'Subfield': 'NONE',
            'Discipline_Confidence': 0,
            'Subfield_Confidence': 0,
            'Classifier': self.name
        }
    
    def _error_result(self, title: str, abstract: str, error: str) -> Dict:
        """Return error result"""
        return {
            'Title': title,
            'Abstract': abstract,
            'Discipline': 'ERROR',
            'Subfield': error,
            'Discipline_Confidence': 0,
            'Subfield_Confidence': 0,
            'Classifier': self.name
        }

    def _validate_response(self, result: Dict) -> bool:
        valid_disciplines = ['CS', 'IS', 'IT']
        if result['Discipline'] not in valid_disciplines:
            return False
        if not 0 <= result['Discipline_Confidence'] <= 100:
            return False
        if not 0 <= result['Subfield_Confidence'] <= 100:
            return False
        if not self._validate_subfield(result['Discipline'], result['Subfield']):
            return False
        return True

    def _validate_subfield(self, discipline: str, subfield: str) -> bool:
        valid_subfields = VALID_SUBFIELDS.get(discipline, [])
        
        if subfield in valid_subfields:
            return True
            
        subfield_upper = subfield.upper()
        for valid in valid_subfields:
            if valid in subfield_upper or subfield_upper.startswith(valid):
                return True
        
        return False

# Checkpoint manager
class CheckpointManager:
    """Manage checkpoints for classification"""
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.lock = threading.Lock()
    
    def save_checkpoint(self, results: List[Tuple[int, Dict]], processed_count: int):
        """Save checkpoint"""
        with self.lock:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "provider_states": {"openai": processed_count},
                "results": sorted(results, key=lambda x: x[0])
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
    
    def load_checkpoint(self) -> Tuple[List[Tuple[int, Dict]], int]:
        """Load checkpoint data"""
        if not os.path.exists(self.checkpoint_file):
            return [], 0
        
        with self.lock:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                results = [(r[0], r[1]) for r in data.get("results", [])]
                processed = data.get("provider_states", {}).get("openai", 0)
                return results, processed

# Main classifier
class OpenAIOnlyClassifier:
    """OpenAI-only paper classifier with checkpoint support"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.checkpoint_manager = CheckpointManager("multi_llm_checkpoint.json")
        self.stop_requested = False
        self.start_time = None  # Will be set when processing starts
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._setup_classifier()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown gracefully"""
        logger.info("\nShutdown requested. Saving progress...")
        self.stop_requested = True
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Extract OpenAI config only
                return {
                    "rate_limit": config["providers"]["openai"]["rate_limit"],
                    "cost": config["providers"]["openai"]["cost"],
                    "max_workers": config["providers"]["openai"]["max_workers"],
                    "batch_size": config.get("batch_size", 50),
                    "checkpoint_frequency": config.get("checkpoint_frequency", 100)
                }
        
        # Default configuration for tier 1 usage
        return {
            "rate_limit": {
                "requests_per_minute": 30,  # Very conservative to avoid 429s
                "tokens_per_minute": 150000,
                "requests_per_day": 10000   # Daily limit for tier 1
            },
            "cost": {
                "input_per_1k": 0.00015,
                "output_per_1k": 0.0006
            },
            "max_workers": 2,  # Reduced for rate limiting
            "batch_size": 10,
            "checkpoint_frequency": 50
        }
    
    def _setup_classifier(self):
        """Setup OpenAI classifier"""
        try:
            # Setup rate limiter
            rate_config = RateLimitConfig(
                requests_per_minute=self.config["rate_limit"]["requests_per_minute"],
                tokens_per_minute=self.config["rate_limit"].get("tokens_per_minute"),
                requests_per_day=self.config["rate_limit"].get("requests_per_day")
            )
            rate_limiter = RateLimiter(rate_config, "openai")
            
            # Setup cost tracker
            cost_tracker = CostTracker(
                input_cost_per_1k=self.config["cost"]["input_per_1k"],
                output_cost_per_1k=self.config["cost"]["output_per_1k"]
            )
            
            # Create classifier
            self.classifier = OpenAIClassifier(rate_limiter, cost_tracker, self.config)
            self.cost_tracker = cost_tracker
            logger.info("✓ OpenAI classifier initialized")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize OpenAI: {e}")
            raise
    
    def process_papers(self, papers: List[Dict], output_file: str, prompt: str, resume: bool = True, detect_duplicates: bool = False):
        """Process papers with OpenAI"""
        # Initialize start time for progress tracking
        self.start_time = time.time()
        
        # Load checkpoint if resuming
        existing_results = []
        start_index = 0
        
        if resume:
            existing_results, start_index = self.checkpoint_manager.load_checkpoint()
            logger.info(f"Resuming from checkpoint: {len(existing_results)} papers already processed")
        
        # Results storage
        all_results = list(existing_results)
        results_lock = threading.Lock()
        
        # Duplicate detection setup
        processed_hashes = set()
        if detect_duplicates:
            for _, result in existing_results:
                if result.get('Title') and result.get('Abstract'):
                    paper_hash = self._get_paper_hash(result)
                    processed_hashes.add(paper_hash)
            logger.info(f"Loaded {len(processed_hashes)} existing paper hashes for duplicate detection")
        
        # Progress tracking
        total_papers = len(papers)
        processed = len(existing_results)
        pbar = tqdm(total=total_papers, initial=processed, desc="Processing papers")
        
        # Process remaining papers
        papers_to_process = [(idx, paper) for idx, paper in enumerate(papers) if idx >= start_index]
        
        if not papers_to_process:
            logger.info("All papers already processed!")
            pbar.close()
            return all_results
        
        # Process papers with thread pool
        futures = []
        checkpoint_counter = 0
        
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            for idx, paper in papers_to_process:
                if self.stop_requested:
                    break
                
                # Skip duplicates if detection is enabled
                if detect_duplicates:
                    paper_hash = self._get_paper_hash(paper)
                    if paper_hash in processed_hashes:
                        logger.debug(f"Skipping duplicate paper {idx}")
                        continue
                    processed_hashes.add(paper_hash)
                
                future = executor.submit(self.classifier.classify_paper, paper, idx, prompt)
                futures.append((future, idx))
            
            # Collect results
            for future, idx in futures:
                if self.stop_requested:
                    break
                
                try:
                    result = future.result(timeout=60)
                    
                    with results_lock:
                        all_results.append((idx, result))
                        processed += 1
                        checkpoint_counter += 1
                        pbar.update(1)
                        
                        # Update progress
                        if processed % 50 == 0:
                            self._update_progress(pbar)
                        
                        # Save checkpoint
                        if checkpoint_counter >= self.config["checkpoint_frequency"]:
                            self.checkpoint_manager.save_checkpoint(all_results, processed)
                            checkpoint_counter = 0
                
                except Exception as e:
                    logger.error(f"Failed to process paper {idx}: {e}")
                    
                    with results_lock:
                        error_result = {
                            'Title': f'Paper {idx}',
                            'Abstract': 'Failed to process',
                            'Discipline': 'ERROR',
                            'Subfield': str(e)[:50],
                            'Discipline_Confidence': 0,
                            'Subfield_Confidence': 0,
                            'Classifier': 'openai'
                        }
                        all_results.append((idx, error_result))
                        pbar.update(1)
        
        pbar.close()
        
        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(all_results, processed)
        
        # Sort results and save to output file
        all_results.sort(key=lambda x: x[0])
        final_results = [r[1] for r in all_results]
        
        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=2)
        else:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                if final_results:
                    fieldnames = ['Title', 'Abstract', 'Discipline', 'Subfield', 
                                 'Discipline_Confidence', 'Subfield_Confidence', 'Classifier']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(final_results)
        
        # Save statistics
        self._save_statistics(output_file)
        
        # Print cost summary
        self._print_cost_summary()
        
        return final_results
    
    def _get_paper_hash(self, paper: Dict) -> str:
        """Generate hash for duplicate detection"""
        title = paper.get('Title', '').strip().lower()
        abstract = paper.get('Abstract', '')[:200].strip().lower()
        return hashlib.md5(f"{title}{abstract}".encode()).hexdigest()
    
    def _update_progress(self, pbar):
        """Update progress bar with statistics and time estimation"""
        if self.cost_tracker.requests > 0:
            stats = {
                "openai": f"{self.cost_tracker.requests}req/${self.cost_tracker.get_cost():.2f}",
                "rpm": f"{self.cost_tracker.requests / max(1, (time.time() - self.start_time) / 60):.1f}"
            }
            pbar.set_postfix(stats)
    
    def _save_statistics(self, output_file: str):
        """Save classification statistics"""
        if not self.start_time:
            return
            
        stats_file = output_file.replace('.csv', '_stats.json').replace('.json', '_stats.json')
        stats = {
            "total_papers": self.cost_tracker.requests,
            "errors": self.cost_tracker.errors,
            "total_cost": self.cost_tracker.get_cost(),
            "avg_cost_per_paper": self.cost_tracker.get_cost() / max(1, self.cost_tracker.requests),
            "processing_time": time.time() - self.start_time,
            "papers_per_minute": self.cost_tracker.requests / max(1, (time.time() - self.start_time) / 60),
            "timestamp": datetime.now().isoformat()
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to: {stats_file}")
    
    def _print_cost_summary(self):
        """Print final cost summary"""
        print("\n" + "="*60)
        print("CLASSIFICATION COMPLETE - COST SUMMARY")
        print("="*60)
        
        summary = self.cost_tracker.get_summary()
        print(f"\nOPENAI (GPT-4o-mini):")
        print(f"  Requests: {summary['requests']}")
        print(f"  Errors: {summary['errors']}")
        print(f"  Total Cost: {summary['total_cost']}")
        print(f"  Avg Cost/Request: {summary['avg_cost_per_request']}")
        print("="*60)

# Load classification prompt from file
def load_prompt(prompt_file: str = "classification_prompt.txt") -> str:
    """Load classification prompt from file"""
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            return f.read()
    
    # Create formatted subfield descriptions for the prompt
    cs_subfields_desc = '\n'.join([f"  - {code}: {desc}" for code, desc in SUBFIELD_DESCRIPTIONS['CS'].items()])
    is_subfields_desc = '\n'.join([f"  - {code}: {desc}" for code, desc in SUBFIELD_DESCRIPTIONS['IS'].items()])
    it_subfields_desc = '\n'.join([f"  - {code}: {desc}" for code, desc in SUBFIELD_DESCRIPTIONS['IT'].items()])
    
    # Default prompt
    return f"""You are an expert classifier for computing research papers. Analyze this paper with maximum precision.

Title: {{title}}
Abstract: {{abstract}}

CLASSIFICATION TASK:
1. Determine the discipline (CS, IS, or IT)
2. Select the MOST SPECIFIC subfield that matches the paper's primary focus

DISCIPLINES:
- CS (Computer Science): Theoretical research, algorithms, software development, AI/ML research, technical computing
- IS (Information Systems): Business applications, organizational technology, enterprise systems, digital business
- IT (Information Technology): Practical implementation, infrastructure, operations, IT services

COMPUTER SCIENCE (CS) SUBFIELDS:
{cs_subfields_desc}

INFORMATION SYSTEMS (IS) SUBFIELDS:
{is_subfields_desc}

INFORMATION TECHNOLOGY (IT) SUBFIELDS:
{it_subfields_desc}

Output EXACTLY in this format (use the subfield code, not the full name):
DISC|SUBFIELD|DISC_CONF|SUB_CONF

Where:
- DISC: CS, IS, or IT
- SUBFIELD: One of the valid codes listed above for the discipline
- DISC_CONF: Confidence in discipline (0-100)
- SUB_CONF: Confidence in subfield (0-100)

Example output: CS|AI/ML|95|90"""

def main():
    import argparse
    parser = argparse.ArgumentParser(description="OpenAI GPT-4o-mini Paper Classifier")
    parser.add_argument('--input', default='Abstracts.csv', help='Input CSV file')
    parser.add_argument('--output', default='classified_openai.csv', help='Output CSV file')
    parser.add_argument('--config', help='Configuration JSON file')
    parser.add_argument('--prompt', help='Prompt text file')
    parser.add_argument('--test', type=int, help='Test with N papers')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--detect-duplicates', action='store_true', help='Enable duplicate detection')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # Create .env file template if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# API Key for OpenAI Classifier\nOPENAI_API_KEY=your_openai_key_here\n")
        print("Created .env file template. Please add your API key and run again.")
        return
    
    # Load papers
    logger.info(f"Loading papers from {args.input}")
    papers = []
    try:
        with open(args.input, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                papers.append(row)
                if args.test and len(papers) >= args.test:
                    break
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return
    
    logger.info(f"Loaded {len(papers)} papers")
    
    # Load prompt
    prompt = load_prompt(args.prompt) if args.prompt else load_prompt()
    
    # Create classifier and process papers
    try:
        classifier = OpenAIOnlyClassifier(args.config)
        results = classifier.process_papers(papers, args.output, prompt, args.resume, args.detect_duplicates)
        
        print(f"\nProcessing complete!")
        print(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()