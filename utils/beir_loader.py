# src/utils/beir_loader.py

import json
import os
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class BEIRDataLoader:
    """
    Loader for BEIR (Benchmarking Information Retrieval) datasets.
    Replaces the ground truth generation with pre-labeled BEIR data.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.available_datasets = self._find_available_datasets()
        
    def _find_available_datasets(self) -> List[str]:
        """Find available BEIR datasets in the data directory."""
        datasets = []
        if not os.path.exists(self.data_dir):
            return datasets
            
        for item in os.listdir(self.data_dir):
            dataset_path = os.path.join(self.data_dir, item)
            if os.path.isdir(dataset_path):
                # Check if it has the required BEIR structure
                required_files = ['corpus.jsonl', 'queries.jsonl']
                qrels_dir = os.path.join(dataset_path, 'qrels')
                
                if (all(os.path.exists(os.path.join(dataset_path, f)) for f in required_files) and
                    os.path.exists(qrels_dir) and 
                    any(f.endswith('.tsv') for f in os.listdir(qrels_dir))):
                    datasets.append(item)
                    
        print(f"Found BEIR datasets: {datasets}")
        return datasets
    
    def load_corpus(self, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """Load corpus documents from BEIR dataset."""
        corpus_path = os.path.join(self.data_dir, dataset_name, 'corpus.jsonl')
        corpus = {}
        
        if not os.path.exists(corpus_path):
            print(f"Corpus file not found: {corpus_path}")
            return corpus
            
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                # Combine title and text for full document content
                title = doc.get('title', '')
                text = doc.get('text', '')
                full_text = f"{title}\n\n{text}" if title else text
                
                corpus[doc['_id']] = {
                    'text': full_text,
                    'title': title,
                    'metadata': doc.get('metadata', {})
                }
                
        print(f"Loaded {len(corpus)} documents from {dataset_name} corpus")
        return corpus
    
    def load_queries(self, dataset_name: str) -> Dict[str, str]:
        """Load queries from BEIR dataset."""
        queries_path = os.path.join(self.data_dir, dataset_name, 'queries.jsonl')
        queries = {}
        
        if not os.path.exists(queries_path):
            print(f"Queries file not found: {queries_path}")
            return queries
            
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                queries[query['_id']] = query['text']
                
        print(f"Loaded {len(queries)} queries from {dataset_name}")
        return queries
    
    def load_qrels(self, dataset_name: str, split: str = 'test') -> Dict[str, Dict[str, int]]:
        """Load relevance judgments (qrels) from BEIR dataset."""
        qrels_path = os.path.join(self.data_dir, dataset_name, 'qrels', f'{split}.tsv')
        qrels = {}
        
        if not os.path.exists(qrels_path):
            print(f"Qrels file not found: {qrels_path}")
            return qrels
            
        df = pd.read_csv(qrels_path, sep='\t')
        
        for _, row in df.iterrows():
            query_id = str(row['query-id'])
            corpus_id = str(row['corpus-id'])
            score = int(row['score'])
            
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][corpus_id] = score
            
        print(f"Loaded qrels for {len(qrels)} queries from {dataset_name} ({split} split)")
        return qrels
    
    def create_ground_truths_from_beir(self, datasets: List[str] | None = None, split: str = 'test') -> Dict[str, Any]:
        """
        Create ground truth data structure compatible with existing evaluation code.
        
        Args:
            datasets: List of BEIR dataset names to use. If None, uses all available.
            split: Which split to use ('train', 'dev', 'test')
            
        Returns:
            Ground truth dictionary in the format expected by existing evaluation code
        """
        if datasets is None:
            datasets = self.available_datasets
            
        ground_truths = {}
        
        for dataset_name in datasets:
            if dataset_name not in self.available_datasets:
                print(f"Dataset {dataset_name} not available. Skipping.")
                continue
                
            print(f"Processing BEIR dataset: {dataset_name}")
            
            # Load data
            corpus = self.load_corpus(dataset_name)
            queries = self.load_queries(dataset_name)
            qrels = self.load_qrels(dataset_name, split)
            
            if not all([corpus, queries, qrels]):
                print(f"Failed to load complete data for {dataset_name}. Skipping.")
                continue
            
            # Convert to ground truth format
            ground_truths[dataset_name] = {
                "queries": [],
                "relevant_chunks": {},
                "corpus": corpus  # Store corpus for chunking
            }
            
            # Process queries and their relevant documents
            for query_id, query_text in queries.items():
                if query_id in qrels:
                    # Get relevant documents (score > 0)
                    relevant_docs = [doc_id for doc_id, score in qrels[query_id].items() if score > 0]
                    
                    if relevant_docs:
                        # Get the text of relevant documents
                        relevant_chunks = []
                        for doc_id in relevant_docs:
                            if doc_id in corpus:
                                relevant_chunks.append(corpus[doc_id]['text'])
                        
                        if relevant_chunks:
                            ground_truths[dataset_name]["queries"].append(query_text)
                            ground_truths[dataset_name]["relevant_chunks"][query_text] = {
                                "chunks": relevant_chunks,
                                "metadata": {
                                    "source": dataset_name,
                                    "query_id": query_id,
                                    "relevant_doc_ids": relevant_docs
                                }
                            }
            
            print(f"Created ground truths for {len(ground_truths[dataset_name]['queries'])} queries in {dataset_name}")
        
        return ground_truths
    
    def get_datasets_for_chunking(self, datasets: List[str] | None = None) -> Dict[str, Dict[str, str]]:
        """
        Get datasets in the format expected by chunking methods.
        
        Returns:
            Dictionary with domain -> {doc_id: text} mapping
        """
        if datasets is None:
            datasets = self.available_datasets
            
        chunking_datasets = {}
        
        for dataset_name in datasets:
            if dataset_name not in self.available_datasets:
                continue
                
            corpus = self.load_corpus(dataset_name)
            chunking_datasets[dataset_name] = {
                doc_id: doc_data['text'] 
                for doc_id, doc_data in corpus.items()
            }
            
        return chunking_datasets
    
    def save_ground_truths(self, ground_truths: Dict[str, Any], output_path: str = 'config/beir_ground_truths.json'):
        """Save ground truths to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Remove corpus data before saving (too large)
        clean_ground_truths = {}
        for domain, data in ground_truths.items():
            clean_ground_truths[domain] = {
                "queries": data["queries"],
                "relevant_chunks": data["relevant_chunks"]
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_ground_truths, f, indent=2, ensure_ascii=False)
            
        print(f"Ground truths saved to {output_path}")

def load_beir_datasets(data_dir: str = "data", datasets: List[str] | None = None, split: str = 'test') -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    """
    Convenience function to load BEIR datasets and ground truths.
    
    Returns:
        Tuple of (datasets_for_chunking, ground_truths)
    """
    loader = BEIRDataLoader(data_dir)
    
    # Get datasets for chunking
    datasets_for_chunking = loader.get_datasets_for_chunking(datasets)
    
    # Create ground truths
    ground_truths = loader.create_ground_truths_from_beir(datasets, split)
    
    return datasets_for_chunking, ground_truths
