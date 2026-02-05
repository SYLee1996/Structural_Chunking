from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from typing import List, Dict
import os
import json
from pathlib import Path
import uuid

class InterquartileChunker:
    """사분위수 범위 기반으로 텍스트를 청크로 분할"""
    
    def __init__(self, embeddings: HuggingFaceEmbeddings, nlp_backend="spacy"):
        self.chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="interquartile",
            # breakpoint_threshold_amount = 95,
            nlp=nlp_backend,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # BEIR 데이터셋 도메인 매핑
        self.domain_mapping = {
            'nfcorpus': 'nfcorpus',
            'scifact': 'scifact', 
            'fiqa': 'fiqa',
            'arguana': 'arguana',
            'scidocs': 'scidocs',
        }
        
        self.process_counter = 0
    
    def split_text(self, text: str, source: str, file_name: str) -> List[str]:
        """텍스트를 사분위수 기반으로 청크 분할"""
        if not text:
            return []
        
        if source not in self.domain_mapping:
            raise ValueError(f"Source must be one of: {list(self.domain_mapping.keys())}")
        
        canonical_source = self.domain_mapping[source]
        self.process_counter += 1
        unique_id = self.generate_unique_id()
        
        try:
            self.logger.info(f"Starting InterquartileChunker.split_text for {canonical_source} document: {unique_id}")
            chunks = self.chunker.create_documents([text])
            
            chunk_texts = []
            for chunk in chunks:
                chunk_texts.append(chunk.page_content)
            
            self.logger.info(f"InterquartileChunker created {len(chunk_texts)} chunks for {canonical_source} document: {unique_id}")
            return chunk_texts
        except Exception as e:
            self.logger.error(f"Error in InterquartileChunker.split_text for {canonical_source} document {self.process_counter} ({file_name}): {e}")
            return [text]

    def generate_unique_id(self) -> str:
        """고유 ID 생성 (UUID의 첫 8자리)"""
        return str(uuid.uuid4())[:8]