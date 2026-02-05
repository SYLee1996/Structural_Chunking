from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from typing import List
import re
import uuid
import numpy as np

class GradientChunker:
    """
    그래디언트 기반 유사도 임계값을 사용하여 텍스트를 청크로 분할
    + 줄바꿈 포함 여부(split_newlines)
    + 구분기호(공백/개행) 보존(preserve_delimiters)
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        min_chunk_size: int = 100,
        merge_short_chunks: bool = False,
        # percentile_threshold: int = 25,
        nlp_backend="spacy"  # "kiwi" or "spacy"
    ):
        self.chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="gradient",
            # min_chunk_size=min_chunk_size,
            # breakpoint_threshold_amount = 95,
            nlp=nlp_backend,
        )
        self.merge_short_chunks = merge_short_chunks
        # self.percentile_threshold = percentile_threshold

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
        """텍스트를 그래디언트 기반으로 청크 분할 (+줄바꿈 포함/구분기호 보존)"""
        if source not in self.domain_mapping:
            raise ValueError(
                f"Source must be one of: {list(self.domain_mapping.keys())}"
            )

        canonical_source = self.domain_mapping[source]
        self.process_counter += 1
        unique_id = self.generate_unique_id()

        try:
            self.logger.info(
                f"Starting GradientChunker.split_text for {canonical_source} document: {unique_id}"
            )

            # 1. langchain의 기본 청크 분리
            chunks = self.chunker.create_documents([text])

            # 2. 결과 후처리: 줄바꿈 포함 및 구분기호 보존
            chunk_texts = []
            for chunk in chunks:
                ctext = chunk.page_content
                chunk_texts.append(ctext)

            # 3. 짧은 청크 병합 (옵션)
            if self.merge_short_chunks and len(chunk_texts) > 1:
                chunk_texts = self._merge_short_chunks(chunk_texts)

            self.logger.info(
                f"GradientChunker created {len(chunk_texts)} chunks for {canonical_source} document: {unique_id}"
            )
            return chunk_texts

        except Exception as e:
            self.logger.error(
                f"Error in GradientChunker.split_text for {canonical_source} document {self.process_counter} ({file_name}): {e}"
            )
            return [text]

    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """전체 청크 길이 분포의 25% 이하인 짧은 청크들을 앞의 청크에 병합"""
        if len(chunks) <= 1:
            return chunks
        
        # 청크 길이 측정 (문자 수 기준)
        lengths = [len(chunk) for chunk in chunks]
        threshold = np.percentile(lengths, self.percentile_threshold)
        
        merged_chunks = []
        buffer = None
        
        for chunk in chunks:
            current_chunk = chunk.strip()
            
            if len(current_chunk) <= threshold:
                # 짧은 청크는 buffer에 저장
                if buffer is None:
                    buffer = current_chunk
                else:
                    buffer += ' ' + current_chunk
            else:
                # 긴 청크가 나오면 buffer와 병합
                if buffer:
                    merged_chunks.append(buffer + ' ' + current_chunk)
                    buffer = None
                else:
                    merged_chunks.append(current_chunk)
        
        # buffer가 남아있으면 마지막에 병합
        if buffer and merged_chunks:
            merged_chunks[-1] += ' ' + buffer
        elif buffer:
            merged_chunks.append(buffer)
        
        return merged_chunks

    def generate_unique_id(self) -> str:
        """고유 ID 생성 (UUID의 첫 8자리)"""
        return str(uuid.uuid4())[:8]
