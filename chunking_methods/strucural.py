from langchain_huggingface import HuggingFaceEmbeddings
import logging
from typing import List, Tuple
import uuid
import re
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


import spacy

class LanguageIndependentStructuralAnalyzer:
    """언어 독립적 구조 분석기"""
    
    def __init__(self):
        pass
    
    def extract_structural_only(self, text):
        """완전히 언어에 독립적인 구조 분석"""
        if not text or not text.strip():
            return self._get_default_features()
        
        words = [w for w in text.split() if w.strip()]
        if not words:
            clean_text = text.strip()
            if clean_text:
                words = [clean_text]
        
        features = {
            # 평균 문장 길이(단어 수 기준)
            'avg_sentence_length': len(words),
            # 문장부호 분포(.,!?;:)
            'punctuation_distribution': Counter([c for c in text if c in '.,!?;:']),
            # 단어 길이 분포
            'word_length_distribution': Counter([len(word) for word in words]),
            # 단락 구조
            'paragraph_structure': self._analyze_paragraph_structure(text),
            # 문자 레벨 분포
            'character_level': self._analyze_character_structure(text),
            
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'punctuation_density': len([c for c in text if c in '.,!?;:']) / len(text) if text else 0,
            'token_level': self._analyze_token_structure(text),
            'spacing_patterns': self._analyze_spacing_patterns(text),
            'length_distributions': self._analyze_length_distributions(text)

        }
        return features

    
    def _get_default_features(self):
        """빈 텍스트 기본값 — 핵심 5개 키만 유지"""
        return {
            'avg_sentence_length': 0,
            'punctuation_distribution': Counter(),
            'word_length_distribution': Counter(),
            
            'avg_word_length': 0,
            'punctuation_density': 0,
            
            
            'paragraph_structure': {
                'paragraph_count': 0,
                'avg_paragraph_length': 0,
                'paragraph_length_variance': 0,
                'sentences_per_paragraph': []
            },
            'character_level': {
                'char_type_distribution': Counter(),
                'char_type_transitions': Counter(),
                'char_density': {
                    'letter_density': 0,
                    'digit_density': 0,
                    'space_density': 0,
                    'punct_density': 0
                }
            },
            'token_level': {
                'token_count': 0,
                'avg_token_length': 0,
                'token_length_variance': 0,
                'capitalization_distribution': Counter(),
                'digit_token_ratio': 0,
                'punct_token_ratio': 0
            },
            'spacing_patterns': {
                'space_distribution': Counter(),
                'avg_space_length': 0,
                'newline_distribution': Counter(),
                'avg_newline_length': 0,
                'total_whitespace_ratio': 0
            },
            'length_distributions': {
                'word_lengths': {'distribution': Counter(), 'mean': 0, 'variance': 0, 'max': 0, 'min': 0},
                'sentence_lengths': {'distribution': Counter(), 'mean': 0, 'variance': 0},
                'paragraph_lengths': {'distribution': Counter(), 'mean': 0, 'variance': 0}
            }
        }

        
    def _get_punctuation_positions(self, text):
        """구두점의 상대적 위치 분석"""
        punct_positions = []
        for i, char in enumerate(text):
            if char in '.,!?;:':
                punct_positions.append(i / len(text) if len(text) > 0 else 0)
        return punct_positions
    
    def _analyze_paragraph_structure(self, text):
        """단락 구조 분석"""
        if not text:
            return {
                'paragraph_count': 0,
                'avg_paragraph_length': 0,
                'paragraph_length_variance': 0,
                'sentences_per_paragraph': []
            }
        
        paragraphs = re.split(r'\n\s*\n+', text.strip())
        valid_paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 0]
        
        if not valid_paragraphs:
            valid_paragraphs = [text.strip()]
        
        paragraph_lengths = [len(p.split()) for p in valid_paragraphs]
        
        return {
            'paragraph_count': len(valid_paragraphs),
            'avg_paragraph_length': np.mean(paragraph_lengths) if paragraph_lengths else 0,
            'paragraph_length_variance': np.var(paragraph_lengths) if len(paragraph_lengths) > 1 else 0,
            'sentences_per_paragraph': [len([s for s in p.split('.') if s.strip()]) for p in valid_paragraphs]
        }
    
    def _analyze_character_structure(self, text):
        """문자 레벨 구조 분석"""
        chars = list(text)
        char_types = []
        
        for char in chars:
            if char.isalpha():
                char_types.append('LETTER')
            elif char.isdigit():
                char_types.append('DIGIT')
            elif char.isspace():
                char_types.append('SPACE')
            elif char in '.,!?;:':
                char_types.append('PUNCT')
            else:
                char_types.append('OTHER')
        
        return {
            'char_type_distribution': Counter(char_types),
            'char_type_transitions': self._get_transitions(char_types),
            'char_density': {
                'letter_density': char_types.count('LETTER') / len(char_types) if char_types else 0,
                'digit_density': char_types.count('DIGIT') / len(char_types) if char_types else 0,
                'space_density': char_types.count('SPACE') / len(char_types) if char_types else 0,
                'punct_density': char_types.count('PUNCT') / len(char_types) if char_types else 0
            }
        }
    
    def _get_transitions(self, sequence):
        """시퀀스의 전환 패턴 분석"""
        transitions = []
        for i in range(len(sequence) - 1):
            transitions.append((sequence[i], sequence[i+1]))
        return Counter(transitions)
    
    def _analyze_token_structure(self, text):
        """토큰 레벨 구조 분석"""
        if not text:
            return {
                'token_count': 0,
                'avg_token_length': 0,
                'token_length_variance': 0,
                'capitalization_distribution': Counter(),
                'digit_token_ratio': 0,
                'punct_token_ratio': 0
            }
        
        tokens = [t for t in text.split() if t.strip()]
        
        if not tokens:
            return {
                'token_count': 0,
                'avg_token_length': 0,
                'token_length_variance': 0,
                'capitalization_distribution': Counter(),
                'digit_token_ratio': 0,
                'punct_token_ratio': 0
            }
        
        token_features = []
        capitalization_patterns = []
        
        for token in tokens:
            features = {
                'length': len(token),
                'has_digit': any(c.isdigit() for c in token),
                'has_punct': any(not c.isalnum() for c in token),
                'is_capitalized': token[0].isupper() if token else False,
                'is_all_caps': token.isupper() if token else False,
                'is_all_lower': token.islower() if token else False
            }
            token_features.append(features)
            
            if features['is_all_caps']:
                capitalization_patterns.append('ALL_CAPS')
            elif features['is_capitalized']:
                capitalization_patterns.append('CAPITALIZED')
            elif features['is_all_lower']:
                capitalization_patterns.append('LOWERCASE')
            else:
                capitalization_patterns.append('MIXED')
        
        return {
            'token_count': len(tokens),
            'avg_token_length': np.mean([f['length'] for f in token_features]) if token_features else 0,
            'token_length_variance': np.var([f['length'] for f in token_features]) if len(token_features) > 1 else 0,
            'capitalization_distribution': Counter(capitalization_patterns),
            'digit_token_ratio': sum(1 for f in token_features if f['has_digit']) / len(token_features) if token_features else 0,
            'punct_token_ratio': sum(1 for f in token_features if f['has_punct']) / len(token_features) if token_features else 0
        }
    
    def _analyze_spacing_patterns(self, text):
        """공백 패턴 분석"""
        if not text:
            return {
                'space_distribution': Counter(),
                'avg_space_length': 0,
                'newline_distribution': Counter(),
                'avg_newline_length': 0,
                'total_whitespace_ratio': 0
            }
        
        space_sequences = re.findall(r' +', text)
        space_lengths = [len(seq) for seq in space_sequences]
        
        newline_sequences = re.findall(r'\n+', text)
        newline_lengths = [len(seq) for seq in newline_sequences]
        
        return {
            'space_distribution': Counter(space_lengths),
            'avg_space_length': np.mean(space_lengths) if space_lengths else 0,
            'newline_distribution': Counter(newline_lengths),
            'avg_newline_length': np.mean(newline_lengths) if newline_lengths else 0,
            'total_whitespace_ratio': (text.count(' ') + text.count('\n') + text.count('\t')) / len(text) if text else 0
        }
    
    def _analyze_length_distributions(self, text):
        """다양한 길이 분포 분석"""
        if not text:
            return {
                'word_lengths': {'distribution': Counter(), 'mean': 0, 'variance': 0, 'max': 0, 'min': 0},
                'sentence_lengths': {'distribution': Counter(), 'mean': 0, 'variance': 0},
                'paragraph_lengths': {'distribution': Counter(), 'mean': 0, 'variance': 0}
            }
        
        words = [w for w in text.split() if w.strip()]
        sentences = [text.strip()] if text.strip() else []
        paragraphs = [text.strip()] if text.strip() else []
        
        return {
            'word_lengths': {
                'distribution': Counter([len(w) for w in words]) if words else Counter(),
                'mean': np.mean([len(w) for w in words]) if words else 0,
                'variance': np.var([len(w) for w in words]) if len(words) > 1 else 0,
                'max': max([len(w) for w in words]) if words else 0,
                'min': min([len(w) for w in words]) if words else 0
            },
            'sentence_lengths': {
                'distribution': Counter([len(s.split()) for s in sentences]) if sentences else Counter(),
                'mean': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
                'variance': np.var([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0
            },
            'paragraph_lengths': {
                'distribution': Counter([len(p.split()) for p in paragraphs]) if paragraphs else Counter(),
                'mean': np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0,
                'variance': np.var([len(p.split()) for p in paragraphs]) if len(paragraphs) > 1 else 0
            }
        }

class LanguageIndependentSimilarity:
    """언어 독립적 유사도 계산기"""
    
    def __init__(self):
        self.analyzer = LanguageIndependentStructuralAnalyzer()
    
    def compare_texts(self, text1, text2):
        """두 텍스트의 언어 독립적 구조 유사도 계산"""
        features1 = self.analyzer.extract_structural_only(text1)
        features2 = self.analyzer.extract_structural_only(text2)

        weights = {
            'paragraph_structure': 0.25,
            'punctuation_distribution': 0.20,
            'word_length_distribution': 0.15,
            'character_level': 0.10,
            'avg_sentence_length': 0.05,
            
            'avg_word_length': 0.02,
            'punctuation_density': 0.02,
            'token_level': 0.01,
            'spacing_patterns': 0.01,
            'length_distributions': 0.01
        }
        

        # 명시된 키만 계산(불필요한 키가 있어도 무시)
        overall_similarity = 0.0
        for cat, w in weights.items():
            sim = self._calculate_feature_similarity(features1.get(cat), features2.get(cat))
            overall_similarity += (sim * w)
        return overall_similarity

    
    def _calculate_feature_similarity(self, feat1, feat2):
        """개별 특징 간 유사도 계산"""
        if isinstance(feat1, Counter) and isinstance(feat2, Counter):
            return self._calculate_counter_similarity(feat1, feat2)
        elif isinstance(feat1, dict) and isinstance(feat2, dict):
            return self._calculate_dict_similarity(feat1, feat2)
        elif isinstance(feat1, list) and isinstance(feat2, list):
            return self._calculate_list_similarity(feat1, feat2)
        elif isinstance(feat1, (int, float)) and isinstance(feat2, (int, float)):
            return self._calculate_numeric_similarity(feat1, feat2)
        else:
            return 0.0
    
    def _calculate_counter_similarity(self, counter1, counter2):
        """Counter 객체 간 코사인 유사도"""
        all_keys = set(counter1.keys()) | set(counter2.keys())
        
        if not all_keys:
            return 1.0
        
        vec1 = [counter1.get(key, 0) for key in all_keys]
        vec2 = [counter2.get(key, 0) for key in all_keys]
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _calculate_dict_similarity(self, dict1, dict2):
        """딕셔너리 간 유사도"""
        similarities = []
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            if key in dict1 and key in dict2:
                sim = self._calculate_feature_similarity(dict1[key], dict2[key])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_list_similarity(self, list1, list2):
        """리스트 간 유사도"""
        if not list1 and not list2:
            return 1.0
        elif not list1 or not list2:
            return 0.0
        
        len_sim = 1 - abs(len(list1) - len(list2)) / max(len(list1), len(list2))
        
        if all(isinstance(x, (int, float)) for x in list1 + list2):
            dist_sim = 1 - abs(np.mean(list1) - np.mean(list2)) / max(np.mean(list1), np.mean(list2), 1)
        else:
            dist_sim = 0.5
        
        return (len_sim + dist_sim) / 2
    
    def _calculate_numeric_similarity(self, num1, num2):
        """수치 간 유사도"""
        if num1 == 0 and num2 == 0:
            return 1.0
        
        max_val = max(abs(num1), abs(num2))
        if max_val == 0:
            return 1.0
        
        return 1 - abs(num1 - num2) / max_val


class StructuralChunker:
    """구조적 + 시맨틱 청킹을 결합한 청커 (+ 최소길이, 개행 포함, 구분기호 보존)"""
    
    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        semantic_weight: float = 0.05,
        structural_weight: float = 0.95,
        # min_chunk_size: int = 100,          #  문자수 기준 최소 길이
        split_newlines: bool = True,        #  개행을 문장 경계로 포함
        preserve_delimiters: bool = True,   #  구분기호(공백/개행) 보존
        merge_short_chunks: bool = False,    #  짧은 청크 병합 여부
        # percentile_threshold: int = 25,     #  병합 임계값 (퍼센타일)
        breakpoint_threshold_amount = 95, # 청크 경계 임계값 (퍼센타일)
        nlp_backend="spacy",
    ):
        self.embeddings = embeddings
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight
        # self.min_chunk_size = min_chunk_size
        self.split_newlines = split_newlines
        self.preserve_delimiters = preserve_delimiters
        self.merge_short_chunks = merge_short_chunks
        # self.percentile_threshold = percentile_threshold
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        
        self.structural_analyzer = LanguageIndependentSimilarity()
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
        if nlp_backend == "kiwi":
            from kiwi import Kiwi
            self._nlp = Kiwi()
        else:
            self._nlp = spacy.load("en_core_web_sm")
        
    def split_text(self, text: str, source: str, file_name: str) -> List[str]:
        """구조적 + 시맨틱 청킹으로 텍스트 분할 (+ min_chunk_size / split_newlines / preserve_delimiters)"""
        try:
            
            # 1) 문장/구분기호 분할
            sentences, delims = self._split_into_sentences_and_delims(text)
            if len(sentences) <= 1:
                return [text]
            
            # 2) 문장별 임베딩 (본문만 사용)
            sentence_embeddings = self.embeddings.embed_documents(sentences)
            
            # 3) 문장 간 거리 계산 (시맨틱 + 구조적)
            distances = []
            for i in range(len(sentences) - 1):
                # 시맨틱 거리
                semantic_sim = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
                semantic_distance = 1 - semantic_sim
                
                # 구조적 거리
                structural_sim = self.structural_analyzer.compare_texts(sentences[i], sentences[i + 1])
                structural_distance = 1 - structural_sim
                
                # 가중합 거리
                weighted_distance = (self.semantic_weight * semantic_distance + 
                                     self.structural_weight * structural_distance)
                distances.append(weighted_distance)
            
            # 4) 청크 경계 결정 (80% 퍼센타일 임계값)
            threshold = np.percentile(distances, self.breakpoint_threshold_amount) if distances else 0.0
            breakpoints = [i for i, dist in enumerate(distances) if dist > threshold]
            
            # 5) 청크 생성 (구분기호 보존 여부 반영)
            chunks = []
            start_idx = 0
            for bp in breakpoints:
                chunk_text = self._join_with_delims(sentences[start_idx:bp+1], delims[start_idx:bp+1])
                chunks.append(chunk_text)
                start_idx = bp + 1
            # 마지막 청크
            if start_idx < len(sentences):
                chunk_text = self._join_with_delims(sentences[start_idx:], delims[start_idx:])
                chunks.append(chunk_text)
            
            # 6) 최소 길이 보장: 문자수 기준으로 병합
            # chunks = self._enforce_min_chunk_size(chunks, self.min_chunk_size)
            
            # 7) 짧은 청크 병합 (옵션)
            if self.merge_short_chunks and len(chunks) > 1:
                chunks = self._merge_short_chunks(chunks)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error in StructuralChunker.split_text for document {self.process_counter} ({file_name}): {e}")
            return [text]
    
    # -------- 내부 유틸 --------
    def _split_into_sentences_and_delims(self, text: str) -> Tuple[List[str], List[str]]:
        """
        텍스트를 (문장, 구분기호)로 분리하여 반환.
        - split_newlines=True면 개행(\n+)도 구분기호로 포함
        - 반환: sentences[i] 다음에 delims[i]가 이어짐
        """
        if not text or not text.strip():
            return [], []

        doc = self._nlp(text)
        sentences: List[str] = []
        delims: List[str] = []

        for sent in doc.sents:
            # 문장 본문
            sentence_text = sent.text.strip()
            if not sentence_text:
                continue
            sentences.append(sentence_text)

            # 문장 끝 다음의 공백/개행 등 구분기호를 원문에서 직접 추출
            end_pos = sent.end_char
            if end_pos < len(text):
                j = end_pos
                # 공백·개행이 연속된 구간을 한 번에 추출
                while j < len(text) and text[j] in (' ', '\n', '\t'):
                    j += 1
                raw_delim = text[end_pos:j]
            else:
                raw_delim = ""

            # 옵션: 구분기호를 그대로 보존할지(space normalize) 선택
            if self.preserve_delimiters:
                delims.append(raw_delim)
            else:
                # 개행 포함 여부 설정
                if self.split_newlines:
                    delims.append(raw_delim if raw_delim else " ")
                else:
                    # 개행이 있더라도 단일 공백으로 정규화
                    delims.append(" " if raw_delim else " ")

        # 길이 보정 (마지막 문장 뒤에 delimiter가 없을 수 있으므로)
        if len(delims) < len(sentences):
            delims += [""] * (len(sentences) - len(delims))

        return sentences, delims
    
    def _join_with_delims(self, sentences: List[str], delims: List[str]) -> str:
        """(문장, 구분기호) 시퀀스를 합쳐 원문 스타일로 재구성"""
        out = []
        for s, d in zip(sentences, delims):
            out.append(s)
            out.append(d if self.preserve_delimiters else (" " if d else " "))
        return "".join(out).strip()
    
    def _enforce_min_chunk_size(self, chunks: List[str], min_len: int) -> List[str]:
        """문자수 기준 최소 길이를 만족하도록 좌→우로 병합"""
        if min_len <= 0 or not chunks:
            return chunks
        
        merged = []
        buf = ""
        for ch in chunks:
            if not buf:
                buf = ch
            else:
                if len(buf) < min_len:
                    buf = (buf + (" " if not buf.endswith((" ", "\n")) else "") + ch)
                else:
                    merged.append(buf)
                    buf = ch
        if buf:
            # 마지막 버퍼가 너무 짧고 앞선 청크가 있다면 앞선 것과 병합 시도
            if len(buf) < min_len and merged:
                merged[-1] = merged[-1] + (" " if not merged[-1].endswith((" ", "\n")) else "") + buf
            else:
                merged.append(buf)
        return merged
    
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
