import logging
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
import matplotlib.pyplot as plt
import json
import warnings

# sklearn 경고 억제
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def evaluate_retrieval_quality(results: Dict[str, Dict[str, List[Dict]]],
                               ground_truths: Dict[str, Dict[str, Dict[str, List[str]]]],
                               embeddings: HuggingFaceEmbeddings,
                               top_k: int = 10):
    """검색 품질을 평가하여 정밀도, 재현율, F1 점수 등을 계산"""
    print("Starting retrieval quality evaluation...")
    try:
        logger.info("Starting evaluation of retrieval quality.")
        metrics = {}
        domains = list(ground_truths.keys())
        
        print(f"Evaluating domains: {domains}")
        
        for method, method_results in results.items():
            print(f"\nProcessing method: {method}")
            metrics[method] = {}
            
            for domain in domains:
                if domain not in method_results or domain not in ground_truths:
                    print(f"  Domain '{domain}' not found. Skipping.")
                    continue

                chunks = method_results[domain]
                if not chunks:
                    print(f"    No chunks for {domain} in {method}. Skipping.")
                    metrics.setdefault(method, {})[domain] = {
                        'precision': 0, 'recall': 0, 'f1_score': 0,
                        'average_precision': 0, 'ndcg': 0
                    }
                    continue
                print(f"  Processing domain: {domain} ({len(chunks)} chunks)")
                
                # Ground truth 데이터 가져오기
                relevant_chunks = ground_truths[domain]['relevant_chunks']
                queries = ground_truths[domain]['queries']
                
                print(f"    Encoding {len(chunks)} chunks...")
                # 모든 청크 인코딩
                chunk_texts = []
                chunk_metadata = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        chunk_texts.append(chunk.get('text') or chunk.get('content', ''))
                        chunk_metadata.append(chunk)
                    else:
                        chunk_texts.append(str(chunk))
                        chunk_metadata.append({})
                
                chunk_embeddings = embeddings.embed_documents(chunk_texts)
                chunk_embeddings = np.asarray(chunk_embeddings, dtype=float)
                if chunk_embeddings.ndim != 2 or chunk_embeddings.shape[0] == 0:
                    print(f"    No valid chunk embeddings for {domain} in {method}. Skipping.")
                    metrics.setdefault(method, {})[domain] = {
                        'precision': 0, 'recall': 0, 'f1_score': 0,
                        'average_precision': 0, 'ndcg': 0
                    }
                    continue
                                
                # 모든 쿼리 처리
                print(f"    Processing {len(queries)} queries...")
                
                precisions, recalls, f1s, aps = [], [], [], []
                
                for i, query in enumerate(queries):
                    # 100개 쿼리마다 진행상황 표시
                    if (i + 1) % 100 == 0:
                        print(f"      Processed {i+1}/{len(queries)} queries...")
                    
                    query_embedding = embeddings.embed_query(query)
                    query_embedding = np.array(query_embedding).reshape(1, -1)
                    
                    # 코사인 유사도 계산
                    similarities = cosine_similarity(query_embedding, chunk_embeddings)
                    
                    # top_k 인덱스 가져오기
                    top_indices = similarities.argsort()[-top_k:][::-1]
                    retrieved_chunks = [chunks[i] for i in top_indices]
                    
                    # 관련 청크 가져오기 (리스트 또는 딕셔너리 형식 지원)
                    query_relevant_raw = relevant_chunks.get(query, [])
                    if isinstance(query_relevant_raw, dict):
                        query_relevant_list = query_relevant_raw.get('chunks', [])
                    else:
                        query_relevant_list = query_relevant_raw

                    # 정밀도, 재현율, F1을 위한 이진 관련성
                    y_true = []
                    for chunk, metadata in zip(retrieved_chunks, [chunk_metadata[i] for i in top_indices]):
                        if isinstance(chunk, dict):
                            chunk_text = chunk.get('text') or chunk.get('content', '')
                        else:
                            chunk_text = str(chunk)
                        
                        is_relevant = 0
                        for relevant_chunk in query_relevant_list:
                            if chunk_text in relevant_chunk or relevant_chunk in chunk_text:
                                is_relevant = 1
                                break
                        y_true.append(is_relevant)
                    
                    y_pred = [1] * len(retrieved_chunks)
                    
                    # 메트릭 계산
                    precision = precision_score(y_true, y_pred, average='binary', zero_division="warn")
                    recall = recall_score(y_true, y_pred, average='binary', zero_division="warn")
                    f1 = f1_score(y_true, y_pred, average='binary', zero_division="warn")
                    ap = average_precision_score(y_true, similarities[top_indices])
                    
                    # 메트릭 수집
                    precisions.append(precision)
                    recalls.append(recall) 
                    f1s.append(f1)
                    aps.append(ap)
                
                # 평균 메트릭 계산
                avg_precision = np.mean(precisions) if precisions else 0
                avg_recall = np.mean(recalls) if recalls else 0
                avg_f1 = np.mean(f1s) if f1s else 0
                avg_ap = np.mean(aps) if aps else 0
                
                metrics[method][domain] = {
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1_score': avg_f1,
                    'average_precision': avg_ap,
                }

                print(f"    Results - Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}, F1: {avg_f1:.3f}, AP: {avg_ap:.3f}")
                
        print("\nCompleted retrieval quality evaluation.")
        return metrics

    except Exception as e:
        logger.error(f"Error in evaluate_retrieval_quality: {e}")
        print(f"Error in evaluating retrieval quality: {e}")
        return {}

def cosine_similarity(query_embedding, chunk_embeddings):
    """쿼리와 청크 임베딩 간의 코사인 유사도 계산"""
    query_norm = np.linalg.norm(query_embedding)
    chunks_norm = np.linalg.norm(chunk_embeddings, axis=1)
    
    similarity = np.dot(chunk_embeddings, query_embedding.T).flatten() / (chunks_norm * query_norm + 1e-10)
    return similarity

def plot_retrieval_quality_metrics(metrics_path: str = 'results/retrieval_quality_metrics.json'):
    """검색 품질 메트릭 비교 차트 생성"""
    print("Starting plot_retrieval_quality_metrics function")
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        if not metrics or not any(metrics.values()):
            print("No metrics to plot")
            return
            
        # 비교 차트 생성
        methods = list(metrics.keys())
        domains = list(next(iter(metrics.values())).keys()) if metrics else []
        
        if not methods or not domains:
            print("No data available for plotting")
            return
        
        metric_names = ['precision', 'recall', 'f1_score', 'average_precision']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, metric in enumerate(metric_names):
            ax = axes[i]
            
            for method in methods:
                values = [metrics[method][domain][metric] for domain in domains if domain in metrics[method]]
                ax.plot(domains[:len(values)], values, marker='o', label=method)
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Domain')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)
        
        # 2x2 그리드로 정확히 4개 메트릭만 표시

        plt.tight_layout()
        plt.savefig('results/retrieval_quality_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Retrieval quality comparison plot saved.")
        
    except Exception as e:
        print(f"Error in plotting retrieval quality metrics: {e}")
        logger.error(f"Error in plotting: {e}")
