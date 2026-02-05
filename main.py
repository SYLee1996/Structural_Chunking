import logging
import argparse
from utils.helpers import save_results
from utils.beir_loader import load_beir_datasets
from evaluation.retrieval_quality_evaluator_clean import evaluate_retrieval_quality, plot_retrieval_quality_metrics
from evaluation.chunk_size_evaluator import evaluate_chunk_sizes
# from evaluation.scoring_system import calculate_scores, plot_scores
from chunking_methods import (
    PercentileChunker, StdDeviationChunker, InterquartileChunker,
    GradientChunker, StructuralChunker, FixedLenChunker
)
from langchain_huggingface import HuggingFaceEmbeddings
import json
import os
from typing import Dict, List, Tuple

# -------------------------------
# Utilities
# -------------------------------
METHOD_ALIASES = {
    # cli arg (lower) -> Canonical ClassName (Pascal) & dir_name (lower)
    'percentile':      ('Percentile', 'percentile', PercentileChunker),
    'stddeviation':    ('StdDeviation', 'stddeviation', StdDeviationChunker),
    'interquartile':   ('Interquartile', 'interquartile', InterquartileChunker),  # FIXED: no "Interquantile"
    'recursive':        ('FixedLen', 'fixedlen', FixedLenChunker),
    'gradient':        ('Gradient', 'gradient', GradientChunker),
    'structural':      ('Structural', 'structural', StructuralChunker),
}

def normalize_methods(methods: List[str]) -> List[Tuple[str, str, object]]:
    """args.method 리스트를 (ClassName, dir_name, ChunkerClass) 튜플 리스트로 정규화"""
    norm = []
    for m in methods:
        key = m.lower()
        if key not in METHOD_ALIASES:
            raise ValueError(f"Unknown method: {m}")
        norm.append(METHOD_ALIASES[key])
    return norm

def make_chunk_dir(dir_name: str, domain: str) -> str:
    return os.path.join('data', 'chunks', dir_name, domain)

def chunks_exist(datasets: Dict, methods: List[str]) -> bool:
    """선택된 메서드/데이터셋에 대해 최소 1개 이상의 청크 파일 존재 여부"""
    valid_domains = list(datasets.keys())
    normalized = normalize_methods(methods)
    for _, dir_name, _ in normalized:
        for domain in valid_domains:
            path = make_chunk_dir(dir_name, domain)
            if not (os.path.isdir(path) and any(f.endswith('_chunks.json') for f in os.listdir(path))):
                return False
    return True

def load_existing_chunks(datasets: Dict, methods: List[str]) -> Dict:
    """선택된 메서드/데이터셋에 대한 기존 청크 로드"""
    results = {}
    valid_domains = list(datasets.keys())
    normalized = normalize_methods(methods)

    for class_name, dir_name, _ in normalized:
        results[class_name] = {}
        for domain in valid_domains:
            path = make_chunk_dir(dir_name, domain)
            results[class_name][domain] = []
            if not os.path.isdir(path):
                print(f"[WARN] Chunk path missing: {path}")
                continue
            chunk_files = [f for f in os.listdir(path) if f.endswith('_chunks.json')]
            print(f"Loading {len(chunk_files)} files from {path}")
            for cf in chunk_files:
                fpath = os.path.join(path, cf)
                try:
                    with open(fpath, 'r') as f:
                        chunks = json.load(f)
                    # 표준화: dict만 허용하고 content 필드 보정
                    normalized_chunks = []
                    for ch in chunks:
                        if isinstance(ch, dict):
                            if 'content' not in ch:
                                ch = {'content': ch.get('text', '')}
                        else:
                            ch = {'content': str(ch)}
                        normalized_chunks.append(ch)
                    results[class_name][domain].extend(normalized_chunks)
                except Exception as e:
                    print(f"[WARN] Failed to load {cf}: {e}")

    # 요약
    for class_name, per_domain in results.items():
        for domain, items in per_domain.items():
            print(f"  {class_name} - {domain}: {len(items)} chunks loaded")
    return results

def save_chunks(dir_name: str, domain: str, doc_id: str, chunks: List[dict]):
    """청크를 파일로 저장 (dir_name은 lower)"""
    chunk_path = make_chunk_dir(dir_name, domain)
    os.makedirs(chunk_path, exist_ok=True)
    # 파일명 안전화
    safe_doc_id = (
        doc_id.replace('<', '').replace('>', '')
             .replace(':', '_').replace('/', '_').replace('\\', '_')
    )
    # 표준 형식으로 변환
    formatted = []
    for ch in chunks:
        if isinstance(ch, dict):
            if 'content' not in ch:
                ch = {'content': ch.get('text', '')}
        else:
            ch = {'content': str(ch)}
        formatted.append(ch)

    out_path = os.path.join(chunk_path, f'{safe_doc_id}_chunks.json')
    with open(out_path, 'w') as f:
        json.dump(formatted, f, indent=2)

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='BEIR Chunking Benchmark')
    parser.add_argument('--mode', choices=['chunk', 'eval'], default='chunk',
                        help='Mode: "chunk" for chunking+evaluation, "eval" for evaluation only')
    parser.add_argument('--datasets', nargs='+',
                        default=['nfcorpus', 'scifact', 'arguana', 'scidocs', 'fiqa'],
                        help='BEIR datasets to use (split=test)')
    parser.add_argument('--method', nargs='+',
                        choices=list(METHOD_ALIASES.keys()),
                        default=list(METHOD_ALIASES.keys()),
                        help='Chunking methods to use')
    parser.add_argument('--nlp', choices=['kiwi', 'spacy'], default='spacy',
                        help='Tokenizer/NLP backend to use inside chunkers')
    args = parser.parse_args()

    # 로깅
    logging.basicConfig(
        filename='logs/main.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    print("=== BEIR Chunking Benchmark ===")
    print(f"Mode: {args.mode}")
    print(f"Datasets (requested): {args.datasets}")
    print(f"Methods: {args.method}")

    try:
        # 데이터셋 로드
        print("Loading BEIR datasets...")
        beir_datasets_list = args.datasets
        datasets, ground_truths = load_beir_datasets(
            data_dir='data',
            datasets=beir_datasets_list,
            split='test'
        )

        if not datasets:
            logger.warning("No BEIR datasets were loaded. Exiting.")
            print("[ERROR] No BEIR datasets were loaded. Please check availability under ./data/")
            return

        # 로드 결과 요약
        loaded_domains = list(datasets.keys())
        print(f"Loaded BEIR datasets: {loaded_domains}")
        print(f"Ground truths available for: {list(ground_truths.keys())}")

        # Ground truth 저장 (후속 평가 모듈에서 재사용)
        from utils.beir_loader import BEIRDataLoader
        beir_loader = BEIRDataLoader('data')
        os.makedirs('config', exist_ok=True)
        beir_loader.save_ground_truths(ground_truths, 'config/beir_ground_truths.json')

        # 임베딩 초기화
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={
                "batch_size": 5012*2,
                "normalize_embeddings": True,
                }
            )
        
        print("HuggingFace Embeddings initialized.")

        # 메서드 정규화
        normalized_methods = normalize_methods(args.method)
        print("Selected chunking methods:",
            [cn for (cn, _, _) in normalized_methods])

        # 기존 청크 재사용 또는 새로 생성
        if args.mode == 'chunk':
            # chunk 모드: 이미 있으면 재사용, 없으면 생성
            if chunks_exist(datasets, args.method):
                print("Chunks already exist for selected methods/datasets. Loading...")
                existing_chunks = load_existing_chunks(datasets, args.method)
            else:
                existing_chunks = {}
                for class_name, dir_name, Chunker in normalized_methods:
                    logger.info(f"Applying chunking method: {class_name}")
                    print(f"Applying chunking method: {class_name}")
                    chunker = Chunker(embeddings=hf_embeddings, nlp_backend=args.nlp) 
                    existing_chunks[class_name] = {}

                    for domain, documents in datasets.items():
                        print(f"  Processing domain: {domain} ({len(documents)} docs)")
                        existing_chunks[class_name][domain] = []
                        for i, (doc_id, text) in enumerate(documents.items(), 1):
                            try:
                                chunks = chunker.split_text(text, source=domain, file_name=doc_id)
                            except Exception as e:
                                print(f"[WARN] Chunking failed ({domain}/{doc_id}): {e}")
                                chunks = []
                            existing_chunks[class_name][domain].extend(chunks)
                            # 저장
                            save_chunks(dir_name, domain, doc_id, chunks)
                            if i % 100 == 0:
                                print(f"    Processed {i}/{len(documents)}...")
                    logger.info(f"Completed chunking for method: {class_name}")
                    print(f"Completed chunking for method: {class_name}")
                    
        else:
            print("Evaluation-only mode: loading existing chunks...")
            existing_chunks = load_existing_chunks(datasets, args.method)
            if not any(len(v) for v in existing_chunks.values()):
                print("[ERROR] No existing chunks found. Run with --mode chunk first.")
                logger.error("No existing chunks found for evaluation.")
                return
            
            # 청크 크기 평가
            print("Evaluating chunk sizes...")
            chunk_size_metrics = evaluate_chunk_sizes(methods=args.method, datasets=args.datasets)
            print("Chunk size evaluation completed.")

            # 검색 품질 평가ㅊ
            print("Evaluating retrieval quality...")
            retrieval_metrics = evaluate_retrieval_quality(existing_chunks, ground_truths, embeddings=hf_embeddings)
            print("Retrieval quality evaluation completed.")
            
            # 결과 저장
            print("Saving results...")
            # save_results(chunk_size_metrics, retrieval_metrics, scores)
            save_results(chunk_size_metrics, retrieval_metrics)
            print("Results saved successfully.")

            # 시각화
            print("Plotting retrieval quality metrics...")
            plot_retrieval_quality_metrics('results/retrieval_quality_metrics.json')
            print("Plotting scores...")
            # plot_scores('results/scores.json')

            logger.info("Completed BEIR Chunking Benchmark successfully.")
            print("Completed BEIR Chunking Benchmark successfully.")

    except Exception as e:
        logging.error(f"An error occurred during benchmarking: {e}")
        print(f"An error occurred during benchmarking: {e}")
        raise

if __name__ == "__main__":
    main()
