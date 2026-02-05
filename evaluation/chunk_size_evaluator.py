import logging
import statistics
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────
# 로깅 준비
# ─────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/chunk_size_evaluator.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# main과 동일한 메서드 별 매핑 (alias → 표시명, 디렉터리명)
# ─────────────────────────────────────────────────────────
_METHOD_ALIASES = {
    # alias(lower): (display_name / class-like, dir_name(lower))
    'percentile':    ('Percentile',    'percentile'),
    'stddeviation':  ('StdDeviation',  'stddeviation'),
    'interquartile': ('Interquartile', 'interquartile'),
    'recursive':     ('FixedLen',      'fixedlen'),     # ⚠ alias와 dir_name 다름
    'gradient':      ('Gradient',      'gradient'),
    'structural':    ('Structural',    'structural'),
}

# 기본 팔레트 (필요 수만큼 순환 사용)
_DEFAULT_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f1c40f",
    "#1abc9c", "#e67e22", "#34495e", "#7f8c8d", "#2c3e50"
]

# 사람이 읽기 좋은 표시명 기본 매핑
_DEFAULT_DISPLAY_NAMES = {
    "nfcorpus": "NFCorpus",
    "scifact": "SciFact",
    "arguana": "ArguAna",
    "scidocs": "SciDocs",
    "fiqa": "FIQA",
}

def _build_domain_maps(datasets: List[str]):
    """데이터셋 목록을 받아 색상/표시명 딕셔너리를 동적으로 만든다."""
    domain_colors: Dict[str, str] = {}
    domain_display_names: Dict[str, str] = {}
    for i, ds in enumerate(datasets):
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        domain_colors[ds] = color
        if ds in _DEFAULT_DISPLAY_NAMES:
            domain_display_names[ds] = _DEFAULT_DISPLAY_NAMES[ds]
        else:
            pretty = " ".join([w.capitalize() for w in ds.replace("_", " ").replace("-", " ").split()])
            domain_display_names[ds] = pretty if pretty else ds
    return domain_colors, domain_display_names

def _normalize_methods(methods: List[str]) -> List[tuple]:
    """alias 리스트를 (alias, display_name, dir_name) 튜플 리스트로 변환"""
    out = []
    for m in methods:
        key = m.lower()
        if key not in _METHOD_ALIASES:
            raise ValueError(f"Unknown method: {m}")
        display_name, dir_name = _METHOD_ALIASES[key]
        out.append((key, display_name, dir_name))
    return out

def evaluate_chunk_sizes(
    output_path: str = 'results/',
    methods: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    청크 크기 분포를 평가하고 분석하여 히스토그램을 생성

    Args:
        output_path: 결과 저장 경로
        methods: 평가할 청킹 메서드 alias 목록 (예: ['gradient','recursive',...])
        datasets: 평가할 데이터셋 목록 (예: ['nfcorpus','scifact',...])
    """
    try:
        logger.info("Starting evaluation of chunk sizes.")
        metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

        os.makedirs(output_path, exist_ok=True)

        # 메서드/데이터셋 기본값
        if methods is None:
            method_aliases = ['gradient', 'interquartile', 'stddeviation', 'percentile', 'structural', 'recursive']
        else:
            method_aliases = methods

        if datasets is None:
            domains = ['nfcorpus', 'scifact', 'arguana', 'scidocs', 'fiqa']
        else:
            domains = datasets

        # 정규화 (alias → display / dir)
        normalized_methods = _normalize_methods(method_aliases)

        # 도메인별 색상/표시명 구성
        domain_colors, domain_display_names = _build_domain_maps(domains)

        for alias, display_name, dir_name in normalized_methods:
            metrics[alias] = {}
            method_path = os.path.join('data', 'chunks', dir_name)

            for domain in domains:
                domain_path = os.path.join(method_path, domain)
                if not os.path.exists(domain_path):
                    logger.warning(f"Directory not found for method '{alias}' (dir='{dir_name}') in domain '{domain}'. Skipping.")
                    continue

                # 청크 크기 수집
                sizes: List[int] = []
                chunk_files = [f for f in os.listdir(domain_path) if f.endswith('_chunks.json')]

                for chunk_file in chunk_files:
                    try:
                        with open(os.path.join(domain_path, chunk_file), 'r', encoding='utf-8') as f:
                            chunks = json.load(f)
                        for ch in chunks:
                            text = (ch.get('content', '') if isinstance(ch, dict) else str(ch)).strip()
                            if text:
                                sizes.append(len(text.split()))
                    except Exception as e:
                        logger.error(f"Error processing {chunk_file}: {e}")
                        continue

                if not sizes:
                    logger.warning(f"No valid chunks found for method '{alias}' in domain '{domain}'.")
                    continue

                # 메트릭 계산
                metrics[alias][domain] = {
                    'mean_size': statistics.mean(sizes),
                    'median_size': statistics.median(sizes),
                    'std_dev': statistics.stdev(sizes) if len(sizes) > 1 else 0.0,
                    'min_size': min(sizes),
                    'max_size': max(sizes),
                    'total_chunks': float(len(sizes))
                }

                # 히스토그램 생성
                plt.figure(figsize=(10, 6))
                plt.hist(
                    sizes,
                    bins=20,
                    alpha=0.85,
                    color=domain_colors[domain],
                    label=(
                        f'{domain_display_names[domain]}'
                        f'\nMean: {metrics[alias][domain]["mean_size"]:.1f}'
                        f'\nTotal: {int(metrics[alias][domain]["total_chunks"])} chunks'
                    )
                )
                plt.title(f'Chunk Size Distribution - {display_name} Method\n{domain_display_names[domain]}')
                plt.xlabel('Words per Chunk')  # 실제 계산은 단어 수 기준
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.legend()

                ax = plt.gca()
                ax.set_facecolor('#f8f9fa')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # 히스토그램 저장
                histogram_path = os.path.join(output_path, f"{alias}_{domain}_distribution.png")
                plt.savefig(histogram_path, bbox_inches='tight', dpi=300)
                plt.close()

                logger.info(f"Generated histogram for {alias} (dir='{dir_name}') in {domain}")

        # 메트릭 저장
        metrics_file = os.path.join(output_path, 'chunk_sizes_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info("Completed evaluation of chunk sizes.")
        return metrics

    except Exception as e:
        logger.error(f"Error in evaluating chunk sizes: {e}")
        raise

