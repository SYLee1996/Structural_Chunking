import logging
import statistics
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Optional, Tuple

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
# 메서드 정규화 매핑
#  - 다양한 입력 형태(alias / dir_name / display_name / snake_case)를 모두 흡수
#    -> (alias, display_name, dir_name) 로 정규화
# ─────────────────────────────────────────────────────────
_METHOD_NORMALIZATION: Dict[str, Tuple[str, str]] = {
    # key(lower) : (display_name, dir_name)
    "percentile": ("Percentile", "percentile"),
    "stddeviation": ("StdDeviation", "stddeviation"),
    "std_deviation": ("StdDeviation", "stddeviation"),
    "interquartile": ("Interquartile", "interquartile"),
    "gradient": ("Gradient", "gradient"),
    "structural": ("Structural", "structural"),
    "recursive": ("FixedLen", "fixedlen"),   # alias → dir
    "fixedlen": ("FixedLen", "fixedlen"),    # dir → dir
    # display_name도 허용
    "percentilechunker": ("Percentile", "percentile"),
    "stddeviationchunker": ("StdDeviation", "stddeviation"),
    "interquartilechunker": ("Interquartile", "interquartile"),
    "gradientchunker": ("Gradient", "gradient"),
    "structuralchunker": ("Structural", "structural"),
    "fixedlenchunker": ("FixedLen", "fixedlen"),
    "fixedlenchunk": ("FixedLen", "fixedlen"),
    "std deviation": ("StdDeviation", "stddeviation"),
}

def _normalize_methods(methods: List[str]) -> List[Tuple[str, str, str]]:
    """
    입력 메서드 이름 목록을 (alias, display_name, dir_name) 리스트로 정규화.
    alias는 호출/저장에서 쓸 key, dir_name은 실제 폴더명.
    """
    norm = []
    seen = set()
    for m in methods:
        key = m.strip().lower().replace(" ", "").replace("-", "").replace("__", "_")
        if key not in _METHOD_NORMALIZATION:
            raise ValueError(f"Unknown method: {m}")
        display_name, dir_name = _METHOD_NORMALIZATION[key]
        alias = key  # 파일명/JSON 키로는 입력 alias를 유지(충돌 방지)
        # 동일 dir_name이라도 서로 다른 alias로 두 번 들어오면 한 번만 처리
        dedup = (alias, display_name, dir_name)
        if dedup not in seen:
            norm.append(dedup)
            seen.add(dedup)
    return norm

# ─────────────────────────────────────────────────────────
# 색상/표시명
# ─────────────────────────────────────────────────────────
_DEFAULT_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f1c40f",
    "#1abc9c", "#e67e22", "#34495e", "#7f8c8d", "#2c3e50"
]

_DEFAULT_DISPLAY_NAMES = {
    "nfcorpus": "NFCorpus",
    "scifact": "SciFact",
    "arguana": "ArguAna",
    "scidocs": "SciDocs",
    "fiqa": "FIQA",
    "trec-covid": "TREC-COVID",
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

# ─────────────────────────────────────────────────────────
# 본 함수
# ─────────────────────────────────────────────────────────
def evaluate_chunk_sizes(
    output_path: str = 'results/',
    methods: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    청크 크기 분포를 평가하고 분석하여 히스토그램을 생성

    Args:
        output_path: 결과 저장 경로
        methods: 평가할 청킹 메서드 목록 (alias/dir/display 혼합 입력 허용)
        datasets: 평가할 데이터셋 목록
    """
    try:
        logger.info("Starting evaluation of chunk sizes.")
        metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        os.makedirs(output_path, exist_ok=True)

        # 기본값
        if methods is None:
            method_aliases = ['gradient', 'interquartile', 'stddeviation', 'percentile', 'structural', 'fixedlen']
        else:
            method_aliases = methods

        if datasets is None:
            domains = ['nfcorpus', 'scifact', 'arguana', 'scidocs', 'fiqa']
        else:
            domains = datasets

        # 메서드 정규화
        normalized_methods = _normalize_methods(method_aliases)

        # 도메인별 색상/표시명
        domain_colors, domain_display_names = _build_domain_maps(domains)

        for alias, display_name, dir_name in normalized_methods:
            metrics[alias] = {}
            method_path = os.path.join('data', 'chunks', dir_name)

            for domain in domains:
                domain_path = os.path.join(method_path, domain)
                if not os.path.exists(domain_path):
                    logger.warning(
                        f"Directory not found for method '{alias}' (dir='{dir_name}') in domain '{domain}'. Skipping."
                    )
                    continue

                # 청크 크기 수집
                sizes: List[int] = []
                try:
                    files = [f for f in os.listdir(domain_path) if f.endswith('_chunks.json')]
                except Exception as e:
                    logger.error(f"Failed to list directory {domain_path}: {e}")
                    continue

                for chunk_file in files:
                    fpath = os.path.join(domain_path, chunk_file)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            chunks = json.load(f)
                        for ch in chunks:
                            text = (ch.get('content', '') if isinstance(ch, dict) else str(ch)).strip()
                            if text:
                                sizes.append(len(text.split()))  # 단어 수 기준
                    except Exception as e:
                        logger.error(f"Error processing {fpath}: {e}")
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

                # 히스토그램 저장 (alias를 파일명에 사용: CLI 입력과 1:1 대응)
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

