# Structural Chunking: A Semantic–Structural Integrated Method for Retrieval-Augmented Generation

Structural Chunking is a **structure-aware document chunking** method for Retrieval-Augmented Generation (RAG). It integrates **semantic cohesion** (embedding-based continuity) with **structural consistency** (surface- and physical-level layout signals) to detect robust chunk boundaries across diverse document domains.

Unlike purely semantic chunking—which can miss paragraph/section cues and is sensitive to typographical noise—Structural Chunking directly computes **language-agnostic structural indicators** (punctuation/spacing/paragraph patterns) and fuses them with semantic embeddings to produce a **semantic–structural cohesion score**. Chunk boundaries are assigned where the fused discontinuity sharply increases (percentile thresholding).

## Motivation

Chunking quality directly impacts RAG retrieval and answer grounding. Existing methods face a trade-off between **semantic coherence** and **computational efficiency**, and often fail on hierarchically structured documents (e.g., scientific, legal, policy, technical).


<p align="center">
  <img width="438" height="467" alt="image" src="https://github.com/user-attachments/assets/55e6f88b-6598-4e7a-b9a1-1749b58a19f9" />
  <br />
</p>



Structural Chunking addresses this by jointly modeling:
- **Semantic continuity** between consecutive sentences
- **Structural discontinuities** caused by paragraph breaks, formatting shifts, and layout cues

## Method Overview

Structural Chunking consists of three main components:

<p align="center">
  <img width="739" height="319" alt="image" src="https://github.com/user-attachments/assets/f9f22305-90ff-46db-8d9e-45f4262351cf" />
  <br />
</p>

- **Semantic Encoder**: generates sentence embeddings and computes semantic distance `D_sem(i, i+1)`.
- **Structural Feature Extractor**: extracts normalized surface/physical features per sentence and computes structural distance `D_stru(i, i+1)`.
- **Fusion + Boundary Detection**: computes fused distance `D_fusion = α·D_sem + (1-α)·D_stru`, then marks boundaries above the **95th percentile** (default setting in the paper).

### Structural Features

- **Surface**: average sentence length, punctuation distribution/density, token patterns, word-length distribution
- **Physical**: paragraph structure, spacing/newline patterns, character-type ratios, length statistics

<p align="center">
  <img width="426" height="560" alt="image" src="https://github.com/user-attachments/assets/7552b195-8311-4e12-8b9d-7620f09121f8" />
  <br />
</p>


---

## Project Structure
```
src/
├── main.py                                   # Main entrypoint for running experiments
├── model_utils.py                            # Model utility functions
├── chunking_methods/                 
│   ├── __init__.py
│   ├── structural.py                         # Structural Chunking (proposed method)
│   ├── percentile.py                         # Percentile-based chunking
│   ├── std_deviation.py                      # Standard deviation-based chunking
│   ├── interquartile.py                      # Interquartile range-based chunking
│   ├── gradient.py                           # Gradient-based chunking
│   └── recursive.py                          # Recursive chunking
├── evaluation/                       
│   ├── __init__.py
│   ├── retrieval_quality_evaluator_clean.py  # Retrieval quality metrics
│   ├── chunk_size_evaluator.py               # Chunk size distribution analysis
│   └── scoring_system.py                     # Scoring and ranking system
├── utils/                            
│   ├── __init__.py
│   ├── beir_loader.py                        # BEIR dataset loader
│   └── helpers.py                            # Helper functions
└── visualization/                    
    ├── combine_distributions.py              # Combine and plot distributions
    └── flowchart.py                          # Generate method flowcharts
```

## Quick Start

### 1) Create environment
```bash
conda create -n structural-chunking python=3.12
conda activate structural-chunking
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Download BEIR datasets

Download the BEIR datasets and place them in the `data/` directory. Each dataset should follow the BEIR structure:
```
data/
├── nfcorpus/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       └── test.tsv
├── scifact/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       └── test.tsv
...
```

**BEIR benchmark**: https://github.com/beir-cellar/beir

### 4) Run chunking + evaluation

The main entrypoint is `src/main.py`. By default, it runs in **chunk mode** which generates chunks and performs evaluation.

**Example 1: Run all methods on all datasets**
```bash
python -m src.main \\
  --mode chunk \\
  --datasets nfcorpus scifact arguana scidocs fiqa \\
  --method percentile stddeviation interquartile gradient structural recursive
```

**Example 2: Run specific method on specific dataset**
```bash
python -m src.main \\
  --mode chunk \\
  --datasets scifact \\
  --method structural
```

**Example 3: Choose NLP backend (spacy or kiwi)**
```bash
python -m src.main \\
  --mode chunk \\
  --datasets nfcorpus \\
  --method structural \\
  --nlp spacy  # or 'kiwi' for Korean text
```

### 5) Evaluation-only mode (use existing chunks)

If chunks are already generated, you can run evaluation only:
```bash
python -m src.main \\
  --mode eval \\
  --datasets nfcorpus scifact arguana scidocs fiqa \\
  --method percentile stddeviation interquartile gradient structural recursive
```

This will:
- Load existing chunks from `data/chunks/`
- Evaluate chunk size distributions
- Evaluate retrieval quality (Precision, Recall, F1, Average Precision)
- Save results to `results/` directory
- Generate comparison plots

### Output Files

After running the pipeline, you will find:
```
results/
├── chunk_sizes_metrics.json                    # Chunk size statistics
├── retrieval_quality_metrics.json              # Retrieval performance metrics
├── retrieval_quality_comparison.png            # Comparison plot
└── [method]_[dataset]_distribution.png         # Individual histograms

logs/
├── main.log                                    # Main execution log
└── chunk_size_evaluator.log                    # Chunk size evaluation log

config/
└── beir_ground_truths.json                     # Generated ground truth data
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | choice | `chunk` | Mode: `chunk` (chunking + eval) or `eval` (eval only) |
| `--datasets` | list | `[nfcorpus, scifact, arguana, scidocs, fiqa]` | BEIR datasets to use |
| `--method` | list | `[all methods]` | Chunking methods: `percentile`, `stddeviation`, `interquartile`, `gradient`, `structural`, `recursive` |
| `--nlp` | choice | `spacy` | NLP backend: `spacy` or `kiwi` (for Korean) |


## Datasets

This project uses BEIR datasets. Please download datasets following BEIR instructions and set paths as required by the loader.

**BEIR benchmark**: https://github.com/beir-cellar/beir

The following datasets are used in the paper:

- **NFCorpus** (medical web documents/FAQs)
- **SciFact** (scientific claim verification abstracts)
- **ArguAna** (argument–counterargument pairs)
- **SCIDOCS** (scientific document relations)
- **FiQA-2018** (finance Q&A / noisy style)

## Citation

If you use this repository, please cite the paper:
```bibtex
@inproceedings{lee2026structuralchunking,
  title = {Structural Chunking: A Semantic--Structural Integrated Method for Retrieval-Augmented Generation},
  author = {Lee, Sangyong and Kim, NaHun and Lee, Junseok},
  booktitle = {ICEIC 2026 (submitted)},
  year = {2026}
}
```

## Contact

**Sangyong Lee**: sangyong1996@gmail.com

Issues and pull requests are welcome!
