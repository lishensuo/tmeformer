# TMEformer

TMEformer is a model that extends [Geneformer](https://github.com/amva13/geneformer) by incorporating the tumor microenvironment (TME) context. While Geneformer focuses on modeling individual cells in isolation, TMEformer considers the influence of neighboring cells on tumor cell states, enabling a more comprehensive understanding of tumor biology.

## Overview

TMEformer introduces multiple **in silico perturbation (ISP)** scenarios to help discover and characterize endogenous factors and microenvironmental signals that influence tumor development and progression. By modeling cell-cell interactions within the TME, the model can predict how perturbations in the microenvironment affect tumor cell states.

## TME Modeling Module (`src/TMEformer/tme/`)

The core TME modeling utilities are located in `src/TMEformer/tme/`:

### BERT-based Architecture (`TmeModeling_bert.py`)
- **CellBertEmbeddings**: Converts TME cell information into 3D embedding representations by combining base cell embeddings, cell type embeddings, and position embeddings
- **CellBertModel**: BERT encoder for TME embeddings with attention or mean pooling
- **TmeBertEmbeddings**: GeneFormer-style input embeddings with TME context fusion
- **TmeBertEncoder/Model**: BERT encoder with optional cross-attention for TME integration, supporting two modes:
  - `"fuse"`: TME context is fused with cell embeddings at the input layer
  - `"cross"`: TME context is integrated via cross-attention in specific transformer layers
- **Downstream Task Heads**:
  - `TmeBertForMaskedLM`: Masked language modeling for pretraining
  - `TmeBertForSequenceClassification`: Sequence classification
  - `TmeBertForMultiGeneExpressionPrediction`: Multi-task regression for gene expression
  - `TmeBertForCellClassification`: Cell type classification

### In Silico Perturbation Utilities

| File | Description |
|------|-------------|
| `TmeModeling_utils.py` | General utilities for TME modeling (tokenization, model prediction, etc.) |
| `TmeModeling_utils_isp_cell.py` | Cell-level perturbation utilities (perturb single cell's gene expression) |
| `TmeModeling_utils_isp_ds.py` | Dataset utilities for ISP (generate ISP score sets, calculate scores) |
| `TmeModeling_utils_isp_gep.py` | Gene expression perturbation analysis (CV evaluation, background gene preparation, ISP score calculation) |
| `TmeModeling_utils_isp_lst.py` | List-based perturbation utilities |
| `TmeModeling_utils_isp_pipe.py` | ISP pipeline orchestration (`TME_ISPipe` class) |
| `TmeModeling_utils_isp_sim.py` | Cell embedding similarity perturbation analysis |
| `TmeModeling_utils_prep.py` | Data preparation utilities |

## CLI Module (`src/TMEformer/cli/`)

Command-line interface scripts for running various analyses:

| Script | Description |
|--------|-------------|
| `emb_isp_target.py` | **Cell Embedding Similarity Perturbation (Target)**: Predicts the change of cell embedding similarity for Target-Rank ISP. Analyzes how perturbing specific genes affects cell embedding similarity. |
| `emb_isp_tme.py` | **Cell Embedding Similarity Perturbation (TME)**: Predicts the change of cell embedding similarity for TME-based ISP. Analyzes how perturbing TME composition/rank affects cell embedding similarity. |
| `gep_ft_model.py` | **Gene Expression Profiling Fine-tuning**: Fine-tunes models for gene expression prediction tasks with cross-validation evaluation. |
| `gep_isp_target.py` | **Gene Expression Perturbation (Target)**: Predicts gene expression changes when perturbing target genes in cells. Supports single gene and gene combination perturbations. |
| `gep_isp_tme.py` | **Gene Expression Perturbation (TME)**: Predicts gene expression changes when perturbing TME composition or cell rank. Supports composition-based and rank-based TME perturbation methods. |

## Installation

```bash
pip install .
```

## Usage

See the [documentation](https://tmeformer.readthedocs.io/) for detailed usage instructions, or refer to the CLI scripts in `src/TMEformer/cli/` for examples.

## Citation

If you use TMEformer in your research, please cite our work.

## License

This project is licensed under the MIT License - see the LICENSE file for details.