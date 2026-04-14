# ruff: noqa: F401
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")  # noqa # isort:skip

GENE_MEDIAN_FILE = Path(__file__).parent / "gene_median_dictionary_gc95M.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent / "token_dictionary_gc95M.pkl"
ENSEMBL_DICTIONARY_FILE = Path(__file__).parent / "gene_name_id_dict_gc95M.pkl"
ENSEMBL_MAPPING_FILE = Path(__file__).parent / "ensembl_mapping_dict_gc95M.pkl"

from . import (
    collator_for_classification,
    emb_extractor,
    in_silico_perturber,
    in_silico_perturber_stats,
    pretrainer,
    tokenizer,
)
from .collator_for_classification import (
    DataCollatorForCellClassification,
    DataCollatorForGeneClassification,
)
from .emb_extractor import EmbExtractor, get_embs
from .in_silico_perturber import InSilicoPerturber
from .in_silico_perturber_stats import InSilicoPerturberStats
from .pretrainer import GeneformerPretrainer
from .tokenizer import TranscriptomeTokenizer

from . import classifier  # noqa # isort:skip
from .classifier import Classifier  # noqa # isort:skip
from .regressor import Regressor  # noqa # isort:skip

from . import mtl_classifier  # noqa # isort:skip
from .mtl_classifier import MTLClassifier  # noqa # isort:skip








'''
# 加载模型的示例

import numpy as np
import TMEformer

from TMEformer.tme.TmeModeling_bert import TmeBertForMaskedLM

from TMEformer import TmeModeling_utils as tu




PR_XE_MODELS_DICT = tu.generate_pr_models_dict("checkpoint_xe")

# PR_XE_MODELS_DICT.keys()
model = TmeBertForMaskedLM.from_pretrained(PR_XE_MODELS_DICT["GF_D1120_06"][0])


TMEformer.PR_XE_MODELS_DICT["GF_D0818_02"][0]

from TMEformer import perturber_utils as pu
model = pu.load_model(
    "Pretrained_TME", 
    # "CellClassifier_TME_MultiEXP",
    num_classes = 4, 
    model_directory = PR_XE_MODELS_DICT["GF_D1120_06"][0], 
    mode = "eval", quantize=False, device=f"cuda:0"
)
print(next(model.parameters()).dtype)  

model

import torch

training_args = torch.load(TMEformer.PR_XE_MODELS_DICT["GF_D0818_02"][0] + "/training_args.bin")

# 打印精度相关参数
print("fp16:", training_args.fp16)  


'''