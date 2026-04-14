"""
Geneformer regressor.
基于Geneformer classifier进行修改, 设计使之也可用于Regressor回归任务
- Cell level regressor: 将cell embedding映射到一个连续变化的状态(e.g. 细胞内某个基因的表达水平)
- Gene level regressor: 在 GF classifier中, 基因分类任务是将一群基因作为一个类别, 进行分类, 而不是某一个基因在不同细胞有不同的类别;
  而对于回归任务, 还没有想好该如何设计


**Input data:**

| Cell state regressor:
| Single-cell transcriptomes as Geneformer rank value encodings with cell state labels in Geneformer .dataset format (generated from single-cell RNAseq data by tokenizer.py)

| Gene regressor:
| Dictionary in format {Gene_label: list(genes)} for gene labels and single-cell transcriptomes as Geneformer rank value encodings in Geneformer .dataset format (generated from single-cell RNAseq data by tokenizer.py)

**Usage:**

.. code-block :: python

    >>> from geneformer import Regressor
    >>> cc = Regressor(regressor="cell",  # example of cell state regressor
    ...                 cell_state_dict={"state_key": "disease", "states": "all"},
    ...                 filter_data={"cell_type":["Cardiomyocyte1","Cardiomyocyte2","Cardiomyocyte3"]},
    ...                 training_args=training_args,
    ...                 freeze_layers = 2,
    ...                 num_crossval_splits = 1,
    ...                 forward_batch_size=200,
    ...                 nproc=16)
    >>> cc.prepare_data(input_data_file="path/to/input_data",
    ...                 output_directory="path/to/output_directory",
    ...                 output_prefix="output_prefix")
    >>> all_metrics = cc.validate(model_directory="path/to/model",
    ...                           prepared_input_data_file=f"path/to/output_directory/{output_prefix}_labeled.dataset",
    ...                           id_class_dict_file=f"path/to/output_directory/{output_prefix}_id_class_dict.pkl",
    ...                           output_directory="path/to/output_directory",
    ...                           output_prefix="output_prefix",
    ...                           predict_eval=True)
    >>> cc.plot_predictions(predictions_file=f"path/to/output_directory/datestamp_geneformer_cellClassifier_{output_prefix}/ksplit1/predictions.pkl",
    ...                     id_class_dict_file=f"path/to/output_directory/{output_prefix}_id_class_dict.pkl",
    ...                     title="disease",
    ...                     output_directory="path/to/output_directory",
    ...                     output_prefix="output_prefix",
    ...                     custom_class_order=["healthy","disease1","disease2"])
"""

import datetime
import logging
import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm, trange
from transformers import Trainer
from transformers.training_args import TrainingArguments

from . import (
    TOKEN_DICTIONARY_FILE,
    DataCollatorForCellClassification,
    DataCollatorForGeneClassification,
)
from . import classifier_utils as cu
from . import evaluation_utils as eu
from . import perturber_utils as pu
from .tme import TmeModeling_utils as tu
from .tme.TmeModeling_bert import TmeBertForMultiGeneExpressionPrediction

sns.set()


logger = logging.getLogger(__name__)


class Regressor:
    valid_option_dict = {
        "regressor": {"cell", "gene"},
        "quantize": {bool, dict},
        "cell_state_dict": {None, dict},
        "gene_class_dict": {None, dict},
        "filter_data": {None, dict},
        "max_ncells": {None, int},
        "training_args": {None, dict},
        "freeze_layers": {int},
        "num_crossval_splits": {0, 1, 5},
        "split_sizes": {None, dict},
        "no_eval": {bool},
        "stratify_splits_col": {None, str},
        "forward_batch_size": {int},
        "token_dictionary_file": {None, str},
        "nproc": {int},
        "ngpu": {int},
        "device": {str},
    }

    def __init__(
        self,
        regressor=None,
        quantize=False,
        cell_state_dict=None,
        gene_class_dict=None,
        filter_data=None,
        max_ncells=None,
        training_args=None,
        freeze_layers=0,
        num_crossval_splits=1,
        split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1},
        stratify_splits_col=None,
        no_eval=False,
        forward_batch_size=100,
        token_dictionary_file=None,
        nproc=4,
        ngpu=1,
        device="cuda:0",
        num_classes=1
    ):
        """
        Initialize Geneformer regressor.

        **Parameters:**

        regressor : {"cell", "gene"}
            | Whether to fine-tune a cell state or gene regressor.
        quantize : bool, dict
            | Whether to fine-tune a quantized model.
            | If True and no config provided, will use default.
            | Will use custom config if provided.
            | Configs should be provided as dictionary of BitsAndBytesConfig (transformers) and LoraConfig (peft).
            | For example: {"bnb_config": BitsAndBytesConfig(...),
            |               "peft_config": LoraConfig(...)}
        cell_state_dict : None, dict
            | Cell states to fine-tune model to distinguish.
            | Two-item dictionary with keys: state_key and states
            | state_key: key specifying name of column in .dataset that defines the states to model
            | states: list of values in the state_key column that specifies the states to model
            | Alternatively, instead of a list of states, can specify "all" to use all states in that state key from input data.
            | Of note, if using "all", states will be defined after data is filtered.
            | Must have at least 2 states to model.
            | For example: {"state_key": "disease",
            |               "states": ["nf", "hcm", "dcm"]}
            |               or
            |               {"state_key": "disease",
            |               "states": "all"}
        gene_class_dict : None, dict
            | Gene classes to fine-tune model to distinguish.
            | Dictionary in format: {Gene_label_A: list(geneA1, geneA2, ...),
            |                        Gene_label_B: list(geneB1, geneB2, ...)}
            | Gene values should be Ensembl IDs.
        filter_data : None, dict
            | Default is to fine-tune with all input data.
            | Otherwise, dictionary specifying .dataset column name and list of values to filter by.
        max_ncells : None, int
            | Maximum number of cells to use for fine-tuning.
            | Default is to fine-tune with all input data.
        training_args : None, dict
            | Training arguments for fine-tuning.
            | If None, defaults will be inferred for 6 layer Geneformer.
            | Otherwise, will use the Hugging Face defaults:
            | https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
            | Note: Hyperparameter tuning is highly recommended, rather than using defaults.
        freeze_layers : int
            | Number of layers to freeze from fine-tuning.
            | 0: no layers will be frozen; 2: first two layers will be frozen; etc.
            | 99: 冻结全部bert参数
            | n [1 : 12]: 冻结encoder的前N层
            | -n [-1 : -12]: 冻结encoder的前N层 + 冻结embedding层
        num_crossval_splits : {0, 1, 5}
            | 0: train on all data without splitting
            | 1: split data into train and eval sets by designated split_sizes["valid"]
            | 5: split data into 5 folds of train and eval sets by designated split_sizes["valid"]
        split_sizes : None, dict
            | Dictionary of proportion of data to hold out for train, validation, and test sets
            | {"train": 0.8, "valid": 0.1, "test": 0.1} if intending 80/10/10 train/valid/test split
        stratify_splits_col : None, str
            | Name of column in .dataset to be used for stratified splitting.
            | Proportion of each class in this column will be the same in the splits as in the original dataset.
        no_eval : bool
            | If True, will skip eval step and use all data for training.
            | Otherwise, will perform eval during training.
        forward_batch_size : int
            | Batch size for forward pass (for evaluation, not training).
        token_dictionary_file : None, str
            | Default is to use token dictionary file from Geneformer
            | Otherwise, will load custom gene token dictionary.
        nproc : int
            | Number of CPU processes to use.
        ngpu : int
            | Number of GPUs available.

        """

        self.regressor = regressor
        if self.regressor == "cell":
            self.model_type = "CellClassifier_TME"   # 可以兼容回归任务 (num_labels == 1)
        elif self.regressor == "gene":
            self.model_type = "GeneClassifier"
        self.quantize = quantize
        self.cell_state_dict = cell_state_dict
        self.gene_class_dict = gene_class_dict
        self.filter_data = filter_data
        self.max_ncells = max_ncells
        self.training_args = training_args
        self.freeze_layers = freeze_layers
        self.num_crossval_splits = num_crossval_splits
        self.split_sizes = split_sizes
        self.train_size = self.split_sizes["train"]
        self.valid_size = self.split_sizes["valid"]
        self.oos_test_size = self.split_sizes["test"]
        # 分为train datasets and test datasets储存，eval size表示train里的evaluation set
        self.eval_size = self.valid_size / (self.train_size + self.valid_size) 
        self.stratify_splits_col = stratify_splits_col
        self.no_eval = no_eval
        self.forward_batch_size = forward_batch_size
        self.token_dictionary_file = token_dictionary_file
        self.nproc = nproc
        self.ngpu = ngpu
        self.device = device
        self.num_classes = num_classes

        if self.training_args is None:
            logger.warning(
                "Hyperparameter tuning is highly recommended for optimal results. "
                "No training_args provided; using default hyperparameters."
            )

        self.validate_options()

        if self.filter_data is None:
            self.filter_data = dict()

        if self.regressor == "cell":
            if self.cell_state_dict["states"] != "all":
                # 进一步增加过滤条件
                self.filter_data[
                    self.cell_state_dict["state_key"]
                ] = self.cell_state_dict["states"]

        # load token dictionary (Ensembl IDs:token)
        if self.token_dictionary_file is None:
            self.token_dictionary_file = TOKEN_DICTIONARY_FILE
        with open(self.token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        self.token_gene_dict = {v: k for k, v in self.gene_token_dict.items()}  #reversal dict -- token id: gene name

        # filter genes for gene regressor for those in token dictionary
        if self.regressor == "gene":
            all_gene_class_values = set(pu.flatten_list(self.gene_class_dict.values())) # merge lists to one list
            missing_genes = [
                gene
                for gene in all_gene_class_values
                if gene not in self.gene_token_dict.keys()
            ]
            if len(missing_genes) == len(all_gene_class_values):
                logger.error(
                    "None of the provided genes to classify are in token dictionary."
                )
                raise
            elif len(missing_genes) > 0:
                logger.warning(
                    f"Genes to classify {missing_genes} are not in token dictionary."
                )
            self.gene_class_dict = {
                k: list(set([self.gene_token_dict.get(gene) for gene in v]))  # replace gene name with token id
                for k, v in self.gene_class_dict.items()
            }
            empty_classes = []
            # 筛选有没有有效基因数为0的class
            for k, v in self.gene_class_dict.items():
                if len(v) == 0:
                    empty_classes += [k]
            if len(empty_classes) > 0:
                logger.error(
                    f"Class(es) {empty_classes} did not contain any genes in the token dictionary."
                )
                raise

    def validate_options(self):
        # confirm arguments are within valid options and compatible with each other
        for attr_name, valid_options in self.valid_option_dict.items():
            attr_value = self.__dict__[attr_name]
            if not isinstance(attr_value, (list, dict)):
                if attr_value in valid_options:
                    continue
            valid_type = False
            for option in valid_options:
                if (option in [int, float, list, dict, bool, str]) and isinstance(
                    attr_value, option
                ):
                    valid_type = True
                    break
            if valid_type:
                continue
            logger.error(
                f"Invalid option for {attr_name}. "
                f"Valid options for {attr_name}: {valid_options}"
            )
            raise

        if self.filter_data is not None:
            for key, value in self.filter_data.items():
                if not isinstance(value, list):
                    self.filter_data[key] = [value]
                    logger.warning(
                        "Values in filter_data dict must be lists. "
                        f"Changing {key} value to list ([{value}])."
                    )

        if self.regressor == "cell":
            if set(self.cell_state_dict.keys()) != set(["state_key", "states"]):
                logger.error(
                    "Invalid keys for cell_state_dict. "
                    "The cell_state_dict should have only 2 keys: state_key and states"
                )
                raise

            if self.cell_state_dict["states"] != "all":
                if not isinstance(self.cell_state_dict["states"], list):
                    logger.error(
                        "States in cell_state_dict should be list of states to model."
                    )
                    raise
                if len(self.cell_state_dict["states"]) < 2:
                    logger.error(
                        "States in cell_state_dict should contain at least 2 states to classify."
                    )
                    raise

        if self.regressor == "gene":
            if len(self.gene_class_dict.keys()) < 2:
                logger.error(
                    "Gene_class_dict should contain at least 2 gene classes to classify."
                )
                raise
        if sum(self.split_sizes.values()) != 1:
            logger.error("Train, validation, and test proportions should sum to 1.")
            raise

    
    def prepare_data(
        self,
        input_data_file,
        output_directory,
        output_prefix,
        split_id_dict=None,
        test_size=None,
        attr_to_split=None,
        attr_to_balance=None,
        max_trials=100,
        pval_threshold=0.1,
    ):
        """
        Prepare data for cell state or gene classification.

        **Parameters**

        input_data_file : Path / Dataset object
            | Path to directory containing .dataset input
        output_directory : Path
            | Path to directory where prepared data will be saved
        output_prefix : str
            | Prefix for output file
        split_id_dict : None, dict 【直接通过设置样本类别划分训练集/测试集】
            | Dictionary of IDs for train and test splits
            | Three-item dictionary with keys: attr_key, train, test
            | attr_key: key specifying name of column in .dataset that contains the IDs for the data splits
            | train: list of IDs in the attr_key column to include in the train split
            | test: list of IDs in the attr_key column to include in the test split
            | For example: {"attr_key": "individual",
            |               "train": ["patient1", "patient2", "patient3", "patient4"],
            |               "test": ["patient5", "patient6"]}
        test_size : None, float
            | Proportion of data to be saved separately and held out for test set
            | (e.g. 0.2 if intending hold out 20%)
            | If None, will inherit from split_sizes["test"] from Classifier
            | The training set will be further split to train / validation in self.validate
            | Note: only available for CellClassifiers
        attr_to_split : None, str
            | Key for attribute on which to split data while balancing potential confounders
            | e.g. "patient_id" for splitting by patient while balancing other characteristics
            | Note: only available for CellClassifiers
        attr_to_balance : None, list
            | List of attribute keys on which to balance data while splitting on attr_to_split
            | e.g. ["age", "sex"] for balancing these characteristics while splitting by patient
            | Note: only available for CellClassifiers
        max_trials : None, int
            | Maximum number of trials of random splitting to try to achieve balanced other attributes
            | If no split is found without significant (p<0.05) differences in other attributes, will select best
            | Note: only available for CellClassifiers
        pval_threshold : None, float
            | P-value threshold to use for attribute balancing across splits
            | E.g. if set to 0.1, will accept trial if p >= 0.1 for all attributes in attr_to_balance
        """

        if test_size is None:
            test_size = self.oos_test_size

        # prepare data and labels for classification
        data = pu.load_and_filter(self.filter_data, self.nproc, input_data_file)

        # 确认数据集先不含有标签列: cell-label, gene-labels [huggingface中常用的key是 labels]
        if self.regressor == "cell":
            if "label" in data.features:
                logger.error(
                    "Column name 'label' must be reserved for class IDs. Please rename column."
                )
                raise
        elif self.regressor == "gene":
            if "labels" in data.features:
                logger.error(
                    "Column name 'labels' must be reserved for class IDs. Please rename column."
                )
                raise

        # 当设置attr_to_split=时，必须设置attr_to_balance
        if (attr_to_split is not None) and (attr_to_balance is None):
            logger.error(
                "Splitting by attribute while balancing confounders requires both attr_to_split and attr_to_balance to be defined."
            )
            raise

        if not isinstance(attr_to_balance, list):
            attr_to_balance = [attr_to_balance]

        if self.regressor == "cell":
            # [SKIP] downsample max cells and max per class
            data = cu.downsample_and_shuffle(
                data, self.max_ncells, max_ncells_per_class = None, cell_state_dict = None
            )
            # rename cell state column to "label"
            data = cu.rename_cols(data, self.cell_state_dict["state_key"])
        # 注意：对于gene regressor, 暂时还没有添加labels数据

        # # [SKIP] convert classes to numerical labels and save as id_class_dict
        # # of note, will label all genes in gene_class_dict
        # # if (cross-)validating, genes will be relabeled in column "labels" for each split
        # # at the time of training with Classifier.validate
        # # 将标签列替换为整型编码（对于基因classifier，会筛选对于在gene_class_dict中的基因具有非零表达的细胞）
        # data, id_class_dict = cu.label_classes(
        #     self.classifier, data, self.gene_class_dict, self.nproc
        # )
        # # {0: 'nf', 1: 'hcm', 2: 'dcm'}

        # # [No Need] save id_class_dict for future reference
        # id_class_output_path = (
        #     Path(output_directory) / f"{output_prefix}_id_class_dict"
        # ).with_suffix(".pkl")
        # with open(id_class_output_path, "wb") as f:
        #     pickle.dump(id_class_dict, f)

        # 根据指定样本类别划分出独立的测试集
        if split_id_dict is not None:
            data_dict = dict()
            data_dict["train"] = pu.filter_by_dict(
                data, {split_id_dict["attr_key"]: split_id_dict["train"]}, self.nproc
            )
            data_dict["test"] = pu.filter_by_dict(
                data, {split_id_dict["attr_key"]: split_id_dict["test"]}, self.nproc
            )
            train_data_output_path = (
                Path(output_directory) / f"{output_prefix}_labeled_train"
            ).with_suffix(".dataset")
            test_data_output_path = (
                Path(output_directory) / f"{output_prefix}_labeled_test"
            ).with_suffix(".dataset")
            data_dict["train"].save_to_disk(str(train_data_output_path))
            data_dict["test"].save_to_disk(str(test_data_output_path))
        # 根据比例，划分出独立的测试集【仅适用于cell classifier | Why not for gene classifier?】
        elif (test_size is not None) and (self.regressor == "cell"):
            if 1 > test_size > 0:
                if attr_to_split is None:
                    data_dict = data.train_test_split(
                        test_size=test_size,
                        stratify_by_column=self.stratify_splits_col,  # 将某一列属性在训练集与测试集中分布平衡
                        seed=42,
                    )
                    train_data_output_path = (
                        Path(output_directory) / f"{output_prefix}_labeled_train"
                    ).with_suffix(".dataset")
                    test_data_output_path = (
                        Path(output_directory) / f"{output_prefix}_labeled_test"
                    ).with_suffix(".dataset")
                    data_dict["train"].save_to_disk(str(train_data_output_path))
                    data_dict["test"].save_to_disk(str(test_data_output_path))
                else:
                    data_dict, balance_df = cu.balance_attr_splits(
                        data,
                        attr_to_split,
                        attr_to_balance,
                        test_size,
                        max_trials,
                        pval_threshold, # p值越大，表明训练集与测试集的分布越相似（差异越小）
                        self.cell_state_dict["state_key"],
                        self.nproc,
                    )
                    balance_df.to_csv(
                        f"{output_directory}/{output_prefix}_train_test_balance_df.csv"
                    )
                    train_data_output_path = (
                        Path(output_directory) / f"{output_prefix}_labeled_train"
                    ).with_suffix(".dataset")
                    test_data_output_path = (
                        Path(output_directory) / f"{output_prefix}_labeled_test"
                    ).with_suffix(".dataset")
                    data_dict["train"].save_to_disk(str(train_data_output_path))
                    data_dict["test"].save_to_disk(str(test_data_output_path))
            else:
                data_output_path = (
                    Path(output_directory) / f"{output_prefix}_labeled"
                ).with_suffix(".dataset")
                data.save_to_disk(str(data_output_path))
                print(data_output_path)
        else:
            data_output_path = (
                Path(output_directory) / f"{output_prefix}_labeled"
            ).with_suffix(".dataset")
            data.save_to_disk(str(data_output_path))

    def train_all_data(
        self,
        model_directory,
        prepared_input_data_file,
        output_directory,
        output_prefix,
        save_eval_output=True,
        gene_balance=False,
    ):
        """
        Train cell state or gene classifier using all data.

        **Parameters**

        model_directory : Path
            | Path to directory containing model
        prepared_input_data_file : Path
            | Path to directory containing _labeled.dataset previously prepared by Classifier.prepare_data
        output_directory : Path
            | Path to directory where model and eval data will be saved
        output_prefix : str
            | Prefix for output files
        save_eval_output : bool
            | Whether to save cross-fold eval output
            | Saves as pickle file of dictionary of eval metrics
        gene_balance : None, bool
            | Whether to automatically balance genes in training set.
            | Only available for binary gene classifications.

        **Output**

        Returns trainer after fine-tuning with all data.

        """
        # # [SKIP] gene_balance参数仅对基因二分类任务有效
        # if (gene_balance is True) and (len(self.gene_class_dict.values()) != 2):
        #     logger.error(
        #         "Automatically balancing gene sets for training is only available for binary gene classifications."
        #     )
        #     raise

        # #####  [SKIP] Load data and prepare output directory #####
        # # load numerical id to class dictionary (id:class)
        # with open(id_class_dict_file, "rb") as f:
        #     id_class_dict = pickle.load(f)
        # class_id_dict = {v: k for k, v in id_class_dict.items()}

        # load previously filtered and prepared data
        data = pu.load_and_filter(None, self.nproc, prepared_input_data_file)
        data = data.shuffle(seed=42)  # reshuffle in case users provide unshuffled data

        # define output directory path
        # current_date = datetime.datetime.now()
        # datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}" # '250125'
        if output_directory[-1:] != "/":  # add slash for dir if not present | 代码很细腻
            output_directory = output_directory + "/"
        # output_dir = f"{output_directory}{datestamp}_geneformer_{self.classifier}Classifier_{output_prefix}/"
        output_dir = f"{output_directory}geneformer_{self.regressor}Regressor_{output_prefix}/"
        subprocess.call(f"mkdir {output_dir}", shell=True)

        # # get number of classes for classifier
        # num_classes = cu.get_num_classes(id_class_dict)

        # # gene_class_dict = {"A":["Gene1","Gene2"], "B":["Gene3","Gene4"]}
        # if self.classifier == "gene":
        #     targets = pu.flatten_list(self.gene_class_dict.values())  #["Gene1","Gene2", "Gene3","Gene4"]
        #     # [0, 0, 1, 1]
        #     labels = pu.flatten_list(
        #         [
        #             [class_id_dict[label]] * len(targets)
        #             for label, targets in self.gene_class_dict.items()
        #         ]
        #     )
        #     assert len(targets) == len(labels)
        #     # 为基因分类任务，添加labels列
        #     data = cu.prep_gene_classifier_all_data(
        #         data, targets, labels, self.max_ncells, self.nproc, gene_balance
        #     )

        trainer = self.train_regressor(
            model_directory, data, None, output_dir
        )

        return trainer

    def validate(
        self,
        model_directory,
        prepared_input_data_file,
        # id_class_dict_file,
        output_directory,
        output_prefix,
        split_id_dict=None,
        attr_to_split=None,
        attr_to_balance=None,
        gene_balance=False,
        max_trials=100,
        pval_threshold=0.1,
        save_eval_output=True,
        predict_eval=True,      # 是否保存evaluate的结果到本地, 一般为True
        # Update: 如果predict_eval为False, 则认为仅使用模型预测, 而不评价
        predict_trainer=False,  # 是否在训练完Trainer后进行预测(trainer.predict), 一般为False
        # save_gene_split_datasets=True,
        # debug_gene_split_datasets=False,
    ):
        """
        (Cross-)validate cell state or gene classifier.

        **Parameters**

        model_directory : Path
            | Path to directory containing model
        prepared_input_data_file : Path
            | Path to directory containing _labeled.dataset previously prepared by Classifier.prepare_data
        id_class_dict_file : Path
            | Path to _id_class_dict.pkl previously prepared by Classifier.prepare_data
            | (dictionary of format: numerical IDs: class_labels)
        output_directory : Path
            | Path to directory where model and eval data will be saved
        output_prefix : str
            | Prefix for output files
        split_id_dict : None, dict
            | Dictionary of IDs for train and eval splits
            | Three-item dictionary with keys: attr_key, train, eval
            | attr_key: key specifying name of column in .dataset that contains the IDs for the data splits
            | train: list of IDs in the attr_key column to include in the train split
            | eval: list of IDs in the attr_key column to include in the eval split
            | For example: {"attr_key": "individual",
            |               "train": ["patient1", "patient2", "patient3", "patient4"],
            |               "eval": ["patient5", "patient6"]}
            | Note: only available for CellClassifiers with 1-fold split (self.classifier="cell"; self.num_crossval_splits=1)
        attr_to_split : None, str
            | Key for attribute on which to split data while balancing potential confounders
            | e.g. "patient_id" for splitting by patient while balancing other characteristics
            | Note: only available for CellClassifiers with 1-fold split (self.classifier="cell"; self.num_crossval_splits=1)
        attr_to_balance : None, list
            | List of attribute keys on which to balance data while splitting on attr_to_split
            | e.g. ["age", "sex"] for balancing these characteristics while splitting by patient
        gene_balance : None, bool
            | Whether to automatically balance genes in training set.
            | Only available for binary gene classifications.
        max_trials : None, int
            | Maximum number of trials of random splitting to try to achieve balanced other attribute
            | If no split is found without significant (p < pval_threshold) differences in other attributes, will select best
        pval_threshold : None, float
            | P-value threshold to use for attribute balancing across splits
            | E.g. if set to 0.1, will accept trial if p >= 0.1 for all attributes in attr_to_balance
        save_eval_output : bool
            | Whether to save cross-fold eval output
            | Saves as pickle file of dictionary of eval metrics
        predict_eval : bool
            | Whether or not to save eval predictions
            | Saves as a pickle file of self.evaluate predictions
        predict_trainer : bool
            | Whether or not to save eval predictions from trainer
            | Saves as a pickle file of trainer predictions
        n_hyperopt_trials : int
            | Number of trials to run for hyperparameter optimization
            | If 0, will not optimize hyperparameters
        save_gene_split_datasets : bool
            | Whether or not to save train, valid, and test gene-labeled datasets
        """
        if self.num_crossval_splits == 0:
            logger.error("num_crossval_splits must be 1 or 5 to validate.")
            raise

        if (gene_balance is True) and (len(self.gene_class_dict.values()) != 2):
            logger.error(
                "Automatically balancing gene sets for training is only available for binary gene classifications."
            )
            raise

        # ensure number of genes in each class is > 5 if validating model
        if self.regressor == "gene":
            insuff_classes = [k for k, v in self.gene_class_dict.items() if len(v) < 5]
            if (self.num_crossval_splits > 0) and (len(insuff_classes) > 0):
                logger.error(
                    f"Insufficient # of members in class(es) {insuff_classes} to (cross-)validate."
                )
                raise

        ##### Load data and prepare output directory #####
        # load numerical id to class dictionary (id:class)
        # with open(id_class_dict_file, "rb") as f:
        #     id_class_dict = pickle.load(f)
        # class_id_dict = {v: k for k, v in id_class_dict.items()}

        # load previously filtered and prepared datasets
        data = pu.load_and_filter(None, self.nproc, prepared_input_data_file)
        data = data.shuffle(seed=42)  # reshuffle in case users provide unshuffled data

        # define output directory path
        # current_date = datetime.datetime.now()
        # datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
        if output_directory[-1:] != "/":  # add slash for dir if not present
            output_directory = output_directory + "/"
        output_dir = f"{output_directory}geneformer_{self.regressor}Regressor_{output_prefix}/"
        subprocess.call(f"mkdir {output_dir}", shell=True)

        # get number of classes for regressor
        # num_classes = cu.get_num_classes(id_class_dict)
        # num_classes = 1

        ##### (Cross-)validate the model #####
        results = []
        iteration_num = 1
        if self.regressor == "cell":
            for i in trange(self.num_crossval_splits):
                print(
                    f"****** Validation split: {iteration_num}/{self.num_crossval_splits} ******\n"
                )
                ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")
                if self.num_crossval_splits == 1:
                    # single 1-eval_size:eval_size split
                    # 首先看是否指定了具体的训练集与验证集样本
                    if split_id_dict is not None:
                        data_dict = dict()
                        data_dict["train"] = pu.filter_by_dict(
                            data,
                            {split_id_dict["attr_key"]: split_id_dict["train"]},
                            self.nproc,
                        )
                        data_dict["test"] = pu.filter_by_dict(
                            data,
                            {split_id_dict["attr_key"]: split_id_dict["eval"]},
                            self.nproc,
                        )
                    # 根据某一列均衡划分
                    elif attr_to_split is not None:
                        data_dict, balance_df = cu.balance_attr_splits(
                            data,
                            attr_to_split,
                            attr_to_balance,
                            self.eval_size,
                            max_trials,
                            pval_threshold,
                            self.cell_state_dict["state_key"],
                            self.nproc,
                        )

                        balance_df.to_csv(
                            f"{output_dir}/{output_prefix}_train_valid_balance_df.csv"
                        )
                    else:
                        data_dict = data.train_test_split(
                            test_size=self.eval_size,
                            stratify_by_column=self.stratify_splits_col,
                            seed=42,
                        )
                    train_data = data_dict["train"]
                    eval_data = data_dict["test"]
                else:
                    # 5-fold cross-validate
                    num_cells = len(data)
                    fifth_cells = int(np.floor(num_cells * 0.2))
                    num_eval = min((self.eval_size * num_cells), fifth_cells)
                    start = i * fifth_cells
                    end = start + int(num_eval)
                    print(start, end, fifth_cells)
                    eval_indices = [j for j in range(start, end)]
                    train_indices = [
                        j for j in range(num_cells) if j not in eval_indices
                    ]
                    eval_data = data.select(eval_indices)
                    train_data = data.select(train_indices)


                trainer = self.train_regressor(
                    model_directory,
                    train_data,
                    eval_data,
                    ksplit_output_dir,
                    predict_trainer,
                )

                print("### eval_data: ", eval_data)
                if predict_eval:
                    print("===> Model evaluating...")
                    result = self.evaluate_model(
                        trainer.model,
                        # num_classes,
                        # id_class_dict,
                        eval_data,
                        predict_eval,       # 是否保存evaluate的结果
                        ksplit_output_dir,  # Fold保存路径
                        output_prefix,      # Fold保存文件名前缀
                    )
                    results += [result]
                else:
                    print("===> Just model predicting...")
                    pred = tu.model_predict(
                        trainer.model,
                        eval_data,
                        self.forward_batch_size,
                        verbose=True,
                    )
                    pred_dict = {
                        "cls_logits": np.array(pred[0]),
                        "reg_preds":  np.array(pred[1]),
                        "labels": np.array(eval_data['label'])
                    }
                    pred_dict_output_path = (
                        Path(ksplit_output_dir) / f"{output_prefix}_pred_dict"
                    ).with_suffix(".pkl")
                    with open(pred_dict_output_path, "wb") as f:
                        pickle.dump(pred_dict, f)

                # all_conf_mat = all_conf_mat + result["conf_mat"]
                iteration_num = iteration_num + 1

        elif self.regressor == "gene":
            raise NotImplementedError

        if len(results) > 0:
            all_metrics = {
                # "y_pred": [result["y_pred"] for result in results],
                # "y_true": [result["y_true"] for result in results],
                "mse": [result["mse"] for result in results],
                "mae": [result["mae"] for result in results],
                "r2": [result["r2"] for result in results],
            }
            if save_eval_output is True:
                eval_metrics_output_path = (
                    Path(output_dir) / f"{output_prefix}_eval_metrics_dict"
                ).with_suffix(".pkl")
                with open(eval_metrics_output_path, "wb") as f:
                    pickle.dump(all_metrics, f)
            return all_metrics

    def train_regressor(
        self,
        model_directory,
        train_data,
        eval_data,
        output_directory,
        predict=False,
    ):
        """
        Fine-tune model for cell state or gene classification.

        **Parameters**

        model_directory : Path
            | Path to directory containing model
        train_data : Dataset
            | Loaded training .dataset input
            | For cell regressor, labels in column "label".
            | For gene regressor, labels in column "labels".
        eval_data : None, Dataset
            | (Optional) Loaded evaluation .dataset input
            | For cell regressor, labels in column "label".
            | For gene regressor, labels in column "labels".
        output_directory : Path
            | Path to directory where fine-tuned model will be saved
        predict : bool
            | Whether or not to save eval predictions from trainer
        """
        num_classes=1

        ##### Load model and training args #####
        # print("Test pu.load_mdoel:", num_classes)
        model = pu.load_model(
            self.model_type,
            self.num_classes,
            model_directory,
            "train",
            quantize=self.quantize,
            device=self.device
        )
        # print(model)


        # e.g. rename colummn "tme_cells1024" to "tme_cells for model to use"
        train_data = tu.modify_tme_dataset(model, train_data)
        if eval_data is not None:
            eval_data = tu.modify_tme_dataset(model, eval_data)

        ##### Validate and prepare data #####
        # 验证数据具有应有的列
        train_data, eval_data = cu.validate_and_clean_cols(
            train_data, eval_data, self.regressor
        )

        if (self.no_eval is True) and (eval_data is not None):
            logger.warning(
                "no_eval set to True; model will be trained without evaluation."
            )
            eval_data = None

        if (self.regressor == "gene") and (predict is True):
            logger.warning(
                "Predictions during training not currently available for gene regressor; setting predict to False."
            )
            predict = False

        # ensure not overwriting previously saved model
        saved_model_test = os.path.join(output_directory, "pytorch_model.bin")
        if os.path.isfile(saved_model_test) is True:
            logger.error("Model already saved to this designated output directory.")
            raise
        # make output directory
        subprocess.call(f"mkdir {output_directory}", shell=True)



        # 获取训练参数和冻结层数
        def_training_args, def_freeze_layers = cu.get_default_train_args(
            model, self.regressor, train_data, output_directory
        )

        if self.training_args is not None:
            def_training_args.update(self.training_args)
        logging_steps = round(
            len(train_data) * def_training_args["num_train_epochs"] / def_training_args["per_device_train_batch_size"] / 10
        ) #每个Epoch，打印10次记录
        def_training_args["logging_steps"] = logging_steps
        def_training_args["output_dir"] = output_directory
        if eval_data is None:
            # train all mode
            print(model)
            # def_training_args["evaluation_ strategy"] = "no"
            def_training_args["eval_strategy"] = "no"
            def_training_args["load_best_model_at_end"] = False
        
        # def_training_args["no_cuda"] = True
        # print(def_training_args)
        
        training_args_init = TrainingArguments(**def_training_args)

        if self.freeze_layers is not None:
            def_freeze_layers = self.freeze_layers

        if def_freeze_layers != 0:
            # 冻结所有bert参数
            if def_freeze_layers == 99:
                for param in model.bert.parameters():
                    param.requires_grad = False
            # 冻结encoder前几层
            else:
                modules_to_freeze = model.bert.encoder.layer[:abs(def_freeze_layers)]
                for module in modules_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False
            # 冻结bert内非bert模块的其它参数
            if def_freeze_layers < 0:
                for name in ["embeddings", "pooler", "cell_bert"]:
                    module = getattr(model.bert, name, None) or getattr(model.bert.encoder, name, None)
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = False


        total_params, frozen_params = 0, 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()

        print(f"===> Frozen parameters: {frozen_params} / {total_params} "
            f"({100 * frozen_params / total_params:.2f}%)")

        ##### Fine-tune the model #####
        # define the data collator
        if self.regressor == "cell":
            data_collator = DataCollatorForCellClassification(
                token_dictionary=self.gene_token_dict
            )
        elif self.regressor == "gene":
            data_collator = DataCollatorForGeneClassification(
                token_dictionary=self.gene_token_dict
            )

        print(train_data)
        print(eval_data)

        # create the trainer
        if isinstance(model, TmeBertForMultiGeneExpressionPrediction):
            # 不太方便计算eval_metrics
            trainer = Trainer(
                model=model,
                args=training_args_init,
                data_collator=data_collator,
                train_dataset=train_data,
                eval_dataset=eval_data, 
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args_init,
                data_collator=data_collator,
                train_dataset=train_data,
                eval_dataset=eval_data,
                compute_metrics=cu.compute_metrics,  # [Acc/F1 metrics] or [Mse/Mae/r2]
            ) 
        # train the regressor
        trainer.train()
        trainer.save_model(output_directory)
        if predict is True:
            # make eval predictions and save predictions and metrics
            predictions = trainer.predict(eval_data)
            prediction_output_path = f"{output_directory}/predictions.pkl"
            with open(prediction_output_path, "wb") as f:
                pickle.dump(predictions, f)
            trainer.save_metrics("eval", predictions.metrics)
        return trainer

    def evaluate_model(
        self,
        model,
        # id_class_dict,
        eval_data,
        predict=False,
        output_directory=None,
        output_prefix=None,
    ):
        """
        Evaluate the fine-tuned model.

        **Parameters**

        model : nn.Module
            | Loaded fine-tuned model (e.g. trainer.model)
        num_classes : int
            | Number of classes for regressor
        id_class_dict : dict
            | Loaded _id_class_dict.pkl previously prepared by regressor.prepare_data
            | (dictionary of format: numerical IDs: class_labels)
        eval_data : Dataset
            | Loaded evaluation .dataset input
        predict : bool
            | Whether or not to save eval predictions
        output_directory : Path
            | Path to directory where eval data will be saved
        output_prefix : str
            | Prefix for output files
        """

        num_classes=1
        ##### Evaluate the model #####
        # labels = id_class_dict.keys()
        labels = None

        print("### eval_data: ", eval_data)
        # print("### model type: ", type(model))
        # print("### model: ", model)

        y_pred, y_true, logits_list = eu.classifier_predict(
            model, self.regressor, eval_data, self.forward_batch_size
        )

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        if predict is True:
            pred_dict = {
                "pred_ids": y_pred,              # 预测label
                "label_ids": y_true,             # 真实label
                "predictions": logits_list,      # 预测概率分数
            }
            pred_dict_output_path = (
                Path(output_directory) / f"{output_prefix}_pred_dict"
            ).with_suffix(".pkl")
            with open(pred_dict_output_path, "wb") as f:
                pickle.dump(pred_dict, f)

        # print(y_pred, y_true, logits_list, num_classes, labels)



        if isinstance(model, TmeBertForMultiGeneExpressionPrediction):
            num_tasks = y_pred.shape[1]
            valid_task_count = 0
            mse, mae, r2 = [], [], []
            for task_idx in range(num_tasks):
                pred = y_pred[:, task_idx]
                target = y_true[:, task_idx]
                mask = target != -1.0
                if mask.sum() > 0:
                    valid_task_count += 1
                    mse_i, mae_i, r2_i = eu.get_metrics(pred[mask], target[mask], logits_list[mask], 
                                                        num_classes, labels)
                else:
                    mse_i, mae_i, r2_i = None, None, None
                mse.append(mse_i)
                mae.append(mae_i)
                r2.append(r2_i)

        else:
            mse, mae, r2 = eu.get_metrics(y_pred, y_true, logits_list, num_classes, labels)

        return {
            # "y_pred": y_pred,
            # "y_true": y_true,
            "mse": mse,
            "mae": mae,
            "r2": r2
        }


    def evaluate_saved_model(
        self,
        model_directory,
        # id_class_dict_file,
        test_data_file,
        output_directory,
        output_prefix,
        predict=True,
    ):
        """
        Evaluate the fine-tuned model.

        **Parameters**

        model_directory : Path
            | Path to directory containing model
        id_class_dict_file : Path
            | Path to _id_class_dict.pkl previously prepared by Classifier.prepare_data
            | (dictionary of format: numerical IDs: class_labels)
        test_data_file : Path
            | Path to directory containing test .dataset
        output_directory : Path
            | Path to directory where eval data will be saved
        output_prefix : str
            | Prefix for output files
        predict : bool
            | Whether or not to save eval predictions
        """

        # # load numerical id to class dictionary (id:class)
        # with open(id_class_dict_file, "rb") as f:
        #     id_class_dict = pickle.load(f)

        # get number of classes for classifier
        # num_classes = cu.get_num_classes(id_class_dict)
        num_classes = 1

        # load previously filtered and prepared data
        test_data = pu.load_and_filter(None, self.nproc, test_data_file)

        # load previously fine-tuned model
        model = pu.load_model(
            self.model_type,
            self.num_classes,
            model_directory,
            "eval",
            quantize=self.quantize,
            device=self.device
        )

        # evaluate the model
        result = self.evaluate_model(
            model,
            # id_class_dict,
            test_data,
            # num_classes=1,
            predict=predict,  # 是否保存预测结果
            output_directory=output_directory,
            output_prefix=output_prefix,
        )
        if not os.path.exists(output_directory):
            raise FileNotFoundError(
                f"Output directory {output_directory} does not exist"
            )
        
        else: 
            all_metrics = result

        test_metrics_output_path = (
            Path(output_directory) / f"{output_prefix}_test_metrics_dict"
        ).with_suffix(".pkl")
        with open(test_metrics_output_path, "wb") as f:
            pickle.dump(all_metrics, f)

        return all_metrics