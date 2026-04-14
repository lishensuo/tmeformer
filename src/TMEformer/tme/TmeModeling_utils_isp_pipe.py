"""
TMEformer ISP Pipeline Utilities
================================

This module provides pipeline utilities for TME (Tumor Microenvironment) In-Silico Perturbation analysis, including:
- Data preparation and initialization
- ISP (In-Silico Perturbation) execution
- Statistics calculation
- Custom state handling

Key Classes:
    - TME_ISPipe: Pipeline for TME In-Silico Perturbation analysis

Example:
    >>> pipe = TME_ISPipe(
    ...     proj="xenium",
    ...     task_dir="./task_test",
    ...     sample_meta_groups=sample_meta_groups,
    ...     model_config=model_config,
    ...     tme_datasets_dict=tme_datasets_dict
    ... )
    >>> pipe.initiate_data()
    >>> pipe.run_isp(gene_symbol=["AR"], direction="Normal2Tumor")
    >>> pipe.stat_isp()
"""

import os
import pickle
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_from_disk

from . import TmeModeling_utils as tu
from . import TmeModeling_utils_isp_lst as tu_isp_lst
from . import TmeModeling_utils_isp_ds as tu_isp_ds
from . import TmeModeling_utils_isp_cell as tu_isp_cell


# =============================================================================
# TME ISP Pipeline
# =============================================================================


class TME_ISPipe:
    """
    Pipeline for TME (Tumor Microenvironment) In-Silico Perturbation analysis.

    Main functionalities:
        1. Prepare states dataset and embedding for ISP operation
        2. Run ISP (In-Silico Perturbation) on target cells
        3. Calculate statistics for ISP scores
        4. Support both cell-level and TME-level ISP

    Attributes:
        proj: Project name.
        task_dir: ISP task directory.
        sample_meta_groups: Sample meta groups of different states.
        model_config: Model configuration dictionary.
        tme_datasets_dict: Dict with key as dataset version name and value as dataset file.
        sample2id_dict_file: Dict file with key as sample name and value as sample id.
        isp_cell_ratio: Ratio of cells to be perturbed.
        isp_run_config: ISP run configuration.
        cell_tme_dict: Cell TME dictionary (for TME ISP).
        cell_cluster_dict: Cell cluster dictionary (for TME ISP).
        custom_state_config: Custom state configuration.
        logger: Logger object.
    """

    def __init__(
        self,
        proj: str,
        task_dir: str,
        sample_meta_groups: Dict[str, List],
        model_config: Dict[str, Any],
        tme_datasets_dict: Dict[str, str],
        sample2id_dict_file: Optional[str] = None,
        isp_cell_ratio: Optional[float] = None,
        isp_run_config: Optional[Dict[str, Any]] = None,
        cell_tme_dict: Optional[Dict[int, int]] = None,
        cell_cluster_dict: Optional[Dict[int, int]] = None,
        custom_state_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the TME_ISPipe class and prepare the directory environment.

        Args:
            proj: Project name.
            task_dir: The ISP task directory.
            sample_meta_groups: The sample meta groups of different states.
                Example with PID (tuple format, multi-groups of isp operation):
                {
                    "Tumor": [("C-038-T",), ("C-039-T",)],
                    "Normal": [("C-038-N",), ("C-039-N",)],
                    "PID": ["C-038", "C-039"]
                }
                Example without PID (simple list format, single group of isp operation):
                {
                    "Tumor": ["C-038-T", "C-039-T"],
                    "Normal": ["C-038-N", "C-039-N"]
                }
            model_config: Model configuration dictionary.
            tme_datasets_dict: Dict with key as dataset version name and value as dataset file.
            sample2id_dict_file: Dict file with key as sample name and value as sample id.
                Example: "/dataSSD7T/liss/work/scPCa/scdata/xenium/processed/xenium_sample_id_dict.pkl"
            isp_cell_ratio: Ratio of cells to be perturbed.
            isp_run_config: ISP run configuration.
            cell_tme_dict: Cell TME dictionary (for TME ISP).
            cell_cluster_dict: Cell cluster dictionary (for TME ISP).
            custom_state_config: Custom state configuration.
            logger: Logger object.

        Example:
            >>> pipe = TME_ISPipe(
            ...     proj="xenium",
            ...     task_dir="./task_test",
            ...     sample_meta_groups=sample_meta_groups,
            ...     model_config=model_config,
            ...     tme_datasets_dict=tme_datasets_dict
            ... )
        """
        self.proj = proj
        self._validate_and_create_dirs(task_dir, custom_task=False)
        self.sample_meta_groups = self._prepare_sample_meta_groups(sample_meta_groups)
        self.model_config = model_config
        self.tme_datasets_dict = tme_datasets_dict
        self.sample2id_dict_file = sample2id_dict_file
        self.isp_cell_ratio = isp_cell_ratio
        self.isp_run_config = isp_run_config
        self.cell_tme_dict = cell_tme_dict
        self.cell_cluster_dict = cell_cluster_dict

        self.custom_state_config = custom_state_config
        if self.custom_state_config is not None:
            self.custom_state_dirname = (
                f"custom_state_{custom_state_config['sample_state']}"
            )
            self._validate_and_create_dirs(
                task_dir + "/" + self.custom_state_dirname,
                custom_task=True
            )

        if logger is not None:
            self.logger = logger
        else:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                ch.setFormatter(formatter)
                logger.addHandler(ch)
            self.logger = logger

    def _validate_and_create_dirs(
        self,
        task_dir: str,
        custom_task: bool = True,
    ) -> None:
        """
        Validate and create necessary directories.

        Args:
            task_dir: Task directory path.
            custom_task: Whether this is a custom task.
        """
        if not os.path.exists(task_dir):
            os.makedirs(task_dir, exist_ok=True)

        if not custom_task:
            self.task_dir = task_dir
            self.task_dataset_dir = os.path.join(task_dir, "dataset")
            self.task_embed_dir = os.path.join(task_dir, "embed")
            self.task_output_dir = os.path.join(task_dir, "output")
            self.task_stat_dir = os.path.join(task_dir, "stat")

            for d in [
                self.task_dataset_dir,
                self.task_embed_dir,
                self.task_output_dir,
                self.task_stat_dir,
            ]:
                os.makedirs(d, exist_ok=True)
        else:
            self.custom_task_dataset_dir = os.path.join(task_dir, "dataset")
            self.custom_task_embed_dir = os.path.join(task_dir, "embed")
            self.custom_task_output_dir = os.path.join(task_dir, "output")
            self.custom_task_stat_dir = os.path.join(task_dir, "stat")

            for d in [
                self.custom_task_dataset_dir,
                self.custom_task_embed_dir,
                self.custom_task_output_dir,
                self.custom_task_stat_dir,
            ]:
                os.makedirs(d, exist_ok=True)

    def _prepare_sample_meta_groups(
        self,
        groups: Dict[str, List],
    ) -> Dict[str, List]:
        """
        Prepare sample meta groups by adding PID if not present.

        Args:
            groups: Sample meta groups dictionary.

        Returns:
            Prepared sample meta groups.
        """
        if "PID" not in groups:
            for group, samples in groups.items():
                groups[group] = [tuple(samples)]
            groups["PID"] = ["State"]
        return groups

    def initiate_data(
        self,
        sample_max_cells: int = 1000,
        model_ids: Optional[List[str]] = None,
        split_file: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Prepare the states dataset and embedding for ISP operation.

        Args:
            sample_max_cells: Maximum number of cells per sample.
            model_ids: List of model ids to calculate model-specific embeddings.
            split_file: Optional file for data splitting.
            force: Whether to force regeneration of existing files.
        """
        if model_ids is None:
            model_ids = ["GF_PR"]

        self.model_ids = model_ids
        tme_versions = [
            self.model_config["model_dict"][model_id][1] for model_id in model_ids
        ]
        tme_versions = sorted(list(set(tme_versions)))

        self.logger.info("#  Step1  : Prepare dataset and embedding.")
        self.logger.info("## Step1-1: Start to save state dataset.")

        for tme_version in tme_versions:
            ds_raw_file = self.tme_datasets_dict[tme_version]
            ds_out_dir = f"{self.task_dataset_dir}/{tme_version}"
            if not os.path.exists(ds_out_dir):
                os.makedirs(ds_out_dir)
            ds_output_file = f"{ds_out_dir}/isp.dataset"

            if split_file is None:
                self.logger.info(f"==> Save path: {ds_output_file}")
                if not force and os.path.exists(ds_output_file):
                    self.logger.info(
                        f"==> [Skipped]: Dataset already existed. ({ds_output_file})"
                    )
                else:
                    tu_isp_ds.group_ft_dataset(
                        self.sample_meta_groups,
                        self.sample2id_dict_file,
                        ds_raw_file,
                        ds_output_file,
                        sample_max_cells,
                    )
            else:
                split_df = pd.read_csv(split_file)
                ds_raw = load_from_disk(ds_raw_file)
                split_df_index = pd.merge(
                    # As pd.merge() will drop index, need to preserve index first
                    pd.DataFrame({"cell_id": ds_raw["cell_id"]}).reset_index(drop=False),
                    split_df,
                ).set_index("index")
                split_dict = (
                    split_df[["isp_split", "group"]]
                    .drop_duplicates()
                    .set_index("isp_split")["group"]
                    .to_dict()
                )

                # {"from": "LowG", "to": "HighG"}
                for split in ["from", "to"]:
                    ds_raw_split = ds_raw.select(
                        split_df_index[split_df_index["isp_split"] == split].index
                    )
                    ds_raw_split = ds_raw_split.add_column(
                        "group",
                        [split_dict[split]] * len(ds_raw_split),
                    )

                    ds_output_file_split = ds_output_file.replace(
                        "isp.dataset",
                        f"isp_{split}.dataset",
                    )
                    self.logger.info(f"==> Save path: {ds_output_file_split}")

                    if not force and os.path.exists(ds_output_file):
                        self.logger.info(
                            f"==> [Skipped]: Dataset already existed. "
                            f"({ds_output_file_split})"
                        )
                    else:
                        tu_isp_ds.group_ft_dataset(
                            self.sample_meta_groups,
                            self.sample2id_dict_file,
                            ds_raw_split,
                            ds_output_file_split,
                            sample_max_cells,
                        )

        self.logger.info("## Step1-2: Start to calculate state embedding.")

        for model_id in model_ids:
            embed_out_dir = f"{self.task_embed_dir}/{model_id}"
            os.makedirs(embed_out_dir, exist_ok=True)
            embed_out_file = (
                f"{embed_out_dir}/L{abs(self.model_config['embed_layer'])}_states_mean.pickle"
            )
            self.logger.info(f"==> Save path: {embed_out_file}")

            if not force and os.path.exists(embed_out_file):
                self.logger.info("==> [Skipped]: Embedding already existed.")
                continue

            tu_isp_ds.embed_ft_dataset(
                self.task_embed_dir,
                self.task_dataset_dir,
                self.sample_meta_groups,
                model_id=model_id,
                model_config=self.model_config,
            )

        if self.custom_state_config is not None:
            self.logger.info("## Step1-3: Start to prepare custom state.")
            self.custom_group_and_embed_ft_dataset(
                custom_state_config=self.custom_state_config,
                model_ids=model_ids,
                model_config=self.model_config,
                ds_output_dir=self.custom_task_dataset_dir,
                embed_output_dir=self.custom_task_embed_dir,
            )

    @staticmethod
    def custom_group_and_embed_ft_dataset(
        custom_state_config: Dict[str, Any],
        model_ids: List[str],
        model_config: Dict[str, Any],
        ds_output_dir: str,
        embed_output_dir: str,
    ) -> None:
        """
        Create custom state datasets and embeddings.

        Args:
            custom_state_config: Custom state configuration.
            model_ids: List of model IDs.
            model_config: Model configuration.
            ds_output_dir: Dataset output directory.
            embed_output_dir: Embedding output directory.
        """
        sample_state = custom_state_config["sample_state"]
        sample_meta_groups = {sample_state: [(sample_state,)]}
        sample_meta_groups["PID"] = [sample_state]
        # {'TKO': [('TKO',)], 'PID': ['TKO']}

        # 1) states datasets
        tme_versions = [
            model_config["model_dict"][model_id][1] for model_id in model_ids
        ]
        tme_versions = sorted(list(set(tme_versions)))

        for tme_version in tme_versions:
            tu_isp_ds.group_ft_dataset(
                sample_meta_groups=sample_meta_groups,
                sample_dict_file=custom_state_config["sample2id_dict_file"],
                ds_raw_file=custom_state_config["tme_datasets_dict"][tme_version],
                ds_output_file=f"{ds_output_dir}/{tme_version}/isp.dataset",
                sample_max_cells=custom_state_config["sample_max_cells"],
            )

        # 2) states embedding
        for model_id in model_ids:
            model_embed_output_dir = f"{embed_output_dir}/{model_id}"
            os.makedirs(model_embed_output_dir, exist_ok=True)
            tu_isp_ds.embed_ft_dataset(
                task_embed_dir=embed_output_dir,
                task_dataset_dir=ds_output_dir,
                sample_meta_groups=sample_meta_groups,
                model_id=model_id,
                model_config=model_config,
            )

    def run_isp(
        self,
        gene_symbol: Optional[List[str]] = None,
        direction: str = "Normal2Tumor",
        force: bool = False,
    ) -> None:
        """
        Run the ISP operation.

        Args:
            gene_symbol: List of gene symbols to be perturbed.
            direction: Direction of ISP operation (e.g., "Normal2Tumor").
            force: Whether to force regeneration of existing files.

        Raises:
            ValueError: If KO and KI are both True in isp_run_config.
        """
        if gene_symbol is None:
            gene_symbol = ["AR"]

        self.logger.info("#  Step2  : Run the isp.")

        cell_ratio = self.isp_cell_ratio
        isp_run_config = self.isp_run_config

        if isp_run_config["ko_method"] and isp_run_config["ki_method"]:
            raise ValueError("KO and KI cannot be both True.")

        if not hasattr(self, "isp_run_pickle_files"):
            self.isp_run_pickle_files = []

        task = self.task_dir.split("/")[-1].replace("task_", "")
        embed_layer = abs(self.model_config["embed_layer"])
        name_suffix = tu_isp_ds.isp_run_suffix(isp_run_config, isp="cell")

        # e.g. "ADT2CRPC_ADT>CRPC-Target_Rank-L1-S1000-W30_EP1_KO0_KI0-ARHGAP35"
        out_name = (
            f"{task}_{direction}-Target_Rank-L{embed_layer}-S{cell_ratio}-"
            f"{name_suffix}-{'+'.join(gene_symbol)}.pickle"
        )

        if out_name not in self.isp_run_pickle_files:
            self.isp_run_pickle_files.append(out_name)

        for model_id in self.model_ids:
            # Check the dirs
            if self.custom_state_config is None:
                out_dir = f"{self.task_output_dir}/{model_id}/target_rank/"
            else:
                out_dir = f"{self.custom_task_output_dir}/{model_id}/target_rank/"

            os.makedirs(out_dir, exist_ok=True)
            out_path = f"{out_dir}/{out_name}"
            self.logger.info(f"==> Save path: {out_path}.")

            if not force and os.path.exists(out_path):
                self.logger.info("==> [Skipped]: Output file already existed.")
                continue

            gene_token = tu.symbol2token(gene_symbol)
            start_state, goal_state = direction.split(">")

            if self.custom_state_config is not None:
                goal_states_embed_file = (
                    f"{self.task_dir}/{self.custom_state_dirname}/embed/"
                    f"{model_id}/L{embed_layer}_states_mean.pickle"
                )

                with open(goal_states_embed_file, "rb") as f:
                    goal_states_embed = pickle.load(f)[goal_state]

                if not isinstance(goal_states_embed, list):
                    raise ValueError("custom_goal_states_embed must be list")

                if len(goal_states_embed) == 1:
                    goal_states_embed = (
                        goal_states_embed * len(self.sample_meta_groups["PID"])
                    )
                elif len(goal_states_embed) != len(self.sample_meta_groups["PID"]):
                    raise ValueError(
                        "custom_goal_states_embed unmatched number of PIDs"
                    )
            else:
                embed_file = (
                    f"{self.task_embed_dir}/{model_id}/"
                    f"L{embed_layer}_states_mean.pickle"
                )
                with open(embed_file, "rb") as f:
                    goal_states_embed = pickle.load(f)[goal_state]

            model, dataset = tu_isp_ds.load_model_and_dataset(
                model_id, self.task_dataset_dir, self.model_config
            )

            sims2goal_samples = {}
            for i, pid in enumerate(self.sample_meta_groups["PID"]):
                # {state_key: sample_name}
                start_state_dict = {
                    start_state: self.sample_meta_groups[start_state][i]
                }
                goal_state_embed = goal_states_embed[i]

                existed = (
                    isp_run_config["ki_method"] is None
                )  # None means not do KI
                # Filter cells datasets (1) belong to start sample; (2) express the isp gene
                token_dataset = tu_isp_ds.filter_token_dataset(
                    dataset,
                    start_state_dict,
                    gene_token,
                    existed=existed,
                    ratio=cell_ratio,
                )

                sims2goal_cells = {}
                if self.custom_state_config is None:
                    cur_direction = (
                        f"{self.sample_meta_groups[start_state][i]} -> "
                        f"{self.sample_meta_groups[goal_state][i]}"
                    )
                else:
                    cur_direction = (
                        f"{self.sample_meta_groups[start_state][i]} -> {goal_state}"
                    )

                self.logger.info(
                    f"## Start model {model_id} | {gene_symbol} | {cur_direction}"
                )
                self.logger.info(
                    f"==> ISP cells: {len(token_dataset)} of {len(dataset)} "
                    f"are perturbed"
                )

                for j in range(len(token_dataset)):
                    if (j + 1) % 100 == 0:
                        self.logger.info(
                            f"==> Processing cell {j+1}/{len(token_dataset)}"
                        )

                    example_cell = token_dataset.select([j])
                    shifted_cell_dataset, shifted_cell_emb = tu_isp_lst.perturb_one_cell(
                        example_cell,
                        gene_token,
                        model,
                        isp_run_config,
                        batch_size=self.model_config["batch_size"],
                        embed_layer=self.model_config["embed_layer"],
                    )

                    sims2goal_relative = tu_isp_ds.calc_relative_similarity(
                        shifted_cell_emb,
                        shifted_cell_dataset["offset"],
                        goal_state_embed,
                    )
                    sims2goal_cells[example_cell["cell_id"][0]] = sims2goal_relative

                sims2goal_samples[pid] = sims2goal_cells

            with open(out_path, "wb") as f:
                pickle.dump(sims2goal_samples, f)

    def stat_isp(
        self,
        direction: str = "Normal2Tumor",
        interval_dict: Optional[Dict[str, List]] = None,
        force: bool = False,
    ) -> None:
        """
        Calculate statistics for ISP scores.

        Args:
            direction: Direction of ISP operation (e.g., "Normal2Tumor").
            interval_dict: Dictionary defining intervals for different score types.
            force: Whether to force regeneration of existing files.
        """
        self.logger.info("#  Step3  : Stat the isp.")

        if interval_dict is None:
            interval_dict = {
                "OE": [[0, 1]],
                "KD": [[0, 1]],
                "KO": [[0, 1]],
                "KI": [None],
            }

        isp_run_config = self.isp_run_config
        start_state = direction.split(">")[0]
        isp_score_sets = tu_isp_ds.generate_cell_isp_score_sets(isp_run_config)
        isp_names = self.isp_run_pickle_files

        for isp_name in isp_names:
            gene_symbol = isp_name.split(".")[0].split("-", 5)[-1].split("+")
            gene_token = tu.symbol2token(gene_symbol)

            for model_id in self.model_ids:
                self.logger.info(f"## Start model {model_id}")

                if self.custom_state_config is None:
                    isp_file = (
                        f"{self.task_output_dir}/{model_id}/target_rank/{isp_name}"
                    )
                    stat_dir = f"{self.task_stat_dir}/{model_id}/target_rank"
                else:
                    isp_file = (
                        f"{self.custom_task_output_dir}/{model_id}/"
                        f"target_rank/{isp_name}"
                    )
                    stat_dir = f"{self.custom_task_stat_dir}/{model_id}/target_rank"

                if not os.path.exists(isp_file):
                    self.logger.info(f"==> [Skipped] ISP file not exists: {isp_file}")
                    continue
                else:
                    with open(isp_file, "rb") as f:
                        sims2goal_stat_merge = pickle.load(f)

                if not os.path.exists(stat_dir):
                    os.makedirs(stat_dir, exist_ok=True)

                stat_file = f"{stat_dir}/{isp_name.replace('.pickle', '.csv')}"
                self.logger.info(f"==> Save path: {stat_file}")

                if not force and os.path.exists(stat_file):
                    self.logger.info("==> [Skipped] Stat file already exists.")
                    continue

                model, dataset = tu_isp_ds.load_model_and_dataset(
                    model_id, self.task_dataset_dir, self.model_config
                )

                score_merge_overall = []
                for score_type, score_method in isp_score_sets:
                    score_merge_samples = []

                    for i, (pid, sims2goal_stat_one) in enumerate(
                        sims2goal_stat_merge.items()
                    ):
                        start_state_dict = {
                            start_state: self.sample_meta_groups[start_state][i]
                        }

                        self.logger.info(
                            f"==> Processing {pid} | {score_type} | {score_method}"
                        )

                        score_one_sample: Dict[str, List] = defaultdict(list)
                        for interval in interval_dict[score_type]:
                            if interval is not None:
                                # OE, KD, KO
                                filt_token_dataset = tu_isp_ds.filter_token_dataset(
                                    dataset,
                                    start_state_dict,
                                    gene_token,
                                    existed=True,
                                    interval=interval,
                                )
                            else:
                                # KI
                                filt_token_dataset = tu_isp_ds.filter_token_dataset(
                                    dataset,
                                    start_state_dict,
                                    gene_token,
                                    existed=False,
                                )

                            interval_cell_ids = set(
                                sims2goal_stat_one.keys()
                            ).intersection(filt_token_dataset["cell_id"])
                            cell_id_scores = []

                            for cell_id in interval_cell_ids:
                                cell_id_dict = {
                                    x[0]: [x[1]] for x in sims2goal_stat_one[cell_id]
                                }
                                cell_id_score = tu_isp_ds.calc_isp_score(
                                    cell_id_dict,
                                    score_type=score_type,
                                    score_method=score_method,
                                )
                                cell_id_scores.append(cell_id_score)

                            score_one_sample["model_id"].extend(
                                [model_id] * len(cell_id_scores)
                            )
                            score_one_sample["sample_id"].extend(
                                [pid] * len(cell_id_scores)
                            )
                            score_one_sample["interval"].extend(
                                [str(interval)] * len(cell_id_scores)
                            )
                            score_one_sample["cell_score"].extend(cell_id_scores)
                            score_one_sample["cell_id"].extend(interval_cell_ids)

                        score_one_sample_df = pd.DataFrame(score_one_sample)
                        score_merge_samples.append(score_one_sample_df)

                    score_merge_samples_df = pd.concat(score_merge_samples)
                    score_merge_samples_df["score_type"] = score_type
                    score_merge_samples_df["score_method"] = score_method
                    score_merge_overall.append(score_merge_samples_df)

                score_merge_overall_df = pd.concat(score_merge_overall)
                score_merge_overall_df.to_csv(stat_file, index=False)

    def run_isp_tme(
        self,
        tme_method: str = "composition",
        isp_cluster_id: int = 1,
        direction: str = "Normal2Tumor",
        force: bool = False,
        fixed_cluster_ids: Optional[List[int]] = None,
        gene_symbol: Optional[List[str]] = None,
        resample: bool = True,
    ) -> None:
        """
        Run the TME ISP operation.

        Args:
            tme_method: TME method to use ('composition' or 'rank').
            isp_cluster_id: ISP cluster identifier.
            direction: Direction of ISP operation (e.g., "Normal2Tumor").
            force: Whether to force regeneration of existing files.
            fixed_cluster_ids: Fixed cluster IDs (composition-specific).
            gene_symbol: Gene symbols for perturbation (rank-specific).
            resample: Whether to resample (rank-specific).

        Raises:
            ValueError: If cell_tme_dict or cell_cluster_dict is None,
                or if model does not use TME.
        """
        self.logger.info("#  Step2  : Run the tme isp.")

        if tme_method == "composition":
            self.fixed_cluster_ids = fixed_cluster_ids
        elif tme_method == "rank":
            self.gene_symbol = gene_symbol
            self.resample = resample
            print(f"### resample: {self.resample}")
            gene_token = tu.symbol2token(gene_symbol)

        self.isp_cluster_id = isp_cluster_id
        if not hasattr(self, "isp_run_pickle_files"):
            self.isp_run_pickle_files = []

        cell_ratio = self.isp_cell_ratio
        isp_run_config = self.isp_run_config
        cell_tme_dict = self.cell_tme_dict
        cell_cluster_dict = self.cell_cluster_dict

        if cell_tme_dict is None:
            raise ValueError("cell_tme_dict is None")
        if cell_cluster_dict is None:
            raise ValueError("cell_cluster_dict is None")

        task = self.task_dir.split("/")[-1].replace("task_", "")
        embed_layer = abs(self.model_config["embed_layer"])
        name_suffix = tu_isp_ds.isp_run_suffix(isp_run_config, isp="tme")
        self.tme_method_label = f"TME_{tme_method.title()}"
        pre_out_name = (
            f"{task}_{direction}-{self.tme_method_label}-L{embed_layer}-"
            f"S{cell_ratio}-{name_suffix}"
        )

        if tme_method == "composition":
            fixed_labels = (
                "all"
                if fixed_cluster_ids is None
                else "+".join(str(i) for i in fixed_cluster_ids)
            )
            out_name = f"{pre_out_name}-TME{isp_cluster_id}_FIX{fixed_labels}.pickle"
        elif tme_method == "rank":
            out_name = (
                f"{pre_out_name}-TME{isp_cluster_id}_{'+'.join(gene_symbol)}.pickle"
            )

        if out_name not in self.isp_run_pickle_files:
            self.isp_run_pickle_files.append(out_name)

        for model_id in self.model_ids:
            # Check the dirs
            if self.custom_state_config is None:
                out_dir = (
                    f"{self.task_output_dir}/{model_id}/"
                    f"{self.tme_method_label.lower()}"
                )
            else:
                out_dir = (
                    f"{self.custom_task_output_dir}/{model_id}/"
                    f"{self.tme_method_label.lower()}"
                )

            os.makedirs(out_dir, exist_ok=True)
            out_path = f"{out_dir}/{out_name}"
            self.logger.info(f"==> Save path: {out_path}.")

            if not force and os.path.exists(out_path):
                self.logger.info("==> [Skipped]: Output file already existed.")
                continue

            start_state, goal_state = direction.split(">")

            if self.custom_state_config is not None:
                goal_states_embed_file = (
                    f"{self.task_dir}/{self.custom_state_dirname}/embed/"
                    f"{model_id}/L{embed_layer}_states_mean.pickle"
                )

                with open(goal_states_embed_file, "rb") as f:
                    goal_states_embed = pickle.load(f)[goal_state]

                if not isinstance(goal_states_embed, list):
                    raise ValueError("custom_goal_states_embed must be list")

                if len(goal_states_embed) == 1:
                    goal_states_embed = (
                        goal_states_embed * len(self.sample_meta_groups["PID"])
                    )
                elif len(goal_states_embed) != len(self.sample_meta_groups["PID"]):
                    raise ValueError(
                        "custom_goal_states_embed unmatched number of PIDs"
                    )
            else:
                embed_file = (
                    f"{self.task_embed_dir}/{model_id}/"
                    f"L{embed_layer}_states_mean.pickle"
                )
                with open(embed_file, "rb") as f:
                    goal_states_embed = pickle.load(f)[goal_state]

            model, dataset = tu_isp_ds.load_model_and_dataset(
                model_id, self.task_dataset_dir, self.model_config
            )

            if not hasattr(model.config, "use_tme") or not model.config.use_tme:
                raise ValueError(f"Model {model_id} does not use TME.")

            # For TME_Composition ISP, add cluster_type column:
            # when 1) cluster_id is one of main types, directly copy tme_types
            # when 2) cluster_id is sub_type id, need to generate new cluster_type
            # column according to cell_cluster_dict
            dataset = tu_isp_ds.add_cluster_types(
                dataset, isp_cluster_id, cell_cluster_dict
            )

            sims2goal_samples = {}
            for i, pid in enumerate(self.sample_meta_groups["PID"]):
                start_state_dict = {
                    start_state: self.sample_meta_groups[start_state][i]
                }
                goal_state_embed = goal_states_embed[i]

                if tme_method == "composition":
                    token_dataset = tu_isp_ds.filter_tme_composition_dataset(
                        dataset,
                        start_state_dict,
                        isp_cluster_id,
                        fixed_cluster_ids,
                        existed=True,
                        ratio=cell_ratio,
                        cluster_endpoint=isp_run_config["endpoints"],
                    )
                elif tme_method == "rank":
                    # First, filter the start state samples that contain the target cluster
                    token_dataset = tu_isp_ds.filter_tme_composition_dataset(
                        dataset, start_state_dict, isp_cluster_id, existed=True
                    )
                    # Then, filter the tme cells that contain the target gene
                    token_dataset = tu_isp_ds.filter_tme_rank_dataset(
                        token_dataset,
                        isp_cluster_id,
                        cell_cluster_dict,
                        proj=self.proj,
                        tokens=gene_token,
                        existed=True,
                        ratio=cell_ratio,
                    )

                if self.custom_state_config is None:
                    cur_direction = (
                        f"{self.sample_meta_groups[start_state][i]} -> "
                        f"{self.sample_meta_groups[goal_state][i]}"
                    )
                else:
                    cur_direction = (
                        f"{self.sample_meta_groups[start_state][i]} -> {goal_state}"
                    )

                self.logger.info(f"## Start model {model_id} | {cur_direction}")
                self.logger.info(
                    f"==> ISP cells: {len(token_dataset)} of {len(dataset)} "
                    f"are perturbed"
                )

                sims2goal_cells = {}
                for j in range(len(token_dataset)):
                    if (j + 1) % 100 == 0:
                        self.logger.info(
                            f"==> Processing cell {j+1}/{len(token_dataset)}"
                        )

                    example_cell = token_dataset.select([j])

                    if tme_method == "composition":
                        shifted_cell_dataset, shifted_cell_emb = (
                            tu_isp_cell.perturb_one_tme_cells_composition(
                                example_cell,
                                isp_cluster_id,
                                fixed_cluster_ids,
                                model,
                                cell_tme_dict,
                                cell_cluster_dict,
                                isp_run_config,
                                batch_size=self.model_config["batch_size"],
                                embed_layer=self.model_config["embed_layer"],
                            )
                        )
                    elif tme_method == "rank":
                        shifted_cell_dataset, shifted_cell_emb = (
                            tu_isp_cell.perturb_one_tme_cells_rank(
                                example_cell,
                                isp_cluster_id,
                                gene_token,
                                model,
                                cell_cluster_dict,
                                isp_run_config,
                                resample=resample,
                                emb_type="cell",
                            )
                        )

                    sims2goal_relative = tu_isp_ds.calc_relative_similarity(
                        shifted_cell_emb,
                        shifted_cell_dataset["offset"],
                        goal_state_embed,
                    )
                    sims2goal_cells[example_cell["cell_id"][0]] = sims2goal_relative

                sims2goal_samples[pid] = sims2goal_cells

            with open(out_path, "wb") as f:
                pickle.dump(sims2goal_samples, f)

    def stat_isp_tme(
        self,
        tme_method: str = "composition",
        direction: str = "Normal2Tumor",
        interval_dict: Optional[Dict[str, List]] = None,
        force: bool = False,
    ) -> None:
        """
        Calculate statistics for TME ISP scores.

        Args:
            tme_method: TME method to use ('composition' or 'rank').
            direction: Direction of ISP operation (e.g., "Normal2Tumor").
            interval_dict: Dictionary defining intervals for different score types.
            force: Whether to force regeneration of existing files.
        """
        self.logger.info("#  Step3  : Stat the tme isp.")

        start_state = direction.split(">")[0]

        isp_run_config = self.isp_run_config
        isp_score_sets = tu_isp_ds.generate_tme_isp_score_sets(
            isp_run_config, tme_method
        )

        isp_run_pickle_files = self.isp_run_pickle_files

        for isp_run_pickle_file in isp_run_pickle_files:
            for model_id in self.model_ids:
                self.logger.info(f"## Start model {model_id}")

                if self.custom_state_config is None:
                    isp_file = (
                        f"{self.task_output_dir}/{model_id}/"
                        f"{self.tme_method_label.lower()}/{isp_run_pickle_file}"
                    )
                    stat_dir = (
                        f"{self.task_stat_dir}/{model_id}/"
                        f"{self.tme_method_label.lower()}"
                    )
                else:
                    isp_file = (
                        f"{self.custom_task_output_dir}/{model_id}/"
                        f"{self.tme_method_label.lower()}/{isp_run_pickle_file}"
                    )
                    stat_dir = (
                        f"{self.custom_task_stat_dir}/{model_id}/"
                        f"{self.tme_method_label.lower()}"
                    )

                if not os.path.exists(isp_file):
                    self.logger.info(f"==> [Skipped] ISP file not exists: {isp_file}")
                    continue
                else:
                    with open(isp_file, "rb") as f:
                        sims2goal_stat_merge = pickle.load(f)

                if not os.path.exists(stat_dir):
                    os.makedirs(stat_dir, exist_ok=True)

                stat_file = f"{stat_dir}/{isp_run_pickle_file.replace('.pickle', '.csv')}"
                self.logger.info(f"==> Save path: {stat_file}")

                if not force and os.path.exists(stat_file):
                    self.logger.info("==> [Skipped] Stat file already exists.")
                    continue

                model, dataset = tu_isp_ds.load_model_and_dataset(
                    model_id, self.task_dataset_dir, self.model_config
                )

                dataset = tu_isp_ds.add_cluster_types(
                    dataset, self.isp_cluster_id, self.cell_cluster_dict
                )

                score_merge_overall = []
                for score_type, score_method in isp_score_sets:
                    score_merge_samples = []

                    for i, (pid, sims2goal_stat_one) in enumerate(
                        sims2goal_stat_merge.items()
                    ):
                        start_state_dict = {
                            start_state: self.sample_meta_groups[start_state][i]
                        }

                        self.logger.info(
                            f"==> Processing {pid} | {score_type} | {score_method}"
                        )

                        score_one_sample: Dict[str, List] = defaultdict(list)
                        fixed_cluster_ids = None

                        for interval in interval_dict[score_type]:
                            if interval is not None:
                                filt_tme_dataset = tu_isp_ds.filter_tme_composition_dataset(
                                    dataset,
                                    start_state_dict,
                                    self.isp_cluster_id,
                                    fixed_cluster_ids,
                                    existed=True,
                                    interval=interval,
                                    cluster_endpoint=isp_run_config["endpoints"],
                                )
                            else:
                                filt_tme_dataset = tu_isp_ds.filter_tme_composition_dataset(
                                    dataset,
                                    start_state_dict,
                                    self.isp_cluster_id,
                                    fixed_cluster_ids,
                                    existed=False,
                                    cluster_endpoint=isp_run_config["endpoints"],
                                )

                            interval_cell_ids = set(
                                sims2goal_stat_one.keys()
                            ).intersection(filt_tme_dataset["cell_id"])
                            cell_id_scores = []

                            for cell_id in interval_cell_ids:
                                cell_id_dict = {
                                    x[0]: [x[1]] for x in sims2goal_stat_one[cell_id]
                                }
                                cell_id_score = tu_isp_ds.calc_isp_score(
                                    cell_id_dict,
                                    score_type=score_type,
                                    score_method=score_method,
                                )
                                cell_id_scores.append(cell_id_score)

                            score_one_sample["model_id"].extend(
                                [model_id] * len(cell_id_scores)
                            )
                            score_one_sample["sample_id"].extend(
                                [pid] * len(cell_id_scores)
                            )
                            score_one_sample["interval"].extend(
                                [str(interval)] * len(cell_id_scores)
                            )
                            score_one_sample["cell_score"].extend(cell_id_scores)
                            score_one_sample["cell_id"].extend(interval_cell_ids)

                        score_one_sample_df = pd.DataFrame(score_one_sample)
                        score_merge_samples.append(score_one_sample_df)

                    score_merge_samples_df = pd.concat(score_merge_samples)
                    score_merge_samples_df["score_type"] = score_type
                    score_merge_samples_df["score_method"] = score_method
                    score_merge_overall.append(score_merge_samples_df)

                score_merge_overall_df = pd.concat(score_merge_overall)
                score_merge_overall_df.to_csv(stat_file, index=False)
                print(
                    "The isp score result has been saved to: \n===> ", stat_file
                )