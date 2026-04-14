"""
TMEformer Spatial Data Preprocessing
====================================

This module provides utilities for spatial data preprocessing, including:
- Single-cell data annotation and tokenization
- TME (Tumor Microenvironment) cell sampling
- Gene expression median computation
- Cell embedding extraction

Key Classes:
    - SpatialDataProcessor: Single-cell data processor for annotation and tokenization
    - TMECellSampler: TME cell sampler for defining neighborhood cells
    - InitEmbExtractor: Cell embedding extractor based on Geneformer

Example:
    >>> processor = SpatialDataProcessor(work_dir="./", proj_name="xenium")
    >>> processor.run(tokenize=True)
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import crick
import faiss
import numpy as np
import pandas as pd
import scanpy as sc
from datasets import load_from_disk
from scipy import sparse
from tqdm import tqdm

from .. import ENSEMBL_MAPPING_FILE, TOKEN_DICTIONARY_FILE
from .. import EmbExtractor, TranscriptomeTokenizer
from . import TmeModeling_utils as tu


# =============================================================================
# Spatial Data Processor
# =============================================================================


class SpatialDataProcessor:
    """
    Single-cell data processor for annotation and tokenization.

    Main functionalities:
        1. Annotate adata.obs and adata.var
        2. Generate tokenized datasets

    Attributes:
        work_dir: Working directory path.
        proj_name: Project name.
        sample_col: Column name for sample identification.
        cell_type_col: Column name for cell type identification.
        spatial_cols: Column names for spatial coordinates.
        use_new_tme_id: Whether to create new TME ID dictionary.
        reference_tme_dict: Reference project for TME ID dictionary.
    """

    def __init__(
        self,
        work_dir: str,
        proj_name: str,
        sample_col: str = "sample_name",
        cell_type_col: str = "cell_type",
        spatial_cols: List[str] = ["spatial_1", "spatial_2"],
        use_new_tme_id: bool = False,
        reference_tme_dict: Optional[Union[str, dict]] = "xenium",
    ) -> None:
        """
        Initialize the processor.

        Args:
            work_dir: Working directory path.
            proj_name: Project name.
            sample_col: Column name for sample identification.
            cell_type_col: Column name for cell type identification.
            spatial_cols: Column names for spatial coordinates.
            use_new_tme_id: Whether to create new TME ID dictionary.
            reference_tme_dict: Reference project for TME ID dictionary or direct TME ID dictionary.

        Example:
            >>> processor = SpatialDataProcessor(
            ...     work_dir="./",
            ...     proj_name="xenium",
            ...     use_new_tme_id=False,
            ...     reference_tme_dict="xenium"
            ... )
        """
        self.work_dir = Path(work_dir)
        self.proj_name = proj_name
        self.sample_col = sample_col
        self.cell_type_col = cell_type_col
        self.spatial_cols = spatial_cols
        self.use_new_tme_id = use_new_tme_id
        self.reference_tme_dict = reference_tme_dict

        # Define paths
        self.processed_dir = self.work_dir / proj_name / "processed"
        self.dataset_dir = self.work_dir / proj_name / "datasets"
        self.h5ad_file = self.processed_dir / f"{proj_name}.h5ad"

        # Create directories if not exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data containers
        self.adata = None
        self.sample_id_dict = None
        self.tme_id_dict = None
        self.gene_median_dict = None

    def load_data(self, adata_path: Optional[str] = None) -> sc.AnnData:
        """
        Load h5ad data.

        Args:
            adata_path: Path to h5ad file. If None, uses default path.

        Returns:
            Loaded AnnData object.

        Raises:
            AssertionError: If 'counts' layer not found.
        """
        if adata_path is None:
            adata_path = self.h5ad_file

        print(f"Loading data from {adata_path}")
        self.adata = sc.read_h5ad(adata_path)

        # Convert to counts
        assert "counts" in self.adata.layers, "'counts' not found in adata.layers"
        if "lognorm" not in self.adata.layers:
            print(
                "Note: adata.X is considered as lognorm layer by default. "
                "Please make sure X is lognorm layer"
            )
            self.adata.layers["lognorm"] = self.adata.X.copy()

        self.adata.X = self.adata.layers["counts"].copy()

        return self.adata

    def annotate_obs(self) -> None:
        """
        Annotate adata.obs with cell_id, sample_id, tme_id.

        Raises:
            ValueError: If cell_type column not specified or n_counts not found.
        """
        print("Annotating adata.obs...")

        # Add cell_id
        self.adata.obs["cell_id"] = range(1, self.adata.shape[0] + 1)

        # Create sample_id mapping
        if self.sample_col is None:
            print("No sample name provided, using project name as sample name")
            self.adata.obs["sample_name"] = self.proj_name
        elif self.sample_col != "sample_name":
            self.adata.obs["sample_name"] = self.adata.obs[self.sample_col]

        self.sample_id_dict = {
            name: i for i, name in enumerate(self.adata.obs["sample_name"].unique())
        }
        self.adata.obs["sample_id"] = self.adata.obs["sample_name"].map(
            self.sample_id_dict
        )

        # Save sample_id_dict
        sample_dict_file = self.processed_dir / f"{self.proj_name}_sample_id_dict.pkl"
        with open(sample_dict_file, "wb") as f:
            pickle.dump(self.sample_id_dict, f)
        print(f"Saved sample_id_dict to {sample_dict_file}")

        # Handle cell_type column
        if self.cell_type_col is None:
            raise ValueError("cell_type must be specified")
        elif self.cell_type_col != "cell_type":
            self.adata.obs["cell_type"] = self.adata.obs[self.cell_type_col]

        # Handle TME ID dictionary
        self._setup_tme_id_dict()

        # Add tme_id
        if not set(self.adata.obs["cell_type"].unique()).issubset(
            self.tme_id_dict.keys()
        ):
            print(
                "tme_id of unknown cell types will be set 0:",
                set(self.adata.obs["cell_type"].unique())
                - set(self.tme_id_dict.keys()),
            )
        self.adata.obs["tme_id"] = (
            self.adata.obs["cell_type"]
            .map(self.tme_id_dict)
            .astype(float)
            .fillna(0)
            .astype(int)
        )

        # Handle n_counts
        if "n_counts" not in self.adata.obs.columns:
            if "total_counts" in self.adata.obs.columns:
                self.adata.obs = self.adata.obs.rename(
                    columns={"total_counts": "n_counts"}
                )
            else:
                raise ValueError("Neither n_counts nor total_counts found in obs")

        print(f"Annotated {self.adata.shape[0]} cells")

    def _setup_tme_id_dict(self) -> None:
        """
        Setup TME ID dictionary.

        Raises:
            FileNotFoundError: If reference TME dict not found when use_new_tme_id is False.
        """
        tme_id_dict_file = self.processed_dir / f"{self.proj_name}_tme_id_dict.pkl"

        if not self.use_new_tme_id:
            # Copy from reference project
            ref_file = (
                self.work_dir
                / self.reference_tme_dict
                / "processed"
                / f"{self.reference_tme_dict}_tme_id_dict.pkl"
            )
            if ref_file.exists():
                shutil.copy(ref_file, tme_id_dict_file)
                with open(tme_id_dict_file, "rb") as f:
                    self.tme_id_dict = pickle.load(f)
                print(f"Copied TME ID dict from {ref_file}")
            else:
                raise FileNotFoundError(f"Reference TME dict not found: {ref_file}")
        else:
            # Create new TME ID dictionary
            self.tme_id_dict = self.reference_tme_dict
            with open(tme_id_dict_file, "wb") as f:
                pickle.dump(self.tme_id_dict, f)
            print(f"Created new TME ID dict at {tme_id_dict_file}")

    def annotate_var(self) -> None:
        """
        Annotate adata.var with n_cells_by_counts and ensembl_id.
        """
        print("Annotating adata.var...")

        # Calculate n_cells_by_counts
        if "n_cells_by_counts" not in self.adata.var.columns:
            X = self.adata.X
            if sparse.issparse(X):
                n_cells = np.asarray((X > 0).sum(axis=0)).ravel()
            else:
                n_cells = (X > 0).sum(axis=0)
            self.adata.var["n_cells_by_counts"] = n_cells

        zero_genes = self.adata.var["n_cells_by_counts"] == 0
        print(f"Filtering {(zero_genes).sum()} genes with 0 count.")
        if (zero_genes).sum() > 0:
            self.adata = self.adata[:, ~zero_genes].copy()

        # Map to ensembl_id
        with open(ENSEMBL_MAPPING_FILE, "rb") as f:
            ensembl_mapping = pickle.load(f)

        self.adata.var["ensembl_id"] = [
            ensembl_mapping.get(gene.upper(), gene) for gene in self.adata.var.index
        ]

        print(f"Annotated {self.adata.shape[1]} genes")

    def save_obsmeta(
        self,
        spatial_cols: List[str] = ["spatial_1", "spatial_2"],
        patch_size: Optional[int] = None,
    ) -> None:
        """
        Save observation metadata to CSV.

        Args:
            spatial_cols: Column names for spatial coordinates.
            patch_size: Patch size for computing patch IDs. If None, skip patch computation.
        """
        print("Saving observation metadata...")

        obsmeta = self.adata.obs.reset_index()
        obsmeta = obsmeta.rename(columns={"index": "barcode"})

        # Select relevant columns
        cols = [
            "barcode",
            "cell_id",
            "sample_name",
            "sample_id",
            "cell_type",
            "tme_id",
        ]

        if not spatial_cols == ["spatial_1", "spatial_2"]:
            obsmeta.rename(
                columns={spatial_cols[0]: "spatial_1", spatial_cols[1]: "spatial_2"},
                inplace=True,
            )

        cols.extend(["spatial_1", "spatial_2"])
        assert set(cols).issubset(
            obsmeta.columns
        ), f"Missing columns: {set(cols) - set(obsmeta.columns)}"

        obsmeta = obsmeta.loc[:, cols]

        # Annotate patch id
        if patch_size is not None:
            obsmeta = obsmeta.groupby("sample_id", group_keys=False).apply(
                tu.compute_patch_ids, patch_size=patch_size, include_groups=False
            )
            obsmeta = obsmeta.rename(columns={"patch_id": f"patch_{patch_size}"})

        obsmeta_file = self.processed_dir / f"{self.proj_name}_obsmeta.csv"
        obsmeta.to_csv(obsmeta_file, index=False)
        print(f"Saved obsmeta to {obsmeta_file}")

        # Save cell_cluster_dict
        cell_cluster_dict = obsmeta.set_index("cell_id")["tme_id"].to_dict()
        cell_cluster_dict_file = (
            self.processed_dir / f"{self.proj_name}_cell_cluster_main_dict.pkl"
        )
        with open(cell_cluster_dict_file, "wb") as f:
            pickle.dump(cell_cluster_dict, f)

    def save_varmeta(self) -> None:
        """
        Save variable metadata (gene information) to CSV.
        """
        with open(ENSEMBL_MAPPING_FILE, "rb") as f:
            gene_mapping_obj = pickle.load(f)

        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
            token_dictionary_obj = pickle.load(f)

        varmeta = self.adata.var.reset_index().rename(columns={"index": "Gene"})[
            ["Gene"]
        ]
        varmeta["ENSEMBL"] = varmeta["Gene"].map(gene_mapping_obj)

        token_ids = [
            token_dictionary_obj.get(gene_mapping_obj.get(ensembl, -999), -999)
            for ensembl in varmeta["ENSEMBL"]
        ]
        varmeta["TOKEN"] = token_ids
        varmeta.to_csv(
            self.processed_dir / f"{self.proj_name}_gene_ids.csv", index=False
        )

        drop_genes = varmeta[varmeta["TOKEN"] == -999].shape[0]
        print(f"{drop_genes} / {varmeta.shape[0]} genes not have matched token ids")

    def compute_gene_medians(self, gene_chunk_size: int = 128) -> Optional[str]:
        """
        Compute gene expression medians using CPM normalization.

        Args:
            gene_chunk_size: Number of genes to process at once.

        Returns:
            Path to gene median file, or None if already exists.
        """
        print("Computing gene expression medians...")

        gene_median_file = (
            self.processed_dir / f"{self.proj_name}_gene_median_dict.pickle"
        )
        if os.path.exists(gene_median_file):
            print("Gene medians already computed: ", gene_median_file)
            return None

        # Load token dictionary
        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
            token_dictionary_obj = pickle.load(f)

        adata = self.adata

        # Filter genes based on ensembl_id
        gene_ids = adata.var["ensembl_id"].values
        coding_miRNA_loc = np.where(
            np.isin(gene_ids, list(token_dictionary_obj.keys()))
        )[0]
        coding_miRNA_genes = gene_ids[coding_miRNA_loc]

        # Initialize TDigest for each gene
        median_digests = [
            crick.tdigest.TDigest() for _ in range(len(coding_miRNA_loc))
        ]
        n_counts = adata.obs["n_counts"].values.astype(np.float32)

        # Process genes in chunks
        progress = tqdm(total=len(coding_miRNA_loc), desc="Processing genes")
        for start in range(0, len(coding_miRNA_loc), gene_chunk_size):
            end = min(start + gene_chunk_size, len(coding_miRNA_loc))
            gene_chunk = coding_miRNA_loc[start:end]

            # Load gene chunk
            X = adata.X[:, gene_chunk]
            if sparse.issparse(X):
                X = X.toarray()

            # CPM normalization
            X = X / n_counts[:, None] * 10_000

            # Convert to float
            if np.issubdtype(X.dtype, np.integer):
                X = X.astype(np.float32)

            # Replace zeros with nan
            X[X == 0] = np.nan

            # Update digests
            for i, gidx in enumerate(range(start, end)):
                median_digests[gidx].update(X[:, i])

            progress.update(end - start)
        progress.close()

        # Create median dictionaries
        median_digest_dict = dict(zip(coding_miRNA_genes, median_digests))
        self.gene_median_dict = {
            k: v.quantile(0.5) for k, v in median_digest_dict.items()
        }

        # Save gene median dictionary
        with open(gene_median_file, "wb") as fp:
            pickle.dump(self.gene_median_dict, fp)

        print(f"Computed medians for {len(self.gene_median_dict)} genes")
        print(f"Saved to {gene_median_file}")

        return gene_median_file

    def tokenize_data(
        self,
        nproc: int = 16,
        special_token: bool = True,
        model_input_size: int = 4096,
        custom_attr_names: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Tokenize the dataset.

        Args:
            nproc: Number of processes.
            special_token: Whether to use special tokens.
            model_input_size: Model input size.
            custom_attr_names: Custom attribute name mapping.
        """
        print("Tokenizing data...")

        dataset_output_file = str(self.dataset_dir / f"{self.proj_name}.dataset")
        if os.path.exists(dataset_output_file):
            print(
                f"Dataset file {dataset_output_file} already exists. Skipping tokenization."
            )
            return

        # Get gene median file
        gene_median_file = (
            self.processed_dir / f"{self.proj_name}_gene_median_dict.pickle"
        )

        if not gene_median_file.exists():
            raise FileNotFoundError(f"Gene median file not found: {gene_median_file}")

        # Default custom attributes
        if custom_attr_names is None:
            custom_attr_names = {"cell_id": "cell_id", "sample_id": "sample_id"}

        # Initialize tokenizer
        tk = TranscriptomeTokenizer(
            nproc=nproc,
            special_token=special_token,
            model_input_size=model_input_size,
            gene_median_file=str(gene_median_file),
            token_dictionary_file=str(TOKEN_DICTIONARY_FILE),
            gene_mapping_file=str(ENSEMBL_MAPPING_FILE),
            custom_attr_name_dict=custom_attr_names,
        )

        # Check h5ad file
        h5ad_files = list(self.processed_dir.glob("*.h5ad"))
        assert len(h5ad_files) == 1, f"Expected 1 h5ad file, found {len(h5ad_files)}"

        # Tokenize
        tk.tokenize_data(
            data_directory=str(self.processed_dir),
            output_directory=str(self.dataset_dir),
            output_prefix=self.proj_name,
            file_format="h5ad",
        )

        # Load and save lengths
        dataset = load_from_disk(dataset_output_file)
        lengths_file = self.dataset_dir / f"{self.proj_name}_lengths.pkl"
        with open(lengths_file, "wb") as f:
            pickle.dump(dataset["length"], f)

        print(f"Tokenization complete. Dataset saved to {self.dataset_dir}")

    def extract_subset_by_celltype(
        self,
        cell_type: str,
        suffix: Optional[str] = None,
    ) -> None:
        """
        Extract subset dataset by cell type.

        Args:
            cell_type: Cell type to extract.
            suffix: Output file suffix (default: cell_type.lower()).
        """
        if suffix is None:
            suffix = cell_type.lower()

        print(f"Extracting {cell_type} cells...")

        # Load obsmeta
        obsmeta_file = self.processed_dir / f"{self.proj_name}_obsmeta.csv"
        obsmeta = pd.read_csv(obsmeta_file)

        # Filter by cell type
        obsmeta_subset = obsmeta[obsmeta["cell_type"] == cell_type]

        if len(obsmeta_subset) == 0:
            print(f"Warning: No cells found for cell type '{cell_type}'")
            return

        # Load full dataset
        dataset = load_from_disk(str(self.dataset_dir / f"{self.proj_name}.dataset"))

        # Select subset (cell_id is 1-indexed)
        ds_cell_id_dict = {cid: i for i, cid in enumerate(dataset["cell_id"])}
        dataset_subset = dataset.select(
            [ds_cell_id_dict[cid] for cid in obsmeta_subset["cell_id"].values]
        )

        # Save subset dataset
        subset_dataset_file = (
            self.dataset_dir / f"{self.proj_name}_{suffix}.dataset"
        )
        dataset_subset.save_to_disk(str(subset_dataset_file))

        # Save lengths
        subset_lengths_file = (
            self.dataset_dir / f"{self.proj_name}_{suffix}_lengths.pkl"
        )
        with open(subset_lengths_file, "wb") as f:
            pickle.dump(dataset_subset["length"], f)

        print(f"Extracted {len(dataset_subset)} {cell_type} cells")
        print(f"Saved to {subset_dataset_file}")

    def run(
        self,
        adata_path: Optional[str] = None,
        tokenize: bool = True,
        extract_celltype: Optional[Dict[str, str]] = {"Epithelia": "epi"},
        **tokenize_kwargs: Any,
    ) -> None:
        """
        Run the complete pipeline.

        Args:
            adata_path: Path to input h5ad file.
            tokenize: Whether to tokenize data.
            extract_celltype: Dict of cell type and its suffix to extract.
            **tokenize_kwargs: Additional arguments for tokenization.
        """
        print("=" * 60)
        print(f"Starting pipeline for project: {self.proj_name}")
        print("=" * 60)

        # Step 1: Load data
        self.load_data(adata_path)

        # Step 2: Annotate obs and var
        self.annotate_obs()
        self.annotate_var()

        # Step 3: Save annotated data
        self.adata.X = self.adata.layers["lognorm"]
        self.adata.write(self.h5ad_file)
        print(f"Saved annotated data to {self.h5ad_file}")

        # Step 4: Save obsmeta and varmeta
        self.save_obsmeta(spatial_cols=self.spatial_cols)
        self.save_varmeta()

        # Step 5: Compute gene medians
        self.adata.X = self.adata.layers["counts"]
        self.compute_gene_medians()

        # Step 6: Tokenize data
        if tokenize:
            self.tokenize_data(**tokenize_kwargs)

        # Step 7: Extract subset by cell type
        if extract_celltype:
            for cell_type, suffix in extract_celltype.items():
                self.extract_subset_by_celltype(cell_type, suffix)

        print("=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)


# =============================================================================
# TME Cell Sampler
# =============================================================================


class TMECellSampler:
    """
    TME cell sampler for defining neighborhood cells and saving to dataset.

    Supports four sampling strategies:
        - v1: Global random sampling (random within sample, no distance constraint)
        - v2: Local neighborhood Top-K sampling (take nearest k cells)
        - v3: Local neighborhood random sampling (randomly select k from nearest 10k)
        - v4: Local neighborhood weighted sampling (weighted by distance rank)

    Attributes:
        work_dir: Working directory path.
        proj_name: Project name.
        tme_version: Sampling version (v1/v2/v3/v4).
        ks: List of sampling sizes.
        search_K: Local search range (for v2-v4).
        chunk_size: Chunk processing size.
        seed: Random seed.
    """

    def __init__(
        self,
        work_dir: str,
        proj_name: str,
        tme_version: Literal["v1", "v2", "v3", "v4"] = "v3",
        ks: Optional[List[int]] = None,
        search_K: int = 10000,
        chunk_size: int = 5000,
        seed: int = 42,
        drop_novel_celltype: bool = True,
        extract_celltype: Optional[Dict[str, str]] = {"Epithelia": "epi"},
    ) -> None:
        """
        Initialize the TME cell sampler.

        Args:
            work_dir: Working directory path.
            proj_name: Project name.
            tme_version: Sampling version (v1/v2/v3/v4).
            ks: List of sampling sizes.
            search_K: Local search range (for v2-v4).
            chunk_size: Chunk processing size.
            seed: Random seed.
            drop_novel_celltype: Whether to drop novel cell types.
            extract_celltype: Dict of cell type and its suffix to extract.
        """
        self.work_dir = work_dir
        self.proj_name = proj_name
        self.drop_novel_celltype = drop_novel_celltype
        self.tme_version = tme_version
        self.ks = ks if ks is not None else [128, 256, 512, 1024]
        self.search_K = search_K
        self.chunk_size = chunk_size
        self.seed = seed

        # Path definitions
        self.processed_dir = os.path.join(work_dir, proj_name, "processed")
        self.dataset_dir = os.path.join(work_dir, proj_name, "datasets")
        self.output_dir = os.path.join(work_dir, proj_name, "tme_cells", tme_version)
        self.tme_cells_paths = {
            k: os.path.join(self.output_dir, f"k_{k}.npy") for k in self.ks
        }

        # Data containers
        self.obsmeta: Optional[pd.DataFrame] = None
        self.cols_spatial: Optional[List[str]] = None
        self.memmaps: Dict[int, np.memmap] = {}
        self.tme_cells_dict: Dict[int, np.ndarray] = {}
        self.cell_to_celltype: Optional[Dict] = None

        self.dataset = load_from_disk(
            os.path.join(self.dataset_dir, f"{self.proj_name}.dataset")
        )

        self.tme_dataset_path = os.path.join(
            self.dataset_dir, f"{self.proj_name}_TME_{self.tme_version}.dataset"
        )

        self.extract_celltype = extract_celltype

        # Prepare weights for v4
        if self.tme_version == "v4":
            self.weights = 1.0 / (np.arange(self.search_K) + 1)
            self.weights = self.weights / self.weights.sum()

        self._tme_cells_dict = None

    def load_metadata(self) -> None:
        """
        Load cell metadata and identify spatial coordinate columns.

        Raises:
            ValueError: If no spatial coordinates found in obsmeta.
        """
        obsmeta_path = os.path.join(
            self.processed_dir, f"{self.proj_name}_obsmeta.csv"
        )
        self.obsmeta = pd.read_csv(obsmeta_path)
        print(f"Loaded obsmeta with shape: {self.obsmeta.shape}")

        # Drop novel cell type
        if self.drop_novel_celltype:
            self.obsmeta = self.obsmeta[self.obsmeta["tme_id"] != 0].copy()
            self.dataset = self.dataset.select(self.obsmeta["cell_id"].values - 1)

        # Identify spatial coordinate columns
        if set(["spatial_1", "spatial_2"]).issubset(self.obsmeta.columns):
            self.cols_spatial = ["spatial_1", "spatial_2"]
        else:
            raise ValueError("No spatial coordinates found in obsmeta")

        print(f"Using spatial columns: {self.cols_spatial}")

    def initialize_memmaps(self) -> None:
        """
        Initialize memory-mapped files.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        n_cells = self.obsmeta.shape[0]

        for k in self.ks:
            self.memmaps[k] = np.memmap(
                os.path.join(self.output_dir, f"k_{k}.dat"),
                dtype="int32",
                mode="w+",
                shape=(n_cells, k),
            )
        print(f"Initialized memmaps for k={self.ks}")

    def sample_cells(self) -> None:
        """
        Sample TME cells for each cell in each sample.

        Uses different sampling strategies based on tme_version.
        """
        print(f"\n{'='*60}")
        print(f"Starting TME cell sampling with version: {self.tme_version}")
        print(f"{'='*60}")

        global_offset = 0
        for sample, obs_sp in self.obsmeta.groupby("sample_name"):
            print(f"\n### Processing sample: {sample}")

            coords = obs_sp[self.cols_spatial].values.astype("float32")
            cell_ids = obs_sp["cell_id"].values.astype("int32")
            n_sp = coords.shape[0]

            # Build sample-specific Faiss index (for v2-v4)
            index = None
            if self.tme_version in {"v2", "v3", "v4"}:
                index = faiss.IndexFlatL2(2)
                index.add(coords)

            # Process in chunks
            for start in range(0, n_sp, self.chunk_size):
                end = min(start + self.chunk_size, n_sp)
                print(f"  Processing cells {start}:{end}")

                if self.tme_version == "v1":
                    self._sample_v1(cell_ids, start, end, global_offset)
                elif self.tme_version in {"v2", "v3", "v4"}:
                    self._sample_local(
                        index, coords, cell_ids, start, end, global_offset
                    )

            global_offset += n_sp

    def _sample_v1(
        self,
        cell_ids: np.ndarray,
        start: int,
        end: int,
        global_offset: int,
    ) -> None:
        """
        v1: Global random sampling (within current sample only).

        Args:
            cell_ids: Array of cell IDs.
            start: Start index of current chunk.
            end: End index of current chunk.
            global_offset: Global offset for current sample.
        """
        for k in self.ks:
            np.random.seed(self.seed)
            sampled = np.random.choice(
                cell_ids,
                size=(end - start, k),
                replace=True,
            )
            self.memmaps[k][
                range(global_offset + start, global_offset + end)
            ] = sampled

    def _sample_local(
        self,
        index: faiss.Index,
        coords: np.ndarray,
        cell_ids: np.ndarray,
        start: int,
        end: int,
        global_offset: int,
    ) -> None:
        """
        v2/v3/v4: Local neighborhood based sampling.

        Args:
            index: Faiss index for nearest neighbor search.
            coords: Spatial coordinates.
            cell_ids: Array of cell IDs.
            start: Start index of current chunk.
            end: End index of current chunk.
            global_offset: Global offset for current sample.
        """
        # Search local neighborhood
        _, I = index.search(coords[start:end], self.search_K + 1)

        # Remove self (first column is self)
        I = I[:, 1:]

        # Convert to global cell_id
        I = cell_ids[I]

        for k in self.ks:
            if self.tme_version == "v2":
                # v2: Take top-k nearest
                sampled = I[:, :k]

            elif self.tme_version == "v3":
                # v3: Randomly sample k from local neighborhood
                np.random.seed(self.seed)
                cols = np.random.choice(self.search_K, size=k, replace=False)
                sampled = I[:, cols]

            elif self.tme_version == "v4":
                # v4: Weighted sampling based on distance rank
                np.random.seed(self.seed)
                cols = np.random.choice(
                    self.search_K,
                    size=k,
                    replace=False,
                    p=self.weights,
                )
                sampled = I[:, cols]

            self.memmaps[k][
                range(global_offset + start, global_offset + end)
            ] = sampled

    def save_memmaps(self) -> None:
        """
        Save memory-mapped files to .npy and clean up temporary files.
        """
        print("\nSaving memmaps to .npy files...")
        for k, mm in self.memmaps.items():
            mm.flush()
            npy_path = os.path.join(self.output_dir, f"k_{k}.npy")
            np.save(npy_path, np.asarray(mm))
            # Remove temporary memory-mapped file
            os.remove(os.path.join(self.output_dir, f"k_{k}.dat"))
            print(f"  Saved k={k} to {npy_path}")

    def load_celltype_mapping(self) -> None:
        """
        Load cell type mapping dictionary.
        """
        celltype_path = os.path.join(
            self.processed_dir, f"{self.proj_name}_cell_cluster_main_dict.pkl"
        )
        with open(celltype_path, "rb") as f:
            self.cell_to_celltype = pickle.load(f)
        print(
            f"Loaded cell-to-celltype mapping with {len(self.cell_to_celltype)} entries"
        )

    def _lazy_load_data(self) -> None:
        """
        Lazy load data, loaded once per process.
        """
        if self._tme_cells_dict is None:
            self._tme_cells_dict = {
                k: np.load(self.tme_cells_paths[k], mmap_mode="r") for k in self.ks
            }
            celltype_path = os.path.join(
                self.processed_dir, f"{self.proj_name}_cell_cluster_main_dict.pkl"
            )
            with open(celltype_path, "rb") as f:
                self._cell_to_celltype = pickle.load(f)

            # Build cell_id to row index mapping
            self._cell_id_to_row = {
                cell_id: idx for idx, cell_id in enumerate(self.obsmeta["cell_id"].values)
            }

    def _process_single_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example.

        Args:
            example: Dataset example dictionary.

        Returns:
            Updated example dictionary with TME cells and types.
        """
        self._lazy_load_data()

        row_index = self._cell_id_to_row[example["cell_id"]]

        for k in self.ks:
            sampled_cells = self._tme_cells_dict[k][row_index]
            example[f"tme_cells{k}"] = sampled_cells.tolist()
            example[f"tme_types{k}"] = [
                self._cell_to_celltype[c] for c in sampled_cells
            ]
        return example

    def add_tme_to_dataset(self, num_proc: int = 10) -> None:
        """
        Add TME cell information to dataset.

        Args:
            num_proc: Number of processes for dataset processing.
        """
        print(f"\nAdding TME cells to dataset with {num_proc} processes...")
        self.dataset = self.dataset.map(
            self._process_single_example,
            num_proc=num_proc,
        )
        self.dataset.save_to_disk(self.tme_dataset_path)
        print(f"Saved TME dataset to {self.tme_dataset_path}")

    def extract_subset_by_celltype(
        self,
        cell_type: str,
        suffix: Optional[str] = None,
    ) -> None:
        """
        Extract subset dataset by cell type.

        Args:
            cell_type: Cell type to extract.
            suffix: Output file suffix (default: cell_type.lower()).
        """
        if suffix is None:
            suffix = cell_type.lower()

        subset_dataset_file = (
            self.dataset_dir + f"/{self.proj_name}_{suffix}_TME_{self.tme_version}.dataset"
        )
        if os.path.exists(subset_dataset_file):
            print(f"{subset_dataset_file} already exists. Skipping...")
            return

        print(f"Extracting {cell_type} cells...")

        obsmeta_subset = self.obsmeta[self.obsmeta["cell_type"] == cell_type]
        if len(obsmeta_subset) == 0:
            print(f"Warning: No cells found for cell type '{cell_type}'")
            return

        # Load full dataset
        dataset = load_from_disk(str(self.tme_dataset_path))
        ds_cell_id_dict = {cid: i for i, cid in enumerate(dataset["cell_id"])}
        dataset_subset = dataset.select(
            [ds_cell_id_dict[cid] for cid in obsmeta_subset["cell_id"].values]
        )
        dataset_subset.save_to_disk(str(subset_dataset_file))

        print(f"Extracted {len(dataset_subset)} {cell_type} cells")
        print(f"Saved to {subset_dataset_file}")

    def run(
        self,
        num_proc: int = 10,
    ) -> None:
        """
        Run the complete TME cell sampling and dataset processing pipeline.

        Args:
            num_proc: Number of processes for dataset processing.
        """
        print(f"\n{'='*60}")
        print(f"TME Cell Sampler - Version {self.tme_version}")
        print(f"Project: {self.proj_name}")
        print(f"{'='*60}\n")

        # Step 1: Load metadata and sample TME cells
        self.load_metadata()
        if not os.path.exists(self.tme_dataset_path):
            if not all(os.path.exists(p) for p in self.tme_cells_paths.values()):
                self.initialize_memmaps()
                self.sample_cells()
                self.save_memmaps()
            else:
                print("TME cells files (.npy) already exist. Skipping sampling step.")

            # Step 2: Load data and add TME information
            self.load_celltype_mapping()
            self.add_tme_to_dataset(num_proc=num_proc)

        if self.extract_celltype:
            for cell_type, suffix in self.extract_celltype.items():
                self.extract_subset_by_celltype(cell_type, suffix)

        print(f"\n{'='*60}")
        print("TME cell sampling completed successfully!")
        print(f"{'='*60}\n")


# =============================================================================
# Initial Embedding Extractor
# =============================================================================


class InitEmbExtractor:
    """
    Cell embedding extractor based on Geneformer.

    Attributes:
        work_dir: Working directory path.
        proj_name: Project name.
        gf_model: Geneformer model identifier.
        emb_layer: Embedding layer to extract.
        device: Device for computation.
        batch_size: Batch size for forward pass.
    """

    def __init__(
        self,
        work_dir: str,
        proj_name: str,
        gf_model: str = "GF_CL",
        emb_layer: int = 0,
        device: int = 0,
        batch_size: int = 48,
    ) -> None:
        """
        Initialize the embedding extractor.

        Args:
            work_dir: Working directory path.
            proj_name: Project name.
            gf_model: Geneformer model identifier.
            emb_layer: Embedding layer to extract.
            device: Device ID for CUDA.
            batch_size: Batch size for forward pass.
        """
        self.work_dir = Path(work_dir)
        self.proj_name = proj_name
        self.gf_model = gf_model
        self.emb_layer = emb_layer
        self.device = device
        self.batch_size = batch_size

        # Path definitions
        self.dataset_path = (
            self.work_dir / proj_name / "datasets" / f"{proj_name}.dataset"
        )
        self.output_dir = (
            self.work_dir / proj_name / "gf_emb" / f"{self.gf_model}_L{emb_layer}"
        )

    def extract(self, save_npy: bool = True) -> np.ndarray:
        """
        Extract embeddings and save to file.

        Args:
            save_npy: Whether to save embeddings to .npy file.

        Returns:
            Extracted embeddings array.
        """
        # Load dataset
        print(f"Loading dataset from: {self.dataset_path}")
        dataset = load_from_disk(str(self.dataset_path))
        print(f"Dataset loaded: {dataset}")

        # Get model path
        pr_models = tu.generate_pr_models_dict()
        model_dir = pr_models[self.gf_model][0]

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract embeddings
        print(f"Extracting embeddings (layer {self.emb_layer})...")
        embex = EmbExtractor(
            model_type="Pretrained",
            emb_mode="cell",
            emb_layer=self.emb_layer,
            forward_batch_size=self.batch_size,
            nproc=16,
            max_ncells=None,
            token_dictionary_file=str(TOKEN_DICTIONARY_FILE),
            device=f"cuda:{self.device}",
        )

        embs = embex.extract_embs(
            model_directory=model_dir,
            input_data_file=str(self.dataset_path),
            output_directory=str(self.output_dir),
            output_prefix="cell_embed",
            output_format="npy",
        )

        # Read and save
        print(f"Embeddings shape: {embs.shape}")
        print(f"Saved to: ", self.output_dir / "cell_embed.npy")
        return embs