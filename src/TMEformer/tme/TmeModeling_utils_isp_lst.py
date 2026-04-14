"""
TMEformer ISP List Utilities
============================

This module provides utilities for in silico perturbation (ISP) on gene lists,
including:
- Gene token perturbation (OE/KD/KO/KI)
- Sliding window operations on gene ranks
- Cell dataset generation for perturbed sequences
- Embedding extraction for perturbed cells

Key Functions:
    - window_slide_one_token: Slide one token index with window
    - get_oe_lst_with_tokens: Over-expression perturbation
    - get_kd_lst_with_tokens: Knock-down perturbation
    - get_ko_lst_with_tokens: Knock-out perturbation
    - get_ki_lst_without_tokens: Knock-in perturbation
    - isp_genelist_in_cell: ISP for genes expressed in cell
    - isp_genelist_not_in_cell: ISP for genes not in cell
    - perturb_one_cell: Main function for cell perturbation
"""

import pickle
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset

from .. import TOKEN_DICTIONARY_FILE
from .. import perturber_utils as pu
from ..emb_extractor import get_embs


# =============================================================================
# List Manipulation Utilities
# =============================================================================


def window_slide_one_token(
    n_lst: int,
    origin_idx: int,
    window: int,
    direction: str = "both",
) -> List[int]:
    """
    Slide one index in a list given a window size and direction.

    Specially designed only for one token perturbation.

    Args:
        n_lst: The length of input list.
        origin_idx: The index of the single isp gene in the list.
        window: Sliding step size.
        direction: The direction to slide. Options:
            - "left": higher expression rank
            - "right": lower expression rank
            - "both": both directions (default)

    Returns:
        List of all possible indices after sliding.

    Raises:
        ValueError: If n_lst < 5 or window <= 0.

    Example:
        >>> n_lst, origin_idx, window = 10, 5, 3
        >>> window_slide_one_token(n_lst, origin_idx, window)
    """
    if n_lst < 5:
        raise ValueError("n_lst must be at least 5 (3 genes at least)")
    if window <= 0:
        raise ValueError("window must be a positive integer")

    min_idx, max_idx = 1, n_lst - 2
    new_indices = []

    max_step = (max_idx - min_idx) // window
    if max_step == 0:
        return []
    for step in range(1, max_step + 1):
        if direction in ("both", "right"):
            new_idx = origin_idx + step * window
            if min_idx <= new_idx <= max_idx:
                new_indices.append(new_idx)

        if direction in ("both", "left"):
            new_idx = origin_idx - step * window
            if min_idx <= new_idx <= max_idx:
                new_indices.append(new_idx)
    return new_indices


def count_tokens_in_list(lst: List[Any], tokens: List[Any]) -> int:
    """
    Count how many tokens from `tokens` appear in the list `lst`.

    Args:
        lst: The input sequence list.
        tokens: The target tokens to check for.

    Returns:
        Number of overlapping tokens.

    Raises:
        TypeError: If lst or tokens is not a list.
    """
    if not isinstance(lst, list) or not isinstance(tokens, list):
        raise TypeError("Both inputs must be lists.")

    return len(set(tokens) & set(lst))


# =============================================================================
# Over-Expression (OE) Utilities
# =============================================================================


def get_oe_lst_with_tokens(lst: List[Any], tokens: List[Any]) -> Dict[int, List[Any]]:
    """
    Move the target tokens to the highest expression rank (first gene rank).

    Args:
        lst: The original sequence list.
        tokens: The target gene tokens to perturb.

    Returns:
        Perturb dictionary with key as offset value (1) and value as perturbed list.

    Raises:
        ValueError: If not all tokens are in the list.

    Example:
        >>> lst, tokens = list(range(10, 22)), [13, 16]
        >>> get_oe_lst_with_tokens(lst, tokens)
    """
    if count_tokens_in_list(lst, tokens) < len(tokens):
        raise ValueError("Not all tokens in list, so that cannot OE!")

    tokens_idxes = [lst.index(x) for x in tokens]
    remaining = [x for i, x in enumerate(lst) if i not in tokens_idxes]
    perturbed_oe = remaining[:1] + tokens + remaining[1:]

    return {1: perturbed_oe}


# =============================================================================
# Knock-Down (KD) Utilities
# =============================================================================


def get_kd_lst_with_tokens(lst: List[Any], tokens: List[Any]) -> Dict[int, List[Any]]:
    """
    Move the target tokens to the lowest expression rank (last gene rank).

    Args:
        lst: The original sequence list.
        tokens: The target gene tokens to perturb.

    Returns:
        Perturb dictionary with key as offset value (-1) and value as perturbed list.

    Raises:
        ValueError: If not all tokens are in the list.

    Example:
        >>> lst, tokens = list(range(10, 22)), [13, 16]
        >>> get_kd_lst_with_tokens(lst, tokens)
    """
    if count_tokens_in_list(lst, tokens) < len(tokens):
        raise ValueError("Not all tokens in list, so that cannot KD!")

    tokens_idxes = [lst.index(x) for x in tokens]
    remaining = [x for i, x in enumerate(lst) if i not in tokens_idxes]
    perturbed_kd = remaining[:-1] + tokens + remaining[-1:]

    return {-1: perturbed_kd}


# =============================================================================
# Knock-Out (KO) Utilities
# =============================================================================


def get_ko_lst_with_tokens(
    lst: List[Any],
    tokens: List[Any],
    ko_method: str = "v3",
    pad_token_id: int = 0,
) -> Dict[int, List[Any]]:
    """
    Knock out target tokens from a sequence list using various methods.

    Args:
        lst: The original sequence list.
        tokens: The target tokens to be knocked out.
        ko_method: Knock-out method. Options:
            - "v1": Remove target tokens directly (length reduction, Geneformer).
            - "v2": Replace target tokens with pad_token (length unchanged).
            - "v3": Remove target tokens, then insert pad_tokens before last element.
            - "v4": Remove target tokens, then append pad_tokens at the end.
        pad_token_id: Token to use as padding (default: 0).

    Returns:
        Perturb dictionary with key as offset value (-9999) and value as perturbed list.

    Raises:
        ValueError: If not all tokens are in the list or invalid ko_method.

    Example:
        >>> lst, tokens = list(range(10, 22)), [13, 16]
        >>> get_ko_lst_with_tokens(lst, tokens, ko_method="v3")
    """
    if count_tokens_in_list(lst, tokens) < len(tokens):
        raise ValueError("Not all tokens in list, so that cannot KO!")
    tokens_idxes = [lst.index(x) for x in tokens]

    lst_copy = lst[:]
    if ko_method == "v1":
        # Directly remove the tokens
        ko_lst = [x for i, x in enumerate(lst_copy) if i not in tokens_idxes]

    elif ko_method == "v2":
        # Replace the tokens with pad_token_id
        ko_lst = [x if i not in tokens_idxes else pad_token_id for i, x in enumerate(lst_copy)]

    elif ko_method == "v3":
        # Remove tokens, then insert pad tokens before the last element
        ko_lst = [x for i, x in enumerate(lst_copy) if i not in tokens_idxes]
        ko_lst[-1:-1] = [pad_token_id] * len(tokens_idxes)

    elif ko_method == "v4":
        # Remove tokens, then append pad tokens at the end
        ko_lst = ko_lst + [pad_token_id] * len(tokens_idxes)

    else:
        raise ValueError("Invalid ko_method. Choose from 'v1', 'v2', 'v3', or 'v4'.")

    return {-9999: ko_lst}


# =============================================================================
# In Silico Perturbation (ISP) for Genes in Cell
# =============================================================================


def isp_genelist_in_cell(
    lst: List[Any],
    tokens: List[Any],
    window: int = 100,
    endpoints: bool = True,
    ko_method: Optional[str] = None,
) -> Dict[int, List[Any]]:
    """
    Get perturbed lists for a given list of tokens expressed in cell.

    Args:
        lst: The original sequence list.
        tokens: The target tokens to be perturbed.
        window: The size of the sliding window. 0 for no window operation.
        endpoints: Whether to move the target tokens to the endpoints of the list.
        ko_method: Knock-out method. Refer to get_ko_lst_with_tokens. None for no knockout.

    Returns:
        Perturb dictionary with key as offset value and value as perturbed list.
            - Offset = 0 for origin, -9999 for KO.
            - For single token ISP, offset is relative perturbed rank to original token (0).
            - For multi-tokens ISP, offset = 1 for OE, -1 for KD.

    Raises:
        ValueError: If tokens is not a list or not all tokens are in the list.

    Examples:
        >>> lst, tokens, window, ko_method = list(range(10, 30)), [13, 15], 2, "v3"
        >>> isp_genelist_in_cell(lst, tokens, window, endpoints=True, ko_method=None)
        >>> isp_genelist_in_cell(lst, tokens[:1], window, endpoints=True, ko_method=ko_method)
    """
    if not isinstance(tokens, list):
        raise ValueError("tokens must be a list, even if it's a single token!")
    if count_tokens_in_list(lst, tokens) < len(tokens):
        raise ValueError("Not all tokens in list, so that cannot ISP(OE/KD/KO)!")

    n_lst = len(lst)
    n_tokens = len(tokens)
    tokens_idxes = [lst.index(x) for x in tokens]

    # Window sliding only for single gene token
    if n_tokens > 1:
        window = 0

    perturb_dict = {0: lst}
    # Perturb one gene token
    if n_tokens == 1:
        new_idxes = []
        if window > 0:
            new_idxes_w = window_slide_one_token(n_lst, tokens_idxes[0], window, direction="both")
            new_idxes.extend(new_idxes_w)
        if endpoints:
            new_idxes.extend([1, n_lst - 2])
            new_idxes = list(set(new_idxes))

        for new_idx in new_idxes:
            new_lst = lst[:]  # copy
            new_lst.pop(tokens_idxes[0])  # remove
            new_lst.insert(new_idx, tokens[0])  # insert
            # offset > 0: OE; offset < 0: KD
            offset = tokens_idxes[0] - new_idx
            perturb_dict[offset] = new_lst
    elif n_tokens > 1 and endpoints:
        perturb_dict.update(get_oe_lst_with_tokens(lst, tokens))  # Offset: 1
        perturb_dict.update(get_kd_lst_with_tokens(lst, tokens))  # Offset: -1

    if ko_method is not None:
        perturb_dict.update(get_ko_lst_with_tokens(lst, tokens, ko_method=ko_method))  # Offset: -9999

    sorted_perturb_dict = {key: perturb_dict[key] for key in sorted(perturb_dict.keys())}
    return sorted_perturb_dict


# =============================================================================
# Knock-In (KI) Utilities
# =============================================================================


def get_ki_lst_without_tokens(
    lst: List[Any],
    tokens: List[Any],
    ki_method: str,
    max_len: int = 4096,
    pad_token_id: int = 0,
) -> Dict[int, List[Any]]:
    """
    Knock in target tokens from a sequence list using various methods.

    Args:
        lst: The original sequence list.
        tokens: The target tokens to be knocked in.
        ki_method: Knock-in method. Options:
            - "v1": Geneformer style (drop last tokens if overflow).
            - "v2": Insert last gene token for shorter lst, replace for full lst.
            - "v3": Replace the last gene token for all (shorter/full) lst.
        max_len: Maximum sequence length (default: 4096).
        pad_token_id: Token to use as padding (default: 0).

    Returns:
        Perturb dictionary with key as offset value (0/1) and value as perturbed list.
            - Offset = 0 for original lst.
            - Offset = 1 for perturbed lst after initial KI.

    Raises:
        ValueError: If some tokens are already in the list or invalid ki_method.

    Examples:
        >>> lst, tokens, ki_method, max_len = list(range(10, 30)), [13, 15], "v2", 20
        >>> out = get_ki_lst_without_tokens(lst, tokens, ki_method, max_len)
    """
    if count_tokens_in_list(lst, tokens) > 0:
        raise ValueError("Some of tokens in list, so that cannot KI!")

    n_tokens = len(tokens)
    lst_will_overflow = len(lst) + n_tokens > max_len
    origin_lst = lst[:]

    if ki_method == "v1":
        if lst_will_overflow:
            # Directly drop
            origin_lst = origin_lst[:-1-n_tokens] + origin_lst[-1:]
        first_ki_lst = origin_lst[:]
        first_ki_lst[-1:-1] = tokens

    elif ki_method == "v2":
        first_ki_lst = origin_lst[:]
        if lst_will_overflow:
            # Replace for full lst
            first_ki_lst[-1-n_tokens:-1] = tokens
        else:
            # Insert for shorter lst
            first_ki_lst[-1:-1] = tokens

    elif ki_method == "v3":
        origin_lst[-1-n_tokens:-1] = [pad_token_id] * n_tokens
        first_ki_lst = origin_lst[:]
        first_ki_lst[-1-n_tokens:-1] = tokens

    else:
        raise ValueError(f"Unsupported ki_method: {ki_method}")

    return {0: origin_lst, 1: first_ki_lst}


# =============================================================================
# In Silico Perturbation (ISP) for Genes Not in Cell
# =============================================================================


def isp_genelist_not_in_cell(
    lst: List[Any],
    tokens: List[Any],
    window: int = 100,
    endpoints: bool = True,
    ki_method: str = "v1",
    max_len: int = 4096,
) -> Dict[int, List[Any]]:
    """
    Get perturbed lists for a given list of tokens not expressed in cell.

    Args:
        lst: The original sequence list.
        tokens: The target tokens to be perturbed (KI).
        window: The size of the sliding window. 0 for no window operation.
        endpoints: Whether to move the target tokens to the endpoints of the list.
        ki_method: Knock-in method. Refer to get_ki_lst_without_tokens.
        max_len: Maximum sequence length (default: 4096).

    Returns:
        Perturb dictionary with key as offset value and value as perturbed list.
            - Offset = 0 for origin.
            - For single token ISP, offset is relative perturbed rank to original token (0).
            - For multi-tokens ISP, offset = 1 for OE.

    Raises:
        ValueError: If tokens is not a list or some tokens are already in the list.

    Examples:
        >>> lst, tokens, window, endpoints, max_len = list(range(10, 20)), [23, 25], 2, True, 10
        >>> ki_method = "v2"
        >>> isp_genelist_not_in_cell(lst, tokens, window, endpoints, ki_method=ki_method, max_len=max_len)
    """
    if not isinstance(tokens, list):
        raise ValueError("tokens must be a list, even if it's a single token!")
    if count_tokens_in_list(lst, tokens) > 0:
        raise ValueError("Some tokens in list, so that cannot ISP(KI)!")

    perturb_dict = get_ki_lst_without_tokens(lst, tokens, ki_method=ki_method, max_len=max_len)
    # do OE based on the lst of offset = 1
    ki_dict = isp_genelist_in_cell(perturb_dict[1], tokens, window=window,
                                   endpoints=endpoints, ko_method=None)
    if len(tokens) == 1:
        perturb_dict.update(ki_dict)
    else:
        perturb_dict.pop(1)
        perturb_dict[1] = ki_dict[1]

    sorted_perturb_dict = {key: perturb_dict[key] for key in sorted(perturb_dict.keys())}
    return sorted_perturb_dict


# =============================================================================
# Cell Dataset and Embedding Utilities
# =============================================================================


def make_shifted_cell_dataset_and_embedding(
    example_cell: Dataset,
    perturb_dict: Dict[int, List[Any]],
    model: torch.nn.Module,
    embed_layer: int = -1,
    batch_size: int = 10,
    do_embed: bool = True,
) -> tuple:
    """
    Get the perturbed cell dataset and embedding for a given cell.

    Args:
        example_cell: The original cell dataset.
        perturb_dict: Key is offset value, value is the perturbed list.
        model: The model to be used for embedding.
        embed_layer: The layer to extract the embedding from (default: -1).
        batch_size: The batch size for embedding extraction (default: 10).
        do_embed: Whether to extract embeddings (default: True).

    Returns:
        Tuple containing:
            - shifted_cell_dataset: The perturbed cell dataset.
            - shifted_cell_emb: The embedding of the perturbed cells (or None if do_embed=False).
    """
    offsets, new_cells = list(perturb_dict.keys()), list(perturb_dict.values())

    shifted_cell_dataset = Dataset.from_dict(
        {
            "input_ids": new_cells,
            "length": [len(cell) for cell in new_cells],
            "offset": offsets,
            "cell_id": example_cell["cell_id"] * len(new_cells),
            "sample_id": example_cell["sample_id"] * len(new_cells),
        }
    )
    if "tme_cells" in example_cell.column_names:
        shifted_cell_dataset = shifted_cell_dataset.add_column(
            "tme_cells", example_cell["tme_cells"] * len(new_cells)
        )
    if "tme_types" in example_cell.column_names:
        shifted_cell_dataset = shifted_cell_dataset.add_column(
            "tme_types", example_cell["tme_types"] * len(new_cells)
        )

    shifted_cell_emb = None
    if do_embed:
        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
            gene_token_dict = pickle.load(f)
        token_gene_dict = {v: k for k, v in gene_token_dict.items()}

        pad_token_id = gene_token_dict.get("<pad>")
        layer_to_quant = pu.quant_layers(model) + embed_layer

        shifted_cell_emb = get_embs(
            model,
            shifted_cell_dataset,
            "cell",
            layer_to_quant,
            pad_token_id,
            batch_size,
            token_gene_dict=token_gene_dict,
            summary_stat=None,
            silent=True,
        ).to("cpu")

    return shifted_cell_dataset, shifted_cell_emb


# =============================================================================
# Main Perturbation Functions
# =============================================================================


def perturb_one_cell(
    example_cell: Dataset,
    gene_tokens: List[Any],
    model: torch.nn.Module,
    isp_run_config: Dict[str, Any],
    embed_layer: int = -1,
    batch_size: int = 10,
    do_embed: bool = True,
) -> tuple:
    """
    Perform ISP (KO/KI) on a given cell.

    Args:
        example_cell: The original cell dataset.
        gene_tokens: The target genes to be knocked out/in.
        model: The model to be used for embedding.
        isp_run_config: Configuration for ISP. Contains:
            - "window": Sliding window size (default: 0).
            - "endpoints": Whether to use endpoints (default: True).
            - "ko_method": Knock-out method (default: None).
            - "ki_method": Knock-in method (default: None).
        embed_layer: The layer to extract the embedding from (default: -1).
        batch_size: The batch size for embedding extraction (default: 10).
        do_embed: Whether to extract embeddings (default: True).

    Returns:
        Tuple containing:
            - shifted_cell_dataset: The perturbed cell dataset.
            - shifted_cell_emb: The embedding of the perturbed cells.

    Raises:
        ValueError: If gene tokens are partially in the list or KI is required but not enabled.

    Example of isp_run_config:
        >>> isp_run_config = {
        ...     "window": 0,
        ...     "endpoints": True,
        ...     "ko_method": None,
        ...     "ki_method": None,
        ... }
    """
    window = isp_run_config.get("window", 0)
    endpoints = isp_run_config.get("endpoints", True)
    ko_method = isp_run_config.get("ko_method")
    ki_method = isp_run_config.get("ki_method")

    enable_ko = ko_method is not None
    enable_ki = ki_method is not None

    input_ids = example_cell["input_ids"][0]
    if count_tokens_in_list(input_ids, gene_tokens) == len(gene_tokens):
        genes_in_cell = True
        gene_tokens_idxes = [input_ids.index(x) for x in gene_tokens]
    elif count_tokens_in_list(input_ids, gene_tokens) == 0:
        genes_in_cell = False
        if not enable_ki:
            raise ValueError(
                f"Gene token {gene_tokens} not found in this cell. "
                "They will be knocked into the cell, but enable_ki is False!"
            )
    else:
        raise ValueError("Only some of tokens are in lst.")

    if genes_in_cell:
        perturb_dict = isp_genelist_in_cell(input_ids, gene_tokens, window, endpoints, ko_method)
    else:
        perturb_dict = isp_genelist_not_in_cell(input_ids, gene_tokens, window, endpoints, ki_method)

    shifted_cell_dataset, shifted_cell_emb = make_shifted_cell_dataset_and_embedding(
        example_cell, perturb_dict, model, embed_layer, batch_size, do_embed
    )

    return shifted_cell_dataset, shifted_cell_emb