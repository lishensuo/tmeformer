"""
TMEformer TME Modeling Module (BERT-based Architecture)
========================================================

This module implements BERT-based architectures for modeling Tumor Microenvironment (TME) 
information in single-cell spatial transcriptomics data.

Key Components:
    - CellBertEmbeddings: Converts TME cell information into 3D embedding sequences
    - CellBertModel: BERT encoder for TME embeddings with pooling
    - TmeBertEmbeddings: GeneFormer-style input embeddings with TME context fusion
    - TmeBertEncoder: BERT encoder with optional cross-attention for TME integration
    - TmeBertModel: Main TME-aware BERT model supporting fuse and cross modes
    - TmeBertForMaskedLM: Masked language modeling head for pretraining
    - TmeBertForSequenceClassification: Sequence classification head
    - TmeBertForMultiGeneExpressionPrediction: Multi-task regression for gene expression
    - TmeBertForCellClassification: Cell type classification head

TME Integration Modes:
    - "fuse": TME context is fused with cell embeddings at the input layer
    - "cross": TME context is integrated via cross-attention in specific transformer layers

Pooling Strategies:
    - "attention": Learnable attention-weighted pooling
    - "mean": Simple mean pooling with mask support

Example:
    >>> from transformers import BertConfig
    >>> config = get_default_tme_config(gf_tme_emb_path="path/to/embeddings")
    >>> model = TmeBertModel(config)
    >>> outputs = model(
    ...     input_ids=input_ids,
    ...     tme_cells=tme_cells,
    ...     tme_types=tme_types
    ... )
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from datasets import load_from_disk

from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertLayer,
    BertEncoder,
    BertPooler,
    BertOnlyMLMHead,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    SequenceClassifierOutput,
    MaskedLMOutput,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
)


# =============================================================================
# Utility Functions
# =============================================================================


def get_default_tme_config(**kwargs) -> BertConfig:
    """
    Create a default TME configuration based on BertConfig.

    This function provides sensible defaults for TME modeling parameters and allows
    customization through keyword arguments.

    Args:
        **kwargs: Override default configuration values.

    Returns:
        BertConfig: Configuration object with TME-specific attributes.

    Default Configuration:
        - hidden_size: 512
        - num_attention_heads: 4
        - num_hidden_layers: 1
        - max_position_embeddings: 512
        - hidden_act: "relu"
        - hidden_dropout_prob: 0.1
        - layer_norm_eps: 1e-12
        - intermediate_size: 1024
        - do_position_embeddings: True
        - do_celltype_embeddings: True
        - total_tme_types: 9
        - pool_type: "attention"
        - tme_add: "fuse"
        - tme_level: "cell"
        - tme_alpha: 0.2

    Example:
        >>> config = get_default_tme_config(
        ...     gf_tme_emb_path="data/xenium/gf_emb/GF_CL_L0/",
        ...     num_hidden_layers=2
        ... )
    """
    defaults = dict(
        hidden_act="relu",
        hidden_size=512,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        do_position_embeddings=True,
        intermediate_size=1024,
        max_position_embeddings=512,
        do_celltype_embeddings=True,
        num_attention_heads=4,
        num_hidden_layers=1,
        total_tme_types=9,
        gf_tme_emb_path=None,
        pool_type="attention",
        tme_add="fuse",
        tme_data="v3",
        tme_level="cell",
        tme_alpha=0.2,
    )
    defaults.update(kwargs)

    config = BertConfig()
    for k, v in defaults.items():
        setattr(config, k, v)
    return config


def build_mlp_head(
    input_dim: int,
    output_dim: int,
    hidden_dim: Optional[int] = None,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> nn.Sequential:
    """
    Build a Multi-Layer Perceptron (MLP) head for downstream tasks.

    Creates a sequential MLP with ReLU activations and dropout between layers.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension (e.g., number of classes or regression targets).
        hidden_dim: Hidden layer dimension. If None, defaults to input_dim.
        num_layers: Total number of linear layers (including input and output).
        dropout: Dropout probability applied after each hidden layer.

    Returns:
        nn.Sequential: MLP module.

    Example:
        >>> mlp = build_mlp_head(
        ...     input_dim=512,
        ...     output_dim=10,
        ...     num_layers=3,
        ...     dropout=0.1
        ... )
        >>> x = torch.randn(32, 512)
        >>> out = mlp(x)  # Shape: (32, 10)
    """
    if hidden_dim is None:
        hidden_dim = input_dim

    layers = []
    for i in range(num_layers):
        in_dim = input_dim if i == 0 else hidden_dim
        out_dim = output_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if i != num_layers - 1:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


# =============================================================================
# Embedding Modules
# =============================================================================


class CellBertEmbeddings(nn.Module):
    """
    Convert TME cell information into 3D embedding representations.

    This module transforms TME cell indices or pre-computed embeddings into
    rich representations by combining:
    - Base cell embeddings (from GeneFormer or similar)
    - Cell type embeddings (optional)
    - Position embeddings (optional)

    The output is a 3D tensor of shape (batch_size, seq_len, hidden_size).

    Attributes:
        tme_config: Configuration object with TME modeling parameters.
        gf_proj: Linear projection layer for base embeddings.
        token_type_embeddings: Cell type embedding table (if enabled).
        position_embeddings: Position embedding table (if enabled).
        LayerNorm: Layer normalization.
        dropout: Dropout layer.

    Input Shapes:
        - tme_cell_ids: (batch_size, seq_len) - TME cell indices (1-based)
        - tme_cell_embs: (batch_size, seq_len, hidden_size) - Pre-computed embeddings
        - tme_celltype_ids: (batch_size, seq_len) - Cell type indices

    Output Shape:
        - torch.Tensor: (batch_size, seq_len, hidden_size)

    Example:
        >>> config = get_default_tme_config(gf_tme_emb_path="path/to/emb")
        >>> embedder = CellBertEmbeddings(config)
        >>> tme_cell_ids = torch.randint(1, 1000, (4, 256))
        >>> tme_types = torch.randint(1, 9, (4, 256))
        >>> embeddings = embedder(
        ...     tme_cell_ids=tme_cell_ids,
        ...     tme_celltype_ids=tme_types
        ... )
    """

    def __init__(self, tme_config: BertConfig) -> None:
        """
        Initialize CellBertEmbeddings.

        Args:
            tme_config: Configuration with embedding parameters.
        """
        super().__init__()
        self.tme_config = tme_config

        # Linear projection for base cell embeddings
        self.gf_proj = nn.Linear(
            self.tme_config.hidden_size, self.tme_config.hidden_size
        )

        # Cell type embeddings (optional)
        if self.tme_config.do_celltype_embeddings:
            self.token_type_embeddings = nn.Embedding(
                self.tme_config.total_tme_types,
                self.tme_config.hidden_size,
                padding_idx=0,
            )

        # Position embeddings (optional)
        if self.tme_config.do_position_embeddings:
            self.register_buffer(
                "position_ids",
                torch.arange(self.tme_config.max_position_embeddings).expand((1, -1)),
                persistent=False,
            )
            self.position_embeddings = nn.Embedding(
                self.tme_config.max_position_embeddings,
                self.tme_config.hidden_size,
            )
            self.position_embedding_type = getattr(
                self.tme_config, "position_embedding_type", "absolute"
            )

        # Normalization and regularization
        self.LayerNorm = nn.LayerNorm(
            self.tme_config.hidden_size, eps=self.tme_config.layer_norm_eps
        )
        self.dropout = nn.Dropout(self.tme_config.hidden_dropout_prob)

    def forward(
        self,
        tme_cell_ids: Optional[torch.LongTensor] = None,
        tme_cell_embs: Optional[torch.FloatTensor] = None,
        tme_celltype_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Generate TME cell embeddings.

        Args:
            tme_cell_ids: Cell indices for lookup (1-based indexing).
                Shape: (batch_size, seq_len)
            tme_cell_embs: Pre-computed cell embeddings. If provided, used instead of lookup.
                Shape: (batch_size, seq_len, hidden_size)
            tme_celltype_ids: Cell type indices for type embeddings.
                Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Combined embeddings.
                Shape: (batch_size, seq_len, hidden_size)

        Note:
            Either tme_cell_ids or tme_cell_embs must be provided.
        """
        dtype = self.gf_proj.weight.dtype

        if tme_cell_embs is None:
            # Lookup embeddings from pre-computed file
            device = tme_cell_ids.device
            gf_cell_embeddings = np.load(
                self.tme_config.gf_tme_emb_path + "/cell_embed.npy", mmap_mode="r"
            )
            # Convert 1-based indices to 0-based for numpy indexing
            input_embeds = torch.from_numpy(
                gf_cell_embeddings[tme_cell_ids.to("cpu") - 1]
            )
        else:
            # Use provided embeddings directly
            device = tme_cell_embs.device
            input_embeds = tme_cell_embs

        # Ensure 3D shape: (batch_size, seq_len, hidden_size)
        if input_embeds.ndim == 2:
            input_embeds = input_embeds.unsqueeze(0)

        # Project and combine embeddings
        input_embeds = input_embeds.to(dtype=dtype, device=device)
        embeddings = self.gf_proj(input_embeds)

        # Add cell type embeddings
        if self.tme_config.do_celltype_embeddings:
            token_type_embeddings = self.token_type_embeddings(tme_celltype_ids)
            embeddings = embeddings + token_type_embeddings

        # Add position embeddings
        if self.tme_config.do_position_embeddings:
            position_ids = self.position_ids[:, : embeddings.size(1)].to(device)
            position_embedding = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embedding

        # Normalize and apply dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# =============================================================================
# Pooling Modules
# =============================================================================


class AttentionPool(nn.Module):
    """
    Attention-based pooling layer for sequence aggregation.

    Computes learnable attention weights for each position in the sequence
    and produces a weighted sum as the pooled representation.

    Attributes:
        attention_weights: Learnable weight vector for attention computation.
            Shape: (hidden_size, 1)

    Input:
        - hidden_states: (batch_size, seq_len, hidden_size)
        - attention_mask: (batch_size, seq_len) - Optional mask for padding

    Output:
        - pooled_output: (batch_size, hidden_size)
        - attention_scores: (batch_size, seq_len, 1)

    Example:
        >>> pooler = AttentionPool(hidden_size=512)
        >>> hidden = torch.randn(4, 256, 512)
        >>> pooled, scores = pooler(hidden)
        >>> assert pooled.shape == (4, 512)
    """

    def __init__(self, hidden_size: int) -> None:
        """
        Initialize attention pooling.

        Args:
            hidden_size: Dimension of input hidden states.
        """
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_size, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention pooling.

        Args:
            hidden_states: Input hidden states.
                Shape: (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask for padding positions.
                Shape: (batch_size, seq_len)

        Returns:
            Tuple containing:
                - pooled_output: Pooled representation.
                  Shape: (batch_size, hidden_size)
                - attention_scores: Attention weights per position.
                  Shape: (batch_size, seq_len, 1)
        """
        # Compute attention scores: (batch_size, seq_len, 1)
        attention_scores = torch.matmul(hidden_states, self.attention_weights)

        # Apply mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            attention_scores = attention_scores + (attention_mask * -1e9)

        # Softmax normalization along sequence dimension
        attention_scores = torch.softmax(attention_scores, dim=1)

        # Weighted sum: (batch_size, hidden_size)
        pooled_output = torch.sum(
            hidden_states * attention_scores, dim=1
        ).to(hidden_states.dtype)

        return pooled_output, attention_scores


class MeanPool(nn.Module):
    """
    Mean-based pooling layer with mask support.

    Computes the mean of all non-padded positions in the sequence.
    Handles variable-length sequences via attention_mask.

    Input:
        - hidden_states: (batch_size, seq_len, hidden_size)
        - attention_mask: (batch_size, seq_len) - Optional mask (0 for padding)

    Output:
        - pooled_output: (batch_size, hidden_size)
        - None: Second return value for API consistency

    Example:
        >>> pooler = MeanPool()
        >>> hidden = torch.randn(4, 256, 512)
        >>> mask = torch.ones(4, 256)
        >>> mask[:, 100:] = 0  # Pad positions 100+
        >>> pooled, _ = pooler(hidden, mask)
        >>> assert pooled.shape == (4, 512)
    """

    def __init__(self) -> None:
        """Initialize mean pooling."""
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Apply mean pooling.

        Args:
            hidden_states: Input hidden states.
                Shape: (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask (1 for valid, 0 for padding).
                Shape: (batch_size, seq_len)

        Returns:
            Tuple containing:
                - pooled_output: Mean-pooled representation.
                  Shape: (batch_size, hidden_size)
                - None: Placeholder for API consistency
        """
        initial_dtype = hidden_states.dtype

        if attention_mask is not None:
            # Expand mask: (batch_size, seq_len, 1)
            attention_mask = attention_mask.unsqueeze(-1).float()

            # Zero out padding
            hidden_states = hidden_states * attention_mask

            # Count valid tokens
            valid_token_count = torch.sum(attention_mask, dim=1)
            valid_token_count = torch.clamp(valid_token_count, min=1e-9)

            # Compute mean
            sum_hidden_states = torch.sum(hidden_states, dim=1)
            pooled_output = sum_hidden_states / valid_token_count
        else:
            # Simple mean without mask
            pooled_output = torch.mean(hidden_states, dim=1)

        pooled_output = pooled_output.to(initial_dtype)
        return pooled_output, None


# =============================================================================
# CellBert Model
# =============================================================================


class CellBertModel(BertPreTrainedModel):
    """
    BERT-based encoder for TME cell embeddings.

    This model processes TME cell information through:
    1. CellBertEmbeddings: Convert cell indices to embeddings
    2. BertEncoder: Transformer encoder layers
    3. Pooling: Attention or mean pooling for sequence aggregation

    Supports both "attention" and "mean" pooling strategies.

    Attributes:
        tme_config: TME configuration.
        embedder: CellBertEmbeddings module.
        encoder: BERT encoder.
        pooler: AttentionPool or MeanPool module.

    Input:
        - tme_cell_ids: (batch_size, seq_len) - Cell indices
        - tme_cell_embs: (batch_size, seq_len, hidden_size) - Pre-computed embeddings
        - tme_celltype_ids: (batch_size, seq_len) - Cell type indices
        - attention_mask: (batch_size, seq_len) - Attention mask

    Output:
        BaseModelOutputWithPoolingAndCrossAttentions with:
        - last_hidden_state: (batch_size, seq_len, hidden_size)
        - pooler_output: (batch_size, hidden_size)
        - pooled_attention_score: (batch_size, seq_len, 1) - From attention pooling

    Example:
        >>> config = get_default_tme_config(
        ...     gf_tme_emb_path="path/to/emb",
        ...     pool_type="attention"
        ... )
        >>> model = CellBertModel(config)
        >>> tme_cells = torch.randint(1, 1000, (4, 256))
        >>> tme_types = torch.randint(1, 9, (4, 256))
        >>> outputs = model(tme_cell_ids=tme_cells, tme_celltype_ids=tme_types)
    """

    def __init__(self, tme_config: BertConfig) -> None:
        """
        Initialize CellBertModel.

        Args:
            tme_config: TME configuration with model parameters.
        """
        super().__init__(tme_config)
        self.tme_config = tme_config

        self.embedder = CellBertEmbeddings(tme_config)
        self.encoder = BertEncoder(tme_config)

        # Initialize pooling layer
        pool_type = getattr(tme_config, "pool_type", "attention")
        if pool_type == "attention":
            self.pooler = AttentionPool(tme_config.hidden_size)
        elif pool_type == "mean":
            self.pooler = MeanPool()

        self.attn_implementation = tme_config._attn_implementation
        self.position_embedding_type = tme_config.position_embedding_type

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings layer."""
        return self.embedder.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the input embeddings layer."""
        self.embedder.word_embeddings = value

    def forward(
        self,
        tme_cell_ids: Optional[torch.Tensor] = None,
        tme_cell_embs: Optional[torch.Tensor] = None,
        tme_celltype_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        Forward pass of CellBertModel.

        Args:
            tme_cell_ids: TME cell indices.
                Shape: (batch_size, seq_len)
            tme_cell_embs: Pre-computed TME cell embeddings.
                Shape: (batch_size, seq_len, hidden_size)
            tme_celltype_ids: TME cell type indices.
                Shape: (batch_size, seq_len)
            attention_mask: Attention mask for padding.
                Shape: (batch_size, seq_len)
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return dict or tuple.

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions with additional
            'pooled_attention_score' key from attention pooling.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.tme_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.tme_config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.tme_config.use_return_dict
        )

        # Get embeddings
        embedding_output = self.embedder(
            tme_cell_ids=tme_cell_ids,
            tme_cell_embs=tme_cell_embs,
            tme_celltype_ids=tme_celltype_ids,
        )

        # Encode
        do_encoder = self.tme_config.num_hidden_layers > 0
        if do_encoder:
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
        else:
            sequence_output = embedding_output

        # Pool
        if self.pooler is not None:
            pooled_output, pooled_attention_score = self.pooler(
                sequence_output, attention_mask
            )
        else:
            pooled_output, pooled_attention_score = None, None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        output = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )
        output["pooled_attention_score"] = pooled_attention_score
        return output


# =============================================================================
# TmeBert Embeddings
# =============================================================================


class TmeBertEmbeddings(nn.Module):
    """
    GeneFormer-style input embeddings with TME context fusion.

    This module creates embeddings for input tokens (genes) and optionally
    fuses them with TME context embeddings.

    The embedding combines:
    - Word (gene) embeddings
    - Position embeddings
    - Token type embeddings
    - TME context embeddings (optional, scaled by tme_alpha)

    Attributes:
        config: Model configuration.
        word_embeddings: Gene/token embedding table.
        position_embeddings: Position embedding table.
        token_type_embeddings: Token type embedding table.
        LayerNorm: Layer normalization.
        dropout: Dropout layer.

    Input:
        - input_ids: (batch_size, seq_len) - Gene indices
        - token_type_ids: (batch_size, seq_len) - Token type indices
        - position_ids: (batch_size, seq_len) - Position indices
        - inputs_embeds: (batch_size, seq_len, hidden_size) - Pre-computed embeddings
        - context_embeds: (batch_size, hidden_size) - TME context to fuse
        - past_key_values_length: int - For incremental decoding

    Output:
        - torch.Tensor: (batch_size, seq_len, hidden_size)

    Example:
        >>> config = BertConfig(vocab_size=1000, hidden_size=512)
        >>> embedder = TmeBertEmbeddings(config)
        >>> input_ids = torch.randint(1, 1000, (4, 128))
        >>> context = torch.randn(4, 512)  # TME context
        >>> embeddings = embedder(input_ids=input_ids, context_embeds=context)
    """

    def __init__(self, config: BertConfig) -> None:
        """
        Initialize TmeBertEmbeddings.

        Args:
            config: Model configuration with embedding parameters.
        """
        super().__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        context_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """
        Generate embeddings with optional TME context fusion.

        Args:
            input_ids: Input token (gene) indices.
                Shape: (batch_size, seq_len)
            token_type_ids: Token type indices.
                Shape: (batch_size, seq_len)
            position_ids: Position indices.
                Shape: (batch_size, seq_len)
            inputs_embeds: Pre-computed embeddings instead of input_ids.
                Shape: (batch_size, seq_len, hidden_size)
            context_embeds: TME context embeddings to fuse.
                Shape: (batch_size, hidden_size)
            past_key_values_length: Length of past keys for incremental decoding.

        Returns:
            torch.Tensor: Combined embeddings.
                Shape: (batch_size, seq_len, hidden_size)
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Compute position IDs
        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Compute token type IDs
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        # Get base embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Fuse TME context if provided
        if context_embeds is not None:
            tme_alpha = self.config.tme_config["tme_alpha"]
            # Expand context: (batch_size, 1, hidden_size) -> (batch_size, seq_len, hidden_size)
            context_embeds = context_embeds.unsqueeze(1).expand(
                -1, embeddings.shape[1], -1
            )
            embeddings = embeddings + tme_alpha * context_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# =============================================================================
# Cross-Attention Module
# =============================================================================


class CrossAttention(nn.Module):
    """
    Cross-attention module for integrating context from another sequence.

    Uses PyTorch's MultiheadAttention to attend from a query sequence
    to a key-value (context) sequence.

    Attributes:
        self: MultiheadAttention layer.
        output: Output projection layer.
        dropout: Dropout after attention.

    Input:
        - query: (batch_size, seq_len_q, hidden_size)
        - key_value: (batch_size, seq_len_kv, hidden_size)
        - attention_mask: (batch_size, seq_len_kv) - Key padding mask

    Output:
        - torch.Tensor: (batch_size, seq_len_q, hidden_size)

    Example:
        >>> config = BertConfig(hidden_size=512, num_attention_heads=8)
        >>> cross_attn = CrossAttention(config)
        >>> query = torch.randn(4, 10, 512)
        >>> context = torch.randn(4, 256, 512)
        >>> output = cross_attn(query, context)
    """

    def __init__(self, config: BertConfig) -> None:
        """
        Initialize cross-attention.

        Args:
            config: Configuration with attention parameters.
        """
        super().__init__()
        self.self = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-attention.

        Args:
            query: Query sequence (e.g., BERT-B hidden states).
                Shape: (batch_size, seq_len_q, hidden_size)
            key_value: Key-value sequence (e.g., BERT-A hidden states).
                Shape: (batch_size, seq_len_kv, hidden_size)
            attention_mask: Key padding mask.
                Shape: (batch_size, seq_len_kv)

        Returns:
            torch.Tensor: Attended output.
                Shape: (batch_size, seq_len_q, hidden_size)
        """
        # Handle 4-bit quantized weights (bitsandbytes)
        from bitsandbytes.functional import dequantize_4bit

        if self.self.out_proj.weight.dtype == torch.uint8:
            quant_state = getattr(self.self.out_proj.weight, "quant_state", None)
            if quant_state is None:
                dequantized_weight = self.self.out_proj.weight.to(self.compute_dtype)
            else:
                dequantized_weight = dequantize_4bit(
                    self.self.out_proj.weight,
                    quant_state=quant_state,
                    blocksize=64,
                    quant_type="nf4",
                )

            original_weight = self.self.out_proj.weight
            self.self.out_proj.weight = torch.nn.Parameter(
                dequantized_weight, requires_grad=False
            )

        # Apply multi-head attention
        attn_output, _ = self.self(
            query, key_value, key_value, key_padding_mask=attention_mask
        )
        attn_output = self.dropout(attn_output)
        attn_output = self.output(attn_output)

        return attn_output


# =============================================================================
# Custom BERT Layer with Cross-Attention
# =============================================================================


class CustomBertLayer(BertLayer):
    """
    BERT layer with optional cross-attention support.

    Extends standard BERT layer with the ability to attend to an external
    context sequence via cross-attention.

    Attributes:
        use_cross_attention: Whether cross-attention is enabled.
        cross_attention: Cross-attention module (if enabled).
        cross_attention_output: Output projection for cross-attention.
        cross_attention_dropout: Dropout for cross-attention output.

    Input:
        - hidden_states: (batch_size, seq_len, hidden_size)
        - attention_mask: (batch_size, seq_len)
        - context_hidden_states: (batch_size, seq_len_ctx, hidden_size)
        - context_attention_mask: (batch_size, seq_len_ctx)

    Output:
        - Tuple: (hidden_states, attentions, ...)
    """

    def __init__(self, config: BertConfig, use_cross_attention: bool = False) -> None:
        """
        Initialize CustomBertLayer.

        Args:
            config: BERT configuration.
            use_cross_attention: Enable cross-attention.
        """
        super().__init__(config)
        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            self.cross_attention = CrossAttention(config)
            self.cross_attention_output = nn.Linear(
                config.hidden_size, config.hidden_size
            )
            self.cross_attention_dropout = nn.Dropout(
                config.attention_probs_dropout_prob
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_hidden_states: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Forward pass with optional cross-attention.

        Args:
            hidden_states: Input hidden states.
                Shape: (batch_size, seq_len, hidden_size)
            attention_mask: Self-attention mask.
                Shape: (batch_size, seq_len)
            context_hidden_states: Context for cross-attention.
                Shape: (batch_size, seq_len_ctx, hidden_size)
            context_attention_mask: Cross-attention mask.
                Shape: (batch_size, seq_len_ctx)

        Returns:
            Tuple containing:
                - layer_output: Output hidden states
                - attention weights (if output_attentions)
        """
        # Standard BERT self-attention
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # Cross-attention with context
        if self.use_cross_attention and context_hidden_states is not None:
            cross_attention_output = self.cross_attention(
                query=attention_output,
                key_value=context_hidden_states,
                attention_mask=context_attention_mask,
            )
            cross_attention_output = self.cross_attention_dropout(
                cross_attention_output
            )
            cross_attention_output = self.cross_attention_output(cross_attention_output)
            # Residual connection
            attention_output = attention_output + cross_attention_output

        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


# =============================================================================
# TmeBert Encoder
# =============================================================================


class TmeBertEncoder(nn.Module):
    """
    BERT encoder with optional TME cross-attention integration.

    This encoder can integrate TME context information via cross-attention
    in specified transformer layers.

    Attributes:
        do_tme_cross: Whether TME cross-attention is enabled.
        tme_cross_layers: List of layer indices for cross-attention.
        layer: ModuleList of CustomBertLayer.

    Input:
        - hidden_states: (batch_size, seq_len, hidden_size)
        - attention_mask: (batch_size, seq_len)
        - output_attentions: bool
        - output_hidden_states: bool
        - return_dict: bool

    Output:
        BaseModelOutputWithPastAndCrossAttentions

    Example:
        >>> config = BertConfig(hidden_size=512, num_hidden_layers=6)
        >>> config.use_tme = True
        >>> config.tme_config = {"tme_add": "cross", "tme_cross_layers": [0, 1]}
        >>> encoder = TmeBertEncoder(config)
    """

    def __init__(self, config: BertConfig) -> None:
        """
        Initialize TmeBertEncoder.

        Args:
            config: Configuration with encoder parameters.
        """
        super().__init__()

        self.do_tme_cross = getattr(config, "use_tme", False) and config.tme_config[
            "tme_add"
        ] == "cross"
        self.tme_cross_layers = []

        self.layer = nn.ModuleList(
            CustomBertLayer(config, use_cross_attention=True)
            if idx in self.tme_cross_layers
            else CustomBertLayer(config, use_cross_attention=False)
            for idx in range(config.num_hidden_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """
        Forward pass of TmeBertEncoder.

        Args:
            hidden_states: Input hidden states.
                Shape: (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask.
                Shape: (batch_size, seq_len)
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return dict or tuple.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# =============================================================================
# TmeBert Model
# =============================================================================


class TmeBertModel(BertPreTrainedModel):
    """
    Main TME-aware BERT model supporting fuse and cross integration modes.

    This model combines:
    1. TmeBertEmbeddings: Gene/token embeddings with optional TME fusion
    2. TmeBertEncoder: Transformer encoder with optional cross-attention
    3. BertPooler: Standard BERT pooler for [CLS] aggregation

    TME Integration Modes:
        - "fuse": TME context fused at embedding layer (scaled by tme_alpha)
        - "cross": TME context integrated via cross-attention in encoder layers

    Attributes:
        config: Model configuration.
        embeddings: TmeBertEmbeddings module.
        encoder: TmeBertEncoder module.
        pooler: BertPooler module.
        cell_bert: CellBertModel for TME context (if fuse mode).

    Input:
        - input_ids: (batch_size, seq_len) - Gene indices
        - attention_mask: (batch_size, seq_len) - Attention mask
        - token_type_ids: (batch_size, seq_len) - Token type indices
        - inputs_embeds: (batch_size, seq_len, hidden_size) - Pre-computed embeddings
        - tme_cells: (batch_size, tme_seq_len) - TME cell indices
        - tme_cell_embs: (batch_size, tme_seq_len, hidden_size) - TME embeddings
        - tme_types: (batch_size, tme_seq_len) - TME cell type indices

    Output:
        BaseModelOutputWithPoolingAndCrossAttentions:
        - last_hidden_state: (batch_size, seq_len, hidden_size)
        - pooler_output: (batch_size, hidden_size)

    Example:
        >>> config = BertConfig(vocab_size=1000, hidden_size=512)
        >>> config.use_tme = True
        >>> config.tme_config = {
        ...     "tme_add": "fuse",
        ...     "tme_alpha": 0.2,
        ...     "gf_tme_emb_path": "path/to/emb",
        ...     ...
        ... }
        >>> model = TmeBertModel(config)
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     tme_cells=tme_cells,
        ...     tme_types=tme_types
        ... )
    """

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True) -> None:
        """
        Initialize TmeBertModel.

        Args:
            config: Model configuration.
            add_pooling_layer: Whether to add pooling layer.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = TmeBertEmbeddings(config)
        self.encoder = TmeBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize TME fusion model if needed
        self.do_tme_fusion = (
            getattr(self.config, "use_tme", False)
            and self.config.tme_config["tme_add"] == "fuse"
        )
        if self.do_tme_fusion:
            self.cell_bert = CellBertModel(BertConfig(**config.tme_config))

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings layer."""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the input embeddings layer."""
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        tme_cells: Optional[torch.Tensor] = None,
        tme_cell_embs: Optional[torch.Tensor] = None,
        tme_types: Optional[torch.Tensor] = None,
        cell_id: Optional[torch.Tensor] = None,   # keep for compatible
        sample_id: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        Forward pass of TmeBertModel.

        Args:
            input_ids: Input token (gene) indices.
                Shape: (batch_size, seq_len)
            attention_mask: Attention mask for padding.
                Shape: (batch_size, seq_len)
            token_type_ids: Token type indices.
                Shape: (batch_size, seq_len)
            inputs_embeds: Pre-computed embeddings.
                Shape: (batch_size, seq_len, hidden_size)
            tme_cells: TME cell indices.
                Shape: (batch_size, tme_seq_len)
            tme_cell_embs: Pre-computed TME cell embeddings.
                Shape: (batch_size, tme_seq_len, hidden_size)
            tme_types: TME cell type indices.
                Shape: (batch_size, tme_seq_len)
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return dict or tuple.

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = (
            input_ids.device if input_ids is not None else inputs_embeds.device
        )

        # Compute token type IDs
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[
                    :, :seq_length
                ]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # Get TME context embeddings (fuse mode)
        if self.do_tme_fusion:
            context_embeds = self.cell_bert(
                tme_cell_ids=tme_cells,
                tme_cell_embs=tme_cell_embs,
                tme_celltype_ids=tme_types,
            ).pooler_output
        else:
            context_embeds = None

        # Get input embeddings (with optional TME fusion)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            context_embeds=context_embeds,
        )

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and not output_attentions
        )

        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask, embedding_output.dtype, tgt_len=seq_length
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape
            )

        # Encode
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        # Pool
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# =============================================================================
# Downstream Task Heads
# =============================================================================


class TmeBertForMaskedLM(BertPreTrainedModel):
    """
    TME-aware BERT for Masked Language Modeling (MLM) pretraining.

    This model uses TmeBertModel as the backbone and adds an MLM head
    for predicting masked tokens.

    Attributes:
        bert: TmeBertModel backbone.
        cls: MLM head (BertOnlyMLMHead).

    Input:
        Same as TmeBertModel, plus:
        - labels: (batch_size, seq_len) - MLM labels (-100 for ignore)

    Output:
        MaskedLMOutput:
        - loss: Scalar MLM loss
        - logits: (batch_size, seq_len, vocab_size)
        - hidden_states, attentions

    Example:
        >>> config = BertConfig(vocab_size=1000, hidden_size=512)
        >>> config.use_tme = True
        >>> config.tme_config = {...}
        >>> model = TmeBertForMaskedLM(config)
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     tme_cells=tme_cells,
        ...     labels=labels
        ... )
        >>> loss = outputs.loss
    """

    _tied_weights_keys = [
        "predictions.decoder.bias",
        "cls.predictions.decoder.weight",
    ]

    def __init__(self, config: BertConfig) -> None:
        """
        Initialize TmeBertForMaskedLM.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

        self.bert = TmeBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.post_init()

    def get_output_embeddings(self) -> nn.Module:
        """Get the output embeddings (decoder)."""
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        """Set the output embeddings."""
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        tme_cells: Optional[torch.Tensor] = None,
        tme_cell_embs: Optional[torch.Tensor] = None,
        tme_types: Optional[torch.Tensor] = None,
        cell_id: Optional[torch.Tensor] = None,   # keep for compatible
        sample_id: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """
        Forward pass for MLM.

        Args:
            input_ids: Input token indices.
                Shape: (batch_size, seq_len)
            attention_mask: Attention mask.
                Shape: (batch_size, seq_len)
            token_type_ids: Token type indices.
                Shape: (batch_size, seq_len)
            inputs_embeds: Pre-computed embeddings.
            tme_cells: TME cell indices.
            tme_cell_embs: TME cell embeddings.
            tme_types: TME cell type indices.
            labels: MLM labels (-100 for ignore/padding).
                Shape: (batch_size, seq_len)
            output_attentions: Output attention weights.
            output_hidden_states: Output hidden states.
            return_dict: Return dict or tuple.

        Returns:
            MaskedLMOutput with loss and logits.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            tme_cells=tme_cells,
            tme_cell_embs=tme_cell_embs,
            tme_types=tme_types,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TmeBertForSequenceClassification(BertPreTrainedModel):
    """
    TME-aware BERT for sequence classification tasks.

    Supports:
    - Regression (single or multi-output)
    - Single-label classification
    - Multi-label classification

    Automatically detects problem type based on num_labels and labels dtype.

    Attributes:
        num_labels: Number of output labels.
        bert: TmeBertModel backbone.
        dropout: Dropout for classifier.
        classifier: Linear classification head.

    Input:
        Same as TmeBertModel, plus:
        - labels: (batch_size, num_labels) or (batch_size,) - Task labels

    Output:
        SequenceClassifierOutput:
        - loss: Scalar loss
        - logits: (batch_size, num_labels)
        - hidden_states, attentions

    Example:
        >>> config = BertConfig(vocab_size=1000, hidden_size=512, num_labels=10)
        >>> config.use_tme = True
        >>> config.tme_config = {...}
        >>> model = TmeBertForSequenceClassification(config)
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     tme_cells=tme_cells,
        ...     labels=labels
        ... )
    """

    def __init__(self, config: BertConfig) -> None:
        """
        Initialize TmeBertForSequenceClassification.

        Args:
            config: Model configuration with num_labels.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = TmeBertModel(config)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        tme_cells: Optional[torch.Tensor] = None,
        tme_cell_embs: Optional[torch.Tensor] = None,
        tme_types: Optional[torch.Tensor] = None,
        cell_id: Optional[torch.Tensor] = None,   # keep for compatible
        sample_id: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        Forward pass for sequence classification.

        Args:
            input_ids: Input token indices.
                Shape: (batch_size, seq_len)
            attention_mask: Attention mask.
                Shape: (batch_size, seq_len)
            token_type_ids: Token type indices.
                Shape: (batch_size, seq_len)
            inputs_embeds: Pre-computed embeddings.
            tme_cells: TME cell indices.
            tme_cell_embs: TME cell embeddings.
            tme_types: TME cell type indices.
            labels: Classification/regression labels.
            output_attentions: Output attention weights.
            output_hidden_states: Output hidden states.
            return_dict: Return dict or tuple.

        Returns:
            SequenceClassifierOutput with loss and logits.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            tme_cells=tme_cells,
            tme_cell_embs=tme_cell_embs,
            tme_types=tme_types,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            dtype = logits.dtype
            labels = labels.to(dtype=dtype)

            # Auto-detect problem type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.float().view(-1, self.num_labels), labels.long().view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TmeBertForMultiGeneExpressionPrediction(BertPreTrainedModel):
    """
    TME-aware BERT for multi-gene expression prediction (multi-task regression).

    Predicts expression levels for multiple genes simultaneously, with support
    for missing values (masked with -1.0).

    Attributes:
        num_labels: Number of genes to predict.
        bert: TmeBertModel backbone.
        dropout: Dropout for regressor.
        regressor: MLP head for regression.

    Input:
        Same as TmeBertModel, plus:
        - labels: (batch_size, num_genes) - Gene expression values (-1.0 for missing)

    Output:
        Dict containing:
        - loss: Average MSE loss across tasks
        - losses: List of per-task losses
        - logits: (batch_size, num_genes) - Predicted expression
        - hidden_states, attentions

    Example:
        >>> config = BertConfig(vocab_size=1000, hidden_size=512, num_labels=50)
        >>> config.use_tme = True
        >>> config.tme_config = {...}
        >>> model = TmeBertForMultiGeneExpressionPrediction(config)
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     tme_cells=tme_cells,
        ...     labels=geps  # Gene expression profiles
        ... )
        >>> predictions = outputs["logits"]
    """

    def __init__(self, config: BertConfig) -> None:
        """
        Initialize TmeBertForMultiGeneExpressionPrediction.

        Args:
            config: Model configuration with num_labels (number of genes).
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = TmeBertModel(config)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.regressor = build_mlp_head(
            config.hidden_size, self.num_labels, num_layers=3, dropout=classifier_dropout
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        tme_cells: Optional[torch.Tensor] = None,
        tme_cell_embs: Optional[torch.Tensor] = None,
        tme_types: Optional[torch.Tensor] = None,
        cell_id: Optional[torch.Tensor] = None,   # keep for compatible
        sample_id: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict:
        """
        Forward pass for multi-gene expression prediction.

        Args:
            input_ids: Input token indices.
            attention_mask: Attention mask.
            token_type_ids: Token type indices.
            inputs_embeds: Pre-computed embeddings.
            tme_cells: TME cell indices.
            tme_cell_embs: TME cell embeddings.
            tme_types: TME cell type indices.
            labels: Gene expression values (-1.0 for missing/masked).
                Shape: (batch_size, num_genes)
            output_attentions: Output attention weights.
            output_hidden_states: Output hidden states.
            return_dict: Return dict or tuple.

        Returns:
            Dict with loss, per-task losses, logits, hidden_states, attentions.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            tme_cells=tme_cells,
            tme_cell_embs=tme_cell_embs,
            tme_types=tme_types,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        reg_output = self.regressor(pooled_output)

        avg_loss, task_losses = None, None
        if labels is not None:
            num_tasks = reg_output.size(1)
            total_loss = 0.0
            mask_value = -1.0
            task_losses = []
            valid_task_count = 0

            for task_idx in range(num_tasks):
                pred = reg_output[:, task_idx]
                target = labels[:, task_idx]
                mask = target != mask_value

                if mask.sum() > 0:
                    task_loss = nn.functional.mse_loss(
                        pred[mask], target[mask], reduction="mean"
                    )
                    total_loss += task_loss
                    task_losses.append(task_loss.item())
                    valid_task_count += 1
                else:
                    task_losses.append(None)

            avg_loss = (
                total_loss / valid_task_count if valid_task_count > 0 else torch.tensor(0.0)
            )

        return {
            "loss": avg_loss,
            "losses": task_losses,
            "logits": reg_output,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }


class TmeBertForCellClassification(BertPreTrainedModel):
    """
    TME-aware BERT for cell-level multi-class classification.

    Classifies cells into predefined cell types based on their gene expression
    and optionally integrated TME context.

    Attributes:
        num_labels: Number of cell types.
        bert: TmeBertModel backbone.
        dropout: Dropout for classifier.
        classifier: MLP classification head.

    Input:
        Same as TmeBertModel, plus:
        - labels: (batch_size,) - Cell type labels (long tensor)

    Output:
        SequenceClassifierOutput:
        - loss: Cross-entropy loss
        - logits: (batch_size, num_labels)
        - hidden_states, attentions

    Example:
        >>> config = BertConfig(vocab_size=1000, hidden_size=512, num_labels=20)
        >>> config.use_tme = True
        >>> config.tme_config = {...}
        >>> model = TmeBertForCellClassification(config)
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     tme_cells=tme_cells,
        ...     labels=cell_types
        ... )
        >>> predictions = outputs.logits.argmax(dim=-1)
    """

    def __init__(self, config: BertConfig) -> None:
        """
        Initialize TmeBertForCellClassification.

        Args:
            config: Model configuration with num_labels (cell types).
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = TmeBertModel(config)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = build_mlp_head(
            config.hidden_size,
            self.num_labels,
            num_layers=3,
            dropout=classifier_dropout,
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        tme_cells: Optional[torch.Tensor] = None,
        tme_cell_embs: Optional[torch.Tensor] = None,
        tme_types: Optional[torch.Tensor] = None,
        cell_id: Optional[torch.Tensor] = None,   # keep for compatible
        sample_id: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        Forward pass for cell classification.

        Args:
            input_ids: Input token indices.
            attention_mask: Attention mask.
            token_type_ids: Token type indices.
            inputs_embeds: Pre-computed embeddings.
            tme_cells: TME cell indices.
            tme_cell_embs: TME cell embeddings.
            tme_types: TME cell type indices.
            labels: Cell type labels (long tensor).
                Shape: (batch_size,)
            output_attentions: Output attention weights.
            output_hidden_states: Output hidden states.
            return_dict: Return dict or tuple.

        Returns:
            SequenceClassifierOutput with loss and logits.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            tme_cells=tme_cells,
            tme_cell_embs=tme_cell_embs,
            tme_types=tme_types,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )