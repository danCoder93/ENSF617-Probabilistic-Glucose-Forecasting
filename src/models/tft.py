# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This code is adapted from the NVIDIA/DeepLearningExamples repository:
# https://github.com/NVIDIA/DeepLearningExamples
# Modifications by danCoder93 (March 24, 2026).
# Updated to integrate PyTorch Lightning.
#
# AI-assisted maintenance note (April 1, 2026):
# The GRN construction sites in this existing module were refined with AI
# assistance under direct user guidance. The updates route shared defaults
# through `GRN.from_tft_config(...)` so the current TFT implementation is easier
# to maintain and slightly more robust, without changing the underlying model
# design or claiming new architectural ideas.
#
# Context:
# this file remains closer to the upstream TFT lineage than several other model
# modules in the repository, so the most valuable comments here are the ones
# that explain how the implementation maps onto the project-specific grouped
# input contract and how the major sub-blocks relate to the fused model.

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.functional import glu, elu
# LazyModules are initialized without any input tensor shape and is dynamically calculated by a dry run before any actual training. 
from torch.nn.modules.lazy import LazyModuleMixin

from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from typing import Any, Dict, Tuple, Optional, List, Mapping, cast

from torch.nn import LayerNorm

from models.grn import GLU, GRN

from config import TFTConfig

# The grouped TFT input contract intentionally allows some feature groups to be
# absent at runtime. For example:
# - a dataset may have no static categorical variables
# - a particular experiment may have no observed categorical history
# - the fused model may pass zero-width tensors for missing groups and convert
#   them to `None` before calling TFT
#
# Using a read-only `Mapping[str, Optional[Tensor]]` instead of `Dict[str,
# Tensor]` captures that reality more accurately than the earlier type:
# - `Mapping` is the right abstraction because TFT only reads from the input; it
#   does not mutate the caller's dictionary
# - `Optional[Tensor]` matches the actual runtime contract used throughout the
#   fused model, where "missing feature group" is represented explicitly as
#   `None`
#
# This also resolves the Pylance invariance issue triggered when
# `FusedModel._build_tft_inputs(...)` returned `dict[str, Tensor | None]`.
TFTInputBatch = Mapping[str, Optional[Tensor]]

def fused_pointwise_linear_v1(x, a, b):
    """
    Apply the learned continuous-feature embedding transform for generic inputs.

    Context:
    this is the compact broadcasted form of multiplying each scalar continuous
    input by its learned embedding vector and then adding a learned bias.
    """
    # Multiply each scalar continuous feature value by its learnable embedding
    # vector and add a per-feature bias, producing one hidden vector per input
    # feature without writing the equivalent einsum more verbosely.
    #
    # Shape mechanics:
    # - `x` starts as `[batch, time?, features]`
    # - `x.unsqueeze(-1)` becomes `[batch, time?, features, 1]`
    # - `a` has shape `[features, hidden]`
    # - broadcasting yields `[batch, time?, features, hidden]`
    #
    # The result is "one hidden embedding vector per continuous feature" while
    # preserving batch and optional time axes.
    out = torch.mul(x.unsqueeze(-1), a)
    out = out + b
    return out

def fused_pointwise_linear_v2(x, a, b):
    """
    Apply the learned continuous-feature embedding transform for target history.

    Context:
    this variant is shaped for the temporal target path, which always carries a
    time dimension and is handled separately from the other continuous groups.
    """
    # Same idea as `fused_pointwise_linear_v1`, but shaped for the target path
    # where the target tensor is always temporal and handled separately from the
    # generic continuous input groups.
    #
    # The target path keeps an explicit temporal axis because TFT uses the
    # historical target trajectory as one of the historical variable groups in
    # its encoder-side stream.
    out = x.unsqueeze(3) * a
    out = out + b
    return out


class TFTEmbedding(Module):
    """
    Convert grouped raw inputs into per-variable hidden embeddings.

    This stage is the front door to TFT:
    - categorical variables become learned lookup embeddings
    - continuous variables become learned linear embeddings
    - each variable keeps its own slot so the later variable-selection network
      can decide which inputs matter most at each stage
    """

    def __init__(self, config, initialize_cont_params=True):
        # `initialize_cont_params=False` is used by `LazyEmbedding` so the
        # continuous embedding parameters can be materialized only after the
        # real grouped-input widths are known at runtime.
        super().__init__()
        self.s_cat_inp_lens    = config.static_categorical_inp_lens
        self.t_cat_k_inp_lens  = config.temporal_known_categorical_inp_lens
        self.t_cat_o_inp_lens  = config.temporal_observed_categorical_inp_lens
        self.s_cont_inp_size   = config.static_continuous_inp_size
        self.t_cont_k_inp_size = config.temporal_known_continuous_inp_size
        self.t_cont_o_inp_size = config.temporal_observed_continuous_inp_size
        self.t_tgt_size        = config.temporal_target_size

        self.hidden_size = config.hidden_size

        # There are 7 types of input:
        # 1. Static categorical
        # 2. Static continuous - numbers
        # 3. Temporal known a priori categorical
        # 4. Temporal known a priori continuous
        # 5. Temporal observed categorical
        # 6. Temporal observed continuous
        # 7. Temporal observed targets (time series obseved so far - Output)

        # Each categorical variable gets its own embedding table so the later
        # variable-selection network can reason about variables separately
        # rather than after they have already been merged together.
        self.s_cat_embed = nn.ModuleList([
            nn.Embedding(n, self.hidden_size) for n in self.s_cat_inp_lens]) if self.s_cat_inp_lens else None
        self.t_cat_k_embed = nn.ModuleList([
            nn.Embedding(n, self.hidden_size) for n in self.t_cat_k_inp_lens]) if self.t_cat_k_inp_lens else None
        self.t_cat_o_embed = nn.ModuleList([
            nn.Embedding(n, self.hidden_size) for n in self.t_cat_o_inp_lens]) if self.t_cat_o_inp_lens else None

        if initialize_cont_params:
            self.s_cont_embedding_vectors = Parameter(
                torch.Tensor(self.s_cont_inp_size, self.hidden_size)
            ) if self.s_cont_inp_size else None
            self.t_cont_k_embedding_vectors = Parameter(
                torch.Tensor(self.t_cont_k_inp_size, self.hidden_size)
            ) if self.t_cont_k_inp_size else None
            self.t_cont_o_embedding_vectors = Parameter(
                torch.Tensor(self.t_cont_o_inp_size, self.hidden_size)
            ) if self.t_cont_o_inp_size else None
            self.t_tgt_embedding_vectors = Parameter(
                torch.Tensor(self.t_tgt_size, self.hidden_size)
            )

            self.s_cont_embedding_bias = Parameter(
                torch.zeros(self.s_cont_inp_size, self.hidden_size)
            ) if self.s_cont_inp_size else None
            self.t_cont_k_embedding_bias = Parameter(
                torch.zeros(self.t_cont_k_inp_size, self.hidden_size)
            ) if self.t_cont_k_inp_size else None
            self.t_cont_o_embedding_bias = Parameter(
                torch.zeros(self.t_cont_o_inp_size, self.hidden_size)
            ) if self.t_cont_o_inp_size else None
            self.t_tgt_embedding_bias = Parameter(
                torch.zeros(self.t_tgt_size, self.hidden_size)
            )

            self.reset_parameters()


    def reset_parameters(self):
        # Continuous embedding matrices are initialized with Xavier so each
        # feature-specific hidden vector starts with a reasonable scale instead
        # of collapsing to zero or exploding. Bias terms start at zero so the
        # learned projection is initially centered around the raw scalar value.
        if self.s_cont_embedding_vectors is not None:
            assert self.s_cont_embedding_bias is not None
            # instead of just initializing weights by all 0 or infinity which will lead to gradient explosion/vanishing, initialize randomly using a xavier normal distribution
            torch.nn.init.xavier_normal_(self.s_cont_embedding_vectors)
            torch.nn.init.zeros_(self.s_cont_embedding_bias)
        if self.t_cont_k_embedding_vectors is not None:
            assert self.t_cont_k_embedding_bias is not None
            torch.nn.init.xavier_normal_(self.t_cont_k_embedding_vectors)
            torch.nn.init.zeros_(self.t_cont_k_embedding_bias)
        if self.t_cont_o_embedding_vectors is not None:
            assert self.t_cont_o_embedding_bias is not None
            torch.nn.init.xavier_normal_(self.t_cont_o_embedding_vectors)
            torch.nn.init.zeros_(self.t_cont_o_embedding_bias)
        if self.t_tgt_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.t_tgt_embedding_vectors)
            torch.nn.init.zeros_(self.t_tgt_embedding_bias)
        if self.s_cat_embed is not None:
            for module in self.s_cat_embed:
                if isinstance(module, nn.Embedding):
                    module.reset_parameters()
        if self.t_cat_k_embed is not None:
            for module in self.t_cat_k_embed:
                if isinstance(module, nn.Embedding):
                    module.reset_parameters()
        if self.t_cat_o_embed is not None:
            for module in self.t_cat_o_embed:
                if isinstance(module, nn.Embedding):
                    module.reset_parameters()

    def _apply_embedding(self,
            cat: Optional[Tensor], # categorical values
            cont: Optional[Tensor], # continuous values
            cat_emb: Optional[nn.ModuleList], # categorical embeddings
            cont_emb: Optional[Tensor], # continuous weights embeddings
            cont_bias: Optional[Tensor], # continuous bias embeddings
            ) -> Optional[Tensor]:
        # Convert one semantic feature group into per-variable hidden vectors.
        # This helper keeps categorical and continuous paths aligned so the rest
        # of the module can work in terms of "embedded variable slots" rather
        # than special-casing raw input types repeatedly.
        #
        # Important output convention:
        # - every returned tensor keeps a dedicated "variable" axis
        # - TFT does not merge variables immediately after embedding
        # - later variable-selection networks need to see the variables as
        #   separate slots so they can learn variable importance weights
        # For each categorical variable:
        #   - take its integer IDs from `cat[..., i]` across the batch (and time)
        #   - pass them through its corresponding embedding layer to get vectors of size hidden_size
        # Stack all variable embeddings along a new dimension (variables),
        # resulting in a tensor of shape:
        #   [batch, time, num_categorical_variables, hidden_size] (or without time for static inputs) 
        e_cat = torch.stack([embed(cat[...,i]) for i, embed in enumerate(cat_emb)], dim=-2) if cat is not None and cat_emb is not None else None
        if cont is not None:
            #the line below is equivalent to following einsums
            #e_cont = torch.einsum('btf,fh->bthf', cont, cont_emb)
            #e_cont = torch.einsum('bf,fh->bhf', cont, cont_emb)
            # take scalars of continuous input value in a [batch, time, feature index]
            # tensor to hidden vectors in a
            # [batch, time, feature index, hidden size] tensor
            e_cont = fused_pointwise_linear_v1(cont, cont_emb, cont_bias)
        else:
            e_cont = None

        if e_cat is not None and e_cont is not None:
            return torch.cat([e_cat, e_cont], dim=-2)
        elif e_cat is not None:
            return e_cat
        elif e_cont is not None:
            return e_cont
        else:
            return None

    def forward(self, x: TFTInputBatch):
        # Pull apart the semantic batch groups that FusedModel assembled for
        # TFT. Every key corresponds to one role in the shared schema.
        s_cat_inp = x.get('s_cat', None)
        s_cont_inp = x.get('s_cont', None)
        t_cat_k_inp = x.get('k_cat', None)
        t_cont_k_inp = x.get('k_cont', None)
        t_cat_o_inp = x.get('o_cat', None)
        t_cont_o_inp = x.get('o_cont', None)
        # `target` is the one temporal group that TFT always requires on the
        # encoder side: it represents the observed target history used to build
        # the historical input stream. Unlike optional covariate groups, this is
        # not something the model can sensibly skip.
        t_tgt_obs = x['target'] # Has to be present
        if t_tgt_obs is None:
            # Keep the runtime failure explicit even though the type alias now
            # allows optional values. The alias reflects the *overall* grouped
            # contract, but this particular key is semantically mandatory for
            # TFT's historical-target path.
            raise ValueError("TFT input batch must include a non-null 'target' tensor.")

        # Static inputs are expected to be equal for all timesteps
        # For memory efficiency there is no assert statement
        #
        # The dataset repeats static values across the encoder axis because that
        # makes batching simpler and keeps all sample groups aligned to the same
        # sequence contract. TFT only needs one copy, so we take the first
        # timestep here instead of carrying the redundant axis forward.
        s_cat_inp = s_cat_inp[:,0,:] if s_cat_inp is not None else None
        s_cont_inp = s_cont_inp[:,0,:] if s_cont_inp is not None else None

        # Convert each semantic input group into a tensor of per-variable
        # embeddings in the common hidden space.
        s_inp = self._apply_embedding(s_cat_inp,
                                      s_cont_inp,
                                      self.s_cat_embed,
                                      self.s_cont_embedding_vectors,
                                      self.s_cont_embedding_bias)
        t_known_inp = self._apply_embedding(t_cat_k_inp,
                                            t_cont_k_inp,
                                            self.t_cat_k_embed,
                                            self.t_cont_k_embedding_vectors,
                                            self.t_cont_k_embedding_bias)
        t_observed_inp = self._apply_embedding(t_cat_o_inp,
                                               t_cont_o_inp,
                                               self.t_cat_o_embed,
                                               self.t_cont_o_embedding_vectors,
                                               self.t_cont_o_embedding_bias)

        # Temporal observed targets
        # t_observed_tgt = torch.einsum('btf,fh->btfh', t_tgt_obs, self.t_tgt_embedding_vectors)
        t_observed_tgt = fused_pointwise_linear_v2(t_tgt_obs, self.t_tgt_embedding_vectors, self.t_tgt_embedding_bias)
        # The four returned tensors line up with the four semantic streams TFT
        # reasons about later:
        # - static inputs
        # - known temporal inputs
        # - observed temporal covariates
        # - observed target history
        #
        # Keeping them separate here is what allows the static encoder and the
        # historical/future variable-selection networks to consume the right
        # groups downstream.

        return s_inp, t_known_inp, t_observed_inp, t_observed_tgt

class LazyEmbedding(LazyModuleMixin, TFTEmbedding):
    """
    Lazily allocate continuous-embedding parameters once real input shapes are known.

    This is helpful because several input widths are only fully known after the
    data pipeline binds runtime metadata such as discovered feature counts.
    """

    cls_to_become = TFTEmbedding

    def __init__(self, config):
        # Call into `TFTEmbedding` without eager continuous-parameter creation.
        # This preserves the same public behavior while deferring shape binding
        # until the first real batch arrives.
        cast(Any, super()).__init__(config, initialize_cont_params=False)

        # if data contains static continuous inputs
        if config.static_continuous_inp_size:
            self.s_cont_embedding_vectors = UninitializedParameter()
            self.s_cont_embedding_bias = UninitializedParameter()
        else:
            self.s_cont_embedding_vectors = None
            self.s_cont_embedding_bias = None

        # if data contains future known continuous values
        if config.temporal_known_continuous_inp_size:
            self.t_cont_k_embedding_vectors = UninitializedParameter()
            self.t_cont_k_embedding_bias = UninitializedParameter()
        else:
            self.t_cont_k_embedding_vectors = None
            self.t_cont_k_embedding_bias = None

        # if data contains past observed values
        if config.temporal_observed_continuous_inp_size:
            self.t_cont_o_embedding_vectors = UninitializedParameter()
            self.t_cont_o_embedding_bias = UninitializedParameter()
        else:
            self.t_cont_o_embedding_vectors = None
            self.t_cont_o_embedding_bias = None

        # target value embeddings
        self.t_tgt_embedding_vectors = UninitializedParameter()
        self.t_tgt_embedding_bias = UninitializedParameter()

    def initialize_parameters(self, x):
        # Materialize any continuous-embedding parameters the first time the
        # module sees a real batch. After this, the module behaves like a normal
        # eagerly initialized embedding block.
        if cast(Any, self).has_uninitialized_params():
            s_cont_inp = x.get('s_cont', None)
            t_cont_k_inp = x.get('k_cont', None)
            t_cont_o_inp = x.get('o_cont', None)
            t_tgt_obs = x['target'] # Has to be present

            if s_cont_inp is not None:
                assert self.s_cont_embedding_vectors is not None
                assert self.s_cont_embedding_bias is not None
                # Materialize one learned embedding vector and one bias vector
                # per static continuous feature. The hidden size comes from the
                # bound TFT config, while the number of features comes from the
                # real runtime batch.
                self.s_cont_embedding_vectors.materialize((s_cont_inp.shape[-1], self.hidden_size))
                self.s_cont_embedding_bias.materialize((s_cont_inp.shape[-1], self.hidden_size))

            if t_cont_k_inp is not None:
                assert self.t_cont_k_embedding_vectors is not None
                assert self.t_cont_k_embedding_bias is not None
                self.t_cont_k_embedding_vectors.materialize((t_cont_k_inp.shape[-1], self.hidden_size))
                self.t_cont_k_embedding_bias.materialize((t_cont_k_inp.shape[-1], self.hidden_size))

            if t_cont_o_inp is not None:
                assert self.t_cont_o_embedding_vectors is not None
                assert self.t_cont_o_embedding_bias is not None
                self.t_cont_o_embedding_vectors.materialize((t_cont_o_inp.shape[-1], self.hidden_size))
                self.t_cont_o_embedding_bias.materialize((t_cont_o_inp.shape[-1], self.hidden_size))

            assert self.t_tgt_embedding_vectors is not None
            assert self.t_tgt_embedding_bias is not None
            self.t_tgt_embedding_vectors.materialize((t_tgt_obs.shape[-1], self.hidden_size))
            self.t_tgt_embedding_bias.materialize((t_tgt_obs.shape[-1], self.hidden_size))

            self.reset_parameters()

class VariableSelectionNetwork(Module):
    """
    Learn which variables matter most at a given stage of TFT.

    Input shape convention:
    - static path:   [batch, num_variables, hidden]
    - temporal path: [batch, time, num_variables, hidden]

    Output:
    - one blended hidden representation per example or timestep
    - one sparse weighting over the original variables
    """

    def __init__(self, config, num_inputs):
        super().__init__()
        # The joint GRN consumes the flattened per-variable embeddings and emits
        # one score per input variable. We keep the input/output sizing explicit
        # here because it is part of the variable-selection architecture, while
        # the shared hidden/dropout defaults come from the TFT config.
        self.joint_grn = GRN.from_tft_config(
            config,
            input_size=config.hidden_size * num_inputs,
            output_size=num_inputs,
            context_hidden_size=config.hidden_size,
        )
        # Each variable-specific GRN transforms one embedded variable into the
        # common hidden space before sparse weighting. These blocks all share
        # the same TFT defaults, so the factory keeps their construction
        # consistent without repeating the same low-level config plumbing.
        self.var_grns = nn.ModuleList(
            [GRN.from_tft_config(config, input_size=config.hidden_size) for _ in range(num_inputs)]
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        # Flatten the per-variable embeddings so the joint GRN can score all
        # variables together, then normalize those scores into attention-like
        # sparse weights over the variable axis.
        #
        # Conceptually this block answers two questions:
        # 1. "How should each variable be transformed into the common hidden
        #    representation space?"
        # 2. "How much should each transformed variable matter right now?"
        #
        # The joint GRN handles the second question by producing weights, while
        # the per-variable GRNs handle the first by transforming each variable
        # slot independently before the weighted reduction.
        Xi = torch.flatten(x, start_dim=-2)
        grn_outputs = self.joint_grn(Xi, c=context)
        sparse_weights = F.softmax(grn_outputs, dim=-1)
        transformed_embed_list = [m(x[...,i,:]) for i, m in enumerate(self.var_grns)]
        transformed_embed = torch.stack(transformed_embed_list, dim=-1)
        # The matmul below computes a weighted sum across the variable axis:
        # - temporal features: [batch, time, hidden, vars] x [batch, time, vars]
        # - static features:   [batch, hidden, vars] x [batch, vars]
        variable_ctx = torch.matmul(transformed_embed, sparse_weights.unsqueeze(-1)).squeeze(-1)
        # After the matmul, the dedicated variable axis has been reduced away.
        # The output is now one hidden vector per batch item or timestep,
        # representing the weighted mixture of the original variable slots.

        return variable_ctx, sparse_weights

class StaticCovariateEncoder(Module):
    """
    Turn static covariates into the context vectors consumed throughout TFT.

    These four vectors seed different downstream roles:
    - `cs`: context for variable selection
    - `ce`: context for static enrichment
    - `ch`: initial hidden state for the temporal LSTMs
    - `cc`: initial cell state for the temporal LSTMs
    """

    def __init__(self, config):
        super().__init__()
        self.vsn = VariableSelectionNetwork(config, config.num_static_vars)
        # The static encoder produces four distinct context vectors used by the
        # downstream recurrent and enrichment stages. The GRNs themselves are
        # architecturally identical, so they now inherit shared defaults from
        # one config-backed construction path.
        self.context_grns = nn.ModuleList(
            [GRN.from_tft_config(config, input_size=config.hidden_size) for _ in range(4)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        variable_ctx, sparse_weights = self.vsn(x)
        # `variable_ctx` is the single static summary vector after TFT has
        # decided which static variables matter most. From that one summary, the
        # model derives multiple context vectors for different downstream roles
        # rather than forcing each downstream stage to relearn the static view
        # from scratch.

        # Context vectors:
        # - `cs`: variable-selection context
        # - `ce`: static-enrichment context
        # - `ch`: initial recurrent hidden state
        # - `cc`: initial recurrent cell state
        cs, ce, ch, cc = [m(variable_ctx) for m in self.context_grns]

        return cs, ce, ch, cc


class InterpretableMultiHeadAttention(Module):
    """
    Causal self-attention over the combined historical+future temporal stream.

    TFT uses attention after recurrent encoding so the model can revisit
    relevant timesteps with a flexible dependency pattern while still obeying a
    strict causal mask.
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        assert config.hidden_size % config.n_head == 0
        self.d_head = config.hidden_size // config.n_head
        self.qkv_linears = nn.Linear(config.hidden_size, (2 * self.n_head + 1) * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_head, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.out_dropout = nn.Dropout(config.dropout)
        self.scale = self.d_head**-0.5
        # The mask prevents any position from attending to positions to its
        # right, preserving the forecasting constraint during both training and
        # inference.
        self.register_buffer("_mask", torch.triu(torch.full((config.example_length, config.example_length), float('-inf')), 1).unsqueeze(0))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        bs, t, h_size = x.shape
        # The linear layer emits:
        # - one query vector per head
        # - one key vector per head
        # - one shared value vector
        #
        # TFT's "interpretable" attention variant shares the value projection
        # across heads, then averages head-specific attention outputs later.
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), dim=-1)
        q = q.view(bs, t, self.n_head, self.d_head)
        k = k.view(bs, t, self.n_head, self.d_head)
        v = v.view(bs, t, self.d_head)

        # Attention score shape after matmul:
        #   [batch, heads, query_time, key_time]
        #
        # Every decoder/history position can attend to earlier positions, but
        # the causal mask below prevents it from attending to anything to its
        # right.
        attn_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        attn_score.mul_(self.scale)

        attn_score = attn_score + self._mask

        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.attn_dropout(attn_prob)

        # Multiply the attention probabilities by the shared value vectors, then
        # average across heads to preserve TFT's interpretable-head design.
        attn_vec = torch.matmul(attn_prob, v.unsqueeze(1))
        m_attn_vec = torch.mean(attn_vec, dim=1)
        out = self.out_proj(m_attn_vec)
        out = self.out_dropout(out)

        return out, attn_prob

class TFTBack(Module):
    """
    Core temporal reasoning stack of TFT after embeddings and static encoding.

    This block:
    - selects variables for historical and future inputs
    - encodes them recurrently
    - enriches them with static context
    - applies causal self-attention
    - returns both latent decoder features and projected quantile outputs
    """

    def __init__(self, config):
        super().__init__()

        self.encoder_length = config.encoder_length
        self.history_vsn = VariableSelectionNetwork(config, config.num_historic_vars)
        self.history_encoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.future_vsn = VariableSelectionNetwork(config, config.num_future_vars)
        self.future_encoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)


        self.input_gate = GLU(config.hidden_size, config.hidden_size)
        # LayerNorm epsilon is now sourced from TFTConfig so the GRN blocks and
        # the surrounding normalization layers can stay numerically aligned.
        self.input_gate_ln = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # This GRN injects static enrichment context into the temporal stream.
        # The context size remains explicit because it is part of the block's
        # role in the TFT architecture rather than a generic shared default.
        self.enrichment_grn = GRN.from_tft_config(
            config,
            input_size=config.hidden_size,
            context_hidden_size=config.hidden_size,
        )
        self.attention = InterpretableMultiHeadAttention(config)
        self.attention_gate = GLU(config.hidden_size, config.hidden_size)
        self.attention_ln = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # The position-wise GRN is another standard TFT sub-block; using the
        # same factory keeps its hidden/dropout/norm defaults synchronized with
        # the other GRN instances in this model.
        self.positionwise_grn = GRN.from_tft_config(
            config,
            input_size=config.hidden_size,
        )

        self.decoder_gate = GLU(config.hidden_size, config.hidden_size)
        self.decoder_ln = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.quantile_proj = nn.Linear(config.hidden_size, len(config.quantiles))
        
    def forward(self, historical_inputs, cs, ch, cc, ce, future_inputs):
        # Historical and future inputs are processed separately at first because
        # TFT treats "already observed" and "known ahead" information
        # differently before joining them into one temporal stream.
        historical_features, _ = self.history_vsn(historical_inputs, cs)
        history, state = self.history_encoder(historical_features, (ch, cc))
        future_features, _ = self.future_vsn(future_inputs, cs)
        future, _ = self.future_encoder(future_features, state)

        # First residual block over the recurrent temporal stream. TFT keeps the
        # original input embedding available as a skip path so later stages do
        # not lose the raw variable-selection output.
        input_embedding = torch.cat([historical_features, future_features], dim=1)
        temporal_features = torch.cat([history, future], dim=1)
        # At this point both tensors are aligned over the full example axis:
        # - `input_embedding` is the pre-LSTM representation after variable selection
        # - `temporal_features` is the post-LSTM temporal representation
        #
        # The gated residual merge lets TFT refine the recurrent output without
        # discarding the direct variable-selection signal.
        temporal_features = self.input_gate(temporal_features)
        temporal_features = temporal_features + input_embedding
        temporal_features = self.input_gate_ln(temporal_features)

        # Static enrichment
        # This is where the static context vector `ce` is injected back into the
        # temporal stream so the per-timestep representation can depend on the
        # subject-level/static context learned earlier.
        enriched = self.enrichment_grn(temporal_features, c=ce)

        # Temporal self attention
        x, _ = self.attention(enriched)

        # The model only needs predictions over the decoder horizon, so all
        # hidden states corresponding to encoder-history positions are removed
        # before the final decoder-side processing.
        x = x[:, self.encoder_length:, :]
        temporal_features = temporal_features[:, self.encoder_length:, :]
        enriched = enriched[:, self.encoder_length:, :]

        x = self.attention_gate(x)
        x = x + enriched
        x = self.attention_ln(x)
        # After causal attention, TFT keeps another residual path from the
        # enriched temporal features so the attention block learns adjustments
        # rather than replacing the whole decoder representation.

        # Position-wise feed-forward
        x = self.positionwise_grn(x)

        # Final decoder-side residual block before quantile projection.
        x = self.decoder_gate(x)
        x = x + temporal_features
        x = self.decoder_ln(x)

        # Return both:
        # - `x`: horizon-aligned latent decoder features for late fusion
        # - `out`: direct TFT quantile outputs for standalone TFT use
        out = self.quantile_proj(x)

        return x, out

class TemporalFusionTransformer(Module):
    """
    Repository TFT wrapper with both latent-feature and quantile-output interfaces.

    In this project the same TFT module serves two roles:
    - as a standalone probabilistic forecaster via `forward(...)`
    - as a representation branch inside the fused model via `forward_features(...)`
    """
    def __init__(self, config: TFTConfig):
        super().__init__()

        # This determines where the example transitions from encoder history to
        # decoder horizon.
        self.encoder_length = config.encoder_length

        # Lazily bind continuous embedding widths to the actual grouped input
        # tensors seen at runtime.
        self.embedding = LazyEmbedding(config)

        self.static_encoder = StaticCovariateEncoder(config)
        # Keep the temporal reasoning stack as a normal eager module.
        #
        # Why this is the future-proof default:
        # - `torch.jit.script(...)` is deprecated upstream
        # - the repository already exposes optional model-wide `torch.compile`
        #   support at the runtime/trainer layer
        # - keeping this block eager avoids binding the model internals to a
        #   second compilation mechanism with different constraints
        #
        # In practice, this means:
        # - plain eager execution works by default
        # - runtime acceleration remains available through the existing
        #   environment/profile/Trainer compile path when enabled
        self.temporal_backbone = TFTBack(config)

    def forward_with_features(self, x: TFTInputBatch) -> Tuple[Tensor, Tensor]:
        # Accept the same optional-group contract as `forward(...)` because the
        # fused model uses this latent-feature path directly. Keeping the two
        # signatures aligned prevents the feature-extraction interface from
        # drifting out of sync with the standard TFT inference interface.
        # Step 1: embed grouped raw inputs into per-variable hidden vectors.
        s_inp, t_known_inp, t_observed_inp, t_observed_tgt = self.embedding(x)

        # Step 2: derive the static context vectors that condition the temporal
        # variable-selection and recurrent blocks.
        cs, ce, ch, cc = self.static_encoder(s_inp)
        ch, cc = ch.unsqueeze(0), cc.unsqueeze(0)

        # Step 3: assemble the historical temporal stream from:
        # - known inputs observed along the encoder axis
        # - the target history itself
        # - optional observed-only covariates
        _historical_inputs = [t_known_inp[:,:self.encoder_length,:], t_observed_tgt[:,:self.encoder_length,:]]
        if t_observed_inp is not None:
            _historical_inputs.insert(0,t_observed_inp[:,:self.encoder_length,:])

        historical_inputs = torch.cat(_historical_inputs, dim=-2)
        # Concatenation happens along the variable axis, not the time axis:
        # - all historical groups already share the same encoder timeline
        # - TFT needs them as separate variable slots available at each
        #   historical timestep
        # The future stream contains only variables known ahead at inference
        # time. Observed-only inputs and the target are not available there.
        future_inputs = t_known_inp[:, self.encoder_length:]
        return self.temporal_backbone(
            historical_inputs,
            cs,
            ch,
            cc,
            ce,
            future_inputs,
        )

    def forward_features(self, x: TFTInputBatch) -> Tensor:
        # Expose the decoder representation before quantile projection so the
        # fused model can combine TFT features with TCN features in latent space.
        features, _ = self.forward_with_features(x)
        return features

    def forward(self, x: TFTInputBatch) -> Tensor:
        # Default standalone interface: return probabilistic forecast outputs.
        _, quantiles = self.forward_with_features(x)
        return quantiles
