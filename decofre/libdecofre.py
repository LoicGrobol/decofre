"""Models and utilities for neural end-to-end coreference resolution."""
# from __future__ import annotations

import itertools as it
import typing as ty

from typing import List, Tuple

import torch
import torch.jit
import torch.nn
import torch.nn.functional
import torch.nn.parallel

from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from typing_extensions import Final, Literal

# Suppress tf messages at loading time
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # noqa: E402
import transformers

# from torch.nn.utils.rnn import pad_sequence

try:
    from allennlp.modules.elmo import Elmo
except ImportError:
    Elmo = None


class MaskedSequence(ty.NamedTuple):
    """A batch of sequences of variable length with an associated mask.

    Attributes
    ==========

    :`sequence`: a `(batch, sequence, *)`-shaped `torch.Tensor` of any dtype
    :`mask`: a boolean tensor of shape `(batch, sequence)` with `mask[i,j]`
      `True` if `sequence[i,j,...]` is an actual sequence element and `False`
      if it is padding.
    """

    sequence: torch.Tensor
    mask: torch.Tensor


@torch.jit.ignore
def elementwise_apply(
    fn: ty.Callable[[torch.Tensor], torch.Tensor],
    packed_sequence: torch.nn.utils.rnn.PackedSequence,
) -> torch.nn.utils.rnn.PackedSequence:
    """Apply a pointwise function `fn` to each element in `packed_sequence`"""
    res = torch.nn.utils.rnn.PackedSequence(
        fn(packed_sequence.data),
        packed_sequence.batch_sizes,
        sorted_indices=packed_sequence.sorted_indices,
        unsorted_indices=packed_sequence.unsorted_indices,
    )
    return res


# TODO: this could be done with a gather (possibly with appending zeros before)
# @torch.jit.script
def slice_and_mask(t: torch.Tensor, indices: torch.Tensor) -> MaskedSequence:
    """Return a `MaskedSequence` corresponding to slices of `t`.

    Arguments
    =========

    :`t`: a `(N, *)` tensor
    :`indices`: a `(N, 2)` int tensor specifying the `start:end` slice indices
      (using the usual Python [i, j[ slice semantic)
    """
    sequence = []
    unbound = t.unbind(0)
    for i in range(t.size(0)):
        start, end = indices[i][0], indices[i][1]
        sequence.append(unbound[i].narrow(0, start, end - start))
    padded_sequence = pad_sequence(sequence, batch_first=True)
    mask = torch.ones(padded_sequence.shape[:2], dtype=torch.bool, device=t.device)
    for i, t in enumerate(sequence):
        mask[i, t.shape[0] :] = torch.zeros(
            mask.shape[1] - t.shape[0], dtype=torch.bool, device=t.device
        )
    return MaskedSequence(padded_sequence, mask)


@torch.jit.script
def select_span_boundaries(t, boundaries, external: bool = False, merge: bool = False):
    # Sequence dimension is 1 so this selects 2 elements per sequence
    # Span [i, j] is representented by boundaries [i, j+1[ so we have to either
    # take [i-1, j+1] or [i, j] depending on `external` to be consistent
    batch_size = t.size(0)
    features_dim = t.size(-1)
    if external:
        fixed_boundaries = boundaries - torch.tensor([1, 0], device=boundaries.device)
    else:
        fixed_boundaries = boundaries - torch.tensor([0, 1], device=boundaries.device)

    boundary_indices = fixed_boundaries.unsqueeze(2).expand(
        (batch_size, 2, features_dim)
    )
    boundaries = t.gather(1, boundary_indices)
    if merge:
        return boundaries.reshape((batch_size, 2 * features_dim))
    return boundaries


class FFNN(torch.jit.ScriptModule):
    """A simple feed-forward neural network.

    ## Arguments

    - `dimensions`: an iterable of integers, the dimensions of the layers. The
      first one is the input dimension of the first layer and the last one the
      output dimension of the last
    - `nonlinearities`: an iterable of callable accepting PyTorch tensors that
      serve as the non-linearities between the layers. `None` indicates the
      identity. The default is `#torch.nn.leaky_relu` for all layers.
    - `dropout`: the factor of the dropout to apply between layers (applied to
      input but not to output)
    - `initializer`: a custom weight initializer for the linear layers,
    - `layer_norms`: for each layer, whether to use layer normalization,
      defaults to `False` for all layers
    - `output_zero_init`: Initialize the weights of the output layer to zero
      possibly good for classifiers (inspired by Zhang et al. (2018)[^1])

    ## Inputs
    - `inpt`: `(N, *, dimensions[0])`-shaped tensor

    ## Outputs
    - `o`: `(N, *, dimensions[-1])`-shaped tensor

    [^1]: Zhang, Hongyi, Yann N. Dauphin, and Tengyu Ma. 2018. ‘Fixup
    Initialization: Residual Learning Without Normalization’, September.
    <https://openreview.net/forum?id=H1gsz30cKX>.
    """

    __constants__ = ["depth"]

    def __init__(
        self,
        dimensions: ty.Iterable[int],
        nonlinearities: ty.Iterable[ty.Optional[torch.nn.Module]] = None,
        dropout: float = 0.2,
        initializer: ty.Optional[ty.Callable[[torch.Tensor], None]] = None,
        layer_norms: ty.Optional[ty.Iterable[bool]] = None,
        output_zero_init: bool = False,
    ):
        super().__init__()
        self.dimensions = list(dimensions)
        self.depth: Final[int] = len(self.dimensions) - 1
        dropouts = [torch.nn.Dropout(dropout) for _ in range(self.depth)]
        lin_layers = [
            torch.nn.Linear(a, b)
            for a, b in zip(self.dimensions[:-1], self.dimensions[1:])
        ]
        self.out_dim: Final[int] = lin_layers[-1].out_features
        if nonlinearities is None:
            nonlinearities = [torch.nn.LeakyReLU() for _ in range(self.depth)]
        else:
            nonlinearities = list(nonlinearities)
        if layer_norms is None:
            layer_norm_layers = [None] * len(
                self.dimensions
            )  # type: ty.List[ty.Optional[torch.nn.LayerNorm]]
        else:
            layer_norm_layers = [
                torch.nn.LayerNorm(d) if n else None
                for n, d in zip(layer_norms, self.dimensions[1:])
            ]

        if initializer is not None:
            for layer in lin_layers:
                initializer(layer.weight)
        if output_zero_init:
            torch.nn.init.constant_(lin_layers[-1].weight, 0.0)
            torch.nn.init.constant_(lin_layers[-1].bias, 0.0)

        self.layers = torch.nn.ModuleList(
            [
                l
                for l in it.chain.from_iterable(
                    zip(dropouts, lin_layers, nonlinearities, layer_norm_layers)
                )
                if l is not None
            ]
        )

    @torch.jit.script_method
    def forward(self, inpt: torch.Tensor):
        res = inpt
        for l in self.layers:
            res = l(res)
        return res


class CategoricalFeaturesBag(torch.jit.ScriptModule):
    def __init__(
        self,
        features: ty.Iterable[ty.Tuple[int, int, ty.Optional[torch.Tensor]]],
        sparse=True,
    ):
        super().__init__()
        embeddings = torch.nn.ModuleList()
        self.out_dim = 0
        for v, e, w in features:
            if w is None:
                layer = torch.nn.Embedding(v, e, sparse=sparse)
                torch.nn.init.xavier_normal_(layer.weight)
            else:
                layer = torch.nn.Embedding.from_pretrained(w, sparse=sparse)
            embeddings.append(layer)
            self.out_dim += e

        self.embeddings = embeddings

    @torch.jit.script_method
    def forward(self, inpt: torch.Tensor):
        res_lst: List[torch.Tensor] = []
        for i, e in enumerate(self.embeddings):
            res_lst.append(e(inpt[..., i]))
        res = torch.cat(res_lst, dim=-1)
        return res


class SeqEncoder(torch.nn.Module):
    """A BiGRU encoder for variable-length sequences.

    ## Inputs
      - `inpt`: Iterable of `$N$` `$(k_i, 1)$`-shaped `#torch.LongTensor` where `$k_i$` is the
        length of the $i$-th sequence.

    ## Outputs
      - `output`: a `$(N, encoding_dim)$` `#torch.Tensor`
    """

    def __init__(
        self,
        vocab_size: int,
        embeddings_dim: int,
        hidden_dim: int,
        out_dim: int,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embeddings_dim = embeddings_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.padding_idx = vocab_size
        if pretrained_embeddings is None:
            self.embeddings = torch.nn.Embedding(
                vocab_size + 1,
                self.embeddings_dim,
                sparse=True,
                padding_idx=self.padding_idx,
            )
        else:
            augmented_embeddings = torch.zeros(
                (vocab_size + 1, self.embeddings_dim),
                device=pretrained_embeddings.device,
            )
            augmented_embeddings[:-1, ...] = pretrained_embeddings
            self.embeddings = torch.nn.Embedding.from_pretrained(
                augmented_embeddings,
                freeze=freeze_embeddings,
                sparse=True,
                padding_idx=self.padding_idx,
            )

        self.drop = torch.nn.Dropout(0.3)
        self.rnn = torch.nn.GRU(
            self.embeddings_dim, self.hidden_dim, bidirectional=True
        )
        self.output = FFNN((2 * self.hidden_dim, self.out_dim), layer_norms=(True,))

    @torch.jit.ignore
    def forward(self, inpt: ty.Sequence[torch.Tensor]):
        lengths = torch.tensor([v.size(0) for v in inpt])
        padded = pad_sequence(inpt, padding_value=self.padding_idx)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths, enforce_sorted=False
        )
        embedded = elementwise_apply(self.embeddings, packed)
        embedded = elementwise_apply(self.drop, embedded)
        _, hidden = self.rnn(embedded)
        # Merge directions
        hidden = hidden.transpose(0, 1).reshape((len(lengths), 2 * self.hidden_dim))
        # !FIXME: double dropout here
        out = self.output(self.drop(hidden))

        return out


class WordEncoderOutput(ty.NamedTuple):
    out: torch.Tensor
    seq_lens: torch.Tensor


class ContextFreeWordEncoder(torch.nn.Module):
    """Encoding words by concatenating word embeddings and characters GRU

    ## Inputs
      - `inpt`: Iterable of `(words, characters)` where `words` is a
        `torch.LongTensor` of word indices and `characters` an iterable
        of `torch.LongTensor`s of char indices

    ## Outputs
      - `output`: `(padded sequences, sequence_lengths)`

    Note that if you set a non-zero word dropout, the 0 index in your word
    embeddings should probably by for <unk>.
    """

    def __init__(
        self,
        words_vocab_size: int,
        words_embeddings_dim: int,
        chars_vocab_size: int,
        chars_embeddings_dim: int,
        chars_encoding_dim: int,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = False,
        unk_word_index: int = 0,
    ):
        super().__init__()
        self.words_embeddings_dim = words_embeddings_dim
        self.chars_embeddings_dim = chars_embeddings_dim

        self.word_padding_idx = words_vocab_size
        if pretrained_embeddings is None:
            self.word_embeddings = torch.nn.Embedding(
                words_vocab_size + 1,
                self.words_embeddings_dim,
                sparse=True,
                padding_idx=self.word_padding_idx,
            )
        else:
            augmented_embeddings = torch.zeros(
                (words_vocab_size + 1, self.words_embeddings_dim),
                device=pretrained_embeddings.device,
            )
            augmented_embeddings[:-1, ...] = pretrained_embeddings
            self.word_embeddings = torch.nn.Embedding.from_pretrained(
                augmented_embeddings,
                freeze=freeze_embeddings,
                sparse=True,
                padding_idx=self.word_padding_idx,
            )

        self.chars_encoder = SeqEncoder(
            vocab_size=chars_vocab_size,
            embeddings_dim=chars_embeddings_dim,
            hidden_dim=150,
            out_dim=chars_encoding_dim,
        )
        self.out_dim = words_embeddings_dim + chars_encoding_dim

    @torch.jit.ignore
    def forward(
        self, tokens: ty.Sequence[ty.Tuple[torch.Tensor, ty.Sequence[torch.Tensor]]]
    ) -> WordEncoderOutput:
        words, chars = zip(*tokens)
        seq_lens = torch.tensor([w.size(0) for w in words])
        padded_words = pad_sequence(
            words, padding_value=self.word_padding_idx, batch_first=True
        )
        padded_embedded_words = self.word_embeddings(padded_words)
        res_chars = []
        # TODO: don't loop, run on all the words in parallel
        for c in chars:
            # shape: (N, self.chars_encoding_dim)
            encoded_chars = self.chars_encoder(c)
            res_chars.append(encoded_chars)
        padded_encoded_chars = pad_sequence(res_chars, batch_first=True)
        res = torch.cat((padded_embedded_words, padded_encoded_chars), dim=-1)
        return WordEncoderOutput(res, seq_lens)


if Elmo is not None:

    class ELMoWordEncoder(torch.nn.Module):
        def __init__(self, options_file: str, weight_file: str):
            super().__init__()
            self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
            self.out_dim = self.elmo.get_output_dim()

        def forward(
            self, character_ids: ty.Sequence[torch.Tensor]
        ) -> WordEncoderOutput:
            # FIXME: this should be dealt with in digitize/collate (or should it ?)
            padded_characters = pad_sequence(
                [c.squeeze(0) for c in character_ids], batch_first=True
            )
            embeddings = self.elmo(padded_characters)
            seq_lens = embeddings["mask"].sum(dim=-1)
            return WordEncoderOutput(embeddings["elmo_representations"][0], seq_lens)


class BERTWordEncoder(torch.nn.Module):
    """A word encoder using a pretrained transformer model as a base.

    Arguments
    =========

    :model_name_or_path: the reference of the pretrained model to load
    :model_class: the class of model to load, defaults to `transformers.AutoModel`.
    :fine_tune: whether to fine-tune the parameters that are already pretrained
      (ie those of the base pretrained model)
    :combine_layers: a list of layer indices. If given, the output will be a scaled scalar mix (à la
      ELMo) of these layers. The layer weights and the scales are trainable parameters.
    :project: whether to apply a trainable linear projection to the outputs of the model

    Inputs
    ======

    :pieces_ids: a liste of `LongTensors`, each one representing the indices of the wordpieces for
    a sequence

    Outputs
    =======

    :out: a `FloatTensor` of shape `(batch_size, max_seq_len, features)` containing the batched,
      0-right-padded output of the model (ie the raw hidden states of the transformer, possibly
      projected, possibly combined depending on the settings)
    :seq_lens: a `LongTensor` of shape `(batch_size,)` containing the true length of the input
      sequences, suitable for `pack_padded_sequence`.

    Notes
    =====

    Using `AutoModel` requires special handling of configurations and is special-cased here, so
    proceed with caution if you want to subclass it.

    """

    def __init__(
        self,
        model_name_or_path: str,
        model_class: ty.Optional[transformers.PreTrainedModel] = None,
        fine_tune: bool = False,
        combine_layers: ty.Optional[ty.Sequence[int]] = None,
        project: bool = False,
    ):
        super().__init__()
        output_hidden_states = combine_layers is not None
        if model_class is None or model_class == transformers.AutoModel:
            # For automodels, we have to load the config first, see
            # <https://github.com/huggingface/transformers/issues/2694>
            model_config = transformers.AutoConfig.from_pretrained(
                model_name_or_path, output_hidden_states=output_hidden_states
            )
            self.model = transformers.AutoModel.from_pretrained(
                model_name_or_path, config=model_config
            )
        else:
            self.model = model_class.from_pretrained(
                model_name_or_path, output_hidden_states=output_hidden_states
            )
        if isinstance(self.model, transformers.BertModel):
            self.hidden_state_indice_in_output = 2
        elif isinstance(self.model, transformers.XLMModel):
            self.hidden_state_indice_in_output = 1
        else:
            logger.warning(
                f"Loading unknown model type {type(self.model)}, defaulting to BERT config"
            )
            self.hidden_state_indice_in_output = 2

        # We can't use layer drop when combining and quite frankly we should not use it in any case.
        if isinstance(self.model, transformers.FlaubertModel):
            self.model.layerdrop = 0.0

        self.out_dim = self.model.config.hidden_size
        self.fine_tune = fine_tune
        # Normalize indices
        self.combine_layers: ty.Optional[ty.List[int]]
        if combine_layers is not None:
            self.combine_layers = sorted(
                n if n >= 0 else self.model.config.num_hidden_layers + n
                for n in combine_layers
            )
            combine_weights = torch.empty((len(self.combine_layers),))
            combine_weights.normal_()
            self.layer_weights = torch.nn.Parameter(combine_weights, requires_grad=True)
            self.combination_scaling = torch.nn.Parameter(
                torch.tensor([1.0]), requires_grad=True
            )
        else:
            self.combine_layers = None
            self.layer_weights = torch.nn.Parameter(
                torch.tensor([1.0]), requires_grad=False
            )
            self.combination_scaling = torch.nn.Parameter(
                torch.tensor([1.0]), requires_grad=False
            )
        self.project = project
        self.output_projection: torch.nn.Module
        if self.project:
            self.output_projection = torch.nn.Linear(self.out_dim, self.out_dim)
        else:
            self.output_projection = torch.nn.Identity()
        if not self.fine_tune:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    @torch.jit.ignore
    def forward(self, pieces_ids: ty.Sequence[torch.Tensor]) -> WordEncoderOutput:
        # FIXME: this should be dealt with in digitize/collate (or should it ?)
        seq_lens = torch.tensor([sent.shape[1] for sent in pieces_ids])
        padded_pieces = pad_sequence(
            [c.squeeze(0) for c in pieces_ids], batch_first=True
        )
        attention_mask = self.make_attention_mask(seq_lens).to(padded_pieces.device)
        model_out = self.model(padded_pieces, attention_mask=attention_mask)
        if self.combine_layers is None:
            embeddings = model_out[0]
        else:
            layers_out = model_out[self.hidden_state_indice_in_output]
            extracted_layers = torch.stack(
                [layers_out[n] for n in self.combine_layers], dim=0
            )
            normalized_weights = torch.nn.functional.softmax(self.layer_weights, dim=0)
            embeddings = self.combination_scaling * torch.einsum(
                "i,ijkl->jkl", (normalized_weights, extracted_layers)
            )
        out = self.output_projection(embeddings)
        return WordEncoderOutput(out, seq_lens)

    def train(self, mode=True):
        res = super().train(mode)
        if not self.fine_tune:
            self.model.eval()
        return res

    @classmethod
    def make_attention_mask(cls, seq_lens: torch.Tensor) -> torch.Tensor:
        mask = torch.ones((seq_lens.shape[0], seq_lens.max().item()), dtype=torch.float64)
        for i, l in enumerate(seq_lens):
            mask[i, l:] = 0.0
        return mask


class FeaturefulWordEncoder(torch.nn.Module):
    def __init__(self, words_encoder, features_encoder):
        super().__init__()
        self.words_encoder = words_encoder
        self.features_encoder = features_encoder

    @torch.jit.ignore
    def forward(self, inpt: ty.Iterable[ty.Tuple]) -> WordEncoderOutput:
        words, features = zip(*inpt)
        encoded_words, seq_lens = self.words_encoder(words)
        batch_features = []
        for sequence_features in features:
            embedded_sequence_features = self.features_encoder(sequence_features)
            batch_features.append(embedded_sequence_features)
        encoded_features = pad_sequence(batch_features, batch_first=True)
        return WordEncoderOutput(
            torch.cat([encoded_words, encoded_features], dim=-1), seq_lens
        )


class MaskedLinearSelfAttention(torch.jit.ScriptModule):
    __constants__ = ["features_dim", "n_heads"]

    def __init__(
        self, features_dim: int, hidden_dim: int, n_heads: int, dropout: float = 0.2
    ):
        super().__init__()
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.nonlinearity = torch.nn.LeakyReLU()
        self.ih = torch.nn.Linear(self.features_dim, self.hidden_dim)
        # self.hh = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ho = torch.nn.Linear(self.hidden_dim, self.n_heads)
        self.drop = torch.nn.Dropout(dropout)

    @torch.jit.script_method
    def forward(self, inpt, mask):
        ff = inpt
        ff = self.nonlinearity(self.ih(self.drop(ff)))
        # ff = self.nonlinearity(self.hh(self.drop(ff)))
        raw_weights = self.ho(self.drop(ff))
        masked_weights = torch.where(
            mask.unsqueeze(2),
            raw_weights,
            # FIXME: Switch this to a scalar as soon as https://github.com/pytorch/pytorch/issues/9190
            # is merged
            torch.tensor(-1e32, device=inpt.device, dtype=torch.float),
        )
        normalized_weights = torch.nn.functional.softmax(masked_weights, dim=-2)
        # shape: (batch_size, features_dim, n_heads)
        attended = torch.einsum("nij,nik->njk", (inpt, normalized_weights))
        # shape: (batch_size, features_dim*n_heads)
        head_merged = attended.reshape((inpt.size(0), self.features_dim * self.n_heads))
        return head_merged


# TODO: add a docstring…
# TODO: use projections + pooling (sum/max/gate) instead of concatenation
class SpanEncoder(torch.nn.Module):
    """Text span embeddings"""

    def __init__(
        self,
        words_encoding_dim: int,
        hidden_dim: int,
        ffnn_dim: int,
        out_dim: int,
        hidden_depth: int = 2,
        attention_heads: int = 2,
        soft_dropout_rate: float = 0.3,
        hard_dropout_rate: float = 0.6,
        external_boundaries: bool = False,
    ):
        super().__init__()

        self.words_encoding_dim: Final[int] = words_encoding_dim
        self.hidden_dim: Final[int] = hidden_dim
        self.hidden_depth: Final[int] = hidden_depth
        self.external_boundaries: Final[bool] = external_boundaries

        self.total_encoding_dim: Final[int] = 2 * self.hidden_dim + words_encoding_dim

        self.soft_drop = torch.nn.Dropout(soft_dropout_rate)
        self.hard_drop = torch.nn.Dropout(hard_dropout_rate)
        self.lstm = torch.nn.LSTM(
            self.words_encoding_dim,
            hidden_dim,
            num_layers=self.hidden_depth,
            bidirectional=True,
        )
        self.attention_heads: Final[int] = attention_heads
        self.attention = MaskedLinearSelfAttention(
            self.total_encoding_dim, ffnn_dim, n_heads=self.attention_heads
        )

        self.out_dim: Final[int] = out_dim
        self.out = FFNN(
            (
                (  # Recurrent embeddings
                    2 * self.total_encoding_dim
                    # Attentional embeddings
                    + (self.attention_heads * self.total_encoding_dim)
                    # Context embedding
                    + 2 * self.hidden_dim
                ),
                self.out_dim,
                self.out_dim,
            ),
            layer_norms=(False, True),
        )

    def forward(
        self, encoded_tokens: WordEncoderOutput, span_boundaries: torch.Tensor
    ) -> torch.Tensor:
        batched_encoded_words, seq_lens = encoded_tokens
        batch_size = batched_encoded_words.size(0)

        batched_encoded_words = self.hard_drop(batched_encoded_words)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            batched_encoded_words, seq_lens, batch_first=True, enforce_sorted=False
        )
        lstm_output, (hn, cn) = self.lstm(packed)
        # shape: (batch_size, seq_len, 2*hidden_dim), 2*[(hidden_depth * 2, batch_size, hidden_dim)]
        hidden, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_output, batch_first=True
        )
        hidden = self.soft_drop(hidden)  # TODO: elementwise apply here?
        # shape: (batch_size, 2*hidden_dim)
        context_embedding = (
            hn.view(self.hidden_depth, 2, batch_size, self.hidden_dim)[-1]
            .transpose(0, 1)
            .reshape(batch_size, 2 * self.hidden_dim)
        )
        context_embedding = self.soft_drop(context_embedding)
        token_embeddings = torch.cat([hidden, batched_encoded_words], dim=-1)
        # If you are reading this, you are awesome

        # shape: (batch_size, 2*self.total_encoding_dim)
        recurrent_embeddings = select_span_boundaries(
            token_embeddings, span_boundaries, self.external_boundaries, merge=True
        )

        # TODO: Masking might still be stealing us some speed
        span_contents, span_mask = slice_and_mask(token_embeddings, span_boundaries)
        # shape: (batch_size, self.total_encoding_dim*attention_heads)
        attentional_embeddings = self.attention(span_contents, span_mask)
        encodings = self.soft_drop(
            torch.cat(
                [recurrent_embeddings, attentional_embeddings, context_embedding],
                dim=-1,
            )
        )

        # !FIXME: double dropout, we drop once in the previous instruction and once in out
        output = self.out(encodings)
        return output


class FeaturefulSpanEncoder(torch.nn.Module):
    def __init__(self, tokens_encoder, span_encoder, features_encoder, out_dim: int):
        super().__init__()
        self.tokens_encoder = tokens_encoder
        self.span_encoder = span_encoder
        self.features_encoder = features_encoder
        self.hidden_dim: Final[int] = self.span_encoder.out_dim + (
            self.features_encoder.out_dim if self.features_encoder is not None else 0
        )
        self.out_dim: Final[int] = out_dim
        self.output_projection = torch.nn.Linear(self.hidden_dim, self.out_dim)
        self.output_nonlinearity = torch.nn.LeakyReLU()
        self.output_norm = torch.nn.LayerNorm(self.out_dim)

    @torch.jit.ignore
    def forward(self, inpt):
        tokens, span_boundaries, features = inpt
        encoded_tokens = self.tokens_encoder(tokens)
        encoded_content = self.span_encoder(encoded_tokens, span_boundaries)
        if self.features_encoder is not None:
            encoded_features = self.features_encoder(features)
            encoded_content = torch.cat([encoded_content, encoded_features], dim=-1)
        outpt = self.output_norm(
            self.output_nonlinearity(self.output_projection(encoded_content))
        )
        return outpt


class MentionDetector(torch.nn.Module):
    def __init__(
        self,
        span_encoder: torch.nn.Module,
        span_encoding_dim: int,
        n_types: int,
        ffnn_dim: int = 150,
        depth: int = 1,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.span_encoder = span_encoder
        self.classifier = FFNN(
            (span_encoding_dim, *([ffnn_dim] * depth), n_types),
            (
                *(torch.nn.LeakyReLU() for _ in range(depth)),
                torch.nn.LogSoftmax(dim=-1),
            ),
            dropout=dropout,
            output_zero_init=False,
        )
        self.n_types: Final[int] = n_types

    def forward(self, inpt):
        encoded = self.span_encoder(inpt)
        return self.classifier(encoded)


class PairsClassifier(FFNN):
    def __init__(self, span_encoding_dim: int, ffnn_dim: int = 150, depth: int = 1):
        super().__init__(
            (2 * span_encoding_dim, *([ffnn_dim] * depth), 2),
            (
                *(torch.nn.LeakyReLU() for _ in range(depth)),
                torch.nn.LogSoftmax(dim=-1),
            ),
        )


class WeightedSumGate(torch.jit.ScriptModule):
    """A learnable feature-wise linear interpolation between two tensors.

    Formally
    ```math
    f = σ(W[old, new]+b)
    outpt = f⋅new + (1-f)⋅old
    ```
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.drop = torch.nn.Dropout(0.2)
        self.gate = torch.nn.Linear(2 * input_dim, input_dim)

    @torch.jit.script_method
    def forward(self, old, new):
        gate_inpt = self.drop(torch.cat([old, new]))
        gate_scores = torch.sigmoid(self.gate(gate_inpt))
        outpt = (gate_scores * new).addcmul((1 - gate_scores), old)
        return outpt


class CorefScorer(torch.nn.Module):
    """Compute coreference scores for a mention and a list of antecedents."""

    def __init__(
        self,
        span_encoder: FeaturefulSpanEncoder,
        span_encoding_dim: int,
        features: ty.Iterable[ty.Tuple[int, int, ty.Optional[torch.Tensor]]],
        ffnn_dim: int = 150,
        depth: int = 1,
        n_coarse: int = 25,
        refine_from_coarse: bool = True,
        mention_new: Literal["zero", "from_raw", "from_refined"] = "from_raw",
        dropout: float = 0.4,
    ):
        super().__init__()
        self.span_encoder = span_encoder
        self.dropout = dropout
        features = list(features)
        # FIXME: if mention new is set to zero this is not used, but having it here makes loading
        # easier. Still, that's ugly.
        self.mention_new_detector = FFNN(
            (span_encoding_dim, ffnn_dim, 1),
            (torch.nn.LeakyReLU(), None),
            dropout=self.dropout,
            output_zero_init=True,
        )
        self.n_coarse = n_coarse
        self.refine_from_coarse = refine_from_coarse
        self.mention_new = mention_new
        if mention_new == "from_refined" and not refine_from_coarse:
            raise ValueError(
                "Using refined mention embeddings for mention new scores makes no sense if you don't actually refine them"
            )
        self.scripted_scorer = ScriptableCorefScorer(
            span_encoding_dim=span_encoding_dim,
            features=features,
            n_coarse=n_coarse,
            refine_from_coarse=refine_from_coarse,
            ffnn_dim=ffnn_dim,
            dropout=dropout,
            depth=depth,
        )

    def forward(
        self, batch: ty.Tuple[ty.Any, ty.Iterable[ty.Tuple[ty.Any, ty.Any]]]
    ) -> torch.Tensor:
        mentions, candidates_meta = batch
        mentions_encodings = self.span_encoder(mentions)
        refined_mentions_encodings_lst = []
        antecedents_lst = []
        for mention, candidates in zip(mentions_encodings.unbind(0), candidates_meta):
            spans, pairs_features = candidates
            spans = self.span_encoder(spans)
            refined_mention, antecedent_scores = self.scripted_scorer(
                mention, spans, pairs_features
            )
            refined_mentions_encodings_lst.append(refined_mention)
            antecedents_lst.append(antecedent_scores)
        padded_antecedents = pad_sequence(
            antecedents_lst, batch_first=True, padding_value=-1e32
        )
        if self.mention_new == "from_refined":
            refined_mentions_encodings = torch.stack(
                refined_mentions_encodings_lst, dim=0
            )
            mention_new_scores = self.mention_new_detector(refined_mentions_encodings)
        elif self.mention_new == "from_raw":
            mention_new_scores = self.mention_new_detector(mentions_encodings)
        elif self.mention_new == "zero":
            mention_new_scores = torch.zeros(
                mentions_encodings.shape[0], 1, device=padded_antecedents.device
            )
        else:
            raise ValueError(f"Unknown mention_new setting: {self.mention_new!r}")
        padded_output = torch.cat([mention_new_scores, padded_antecedents], dim=-1)
        return padded_output


class ScriptableCorefScorer(torch.nn.Module):
    def __init__(
        self,
        span_encoding_dim: int,
        features: ty.Iterable[ty.Tuple[int, int, ty.Optional[torch.Tensor]]],
        n_coarse: int,
        refine_from_coarse: bool,
        dropout: float,
        ffnn_dim: int,
        depth: int,
    ):
        super().__init__()
        self.span_encoding_dim: Final[int] = span_encoding_dim
        self.coarse_scorer = torch.nn.Bilinear(
            self.span_encoding_dim, self.span_encoding_dim, 1, bias=True
        )
        self.refining_gate = WeightedSumGate(self.span_encoding_dim)
        self.pair_features_embeddings = CategoricalFeaturesBag(features)
        self.n_coarse: Final[int] = n_coarse
        self.refine_from_coarse: Final[bool] = refine_from_coarse
        # TODO: add option to use some pair features in the coarse scorer
        self.pair_scorer = FFNN(
            (
                2 * span_encoding_dim + sum(e for _, e, _ in features),
                *([ffnn_dim] * depth),
                1,
            ),
            (*(torch.nn.LeakyReLU() for _ in range(depth)), None),
            dropout=dropout,
        )

    def forward(
        self,
        mention: torch.Tensor,
        antecedent_candidates: torch.Tensor,
        pairs_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_antecedents = antecedent_candidates.size(0)
        expanded_mention = mention.unsqueeze(0).expand((n_antecedents, -1))
        coarse_scores = self.coarse_scorer(
            expanded_mention, antecedent_candidates
        ).squeeze(1)

        # Refining mention representation from coarse scores
        if self.refine_from_coarse:
            antecedents_summary = torch.einsum(
                "i,ij->j",
                torch.nn.functional.softmax(coarse_scores, dim=-1),
                antecedent_candidates,
            )
            refined_mention = self.refining_gate(mention, antecedents_summary)
        else:
            refined_mention = mention

        # Pruning antecedents
        if self.n_coarse < n_antecedents:
            _, selected_indices = coarse_scores.topk(k=self.n_coarse)
            antecedent_candidates = antecedent_candidates.index_select(
                0, selected_indices
            )
            pairs_features = pairs_features.index_select(0, selected_indices)
            n_antecedents = self.n_coarse
        else:
            selected_indices = torch.arange(
                n_antecedents, device=antecedent_candidates.device, dtype=torch.long
            )

        expanded_refined_mention = refined_mention.unsqueeze(0).expand(
            (n_antecedents, -1)
        )
        pairs_features = self.pair_features_embeddings(pairs_features)
        pairs_cat = torch.cat(
            [expanded_refined_mention, antecedent_candidates, pairs_features], dim=-1
        )
        fine_scores = self.pair_scorer(pairs_cat).squeeze(1)
        final_scores = coarse_scores.index_put(
            (selected_indices,), fine_scores, accumulate=True
        )
        return refined_mention, final_scores


@torch.jit.script
def masked_multi_cross_entropy(
    inpt: torch.Tensor, target: torch.Tensor, reduction: str = "elementwise_mean"
):
    """
    Take a batch of vectors of unnormalized class scores and a mask of correct answers
    and return the negative marginal log likelihood.

    The class scores can be padded with `-1e32` for classes that are not relevant
    """
    padded_log_scores = torch.log_softmax(inpt, dim=-1)
    scores = -torch.where(
        target,
        padded_log_scores,
        # FIXME: Switch this to a scalar as soon as https://github.com/pytorch/pytorch/issues/9190
        # is merged
        torch.tensor(-1e32, device=inpt.device, dtype=torch.float),
    ).logsumexp(dim=-1)
    if reduction == "none":
        return scores
    if reduction == "sum":
        return scores.sum()
    return scores.mean()


def multi_cross_entropy(
    inpt: ty.Iterable[torch.Tensor],
    target: ty.Iterable[torch.Tensor],
    reduction: str = "elementwise_mean",
) -> torch.Tensor:
    """
    Take a batch of vectors of unnormalized class scores and a batch of multi-class target
    indices and return the negative marginal log likelihood
    """
    outp = [
        -response.log_softmax(dim=-1).index_select(-1, key).logsumexp(dim=-1)
        for response, key in zip(inpt, target)
    ]
    scores = torch.stack(outp)
    if reduction == "none":
        return scores
    elif reduction == "elementwise_mean":
        return scores.mean()
    elif reduction == "sum":
        return scores.sum()
    raise ValueError(f"Unknown reduction {reduction!r}.")


def slack_rescaled_max_margin(
    inpt: ty.Iterable[torch.Tensor],
    target: ty.Iterable[torch.Tensor],
    fl_weight: float = 1.0,
    fn_weight: float = 1.0,
    wl_weight: float = 1.0,
    batch_reduction: str = "elementwise_mean",
) -> torch.Tensor:
    """Compute the slack-rescaled max margin loss as in Wiseman et al. (2015)

    ## Arguments
      - `inpt` A float tensor representing the scores of each antecedent candidate
      - `target` An int tensor containing the indices of the correct antecedents

    Wiseman, Sam, Alexander M. Rush, Stuart M. Shieber, and Jason Weston. 2015.
    ‘Learning Anaphoricity and Antecedent Ranking Features for Coreference
    Resolution’. In *Proceedings of the 53rd Annual Meeting of the Association
    for Computational Linguistics and the 7th International Joint Conference on
    Natural Language Processing of the Asian Federation of Natural Language
    Processing*, Beijing, China.
    <http://aclweb.org/anthology/P/P15/P15-1137.pdf>.
    """
    scores = []
    for response, key in zip(inpt, target):
        # The latent antecedent is the correct antecedent with the biggest response score
        latent_antecedent_score = response.index_select(-1, key).max()
        errors = torch.ones_like(response, dtype=torch.bool).scatter_(
            -1, key, value=False
        )
        # We need ReLU here since we are selecting only the wrong antecedents scores
        margins = torch.relu(
            1 + response.masked_select(errors) - latent_antecedent_score
        )
        if key.size(0) > 1 or key.is_nonzero():
            err_weights = torch.full_like(margins, wl_weight)
            err_weights[0] = fn_weight
        else:  # The mention is non-anaphoric (MENTION-NEW), create a scalar weight
            err_weights = torch.tensor(fl_weight)
        margins = margins * err_weights
        scores.append(margins.max())
    scores = torch.stack(scores)
    if batch_reduction == "none":
        return scores
    elif batch_reduction == "elementwise_mean":
        return scores.mean()
    elif batch_reduction == "sum":
        return scores.sum()
    raise ValueError(f"Unknown reduction {batch_reduction!r}.")


@torch.jit.script
def weighted_nll_loss(inpt: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
    """What `torch.nll_loss` with `reduction='weighted_average'` should be"""
    return (
        torch.nn.functional.nll_loss(inpt, target, weight=weight, reduction="sum")
        / weight.take(target).sum()
    )


@torch.jit.script
def averaged_nll_loss(inpt: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
    """`torch.nll_loss` with `reduction='sum'` divided by batch len"""
    return (
        torch.nn.functional.nll_loss(inpt, target, weight=weight, reduction="sum")
        / target.numel()
    )
