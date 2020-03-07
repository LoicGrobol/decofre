import pathlib

import hypothesis
import torch

import numpy as np

from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp

from decofre import libdecofre


test_dir = pathlib.Path(__file__).parent


def test_contextfree_encoder():
    torch.manual_seed(0)
    words_voc_size = 2713
    words_emb_dim = 300
    pretrained_weights = torch.rand((words_voc_size, words_emb_dim))
    encoder = libdecofre.ContextFreeWordEncoder(
        words_voc_size, words_emb_dim, 26, 50, 50, pretrained_weights
    )
    encoder.eval()
    sent_words = [
        torch.tensor([1, 3, 5]),
        torch.tensor([6, 7, 8, 9]),
        torch.tensor([0, 1000]),
    ]
    chars = [[torch.tensor([i]) for i in range(t.shape[0])] for t in sent_words]
    batch = list(zip(sent_words, chars))
    out, seq_lens = encoder(batch)
    assert seq_lens.tolist() == [t.shape[0] for t in sent_words]
    assert list(out.shape) == [3, 4, words_emb_dim + 50]
    for i, (w, c) in enumerate(batch):
        padding = out[i, w.shape[0] :, :]
        assert out[i, : w.shape[0], :words_emb_dim].allclose(
            pretrained_weights.index_select(0, w)
        )
        assert padding.equal(torch.zeros_like(padding))
        chars_target = encoder.chars_encoder(c)
        assert out[i, : w.shape[0], words_emb_dim:].allclose(chars_target)


def test_slice_and_mask():
    torch.manual_seed(0)
    t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float)
    indices = torch.tensor([[0, 3], [3, 4], [1, 3]])
    out = libdecofre.slice_and_mask(t, indices)
    target = libdecofre.MaskedSequence(
        sequence=torch.tensor([[1, 2, 3], [8, 0, 0], [10, 11, 0]], dtype=torch.float),
        mask=torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.bool),
    )
    assert out.sequence.equal(target.sequence)
    assert out.mask.equal(target.mask)


def test_elementwise_apply():
    torch.manual_seed(0)
    source = [
        torch.tensor([4, 5]),
        torch.tensor([0, 1, 2, 3, 4, 5, 6]),
        torch.tensor([6, 7, 8]),
        torch.tensor([4, 3, 2, 1, 0]),
        torch.tensor([9]),
    ]
    lengths = [t.size(0) for t in source]
    e = torch.nn.Embedding(11, embedding_dim=13, padding_idx=10)
    e.eval()
    padded = torch.nn.utils.rnn.pad_sequence(source, padding_value=10)
    direct = e(padded)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        padded, lengths, enforce_sorted=False
    )
    embedded = libdecofre.elementwise_apply(e, packed)
    unpacked, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(embedded)
    assert seq_lens.tolist() == lengths
    assert unpacked.equal(direct)

    g = torch.nn.GRU(13, 23, 1, bidirectional=True)
    g.eval()
    output, hn = g(embedded)
    forward = (
        torch.nn.utils.rnn.pad_packed_sequence(output)[0]
        .view(max(lengths), len(source), 2, 23)[:, :, 0, :]
        .transpose(0, 1)
    )
    direct_output, direct_hn = g(direct)
    direct_forward = direct_output.view(max(lengths), len(source), 2, 23)[
        :, :, 0, :
    ].transpose(0, 1)
    assert list(forward.shape) == list(direct_forward.shape)
    assert list(forward.shape) == [len(source), max(lengths), 23]
    for l, p, d, f in zip(lengths, forward, direct_forward, hn[0]):
        assert p[:l, :].allclose(d[:l, :])
        assert f.allclose(d[l - 1, :])


@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow], deadline=None)
@given(data=st.data(), external=st.booleans(), merge=st.booleans())
def test_select_span_boundaries(data, external, merge):
    m = data.draw(st.integers(min_value=1, max_value=128), label="Batch dimension")
    n = data.draw(
        st.integers(min_value=3 if external else 1, max_value=256),
        label="Sequence length",
    )
    k = data.draw(st.integers(min_value=1, max_value=512), label="Features dimension")
    spans_val = data.draw(hnp.arrays(np.float64, (m, n, k)), label="Sequences")
    indices_val = data.draw(
        st.lists(
            st.tuples(
                st.integers(
                    min_value=1 if external else 0,
                    max_value=n - 2 if external else n - 1,
                ),
                st.integers(
                    min_value=2 if external else 1, max_value=n - 1 if external else n
                ),
            ).map(sorted),
            min_size=m,
            max_size=m,
        ),
        label="Span indices",
    )
    spans = torch.from_numpy(spans_val)
    indices = torch.tensor(indices_val)
    target = torch.empty((m, 2, k), dtype=spans.dtype)
    for i, (start, stop) in enumerate(indices):
        if external:
            target[i, 0, :] = spans[i, start - 1, :]
            target[i, 1, :] = spans[i, stop, :]
        else:
            target[i, 0, :] = spans[i, start, :]
            target[i, 1, :] = spans[i, stop - 1, :]
    if merge:
        target = target.view((m, 2 * k))
    out = libdecofre.select_span_boundaries(spans, indices, external, merge)
    assert out.allclose(target, equal_nan=True)
