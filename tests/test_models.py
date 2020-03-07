import itertools as it
import pathlib

import pytest
import torch

from decofre.models import utils, encoders
from decofre import lexicon, datatools


test_dir = pathlib.Path(__file__).parent
fixtures_dir = test_dir / "fixtures"


def test_embeddings_loading():
    fix_path = fixtures_dir / "pretrained_embeddings.word2vec"
    with open(fix_path) as in_stream:
        n, d = map(int, in_stream.readline().split())
        embs = [
            (w, [float(c) for c in v])
            for l in in_stream
            for w, *v in (l.rstrip().split(" "),)
        ]
    initial_lex_voc = zip(
        ("something_not_in_the_lex", *(w for w, _ in embs[: len(embs) // 2])),
        it.repeat(1),
    )
    initial_lex = lexicon.Lexicon(vocabulary=initial_lex_voc)

    # Keep all words
    new_lex, weights = utils.load_embeddings(fix_path, initial_lex)
    assert weights.shape == (len(initial_lex.i2t), d)
    assert len(new_lex.i2t) == len(initial_lex.i2t)
    for w, v in embs[: len(embs) // 2]:
        loaded_weights = weights[new_lex.t2i(w)].tolist()
        assert loaded_weights == pytest.approx(v)

    # Keep only the intersection of pretrained and observed
    new_lex, weights = utils.load_embeddings(fix_path, initial_lex, keep_original=False)
    assert weights.shape == (1 + len(embs) // 2, d)  # +1 because of <unk>
    assert len(new_lex.i2t) == 1 + len(embs) // 2  # +1 because of <unk>
    for w, v in embs[: len(embs) // 2]:
        loaded_weights = weights[new_lex.t2i(w)].tolist()
        assert loaded_weights == pytest.approx(v)


def test_contextfree_digitize():
    vocab = ["spam", "ham", "sausages", "eggs"]
    words_lexicon = lexicon.Lexicon.from_instances(["spam", "ham", "sausages", "eggs"])
    chars_lexicon = lexicon.Lexicon.from_instances((c for w in vocab for c in w))
    features = [{"name": 'length', "vocabulary_size": 10, "embeddings_dim": 20}]
    features_digitizers = {"length": int}
    model = encoders.ContextFreeEncoder.default_model(
        len(words_lexicon.i2t),
        len(chars_lexicon.i2t),
        features=features,
        word_embeddings_dim=7,
        chars_embeddings_dim=5,
        span_encoding_dim=13,
        hidden_dim=23,
    )
    encoder = encoders.ContextFreeEncoder(
        model, words_lexicon, chars_lexicon, features, features_digitizers
    )
    span = {
        "left_context": ["spam", "eggs"],
        "content": ["sausages", "and", "spam"],
        "right_context": ["ham"],
        "length": 2,
    }
    digitized = encoder.digitize(span)
    target = datatools.FeaturefulSpan(
        tokens=(
            torch.tensor([1, 4, 3, 0, 1, 2]),
            (
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([8, 7, 7, 1]),
                torch.tensor([1, 3, 6, 1, 3, 7, 8, 1]),
                torch.tensor([3, 0, 0]),
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([5, 3, 4]),
            ),
        ),
        span_boundaries=(2, 5),
        features=(2,),
    )
    # The words are correctly digitized
    assert digitized.tokens[0].equal(target.tokens[0])
    # The characters are correctly digitized
    for d, t in zip(digitized.tokens[1], target.tokens[1]):
        assert d.equal(t)
    assert digitized.span_boundaries == target.span_boundaries
    assert digitized.features == target.features

