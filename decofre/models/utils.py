import pathlib

import typing as ty

from collections import Counter

import torch
import tqdm

from loguru import logger

import ujson as json

from decofre import lexicon

Digitizer = ty.Callable[[ty.Union[str, int, ty.Iterable[str]]], int]


def get_digitizers(
    features: ty.Sequence[ty.Dict[str, ty.Any]]
) -> ty.Dict[str, Digitizer]:
    """Read features configurations and return the necessary digitizers."""
    res = dict()  # type: ty.Dict[str, Digitizer]
    for f in features:
        name = f["name"]
        digitization = f.get("digitization", None)
        if digitization is None:
            res[name] = int  # type: ignore
        else:
            if digitization in ("lexicon", "word"):
                lex = f["lexicon"]
            else:
                raise ValueError(f"Unknow digitization name {digitization!r}")
            digitizer = lex.t2i

            config_vocabulary_size = f.get("vocabulary_size", None)
            digitizer_vocabulary_size = len(lex.i2t)
            if config_vocabulary_size is not None:
                if config_vocabulary_size != digitizer_vocabulary_size:
                    raise ValueError(
                        f"Bad vocabulary size for feature {name}:"
                        f"{config_vocabulary_size} != {digitizer_vocabulary_size}"
                    )
            else:
                f["vocabulary_size"] = digitizer_vocabulary_size
            res[name] = digitizer
    return res


def normal_tensor(size, mean=0.0, std=1.0, device=None):
    t = torch.empty(size, device=device)
    with torch.no_grad():
        t.normal_(mean, std)
    return t


# FIXME: allow to keep some original common words e.g. "euh"…
def load_embeddings(
    path: ty.Union[str, pathlib.Path], lex: lexicon.Lexicon, keep_original: bool = True
) -> ty.Tuple[lexicon.Lexicon, torch.Tensor]:
    """
    Load pretrained embeddings from a text word2vec file in the word2vec text
    format.

    See <https://fasttext.cc/docs/en/crawl-vectors.html> to get some. OOV
    embeddings are initialized at 0.
    """
    with open(path) as in_stream:
        n, d = map(int, in_stream.readline().split())
        weights = []
        if lex.allow_unknown:
            weights.append(normal_tensor((d,)))
        for _ in lex.specials:
            weights.append(normal_tensor((d,)))
        tokens = []
        pbar = tqdm.tqdm(
            in_stream,
            total=n,
            unit="words",
            dynamic_ncols=True,
            leave=False,
            unit_scale=True,
            desc="Loading embeddings",
            disable=None,
        )
        for line in pbar:
            word, *vec = line.rstrip().split(" ")
            i, f = lex.tokens_data.get(word, (None, None))
            if i is not None:
                assert f is not None
                weights.append(torch.tensor([float(x) for x in vec]))
                tokens.append((word, f))
    logger.info(
        f"Loaded {len(tokens)} pretrained embeddings (lexicon size {len(lex.i2t)})"
    )
    if keep_original:
        logger.debug("Generating weights for the tokens not found in pretrained")
        loaded = set((w for w, _ in tokens))
        generated = 0
        for w, (i, f) in lex.tokens_data.items():
            if f is not None and w not in loaded:
                weights.append(normal_tensor((d,)))
                tokens.append((w, f))
                generated += 1
        logger.debug(f"Generated {generated} vectors")
    new_lex = lexicon.Lexicon(
        vocabulary=tokens, specials=lex.specials, allow_unknown=lex.allow_unknown
    )
    return new_lex, torch.stack(weights)


def generate_lexicons(
    features_names: ty.Iterable[str],
    sources: ty.Iterable[ty.Union[str, pathlib.Path]],
    allow_unknown: ty.Optional[ty.Iterable[str]] = None,
) -> ty.Dict[str, lexicon.Lexicon]:
    """Generate features lexicons from json files"""
    allow_unknown = set(allow_unknown) if allow_unknown is not None else set()
    counters = {n: Counter() for n in features_names}  # type: ty.Dict[str, Counter]
    for f in tqdm.tqdm(
        sources,
        desc="Generating features lexicon",
        unit="files",
        leave=False,
        disable=None,
    ):
        with open(f) as in_stream:
            data = json.load(in_stream)
            # FIXME: this is brittle and should be fixed asap (2019-06-16) ← lol (2019-12)
            if isinstance(data, ty.Mapping):  # This is an antecedent file
                data = (
                    candidate
                    for candidates_lst in data["antecedents"].values()
                    for candidate in candidates_lst.values()
                )
            for sample in data:
                for n, count in counters.items():
                    count[sample[n]] += 1
    return {
        n: lexicon.Lexicon.from_counter(
            counters[n].items(), allow_unknown=n in allow_unknown
        )
        for n in features_names
    }


def generate_tokens_lexicons(
    features_names: ty.Iterable[str],
    sources: ty.Iterable[ty.Union[str, pathlib.Path]],
    allow_unknown: ty.Optional[ty.Iterable[str]] = None,
) -> ty.Dict[str, lexicon.Lexicon]:
    """Generate features lexicons from json files"""
    allow_unknown = set(allow_unknown) if allow_unknown is not None else set()
    counters = {n: Counter() for n in features_names}  # type: ty.Dict[str, Counter]
    for f in tqdm.tqdm(
        sources,
        desc="Generating token features lexicon",
        unit="files",
        leave=False,
        disable=None,
    ):
        with open(f) as in_stream:
            data = json.load(in_stream)
            # FIXME: this is brittle and should be fixed asap (2019-06-16)
            # FIXME: yes indeed (2020-02-14)
            if isinstance(data, ty.Mapping):  # This is an antecedent file
                data = (
                    candidate
                    for candidates_lst in data["antecedents"].values()
                    for candidate in candidates_lst.values()
                )
            for sample in data:
                for n, count in counters.items():
                    seq_values = sample[n]
                    # single-valued feats
                    if isinstance(seq_values[0], str):
                        count.update(seq_values)
                    # multivalued
                    else:
                        try:
                            count.update(v for value in seq_values for v in value)
                        except TypeError as e:
                            raise ValueError(
                                f"{seq_values} is not an acceptable list of token feature values"
                            ) from e
    return {
        n: lexicon.Lexicon.from_counter(
            counters[n].items(), allow_unknown=n in allow_unknown
        )
        for n in features_names
    }
