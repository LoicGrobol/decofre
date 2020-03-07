import json
import pathlib
import warnings

import typing as ty


__serialization_version__ = (0, 1, 0)


class LexiconItem(ty.NamedTuple):
    idx: int
    frequency: ty.Optional[int]


class Lexicon:
    """A two way mapping from tokens to integers."""

    def __init__(
        self,
        vocabulary: ty.Optional[ty.Iterable[ty.Tuple[ty.Hashable, int]]] = None,
        allow_unknown=True,
        specials: ty.Optional[ty.Iterable[ty.Hashable]] = None,
        unknow_token: ty.Hashable = "<unk>",
    ):
        self.allow_unknown = allow_unknown
        self.unknow_token = unknow_token

        # And this maps indices to the corresponding token
        self.i2t = []  # type: ty.List[ty.Hashable]
        # This maps tokens to their index and frequency (in that order)
        self.tokens_data = dict()  # type: ty.Dict[ty.Hashable, LexiconItem]
        self.total_freq = 0

        if allow_unknown:
            self.tokens_data[self.unknow_token] = LexiconItem(0, None)
            self.i2t.append(self.unknow_token)

        if specials is None:
            self.specials = set()  # type: ty.Set[ty.Hashable]
        else:
            for t in specials:
                self.tokens_data[t] = LexiconItem(len(self.i2t), None)
                self.i2t.append(t)
            self.specials = set(self.i2t[1:]) if self.allow_unknown else set(self.i2t)

        if vocabulary is not None:
            self.extend(vocabulary)

    def extend_from_instances(self, tokens: ty.Iterable[ty.Hashable]):
        """Add `#tokens` to the lexicon if they are not already present, update
           their frequencies if they are."""
        self.extend((t, 1) for t in tokens)

    def extend(self, counter: ty.Iterable[ty.Tuple[ty.Hashable, int]]):
        for token, freq in counter:
            old_item = self.tokens_data.get(token, None)
            if old_item is not None and token not in self.specials:
                assert (
                    old_item.frequency is not None
                ), "Only specials may have a `None` frequency"
                self.tokens_data[token] = LexiconItem(
                    old_item.idx, old_item.frequency + freq
                )
            else:
                self.tokens_data[token] = LexiconItem(len(self.i2t), freq)
                self.i2t.append(token)
            self.total_freq += freq

    def get_data(self, token: ty.Hashable) -> LexiconItem:
        data = self.tokens_data.get(token, None)
        if data is None:
            if self.allow_unknown:
                return self.tokens_data[self.unknow_token]
            raise ValueError(f"Token {token!r} is not in the lexicon")
        return data

    def t2i(self, t: ty.Hashable) -> int:
        """Get the index of token `#t`. Return `0` if `#t` is not in the lexicon."""
        return self.get_data(t).idx

    def freq(self, t: ty.Hashable) -> ty.Union[int, None]:
        """Get the frequency of token `#t`."""
        return self.get_data(t).frequency

    @classmethod
    def from_instances(
        cls, tokens: ty.Iterable[ty.Hashable], *args, **kwargs
    ) -> "Lexicon":
        """Create a new `#::Lexicon` from an iterable of tokens."""
        res = cls(*args, **kwargs)
        res.extend_from_instances(tokens)
        return res

    @classmethod
    def from_counter(
        cls, counter: ty.Iterable[ty.Tuple[ty.Hashable, int]], *args, **kwargs
    ) -> "Lexicon":
        res = cls(*args, **kwargs)
        res.extend(counter)
        return res

    def __str__(self):
        return str(self.i2t)

    def __len__(self):
        return len(self.i2t)


def load(lex_path: ty.Union[str, pathlib.Path]) -> Lexicon:
    """Load a lexicon from a tsv file."""
    with open(lex_path) as in_stream:
        params = json.load(in_stream)
    version = params.pop("version")
    if version[0] != __serialization_version__[0]:
        raise ValueError(
            f"Trying to load a lexicon using incompatible serialization ({version})"
        )
    elif version[1] < __serialization_version__[1]:
        warnings.warn(
            f"Loading a lexicon with deprecated serialization, ({version}),"
            " please consider updating it before it is too late."
        )
    return Lexicon(**params)


def dump(lex: Lexicon, lex_path: ty.Union[str, pathlib.Path]):
    """Save a lexicon to a file."""
    unknown_offset = 1 if lex.allow_unknown else 0
    specials = lex.i2t[unknown_offset : len(lex.specials) + unknown_offset]
    regular_tokens = lex.i2t[unknown_offset + len(lex.specials) :]

    with open(lex_path, "w") as out_stream:
        json.dump(
            {
                "allow_unknown": lex.allow_unknown,
                "unknow_token": lex.unknow_token,
                "specials": specials,
                "vocabulary": [(t, lex.freq(t)) for t in regular_tokens],
                "version": __serialization_version__,
            },
            out_stream,
        )
