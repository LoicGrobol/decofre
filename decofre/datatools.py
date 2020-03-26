"""Manage decofre data."""
from __future__ import annotations

import collections.abc
import io
import pathlib
import pickle  # nosec
import shutil

import typing as ty

import lmdb
import orjson
import torch
import torch.utils.data
import tqdm

from loguru import logger

from decofre import lexicon


class Batch(ty.NamedTuple):
    sources: ty.Sequence
    targets: torch.Tensor

    def to(self, device: ty.Union[torch.device, str]):
        return type(self)(
            sources=move(self.sources, device),
            targets=self.targets.to(device=device, non_blocking=True),
        )


class FeaturefulSpan(ty.NamedTuple):
    """A digitized text span with context and additional features

    Attributes
    ----------

    :tokens: a representation of the token of the span *and* its
      left and right context, iteration order is the same as document
      order
    :span_boundaries: the boundaries of the slice of `tokens` that is
      the span proper. Following Python's semantics, they are half-open,
      so if `span_boundaries == (i, j)`, the span is `tokens[i:j]`.
    :features: a sequence of digitized features value indices.
    """

    tokens: ty.Union[torch.Tensor, ty.Sequence]
    span_boundaries: ty.Tuple[int, int]
    features: ty.Sequence[int]

    @staticmethod
    def collate(
        batch: ty.Sequence["FeaturefulSpan"],
    ) -> ty.Tuple[
        ty.Tuple[ty.Union[torch.Tensor, ty.Sequence], ...], torch.Tensor, torch.Tensor
    ]:
        tokens, boundaries, feats = zip(*batch)
        boundaries = torch.tensor(boundaries, dtype=torch.long)
        feats = torch.tensor(feats, dtype=torch.long)
        return tokens, boundaries, feats


class AntecedentCandidate(ty.NamedTuple):
    mention_id: str
    pair_feats: ty.Tuple[int, ...]
    coref: bool


def move(t: ty.Union[torch.Tensor, ty.Iterable], device: ty.Union[torch.device, str]):
    """Recursively move `torch.Tensors` from an iterable to `#device`."""
    if torch.is_tensor(t):
        return ty.cast(torch.Tensor, t).to(device=device, non_blocking=True)
    elif hasattr(t, "to"):
        return t.to(device)  # type: ignore
    elif isinstance(t, collections.abc.Mapping):
        return {k: move(v, device) for k, v in t.items()}
    elif isinstance(t, collections.abc.Iterable):
        return [move(c, device) for c in t]
    raise TypeError(f"{type(t)} is not movable")


def to_bytes(obj: ty.Any) -> bytes:
    """Pickle an object to bytes using `torch.save`."""
    res = io.BytesIO()
    torch.save(obj, res, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    return res.getvalue()


def load_bytes(b: bytes) -> ty.Any:
    inpt = io.BytesIO(b)
    return torch.load(inpt)


# TODO: Allow dynamic map size (e.g. double when full)
class LmdbDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: ty.Union[str, pathlib.Path],
        set_name: str,
        data: ty.Optional[ty.Iterable] = None,
        overwrite_cache: bool = False,
        map_size: int = int(2e11),
        **kwargs,
    ):
        super().__init__()
        self.map_size = map_size
        cache_dir = pathlib.Path(cache_dir)
        self.db_path = cache_dir / set_name
        if self.db_path.exists():
            if overwrite_cache:
                logger.warning(f"Overwriting data cache at {self.db_path}")
                shutil.rmtree(str(self.db_path))
            else:
                raise ValueError(
                    f"Cache at {self.db_path} already exists and won't be overwritten"
                )
        cache_dir.mkdir(parents=True, exist_ok=True)
        with lmdb.open(
            str(self.db_path),
            map_size=self.map_size,
            meminit=False,
            readahead=False,
            map_async=True,
            writemap=True,
            subdir=True,
            **kwargs,
        ):
            # Just create the db
            pass

        self._len = 0
        if data is not None:
            self.add_data(data)

    def add_data(self, data: ty.Iterable):
        with lmdb.open(
            str(self.db_path),
            map_size=self.map_size,
            meminit=False,
            readahead=False,
            map_async=True,
            writemap=True,
        ) as env:
            with env.begin(write=True, buffers=True) as txn:
                added, consumed = txn.cursor().putmulti(
                    (
                        (i.to_bytes(64, "big"), to_bytes(d))
                        for i, d in enumerate(data, start=len(self))
                    ),
                    append=True,
                )
                if added != consumed:
                    raise ValueError(
                        f"Database overwrite: only {added} new elements among {consumed}"
                    )
                new_len = txn.stat()["entries"]
            if new_len != self._len + added:
                raise ValueError(
                    f"Data corruption: {new_len} database entries instead of {self._len + added}"
                )
            self._len = new_len

    def __len__(self):
        return self._len

    def __getitem__(self, index: ty.Union[int, ty.Iterable[int]]):
        with lmdb.open(
            str(self.db_path),
            meminit=False,
            readahead=False,
            subdir=True,
            readonly=True,
        ) as env:
            with env.begin(write=False, buffers=True) as txn:
                if isinstance(index, int):
                    return load_bytes(txn.get(index.to_bytes(64, "big")))
                else:
                    return [load_bytes(txn.get(i.to_bytes(64, "big"))) for i in index]


class SpansDataset(LmdbDataset):
    def __init__(
        self,
        span_digitizer: ty.Callable[[ty.Mapping[str, ty.Any]], FeaturefulSpan],
        tags_lexicon: lexicon.Lexicon,
        cache_dir: ty.Union[str, pathlib.Path],
        data: ty.Optional[ty.Iterable] = None,
        set_name: str = "spans",
    ):
        self.span_digitizer = span_digitizer
        self.tags_lexicon = tags_lexicon
        super().__init__(cache_dir, set_name, data=data)

    def add_data(self, data: ty.Iterable):
        if isinstance(data, ty.Sized):
            data_len = len(data)
        else:
            data_len = None
        digitized = (
            (self.span_digitizer(span), self.tags_lexicon.t2i(span["type"]))
            for span in data
        )
        pbar = tqdm.tqdm(
            digitized,
            desc="Digitizing spans",
            unit="spans",
            total=data_len,
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            leave=False,
            mininterval=1,
            disable=None,
        )
        super().add_data(pbar)

    @ty.overload
    def __getitem__(self, index: int) -> FeaturefulSpan:
        pass

    @ty.overload
    def __getitem__(self, index: ty.Iterable[int]) -> Batch:
        pass

    def __getitem__(self, index):
        loaded = super().__getitem__(index)
        if isinstance(index, int):
            return loaded
        else:
            return type(self).collate(loaded)

    @staticmethod
    def collate(batch: ty.Iterable[ty.Tuple[FeaturefulSpan, int]]) -> Batch:
        spans, targets = zip(*batch)
        targets = torch.tensor(targets, dtype=torch.long)
        res = Batch(FeaturefulSpan.collate(spans), targets)
        return res

    @classmethod
    def from_json(
        cls, json_path: ty.Union[pathlib.Path, str], cache_dir=None, *args, **kwargs
    ):
        json_path = pathlib.Path(json_path)
        if cache_dir is None:
            cache_dir = json_path.parent / ".data_cache"
        else:
            cache_dir = pathlib.Path(cache_dir)
        kwargs.setdefault("cache_dir", cache_dir)

        res = cls(*args, **kwargs)

        if json_path.is_file():
            json_files = [json_path]
        else:
            json_files = list(json_path.glob("*.json"))

        pbar = tqdm.tqdm(json_files, unit="files", leave=False, disable=None)
        for f in pbar:
            pbar.set_description(f"Loading {f.name}")
            with open(f, "rb") as in_stream:
                data = orjson.loads(in_stream.read())
                res.add_data(data)
        logger.debug(f"Loaded {res._len} samples in db at {res.db_path}")
        return res


# FIXME: The collate/prepare batch logic is flaky
# FIXME: Figure out a proper interface to data management that is less brittle and more
# Liskov-compliant
class AntecedentsDataset(LmdbDataset):
    def __init__(
        self,
        cache_dir: ty.Union[str, pathlib.Path],
        mentions: ty.Optional[ty.Dict[str, FeaturefulSpan]] = None,
        antecedents: ty.Optional[
            ty.Iterable[
                ty.Tuple[str, ty.Sequence[AntecedentCandidate], ty.Sequence[int]]
            ]
        ] = None,
        set_name: str = "antecedents",
        **kwargs,
    ):
        super().__init__(cache_dir, set_name, data=None, max_dbs=2, **kwargs)
        self.add_data(mentions, antecedents)

    def add_data(
        self,
        mentions: ty.Optional[ty.Dict[str, FeaturefulSpan]] = None,
        antecedents: ty.Optional[
            ty.Iterable[
                ty.Tuple[str, ty.Sequence[AntecedentCandidate], ty.Sequence[int]]
            ]
        ] = None,
    ):
        with lmdb.open(
            str(self.db_path),
            meminit=False,
            readahead=False,
            map_async=True,
            writemap=True,
            max_dbs=2,
            map_size=self.map_size,
        ) as env:
            if mentions is not None:
                if isinstance(mentions, ty.Sized):
                    total_mentions = len(mentions)
                else:
                    total_mentions = None
                mentions_db = env.open_db(key="mentions_db".encode())
                with env.begin(db=mentions_db, write=True, buffers=True) as txn:
                    txn.cursor().putmulti(
                        tqdm.tqdm(
                            (
                                (
                                    k.encode(),
                                    to_bytes(v),
                                )
                                for k, v in mentions.items()
                            ),
                            desc="Writing mentions to database",
                            total=total_mentions,
                            unit_scale=True,
                            unit="mentions",
                            unit_divisor=1024,
                            dynamic_ncols=True,
                            leave=False,
                            mininterval=1,
                            disable=None,
                        )
                    )
            if antecedents is not None:
                if isinstance(antecedents, ty.Sized):
                    total_antecedents = len(antecedents)  # type: ty.Optional[int]
                else:
                    total_antecedents = None
                antecedents_db = env.open_db(key="antecedents_db".encode())
                with env.begin(db=antecedents_db, write=True, buffers=True) as txn:
                    added, consumed = txn.cursor().putmulti(
                        tqdm.tqdm(
                            (
                                (
                                    i.to_bytes(64, "big"),
                                    to_bytes(
                                        (
                                            m_id.encode(),
                                            tuple(
                                                (
                                                    ant.mention_id.encode(),
                                                    ant.pair_feats,
                                                )
                                                for ant in a
                                            ),
                                            t,
                                        ),
                                    ),
                                )
                                for i, (m_id, a, t) in enumerate(
                                    antecedents, start=len(self)
                                )
                            ),
                            desc="Writing antecedents to database",
                            total=total_antecedents,
                            unit_scale=True,
                            unit="mentions",
                            unit_divisor=1024,
                            dynamic_ncols=True,
                            leave=False,
                            mininterval=1,
                            disable=None,
                        ),
                        append=True,
                    )
                    if added != consumed:
                        raise ValueError(
                            f"Database overwrite: only {added} new elements among {consumed}"
                        )
                    new_len = txn.stat()["entries"]
                if new_len != self._len + added:
                    raise ValueError(
                        f"Data corruption: {new_len} database entries instead of {self._len + added}"
                    )
                self._len = new_len

    @ty.overload
    def __getitem__(
        self, index: int
    ) -> ty.Tuple[
        ty.Tuple[
            FeaturefulSpan, ty.Tuple[ty.Tuple[FeaturefulSpan, ty.Sequence[int]], ...]
        ],
        ty.Sequence[int],
    ]:
        pass

    @ty.overload
    def __getitem__(self, index: ty.Iterable[int]) -> Batch:
        pass

    def __getitem__(self, index):
        if isinstance(index, int):
            indices = [index]
        else:
            indices = index

        with lmdb.open(
            str(self.db_path),
            meminit=False,
            readahead=False,
            subdir=True,
            readonly=True,
            max_dbs=2,
        ) as env:
            mentions_db = env.open_db(key="mentions_db".encode())
            antecedents_db = env.open_db(key="antecedents_db".encode())
            with env.begin(db=antecedents_db, write=False, buffers=True) as txn:
                loaded_indices = [
                    load_bytes(txn.get(i.to_bytes(64, "big"))) for i in indices
                ]

            with env.begin(db=mentions_db, write=False, buffers=True) as txn:
                loaded_data = [
                    (
                        (
                            load_bytes(txn.get(mention)),
                            tuple(
                                (load_bytes(txn.get(a_id)), pairs_feats)
                                for a_id, pairs_feats in antecedents
                            ),
                        ),
                        target,
                    )
                    for mention, antecedents, target in loaded_indices
                ]
        if isinstance(index, int):
            return loaded_data[0]
        else:
            return type(self).collate(loaded_data)

    @staticmethod
    def prepare_source_batch(batch) -> ty.Sequence:
        mentions, antecedents = zip(*batch)
        mentions = FeaturefulSpan.collate(mentions)
        collated_antecedents = []
        for a in antecedents:
            spans, pairs_feats = zip(*a)
            spans = FeaturefulSpan.collate(spans)
            pairs_feats = torch.tensor(pairs_feats, dtype=torch.long)
            collated_antecedents.append((spans, pairs_feats))
        return (mentions, collated_antecedents)

    @classmethod
    def collate(cls, batch) -> Batch:
        raw_sources, targets = zip(*batch)
        sources = cls.prepare_source_batch(raw_sources)
        max_antecedents = max(len(a) for m, a in raw_sources)
        target_masks = tuple(
            torch.zeros(max_antecedents + 1, dtype=torch.bool).scatter_(
                0, torch.tensor(t), True
            )
            for t in targets
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(target_masks, batch_first=True)
        return Batch(sources, padded_targets)

    # TODO: allow reading from multiple files
    @classmethod
    def from_json(
        cls,
        json_path: ty.Union[str, pathlib.Path],
        span_digitizer: ty.Callable[[ty.Mapping[str, ty.Any]], FeaturefulSpan],
        pair_feats_digitizer: ty.Callable[[ty.Mapping[str, str]], ty.Iterable[int]],
        cache_dir: ty.Optional[ty.Union[str, pathlib.Path]] = None,
        *args,
        **kwargs,
    ):
        json_path = pathlib.Path(json_path)
        if cache_dir is None:
            cache_dir = json_path.parent / ".data_cache"
        else:
            cache_dir = pathlib.Path(cache_dir)
        kwargs.setdefault("cache_dir", cache_dir)

        res = cls(*args, **kwargs)

        if json_path.is_file():
            json_files = [json_path]
        else:
            json_files = list(json_path.glob("*.json"))

        pbar = tqdm.tqdm(json_files, unit="files", leave=False, disable=None)
        for f in pbar:
            pbar.set_description(f"Loading {f.name}")
            with open(f, "rb") as in_stream:
                data = orjson.loads(in_stream.read())
            mentions = {k: span_digitizer(v) for k, v in data["mentions"].items()}
            antecedents = dict()
            for m_id, candidates in data["antecedents"].items():
                current_ant = []
                targets = []
                for i, (c_id, c) in enumerate(candidates.items(), start=1):
                    feats = tuple(pair_feats_digitizer(c))
                    current_ant.append(
                        AntecedentCandidate(
                            mention_id=c_id, pair_feats=feats, coref=c["coref"]
                        )
                    )
                    if c["coref"]:
                        targets.append(i)
                if not targets:
                    targets = [0]
                antecedents[m_id] = (tuple(current_ant), tuple(targets))

            res.add_data(mentions, ((m, a, t) for m, (a, t) in antecedents.items()))

        return res
