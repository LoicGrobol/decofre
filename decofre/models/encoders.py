import abc
import copy
import pathlib
import tempfile
import shutil

import itertools as it
import typing as ty

from collections import Counter
from pprint import pformat

import schema
import tqdm
import torch

# Suppress tf messages at loading time
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # noqa: E402
import transformers

import ujson as json

from loguru import logger

from decofre import datatools, libdecofre, lexicon
from decofre.models import Model, InvalidModelException, config, utils

try:
    from allennlp.modules import elmo
except ImportError:
    elmo = None


class Encoder(Model, metaclass=abc.ABCMeta):
    model: libdecofre.FeaturefulSpanEncoder
    out_dim: int
    _features: ty.Sequence[ty.Dict[str, ty.Any]]
    _features_digitizers: ty.Dict[str, utils.Digitizer]

    def __init__(
        self,
        model: libdecofre.FeaturefulSpanEncoder,
        out_dim: int,
        features: ty.Optional[ty.Sequence[ty.Dict[str, ty.Any]]],
        features_digitizers: ty.Optional[ty.Dict[str, utils.Digitizer]],
    ):
        self._features = features if features is not None else tuple()
        self._features_digitizers = (
            features_digitizers if features_digitizers is not None else dict()
        )
        self.model = model
        self.out_dim = out_dim

    @abc.abstractmethod
    def digitize(
        self, span: ty.Mapping[str, ty.Union[str, int, ty.Sequence[str]]]
    ) -> datatools.FeaturefulSpan:
        pass

    @abc.abstractmethod
    def save(self, path: ty.Union[str, pathlib.Path]):
        pass

    def __call__(self, span):
        return self.model(self.digitize(span))

    def digitize_feats(
        self, span: ty.Mapping[str, ty.Union[str, int, ty.Sequence[str]]]
    ) -> ty.Tuple[int, ...]:
        feats = []
        for f in self._features:
            n = f["name"]
            digitizer = self._features_digitizers[n]
            try:
                raw_feature = span[n]
            except KeyError as e:
                logger.error(
                    f"Missing feature {n} in the following span\n{pformat(span)}"
                )
                raise e
            feats.append(digitizer(raw_feature))
        return tuple(feats)

    @classmethod
    def load(cls, model_path: ty.Union[str, pathlib.Path]) -> "Encoder":
        """Load a model archive."""
        with tempfile.TemporaryDirectory() as _tempdir:
            tempdir = pathlib.Path(_tempdir)
            try:
                shutil.unpack_archive(str(model_path), tempdir, format="gztar")
            except shutil.ReadError as e:
                raise InvalidModelException(f"Couldn't unpack {model_path}") from e
            try:
                with open(tempdir / "config.json") as config_stream:
                    encoder_config = config.encoder_schema.validate(
                        json.load(config_stream)
                    )
            except FileNotFoundError as e:
                raise InvalidModelException(f"Files missing in {model_path}") from e
            except schema.SchemaError as e:
                raise InvalidModelException(f"Invalid config in {model_path}") from e
        if encoder_config["type"] == "context-free":
            return ContextFreeEncoder.load(model_path)
        elif encoder_config["type"] == "elmo":
            return ELMoEncoder.load(model_path)
        elif encoder_config["type"] == "bert":
            return BERTEncoder.load(model_path)
        else:
            raise ValueError(f"Unknown encoder type {encoder_config['type']}")

    @classmethod
    def initialize(
        cls, model_config: ty.Dict[str, ty.Any], initialisation: ty.Dict[str, ty.Any]
    ) -> "Encoder":
        if model_config["type"] == "context-free":
            return ContextFreeEncoder.initialize(model_config, initialisation)
        elif model_config["type"] == "elmo":
            return ELMoEncoder.initialize(model_config, initialisation)
        elif model_config["type"] == "bert":
            return BERTEncoder.initialize(model_config, initialisation)
        else:
            raise ValueError(f"Unknown encoder type {model_config['type']}")


class ContextFreeEncoder(Encoder):
    """
    A wrapper around a `libdecofre.FeaturefulSpanEncoder` that provides digitization
    and serialization.
    """

    def __init__(
        self,
        model: libdecofre.FeaturefulSpanEncoder,
        words_lexicon: lexicon.Lexicon,
        chars_lexicon: lexicon.Lexicon,
        features: ty.Sequence[ty.Dict[str, ty.Any]],
        features_digitizers: ty.Dict[str, utils.Digitizer],
        token_features: ty.Optional[ty.Sequence[ty.Dict[str, ty.Any]]] = None,
    ):
        super().__init__(model, model.out_dim, features, features_digitizers)
        self._words_lexicon = words_lexicon
        self._chars_lexicon = chars_lexicon
        self._token_features = token_features if token_features is not None else tuple()

    def digitize(
        self, span: ty.Mapping[str, ty.Union[str, int, ty.Sequence[str]]]
    ) -> datatools.FeaturefulSpan:
        """Digitize a span."""
        left_context = ty.cast(ty.Sequence[str], span["left_context"])
        content = ty.cast(ty.Sequence[str], span["content"])
        right_context = ty.cast(ty.Sequence[str], span["right_context"])
        all_tokens = [*left_context, *content, *right_context]
        span_boundaries = (len(left_context), len(left_context) + len(content))

        # FIXME: or short-circuiting here is brittle and ugly
        words = torch.tensor(
            [
                self._words_lexicon.t2i(w) or self._words_lexicon.t2i(w.lower())
                for w in all_tokens
            ]
        )
        chars = tuple(
            torch.tensor([self._chars_lexicon.t2i(c) for c in w])
            if w not in self._words_lexicon.specials
            else torch.tensor([self._chars_lexicon.t2i("<no>")])
            for w in all_tokens
        )

        feats = self.digitize_feats(span)

        if self._token_features:
            token_feats = []
            for f in self._token_features:
                n = f["name"]
                d = self._features_digitizers[n]
                try:
                    v = ty.cast(ty.Sequence[str], span[n])
                except KeyError as e:
                    logger.error(
                        f"Missing feature {n} in the following span\n{pformat(span)}"
                    )
                    raise e
                token_feats.append([d(t) for t in v])
            res = datatools.FeaturefulSpan(
                ((words, chars), torch.tensor(token_feats, dtype=torch.int64).t()),
                span_boundaries,
                tuple(feats),
            )
        else:
            res = datatools.FeaturefulSpan(
                (words, chars), span_boundaries, tuple(feats)
            )
        return res

    def save(self, path: ty.Union[str, pathlib.Path]):
        """Save as a model archive."""
        logger.debug(f"Saving the model to {path}")
        with tempfile.TemporaryDirectory() as _tempdir:
            tempdir = pathlib.Path(_tempdir)
            lexicon.dump(self._words_lexicon, tempdir / "words.lexicon")
            lexicon.dump(self._chars_lexicon, tempdir / "chars.lexicon")
            torch.save(self.model.state_dict(), tempdir / "weights.dat")

            features_dump = copy.deepcopy(self._features)
            for f in features_dump:
                digitization = f.get("digitization", None)
                if digitization is None:
                    continue
                if digitization == "lexicon":
                    lexicon_filename = f"{f['name']}.lexicon"
                    lexicon.dump(f["lexicon"], tempdir / lexicon_filename)
                elif digitization == "word":
                    lexicon_filename = "words.lexicon"
                f["lexicon"] = lexicon_filename

            # FIXME: deduplicate please
            token_features_dump = copy.deepcopy(self._token_features)
            for f in token_features_dump:
                lexicon_filename = f"{f['name']}.lexicon"
                lexicon.dump(f["lexicon"], tempdir / lexicon_filename)
                f["lexicon"] = lexicon_filename

            if isinstance(self.model.tokens_encoder, libdecofre.ContextFreeWordEncoder):
                word_embeddings_dim = self.model.tokens_encoder.words_embeddings_dim
                chars_embeddings_dim = self.model.tokens_encoder.chars_embeddings_dim
            elif isinstance(
                self.model.tokens_encoder, libdecofre.FeaturefulWordEncoder
            ):
                word_embeddings_dim = (
                    self.model.tokens_encoder.words_encoder.words_embeddings_dim
                )
                chars_embeddings_dim = (
                    self.model.tokens_encoder.words_encoder.chars_embeddings_dim
                )
            else:
                raise ValueError(
                    f"{type(self.model.tokens_encoder)} should not have been used for model word encoder"
                )

            encoder_config = {
                "type": "context-free",
                "span_encoding_dim": self.model.out_dim,
                "word_embeddings_dim": word_embeddings_dim,
                "chars_embeddings_dim": chars_embeddings_dim,
                "hidden_dim": self.model.span_encoder.hidden_dim,
                "features": features_dump,
                "token_features": token_features_dump,
            }
            with open(tempdir / "config.json", "w") as config_stream:
                json.dump(encoder_config, config_stream)

            archive = shutil.make_archive(str(tempdir), "gztar", root_dir=tempdir)
            shutil.move(archive, path)

    @classmethod
    def load(cls, model_path: ty.Union[str, pathlib.Path]) -> "ContextFreeEncoder":
        """Load a model archive."""
        with tempfile.TemporaryDirectory() as _tempdir:
            tempdir = pathlib.Path(_tempdir)
            try:
                shutil.unpack_archive(str(model_path), tempdir, format="gztar")
            except shutil.ReadError as e:
                raise InvalidModelException(f"Couldn't unpack {model_path}") from e
            try:
                words_lexicon = lexicon.load(tempdir / "words.lexicon")
                chars_lexicon = lexicon.load(tempdir / "chars.lexicon")
                with open(tempdir / "config.json") as config_stream:
                    encoder_config = config.encoder_schema.validate(
                        json.load(config_stream)
                    )
                features = encoder_config["features"]
                feature_digitizers: ty.Dict[str, utils.Digitizer]
                if features is None:
                    feature_digitizers = dict()
                else:
                    for f in features:
                        digitization = f.get("digitization", None)
                        if digitization == "lexicon":
                            f["lexicon"] = lexicon.load(tempdir / f["lexicon"])
                        elif digitization == "word":
                            f["lexicon"] = words_lexicon
                    feature_digitizers = utils.get_digitizers(features)

                token_features = encoder_config["token_features"]
                if token_features is not None:
                    for f in token_features:
                        f_lex = lexicon.load(tempdir / f["lexicon"])
                        f["lexicon"] = f_lex
                        feature_digitizers[f["name"]] = f_lex.t2i

                model = cls.default_model(
                    len(words_lexicon.i2t), len(chars_lexicon.i2t), **encoder_config
                )

                weights_path = tempdir / "weights.dat"
                weights = torch.load(weights_path, map_location="cpu")
            except FileNotFoundError as e:
                raise InvalidModelException(f"Files missing in {model_path}") from e
            model.load_state_dict(weights)

            return cls(
                model=model,
                words_lexicon=words_lexicon,
                chars_lexicon=chars_lexicon,
                features=features,
                features_digitizers=feature_digitizers,
                token_features=token_features,
            )

    # FIXME: there has to be a better way to deal with word features
    @staticmethod
    def default_model(
        words_vocab_size: int,
        chars_vocab_size: int,
        span_encoding_dim: int,
        word_embeddings_dim: int,
        chars_embeddings_dim: int,
        hidden_dim: int,
        features: ty.Optional[ty.Iterable[ty.Dict[str, ty.Any]]] = None,
        token_features: ty.Optional[ty.Iterable[ty.Dict[str, ty.Any]]] = None,
        embeddings: ty.Optional[torch.Tensor] = None,
        external_boundaries: bool = False,
        **kwargs,
    ) -> libdecofre.FeaturefulSpanEncoder:
        featureless_words_encoder = libdecofre.ContextFreeWordEncoder(
            words_vocab_size=words_vocab_size,
            chars_vocab_size=chars_vocab_size,
            words_embeddings_dim=word_embeddings_dim,
            chars_embeddings_dim=chars_embeddings_dim,
            chars_encoding_dim=chars_embeddings_dim,
            pretrained_embeddings=embeddings,
        )

        words_encoder: ty.Union[
            libdecofre.FeaturefulWordEncoder, libdecofre.ContextFreeWordEncoder
        ]
        if token_features:
            token_features_lst = tuple(
                (f["vocabulary_size"], f["embeddings_dim"], None)
                for f in token_features
            )
            token_features_encoder = libdecofre.CategoricalFeaturesBag(
                token_features_lst
            )
            words_encoder = libdecofre.FeaturefulWordEncoder(
                words_encoder=featureless_words_encoder,
                features_encoder=token_features_encoder,
            )
            words_encoding_dim = (
                featureless_words_encoder.out_dim + token_features_encoder.out_dim
            )
        else:
            words_encoder = featureless_words_encoder
            words_encoding_dim = words_encoder.out_dim

        featureless_span_encoder = libdecofre.SpanEncoder(
            words_encoding_dim=words_encoding_dim,
            hidden_dim=hidden_dim,
            ffnn_dim=hidden_dim,
            out_dim=span_encoding_dim,
            external_boundaries=external_boundaries,
        )

        features_encoder: ty.Optional[libdecofre.CategoricalFeaturesBag]
        if features:
            features_lst = tuple(
                (
                    f["vocabulary_size"],
                    f["embeddings_dim"],
                    embeddings if f.get("digitization", None) == "word" else None,
                )
                for f in features
            )
            features_encoder = libdecofre.CategoricalFeaturesBag(features_lst)
        else:
            features_encoder = None

        enc = libdecofre.FeaturefulSpanEncoder(
            tokens_encoder=words_encoder,
            span_encoder=featureless_span_encoder,
            features_encoder=features_encoder,
            out_dim=span_encoding_dim,
        )
        return enc

    @staticmethod
    def generate_lexicons(
        spans_path: ty.Union[str, pathlib.Path], min_freq: int = 1
    ) -> ty.Tuple[lexicon.Lexicon, lexicon.Lexicon]:
        """Generate words and characters lexicons from the spans of a corpus."""
        spans_path = pathlib.Path(spans_path)
        if spans_path.is_file():
            spans_files = [spans_path]
        else:
            spans_files = list(spans_path.glob("*.json"))
        toks = Counter()  # type: ty.Counter[str]
        for f in tqdm.tqdm(
            spans_files,
            desc="Generating tokens lexicons",
            unit="files",
            leave=False,
            disable=None,
        ):
            with open(f) as in_stream:
                data = json.load(in_stream)
            for r in data:
                for w in r["content"]:
                    toks[w] += 1

        words_lex = lexicon.Lexicon.from_counter(
            tqdm.tqdm(
                ((w, c) for w, c in toks.items() if c > min_freq),
                desc="Building words lexicon",
                leave=False,
                disable=None,
            ),
            specials=("<pad>", "<start>", "<end>"),
        )
        chars_lex = lexicon.Lexicon.from_counter(
            tqdm.tqdm(
                ((c, count) for w, count in toks.items() for c in w),
                desc="Building chars lexicon",
                leave=False,
                disable=None,
            ),
            specials=("<no>",),
        )

        return words_lex, chars_lex

    @classmethod
    def initialize(
        cls, model_config: ty.Dict[str, ty.Any], initialisation: ty.Dict[str, ty.Any]
    ) -> "ContextFreeEncoder":
        """Create a new encoder from a model config and weights/lexicons initialisation."""
        words_lexicon, chars_lexicon = cls.generate_lexicons(
            initialisation["lexicon-source"]
        )
        embeddings: ty.Optional[torch.Tensor]
        word_embeddings_path = initialisation.get("word_embeddings_path", None)
        if word_embeddings_path is not None:
            logger.debug(f"Loading embeddings from {word_embeddings_path}")
            words_lexicon, embeddings = utils.load_embeddings(
                word_embeddings_path, words_lexicon
            )
        else:
            logger.info("Not using pretrained embeddings")
            embeddings = None

        features_config = model_config.pop("features")
        if features_config is None:
            features: ty.List[ty.Dict[str, ty.Any]] = []
            features_digitizers: ty.Dict[str, utils.Digitizer] = dict()
        else:
            features = config.load_features(
                features_config, initialisation["lexicon-source"], words_lexicon
            )
            features_digitizers = utils.get_digitizers(features)

        token_features = model_config.pop("token_features")
        if token_features is not None:
            token_features = config.load_token_features(
                token_features, initialisation["lexicon-source"]
            )
            features_digitizers.update(
                {f["name"]: f["lexicon"].t2i for f in token_features}
            )
        model = cls.default_model(
            len(words_lexicon.i2t),
            len(chars_lexicon.i2t),
            embeddings=embeddings,
            features=features,
            token_features=token_features,
            **model_config,
        )
        return cls(
            model=model,
            words_lexicon=words_lexicon,
            chars_lexicon=chars_lexicon,
            features=features,
            features_digitizers=features_digitizers,
            token_features=token_features,
        )


if elmo is not None:

    class ELMoEncoder(Encoder):
        """
        A wrapper around a `libdecofre.FeaturefulSpanEncoder` that provides digitization
        and serialization.
        """

        def __init__(
            self,
            model: libdecofre.FeaturefulSpanEncoder,
            features: ty.Sequence[ty.Dict[str, ty.Any]],
            features_digitizers: ty.Dict[str, utils.Digitizer],
            elmo_weight_file: str,
            elmo_options_file: str,
        ):
            super().__init__(model, model.out_dim, features, features_digitizers)
            self.elmo_weight_file = elmo_weight_file
            self.elmo_options_file = elmo_options_file

        def digitize(
            self, span: ty.Mapping[str, ty.Union[str, int, ty.Sequence[str]]]
        ) -> datatools.FeaturefulSpan:
            """Digitize a span."""
            left_context = ty.cast(ty.Sequence[str], span["left_context"])
            content = ty.cast(ty.Sequence[str], span["content"])
            right_context = ty.cast(ty.Sequence[str], span["right_context"])
            words = elmo.batch_to_ids([[*left_context, *content, *right_context]])
            # FIXME: :art:
            span_boundaries = (len(left_context), len(left_context) + len(content))
            feats = self.digitize_feats(span)
            return datatools.FeaturefulSpan(words, span_boundaries, feats)

        def save(self, path: ty.Union[str, pathlib.Path]):
            """Save as a model archive."""
            logger.debug(f"Saving the model to {path}")
            with tempfile.TemporaryDirectory() as _tempdir:
                tempdir = pathlib.Path(_tempdir)
                torch.save(self.model.state_dict(), tempdir / "weights.dat")

                features_dump = copy.deepcopy(self._features)
                for f in features_dump:
                    digitization = f.get("digitization", None)
                    if digitization is None:
                        continue
                    if digitization == "lexicon":
                        lexicon_filename = f"{f['name']}.lexicon"
                        lexicon.dump(f["lexicon"], tempdir / lexicon_filename)
                    f["lexicon"] = lexicon_filename

                encoder_config = {
                    "type": "elmo",
                    "span_encoding_dim": self.model.out_dim,
                    "hidden_dim": self.model.span_encoder.hidden_dim,
                    "features": features_dump,
                    "elmo_weight_file": self.elmo_weight_file,
                    "elmo_options_file": self.elmo_options_file,
                }
                with open(tempdir / "config.json", "w") as config_stream:
                    json.dump(encoder_config, config_stream)

                archive = shutil.make_archive(str(tempdir), "gztar", root_dir=tempdir)
                shutil.move(archive, path)

        @classmethod
        def load(cls, model_path: ty.Union[str, pathlib.Path]) -> "ELMoEncoder":
            """Load a model archive."""
            with tempfile.TemporaryDirectory() as _tempdir:
                tempdir = pathlib.Path(_tempdir)
                try:
                    shutil.unpack_archive(str(model_path), tempdir, format="gztar")
                except shutil.ReadError as e:
                    raise InvalidModelException(f"Couldn't unpack {model_path}") from e
                try:
                    with open(tempdir / "config.json") as config_stream:
                        encoder_config = config.encoder_schema.validate(
                            json.load(config_stream)
                        )
                except FileNotFoundError as e:
                    raise InvalidModelException(f"Files missing in {model_path}") from e
                except schema.SchemaError as e:
                    raise InvalidModelException(
                        f"Invalid config in {model_path}"
                    ) from e

                try:
                    features = encoder_config["features"]
                    for f in features:
                        digitization = f.get("digitization", None)
                        if digitization == "lexicon":
                            f["lexicon"] = lexicon.load(tempdir / f["lexicon"])
                    feature_digitizers = utils.get_digitizers(features)
                    model = cls.default_model(**encoder_config)

                    weights_path = tempdir / "weights.dat"
                    weights = torch.load(weights_path, map_location="cpu")
                except FileNotFoundError as e:
                    raise InvalidModelException(f"Files missing in {model_path}") from e
                model.load_state_dict(weights)

                return cls(
                    model=model,
                    features=features,
                    features_digitizers=feature_digitizers,
                    elmo_options_file=encoder_config["elmo_options_file"],
                    elmo_weight_file=encoder_config["elmo_weight_file"],
                )

        @staticmethod
        def default_model(
            elmo_options_file: ty.Union[str, pathlib.Path],
            elmo_weight_file: ty.Union[str, pathlib.Path],
            span_encoding_dim: int,
            hidden_dim: int,
            features: ty.Iterable[ty.Dict[str, ty.Any]] = None,
            embeddings: ty.Optional[torch.Tensor] = None,
            external_boundaries: bool = False,
            **kwargs,
        ) -> libdecofre.FeaturefulSpanEncoder:
            words_encoder = libdecofre.ELMoWordEncoder(
                options_file=str(elmo_options_file), weight_file=str(elmo_weight_file)
            )
            featureless_span_encoder = libdecofre.SpanEncoder(
                words_encoding_dim=words_encoder.out_dim,
                hidden_dim=hidden_dim,
                ffnn_dim=hidden_dim,
                out_dim=span_encoding_dim,
                external_boundaries=external_boundaries,
            )

            if features is None:
                raise NotImplementedError("Featureless encoder not yet supported")
            else:
                # instead of just having the same initialization
                features_lst = tuple(
                    (f["vocabulary_size"], f["embeddings_dim"], None) for f in features
                )

                enc = libdecofre.FeaturefulSpanEncoder(
                    tokens_encoder=words_encoder,
                    span_encoder=featureless_span_encoder,
                    features_encoder=libdecofre.CategoricalFeaturesBag(features_lst),
                    out_dim=span_encoding_dim,
                )
                return enc

        @classmethod
        def initialize(
            cls,
            model_config: ty.Dict[str, ty.Any],
            initialisation: ty.Dict[str, ty.Any],
        ) -> "ELMoEncoder":
            """Create a new encoder from a model config and weights/lexicons initialisation."""
            features = config.load_features(
                model_config.pop("features"), initialisation["lexicon-source"]
            )
            features_digitizers = utils.get_digitizers(features)
            model = cls.default_model(features=features, **model_config)
            return cls(
                model=model,
                features=features,
                features_digitizers=features_digitizers,
                elmo_options_file=model_config["elmo_options_file"],
                elmo_weight_file=model_config["elmo_weight_file"],
            )


class BERTEncoder(Encoder):
    def __init__(
        self,
        model: libdecofre.FeaturefulSpanEncoder,
        features: ty.Sequence[ty.Dict[str, ty.Any]],
        features_digitizers: ty.Dict[str, utils.Digitizer],
        pretrained: str,
    ):
        super().__init__(model, model.out_dim, features, features_digitizers)
        self.pretrained = pretrained
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.pretrained, do_lowercase_and_remove_accent=False, use_fast=True
        )

    def digitize(
        self, span: ty.Mapping[str, ty.Union[str, int, ty.Sequence[str]]]
    ) -> datatools.FeaturefulSpan:
        """Digitize a span."""
        left_context = ty.cast(ty.Sequence[str], span["left_context"])
        tokenized_left_context = [
            piece for token in left_context for piece in self.tokenizer.tokenize(token)
        ]
        content = ty.cast(ty.Sequence[str], span["content"])
        tokenized_content = [
            piece for token in content for piece in self.tokenizer.tokenize(token)
        ]
        right_context = ty.cast(ty.Sequence[str], span["right_context"])
        tokenized_right_context = [
            piece for token in right_context for piece in self.tokenizer.tokenize(token)
        ]

        pieces = [*tokenized_left_context, *tokenized_content, *tokenized_right_context]
        encoded = self.tokenizer.encode_plus(
            pieces,
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        # Mask processing to get the correct boundary indices, this could probably be made more
        # efficient if/when we get a saner api for BERT digitization
        shift = self.shift_from_mask(encoded["special_tokens_mask"])
        # FIXME: :art:
        span_boundaries = (
            shift + len(tokenized_left_context),
            shift + len(tokenized_left_context) + len(tokenized_content),
        )
        feats = self.digitize_feats(span)
        return datatools.FeaturefulSpan(encoded["input_ids"], span_boundaries, feats)

    def save(self, path: ty.Union[str, pathlib.Path]):
        """Save as a model archive."""
        logger.debug(f"Saving the model to {path}")
        with tempfile.TemporaryDirectory() as _tempdir:
            tempdir = pathlib.Path(_tempdir)
            torch.save(self.model.state_dict(), tempdir / "weights.dat")

            features_dump = copy.deepcopy(self._features)
            for f in features_dump:
                digitization = f.get("digitization", None)
                if digitization is None:
                    continue
                if digitization == "lexicon":
                    lexicon_filename = f"{f['name']}.lexicon"
                    lexicon.dump(f["lexicon"], tempdir / lexicon_filename)
                f["lexicon"] = lexicon_filename

            encoder_config = {
                "type": "bert",
                "span_encoding_dim": self.model.out_dim,
                "hidden_dim": self.model.span_encoder.hidden_dim,
                "features": features_dump,
                "pretrained": self.pretrained,
            }
            with open(tempdir / "config.json", "w") as config_stream:
                json.dump(encoder_config, config_stream)

            archive = shutil.make_archive(str(tempdir), "gztar", root_dir=tempdir)
            shutil.move(archive, path)

    @classmethod
    def load(cls, model_path: ty.Union[str, pathlib.Path]) -> "BERTEncoder":
        """Load a model archive."""
        with tempfile.TemporaryDirectory() as _tempdir:
            tempdir = pathlib.Path(_tempdir)
            try:
                shutil.unpack_archive(str(model_path), tempdir, format="gztar")
            except shutil.ReadError as e:
                raise InvalidModelException(f"Couldn't unpack {model_path}") from e
            try:
                with open(tempdir / "config.json") as config_stream:
                    encoder_config = config.encoder_schema.validate(
                        json.load(config_stream)
                    )
            except FileNotFoundError as e:
                raise InvalidModelException(f"Files missing in {model_path}") from e
            except schema.SchemaError as e:
                raise InvalidModelException(f"Invalid config in {model_path}") from e

            try:
                features = encoder_config["features"]
                for f in features:
                    digitization = f.get("digitization", None)
                    if digitization == "lexicon":
                        f["lexicon"] = lexicon.load(tempdir / f["lexicon"])
                feature_digitizers = utils.get_digitizers(features)
                model = cls.default_model(**encoder_config)

                weights_path = tempdir / "weights.dat"
                weights = torch.load(weights_path, map_location="cpu")
            except FileNotFoundError as e:
                raise InvalidModelException(f"Files missing in {model_path}") from e
            model.load_state_dict(weights)

            return cls(
                model=model,
                features=features,
                features_digitizers=feature_digitizers,
                pretrained=encoder_config["pretrained"],
            )

    @staticmethod
    def default_model(
        pretrained: str,
        span_encoding_dim: int,
        hidden_dim: int,
        features: ty.Iterable[ty.Dict[str, ty.Any]] = None,
        embeddings: ty.Optional[torch.Tensor] = None,
        external_boundaries: bool = False,
        fine_tune: bool = False,
        combine_layers: ty.Optional[ty.Sequence[int]] = None,
        project: bool = False,
        **kwargs,
    ) -> libdecofre.FeaturefulSpanEncoder:
        words_encoder = libdecofre.BERTWordEncoder(
            model_name_or_path=pretrained,
            model_class=transformers.AutoModel,
            combine_layers=combine_layers,
            project=project,
            fine_tune=fine_tune,
        )
        featureless_span_encoder = libdecofre.SpanEncoder(
            words_encoding_dim=words_encoder.out_dim,
            hidden_dim=hidden_dim,
            ffnn_dim=hidden_dim,
            out_dim=span_encoding_dim,
            external_boundaries=external_boundaries,
        )

        if features is None:
            raise NotImplementedError("Featureless encoder not yet supported")
        else:
            # instead of just having the same initialization
            features_lst = tuple(
                (f["vocabulary_size"], f["embeddings_dim"], None) for f in features
            )

            enc = libdecofre.FeaturefulSpanEncoder(
                tokens_encoder=words_encoder,
                span_encoder=featureless_span_encoder,
                features_encoder=libdecofre.CategoricalFeaturesBag(features_lst),
                out_dim=span_encoding_dim,
            )
            return enc

    @classmethod
    def initialize(
        cls, model_config: ty.Dict[str, ty.Any], initialisation: ty.Dict[str, ty.Any]
    ) -> "BERTEncoder":
        """Create a new encoder from a model config and weights/lexicons initialisation."""
        features = config.load_features(
            model_config.pop("features"), initialisation["lexicon-source"]
        )
        features_digitizers = utils.get_digitizers(features)
        model = cls.default_model(features=features, **model_config)
        return cls(
            model=model,
            features=features,
            features_digitizers=features_digitizers,
            pretrained=model_config["pretrained"],
        )

    @staticmethod
    def shift_from_mask(mask: ty.Iterable[int]) -> int:
        values, lengths = zip(*((k, sum(1 for _ in g)) for k, g in it.groupby(mask)))
        # TODO: THIS IS THE OPPOSITE OF THE DOC BUT CONSISTENT WITH THE cODE WHAT THE HELL
        if values != (1, 0, 1):
            raise ValueError("Invalid mask")
        return lengths[0]
