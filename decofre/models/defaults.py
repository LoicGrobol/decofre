# TODO: support all features for detector and scorer
# TODO: unified model superclass for dump/loading

import copy
import pathlib
import pprint
import tempfile
import shutil

import typing as ty

import torch

import ujson as json

from loguru import logger

from decofre import libdecofre, lexicon
import decofre.utils
from decofre.models import Model, InvalidModelException, config, utils
from decofre.models.encoders import Encoder


class Detector(Model):
    """
    A wrapper around a `libdecofre.MentionDetector` that provides digitization
    and serialization.
    """

    def __init__(
        self,
        model: libdecofre.MentionDetector,
        model_config: ty.Dict[str, ty.Any],
        encoder: Encoder,
        span_types_lexicon: lexicon.Lexicon,
    ):
        self.model = model
        self.model_config = model_config
        self.encoder = encoder
        self.span_types_lexicon = span_types_lexicon
        self.digitize_span = self.encoder.digitize

    def save(self, path: ty.Union[str, pathlib.Path]):
        logger.debug(f"Saving the model to {path}")
        with tempfile.TemporaryDirectory() as _tempdir:
            tempdir = pathlib.Path(_tempdir)
            self.encoder.save(tempdir / "encoder.model")
            lexicon.dump(self.span_types_lexicon, tempdir / "span_types.lexicon")
            torch.save(self.model.state_dict(), tempdir / "weights.dat")
            with open(tempdir / "model_config.json", "w") as config_stream:
                json.dump(self.model_config, config_stream)
            archive = shutil.make_archive(str(tempdir), "gztar", root_dir=tempdir)
            shutil.move(archive, path)

    @classmethod
    def load(cls, model_path: ty.Union[str, pathlib.Path]) -> "Detector":
        with tempfile.TemporaryDirectory() as _tempdir:
            tempdir = pathlib.Path(_tempdir)
            try:
                shutil.unpack_archive(str(model_path), tempdir, format="gztar")
            except shutil.ReadError as e:
                raise InvalidModelException(f"Couldn't unpack {model_path}") from e
            try:
                span_types_lexicon = lexicon.load(tempdir / "span_types.lexicon")
                encoder = Encoder.load(tempdir / "encoder.model")
            except FileNotFoundError as e:
                raise InvalidModelException(f"Files missing in {model_path}") from e

            try:
                with open(tempdir / "model_config.json") as config_stream:
                    model_config = json.load(config_stream)
            except FileNotFoundError:
                model_config = dict()
            model = cls.default_model(
                encoder=encoder.model,
                n_span_types=len(span_types_lexicon),
                **model_config,
            )
            try:
                det_weights = torch.load(tempdir / "weights.dat", map_location="cpu")
            except FileNotFoundError as e:
                raise InvalidModelException(f"Files missing in {model_path}") from e
            model.load_state_dict(det_weights)

            return cls(
                model=model,
                model_config=model_config,
                encoder=encoder,
                span_types_lexicon=span_types_lexicon,
            )

    @staticmethod
    def default_model(
        encoder: libdecofre.FeaturefulSpanEncoder, n_span_types: int, **kwargs
    ):
        model = libdecofre.MentionDetector(
            span_encoder=encoder,
            span_encoding_dim=encoder.out_dim,
            n_types=n_span_types,
            **kwargs,
        )
        return model

    @classmethod
    def initialize(
        cls,
        model_config: ty.Dict[str, ty.Any],
        initialisation: ty.Dict[str, ty.Any],
        encoder: Encoder,
    ) -> "Detector":
        """Create a new scorer from a model config and weights/lexicons initialisation."""
        lexicon_source_path = pathlib.Path(initialisation["lexicon-source"])
        if lexicon_source_path.is_file():
            lexicon_sources = [lexicon_source_path]
        else:
            lexicon_sources = list(lexicon_source_path.glob("*.json"))
        span_types_lexicon = utils.generate_lexicons(("type",), lexicon_sources)["type"]
        # TODO: log loaded frequencies
        model = cls.default_model(
            encoder=encoder.model,
            n_span_types=len(span_types_lexicon.i2t),
            **model_config,
        )
        return cls(
            model=model,
            model_config=model_config,
            encoder=encoder,
            span_types_lexicon=span_types_lexicon,
        )


class Scorer(Model):
    """
    A wrapper around a `libdecofre.CorefScorer` that provides digitization
    and serialization.
    """

    def __init__(
        self,
        model: libdecofre.CorefScorer,
        encoder: Encoder,
        features: ty.Sequence[ty.Dict[str, ty.Any]],
        features_digitizers: ty.Dict[str, utils.Digitizer],
    ):
        self.model = model
        self.encoder = encoder
        self.digitize_span = self.encoder.digitize
        self._features = features
        self._feature_digitizers = features_digitizers

    def get_pair_feats(
        self, row: ty.Mapping[str, ty.Union[str, ty.Iterable[str]]]
    ) -> ty.Tuple[int, ...]:
        return tuple(
            self._feature_digitizers[f["name"]](row[f["name"]]) for f in self._features
        )

    def save(self, path: ty.Union[str, pathlib.Path]):
        logger.debug(f"Saving the model to {path}")
        with tempfile.TemporaryDirectory() as _tempdir:
            tempdir = pathlib.Path(_tempdir)
            self.encoder.save(tempdir / "encoder.model")
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
            model_config = {"features": features_dump}

            with open(tempdir / "config.json", "w") as config_stream:
                json.dump(model_config, config_stream)
            archive = shutil.make_archive(str(tempdir), "gztar", root_dir=tempdir)
            shutil.move(archive, path)

    @classmethod
    def load(cls, model_path: ty.Union[str, pathlib.Path]) -> "Scorer":
        with tempfile.TemporaryDirectory() as _tempdir:
            tempdir = pathlib.Path(_tempdir)
            try:
                shutil.unpack_archive(str(model_path), tempdir, format="gztar")
            except shutil.ReadError as e:
                raise InvalidModelException(f"Couldn't unpack {model_path}") from e
            encoder = Encoder.load(tempdir / "encoder.model")

            try:
                with open(tempdir / "config.json") as config_stream:
                    model_config = config.scorer_schema.validate(
                        json.load(config_stream)
                    )

                features = model_config["features"]
                for f in features:
                    digitization = f.get("digitization", None)
                    if digitization == "lexicon":
                        f["lexicon"] = lexicon.load(tempdir / f["lexicon"])
                    elif digitization == "word":
                        f["lexicon"] = getattr(encoder, "_words_lexicon", None)
                        if f["lexicon"] is None:
                            raise ValueError(f"{type(encoder)} has no word lexicon")
                feature_digitizers = utils.get_digitizers(features)

                model = cls.default_model(encoder=encoder.model, features=features)
                weights = torch.load(tempdir / "weights.dat", map_location="cpu")
            except FileNotFoundError as e:
                raise InvalidModelException(f"Files missing in {model_path}") from e
            model.load_state_dict(weights)

            return cls(
                model=model,
                encoder=encoder,
                features=features,
                features_digitizers=feature_digitizers,
            )

    @staticmethod
    def default_model(
        encoder: libdecofre.FeaturefulSpanEncoder,
        features: ty.Optional[ty.Sequence[ty.Mapping[str, ty.Any]]] = None,
    ):
        if features is None:
            raise NotImplementedError("Featureless scorer not yet supported")
        else:
            features_lst = []
            for f in features:
                if f.get("digitization", None) == "word":
                    word_embedding_layer = getattr(
                        encoder.tokens_encoder, "word_embeddings"
                    )
                    if word_embedding_layer is not None:
                        weights = word_embedding_layer.weights[:-1, ...]
                    else:
                        weights = None
                else:
                    weights = None
                features_lst.append(
                    (f["vocabulary_size"], f["embeddings_dim"], weights)
                )

        scor = libdecofre.CorefScorer(
            span_encoder=encoder,
            span_encoding_dim=encoder.out_dim,
            features=features_lst,
        )
        return scor

    @classmethod
    def initialize(
        cls,
        model_config: ty.Dict[str, ty.Any],
        initialisation: ty.Dict[str, ty.Any],
        encoder: Encoder,
    ) -> "Scorer":
        """Create a new scorer from a model config and weights/lexicons initialisation."""
        words_lexicon = getattr(encoder, "_words_lexicon", None)
        features = config.load_features(
            model_config["features"], initialisation["lexicon-source"], words_lexicon
        )
        features_digitizers = utils.get_digitizers(features)
        model = cls.default_model(encoder=encoder.model, features=features)
        return cls(
            model=model,
            encoder=encoder,
            features=features,
            features_digitizers=features_digitizers,
        )


def initialize_models_from_config(
    config_path: ty.Union[str, pathlib.Path],
    initialisation: ty.Dict[str, ty.Any],
    device: ty.Union[str, torch.device] = "cpu",
) -> ty.Tuple[Detector, Scorer]:
    model_config = config.model_config_schema.validate(
        decofre.utils.load_jsonnet(config_path)
    )
    logger.debug(f"Model config:\n{pprint.pformat(dict(model_config))}")
    logger.debug(f"initialisation:\n{pprint.pformat(dict(initialisation))}")

    encoder_init = initialisation["encoder"]
    # TODO: missing test, check that this actually give the right weights
    if isinstance(encoder_init, str) or isinstance(encoder_init, pathlib.Path):
        logger.info(f"Loading encoder model from {encoder_init}")
        try:
            encoder = Encoder.load(encoder_init)
        except InvalidModelException:
            try:
                logger.debug(
                    f"{encoder_init} is not an encoder model, retrying assuming it is a detector"
                )
                encoder = Detector.load(encoder_init).encoder
            except InvalidModelException:
                try:
                    logger.debug(
                        f"{encoder_init} is not an detector model, retrying assuming it is a scorer"
                    )
                    encoder = Scorer.load(encoder_init).encoder
                except InvalidModelException:
                    raise InvalidModelException(
                        f"Couldn't extract an encoder from {encoder_init}"
                    )
        logger.debug(f"Encoder model successfully loaded")
    else:
        encoder = Encoder.initialize(
            model_config.get("encoder", dict()), initialisation["encoder"]
        )
    mention_detector = Detector.initialize(
        model_config.get("detector", dict()), initialisation["detector"], encoder
    )
    antecedent_scorer = Scorer.initialize(
        model_config.get("scorer", dict()), initialisation["scorer"], encoder
    )

    mention_detector.model.to(device)
    antecedent_scorer.model.to(device)

    return mention_detector, antecedent_scorer
