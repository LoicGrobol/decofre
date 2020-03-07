import copy
import pathlib

import typing as ty

import schema

from decofre import lexicon
from decofre.models import utils

raw_features_schema = schema.Schema(
    {"name": str, "vocabulary_size": schema.Use(int), "embeddings_dim": schema.Use(int)}
)

categorical_features_schema = schema.Schema(
    {
        "name": str,
        schema.Optional("digitization"): str,
        schema.Optional("lexicon"): str,
        schema.Optional("allow_unknown", default=False): bool,
        schema.Optional("vocabulary_size"): schema.Use(int),
        "embeddings_dim": schema.Use(int),
    }
)

context_free_encoder_schema = schema.Schema(
    {
        schema.Optional("type", default="context-free"): "context-free",
        "span_encoding_dim": schema.Use(int),
        "word_embeddings_dim": schema.Use(int),
        "chars_embeddings_dim": schema.Use(int),
        "hidden_dim": schema.Use(int),
        schema.Optional("features", default=None): [
            schema.Or(raw_features_schema, categorical_features_schema)
        ],
        schema.Optional("token_features", default=None): [
            schema.Or(raw_features_schema, categorical_features_schema)
        ],
        schema.Optional("external_boundaries", default=False): schema.Use(bool),
    }
)


elmo_encoder_schema = schema.Schema(
    {
        "type": "elmo",
        "span_encoding_dim": schema.Use(int),
        "elmo_options_file": schema.Use(str),
        "elmo_weight_file": schema.Use(str),
        "hidden_dim": schema.Use(int),
        "features": [schema.Or(raw_features_schema, categorical_features_schema)],
        schema.Optional("external_boundaries", default=False): bool,
    }
)


bert_encoder_schema = schema.Schema(
    {
        "type": "bert",
        "span_encoding_dim": schema.Use(int),
        "pretrained": schema.Use(str),
        schema.Optional("project", default=False): bool,
        schema.Optional("fine_tune", default=False): bool,
        schema.Optional("combine_layers", default=None): [schema.Use(int)],
        "hidden_dim": schema.Use(int),
        "features": [schema.Or(raw_features_schema, categorical_features_schema)],
        schema.Optional("external_boundaries", default=False): bool,
    }
)


encoder_schema = schema.Or(
    context_free_encoder_schema, elmo_encoder_schema, bert_encoder_schema
)

scorer_schema = schema.Schema(
    {
        schema.Optional("features"): [
            schema.Or(raw_features_schema, categorical_features_schema)
        ]
    }
)

detector_schema = schema.Schema(
    {
        schema.Optional("ffnn_dim"): schema.Use(int),
        schema.Optional("dropout"): schema.Use(float),
    }
)


model_config_schema = schema.Schema(
    {
        "encoder": encoder_schema,
        schema.Optional("scorer"): scorer_schema,
        schema.Optional("detector"): detector_schema,
    }
)


def load_features(
    feature_configs: ty.List[ty.Dict[str, ty.Any]],
    lexicon_source: ty.Optional[ty.Union[str, pathlib.Path]],
    words_lex: ty.Optional[lexicon.Lexicon] = None,
) -> ty.List[ty.Dict[str, ty.Any]]:
    lexicon_source = (
        pathlib.Path(lexicon_source) if lexicon_source is not None else None
    )
    feature_configs = copy.deepcopy(feature_configs)
    lexicons_to_generate = []  # ty.List[str]
    features_lexicons = dict()  # ty.Dict[str, lexicon.Lexicon]
    allow_unknown = []
    for f in feature_configs:
        digitization = f.get("digitization", None)
        if digitization == "lexicon":
            lexicon_path = f.get("lexicon", None)
            if lexicon_path is not None:
                features_lexicons[f["name"]] = lexicon.load(lexicon_path)
            else:
                lexicons_to_generate.append(f["name"])
                if f["allow_unknown"]:
                    allow_unknown.append(f["name"])

    if lexicons_to_generate:
        if lexicon_source is None:
            raise ValueError(
                "A lexicon source is needed if a feature has no predefined lexicon"
            )
        if lexicon_source.is_file():
            lexicon_source_lst = [lexicon_source]
        else:
            lexicon_source_lst = list(lexicon_source.glob("*.json"))
        features_lexicons.update(
            utils.generate_lexicons(
                lexicons_to_generate,
                sources=lexicon_source_lst,
                allow_unknown=allow_unknown,
            )
        )
    for f in feature_configs:
        name = f["name"]
        digitization = f.get("digitization", None)
        if digitization == "lexicon":
            if "lexicon" not in f:
                f["lexicon"] = features_lexicons[name]

            config_vocabulary_size = f.get("vocabulary_size", None)
            lexicon_size = len(f["lexicon"].i2t)

            if config_vocabulary_size is not None:
                if config_vocabulary_size != lexicon_size:
                    raise ValueError(
                        f"Bad vocabulary size for feature {f['name']}:"
                        f"{config_vocabulary_size} != {lexicon_size}"
                    )
            else:
                f["vocabulary_size"] = lexicon_size
        elif digitization == "word":
            if words_lex is None:
                raise ValueError(f"A words lexicon is needed for feature {f['name']}")
            f["lexicon"] = words_lex
            lexicon_size = len(words_lex.i2t)
            if f.setdefault("vocabulary_size", lexicon_size) != lexicon_size:
                raise ValueError(
                    f"Incompatible vocabulary size for feature {f['name']}"
                )

    return feature_configs


def load_token_features(
    feature_configs: ty.List[ty.Dict[str, ty.Any]],
    lexicon_source: ty.Optional[ty.Union[str, pathlib.Path]],
) -> ty.List[ty.Dict[str, ty.Any]]:
    feature_configs = copy.deepcopy(feature_configs)
    lexicon_source = (
        pathlib.Path(lexicon_source) if lexicon_source is not None else None
    )
    lexicons_to_generate = []  # ty.List[str]
    features_lexicons = dict()  # ty.Dict[str, lexicon.Lexicon]
    allow_unknown = []
    for f in feature_configs:
        lexicon_path = f.get("lexicon", None)
        if lexicon_path is not None:
            features_lexicons[f["name"]] = lexicon.load(lexicon_path)
        else:
            lexicons_to_generate.append(f["name"])
            if f["allow_unknown"]:
                allow_unknown.append(f["name"])

    if lexicons_to_generate:
        if lexicon_source is None:
            raise ValueError(
                "A lexicon source is needed if a feature has no predefined lexicon"
            )
        if lexicon_source.is_file():
            lexicon_source_lst = [lexicon_source]
        else:
            lexicon_source_lst = list(lexicon_source.glob("*.json"))

        features_lexicons.update(
            utils.generate_tokens_lexicons(
                lexicons_to_generate,
                sources=lexicon_source_lst,
                allow_unknown=allow_unknown,
            )
        )
    for f in feature_configs:
        name = f["name"]
        if "lexicon" not in f:
            f["lexicon"] = features_lexicons[name]

        config_vocabulary_size = f.get("vocabulary_size", None)
        lexicon_size = len(f["lexicon"].i2t)

        if config_vocabulary_size is not None:
            if config_vocabulary_size != lexicon_size:
                raise ValueError(
                    f"Bad vocabulary size for feature {f['name']}:"
                    f"{config_vocabulary_size} != {lexicon_size}"
                )
        else:
            f["vocabulary_size"] = lexicon_size

    return feature_configs
