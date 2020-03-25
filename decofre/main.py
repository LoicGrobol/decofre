#! /usr/bin/env python3
"""Train a neural end-to-end coreference detector.

Usage:
  decofre [options]

Options:
  --config <f>  Path to a JSON train config file (mandatory)
  --debug  Verbose training display
  --device <d>  The device to use for computations (defaults to `cuda:0` or `cpu`)
  --encoder <f>  Path to a serialized span encoding model to preload
  --model-config <f>  Path to a JSON model configuration
  --num-workers <n>  Number of data loading workers to use [default: 0]
  --out-dir <p>  Output directory, end with `/` to make it a timestamped subdir (default:
                 decofre-<timestamp>)
  -h, --help  Show this screen.
"""

import datetime
import json
import os
import pathlib
import pkg_resources
import pprint
import signal
import subprocess  # nosec
import sys

import typing as ty

from collections import defaultdict

import docopt

import schema

import torch
import torch.multiprocessing
import torch.optim
import torch.utils.data

from loguru import logger

import decofre.models.defaults

from decofre import utils

from decofre import __version__


def load_train_config(config_path=ty.Union[str, pathlib.Path]):
    config_path = pathlib.Path(config_path).resolve()

    def to_path(value):
        return (config_path.parent / value).resolve()

    config_schema = schema.Schema(
        {
            "mention-detection": {
                "train-file": schema.Use(to_path),
                schema.Optional("lr", default=3e-4): schema.Use(float),
                schema.Optional("lr-schedule", default=None): str,
                schema.Optional("weight-decay", default=0.0): schema.Use(float),
                schema.Optional("mention-boost", default=None): schema.Use(float),
                schema.Optional("epochs", default=10): schema.Use(int),
                schema.Optional("train-batch-size", default=32): schema.Use(int),
                schema.Optional("eval-batch-size", default=128): schema.Use(int),
                schema.Optional("patience", default=None): schema.Use(int),
                schema.Optional("dev-file"): utils.PathUri(must_exist=True),
                schema.Optional("test-file"): utils.PathUri(must_exist=True),
            },
            "antecedent-scoring": {
                "train-file": schema.Use(to_path),
                schema.Optional("lr", default=1e-4): schema.Use(float),
                schema.Optional("weight-decay", default=0.0): schema.Use(float),
                schema.Optional("epochs", default=10): schema.Use(int),
                schema.Optional("train-batch-size", default=8): schema.Use(int),
                schema.Optional("patience", default=None): schema.Use(int),
                schema.Optional("dev-file"): utils.PathUri(must_exist=True),
                schema.Optional("test-file"): utils.PathUri(must_exist=True),
                schema.Optional("lexicon-source"): utils.PathUri(must_exist=True),
            },
            schema.Optional("word-embeddings"): utils.PathUri(must_exist=True),
            schema.Optional("lexicon-source"): utils.PathUri(must_exist=True),
            schema.Optional(
                "training-scheme", default={"type": "sequential"}
            ): schema.Or(
                {"type": "sequential"},
                {
                    "type": "quasisimultaneous",
                    schema.Optional("steps", default=None): {
                        "detection": schema.Use(int),
                        "antecedent": schema.Use(int),
                    },
                },
            ),
        }
    )
    config = config_schema.validate(utils.load_jsonnet(config_path))
    if "lexicon-source" not in config:
        config["lexicon-source"] = config["mention-detection"]["train-file"]
    return config


# Deal with piping output in a standard-compliant way
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def main_entry_point(argv=None):
    arguments = docopt.docopt(__doc__, version=__version__, argv=argv)
    logger.remove(0)  # Don't log directly to stderr

    if arguments["--debug"]:
        log_level = "DEBUG"
        log_fmt = (
            "[decofre] "
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>@<cyan>{line}</cyan>: "
            "<level>{message}</level>"
        )
    else:
        log_level = "INFO"
        log_fmt = (
            "[decofre] "
            "<green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            "<level>{message}</level>"
        )
    logger.add(
        utils.TqdmCompatibleStream(sys.stderr),
        level=log_level,
        format=log_fmt,
        colorize=True,
    )

    script_dir = pathlib.Path(__file__).resolve().parent
    if arguments["--out-dir"] is None:
        out_dir = pathlib.Path(
            f"decofre-{datetime.datetime.now().isoformat(timespec='seconds')}"
        )
    else:
        out_dir = pathlib.Path(arguments["--out-dir"]).resolve()
        if arguments["--out-dir"].endswith(os.sep):
            out_dir = out_dir / datetime.datetime.now().isoformat(timespec="seconds")
            logger.info(f"Outputs in {out_dir} (from {arguments['--out-dir']!r})")
        else:
            logger.info(f"Outputs in {out_dir}")

    try:
        out_dir.mkdir(parents=True)
    except FileExistsError:
        # TODO: This doesn't avoid name clash, maybe rmtree if existing?
        # FIXME: This blocks non-interactive usage
        # FIXME: There probably is a package to do it already
        while True:
            ans = input(f"{out_dir} already exists. Use it anyway? [yN] ")  # nosec
            if not ans or "no".startswith(ans.lower()):
                return
            elif "yes".startswith(ans.lower()):
                out_dir.mkdir(exist_ok=True, parents=True)
                break
            print(f"{ans!r} is not a valid answer")
    logger.add(out_dir / "train.log", level="DEBUG")
    logger.debug(f"This is decofre trainer version {__version__}")
    git_commit = utils.git_commit_hash(script_dir)
    if git_commit is not None:
        logger.debug(f"State as of git commit {git_commit}")
        if utils.git_dirty(script_dir):
            logger.warning(
                "Running with uncommitted changes in tracked files, see `git status` for details"
            )
    logger.debug(f"Pytorch version {torch.__version__}")
    pip_freeze_file = out_dir / "frozen_libs.txt"
    logger.debug(f"recording frozen requirements in {pip_freeze_file}")
    try:
        with open(pip_freeze_file, "w") as freeze_stream:
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
                check=True,
                stdout=freeze_stream,
            )
    except subprocess.CalledProcessError as e:
        logger.warning(f"Couldn't freeze pip state: {e}")
    logger.debug(f"Running {pprint.pformat(sys.argv)}")

    if arguments["--device"] is not None:
        device = torch.device(arguments["--device"])
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    logger.debug(
        f"Training on device {device}"
        f" (CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')})"
    )
    if arguments["--config"] is not None:
        train_config = load_train_config(arguments["--config"])
    else:
        raise NotImplementedError
    with open(out_dir / "train_config.json", "w") as out_stream:
        json.dump(utils.dumpable_config(train_config), out_stream)
    logger.debug(f"Training config:\n{pprint.pformat(train_config)}")
    train_config = defaultdict(lambda: None, train_config)

    if arguments["--model-config"] is None:
        arguments["--model-config"] = pkg_resources.resource_filename(
            __name__, "models/default.jsonnet"
        )

    if arguments["--encoder"] is not None:
        encoder_init = arguments["--encoder"]
    else:
        encoder_init = {
            "lexicon-source": train_config["lexicon-source"],
            "word_embeddings_path": train_config["word-embeddings"],
        }

    det, cor = decofre.models.defaults.initialize_models_from_config(
        config_path=arguments["--model-config"],
        initialisation={
            "encoder": encoder_init,
            "detector": {"lexicon-source": train_config["lexicon-source"]},
            "scorer": {
                "lexicon-source": train_config["antecedent-scoring"]["train-file"]
            },
        },
        device=arguments["--device"],
    )

    if train_config["training-scheme"]["type"] == "sequential":
        from decofre.training import sequential as training_scheme
    elif train_config["training-scheme"]["type"] == "quasisimultaneous":
        raise NotImplementedError(
            "Quasisimultaneous training scheme currently unavailable, bear with us!"
        )
    else:
        raise ValueError(f'Unknown training scheme {train_config["training-scheme"]}')
    training_scheme.train(
        det,
        cor,
        train_config,
        out_dir,
        device,
        num_workers=int(arguments["--num-workers"]),
        debug=arguments["--debug"],
    )


if __name__ == "__main__":
    # See https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("forkserver", force=True)
    main_entry_point()
