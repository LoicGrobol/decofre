#! /usr/bin/env python3
r"""Coreference antecedent scoring

Usage:
  score [options] <model> <dataset> [<output>]

Options:
  --device <d>      The device to use for computations (defaults to `cuda:0` or `cpu`)
  -h, --help  Show this screen.
"""
import contextlib
import json
import sys

import typing as ty

import docopt
import torch

from loguru import logger

from decofre import datatools
from decofre import utils

from decofre.models.defaults import Scorer
from decofre.runners import run_model
from decofre import __version__


@contextlib.contextmanager
def smart_open(filename: str, mode: str = "r", *args, **kwargs):
    """Open files and i/o streams transparently."""
    fh: ty.IO
    if filename == "-":
        if "r" in mode:
            stream = sys.stdin
        else:
            stream = sys.stdout
        if "b" in mode:
            fh = stream.buffer
        else:
            fh = stream
        close = False
    else:
        fh = open(filename, mode, *args, **kwargs)
        close = True

    try:
        yield fh
    finally:
        if close:
            try:
                fh.close()
            except AttributeError:
                pass


def main_entry_point(argv=None):
    arguments = docopt.docopt(__doc__, version=__version__, argv=argv)
    logger.add(
        utils.TqdmCompatibleStream(),
        format=(
            "[decofre] "
            "<green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    if arguments["<output>"] is None:
        arguments["<output>"] = "-"

    if arguments["--device"] is None:
        if torch.cuda.is_available():
            arguments["--device"] = "cuda:0"
        else:
            arguments["--device"] = "cpu"
    device = torch.device(arguments["--device"])

    cor = Scorer.load(arguments["<model>"])
    cor.model.eval()
    cor.model.to(device)

    with open(arguments["<dataset>"]) as in_stream:
        data = json.load(in_stream)

    # FIXME: this is barely readable, needs heavy refactoring
    mentions = data["mentions"]
    antecedents = data["antecedents"]
    sorted_mentions_ids = sorted(antecedents.keys())

    # FIXME: this might OOM for large files but it really speeds things up
    digitized_mentions = {m_id: cor.digitize_span(m) for m_id, m in mentions.items()}

    dataset = (
        (
            digitized_mentions[m_id],
            [
                (digitized_mentions[c_id], cor.get_pair_feats(antecedents[m_id][c_id]))
                for c_id in sorted(antecedents[m_id].keys())
            ],
        )
        for m_id in sorted_mentions_ids
    )

    with torch.no_grad():
        sys_out = run_model(
            cor.model,
            dataset,
            prepare_batch=datatools.AntecedentsDataset.prepare_source_batch,
            batch_size=64,
            data_len=(len(sorted_mentions_ids) - 1) // 64 + 1,
            join="chain",
        )

    scores = dict(zip(sorted_mentions_ids, sys_out))

    out_dict = {"mentions": data["mentions"], "antecedents": dict()}
    for mention_id, antecedent_scores in scores.items():
        scores = antecedent_scores.tolist()
        out_dict["antecedents"][mention_id] = {
            "candidates": {
                c_id: {**antecedents[mention_id][c_id], "score": s}
                for c_id, s in zip(sorted(antecedents[mention_id].keys()), scores[1:])
            },
            "anaphoricity": scores[0],
        }

    with smart_open(arguments["<output>"], "w") as out_stream:
        json.dump(out_dict, out_stream, ensure_ascii=False)


if __name__ == "__main__":
    main_entry_point()
