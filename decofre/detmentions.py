#! /usr/bin/env python3
r"""Detect mentions

Usage:
  detmentions [options] <model> <spans> [<output>]

Options:
  --device <d>  The device to use for computations (defaults to `cuda:0` or `cpu`)
  --eval  Also evaluate the quality of the detection (requires --gold-key)
  --format <f>  The output format (see below) [default: prettyjson]
  --gold-key <g>  Use key <g> as the gold mention tag
  --mentions  Output only mentions (gold and predicted)
  --mistakes  Output only mislassifications (requires --gold-key)
  --no-overlap  Filter out overlapping mentions with a greedy heuristic,
                doesn't make much sense without --mentions
  -h, --help  Show this screen.

Formats:
  - `csv` designed for human- rather than manchine- reading
  - `json` compact JSON for forwarding to other tools
  - `prettyjson` nicer JSON, possibly for easier debugging
"""

import contextlib
import csv
import json
import pathlib
import sys

import typing as ty

import docopt
import torch
import tqdm

from boltons import iterutils as itu
from loguru import logger

from decofre import datatools
from decofre import utils

from decofre.models.defaults import Detector
from decofre import __version__


def load_spans(
    spans_file: ty.Union[str, pathlib.Path],
    span_digitizer: ty.Callable[[ty.Dict[str, str]], ty.Any],
) -> ty.Tuple[
    ty.Tuple[ty.Dict[str, ty.Union[str, ty.List[str]]], ...], ty.Tuple[ty.Any, ...]
]:
    """Load and digitize a span file.

    Output: (raw_spans, digitized_spans)
    """
    with open(spans_file) as in_stream:
        raw = json.load(in_stream)
    digitized = tuple(
        span_digitizer(row)
        for row in tqdm.tqdm(
            raw,
            unit="spans",
            desc="Digitizing",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            leave=False,
            disable=None,
        )
    )

    return raw, digitized


def tag(
    model: torch.nn.Module, data: ty.Iterable, batch_size: int = 128
) -> ty.List[ty.Tuple[int, ty.List[float]]]:
    """Tag a dataset

    Output: (tag, scores)
    """
    device = next(model.parameters()).device
    model.eval()
    sys_out = []  # type: ty.List[ty.Tuple[int, ty.List[float]]]
    if isinstance(data, ty.Sized):
        data_len = (len(data) - 1) // batch_size + 1  # type: ty.Optional[int]
    else:
        data_len = None
    data = map(datatools.FeaturefulSpan.collate, itu.chunked_iter(data, batch_size))
    pbar = tqdm.tqdm(
        data,
        total=data_len,
        unit="batch",
        desc="Tagging",
        mininterval=2,
        unit_scale=True,
        dynamic_ncols=True,
        disable=None,
        leave=False,
    )
    with torch.no_grad():
        for d in pbar:
            r = model(datatools.move(d, device=device))
            sys_tags = r.argmax(dim=-1).tolist()
            scores = r.exp().tolist()
            sys_out.extend(zip(sys_tags, scores))
    return sys_out


def resolve_overlap(
    mentions_lst: ty.Iterable[ty.Dict[str, ty.Any]]
) -> ty.Iterable[ty.Dict[str, ty.Any]]:
    """Greedily resolve improper mentions overlaps."""
    # Sort the spans
    sorted_by_score = sorted(mentions_lst, key=lambda m: m["scores"][None])
    res: ty.List[ty.Dict[str, ty.Any]] = []
    for mention in sorted_by_score:
        mention_start, mention_end = mention["start"], mention["end"]
        keep = True
        for other in res:
            other_start, other_end = other["start"], other["end"]
            if (other_start < mention_start <= other_end < mention_end) or (
                mention_start < other_start <= mention_end < other_end
            ):
                keep = False
                logger.debug(f"Discarding {mention!r} in favour of {other!r}")
                break
        if keep:
            res.append(mention)
    return res


@contextlib.contextmanager
def smart_open(
    filename: str, mode: str = "r", *args, **kwargs
) -> ty.Generator[ty.IO, None, None]:
    """Open files and i/o streams transparently."""
    if filename == "-":
        if "r" in mode:
            stream = sys.stdin
        else:
            stream = sys.stdout
        if "b" in mode:
            fh = stream.buffer  # type: ty.IO
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
            arguments["--device"] = "cuda"
        else:
            arguments["--device"] = "cpu"

    gold_key = arguments["--gold-key"]
    if gold_key is None:
        for k in ("--mistakes", "--eval"):
            if arguments[k]:
                raise ValueError(f"{k} without --gold-key makes no sense")

    device = torch.device(arguments["--device"])

    detector = Detector.load(arguments["<model>"])
    detector.model.eval()
    detector.model.to(device)

    raw, digitized = load_spans(arguments["<spans>"], detector.digitize_span)

    sys_out = tag(detector.model, digitized)

    results = []
    confusion = torch.zeros(
        (len(detector.span_types_lexicon.i2t), len(detector.span_types_lexicon.i2t))
    )
    for row, (sys_t, scores) in zip(raw, sys_out):
        sys_tag = detector.span_types_lexicon.i2t[sys_t]
        if gold_key is not None:
            confusion[sys_t, detector.span_types_lexicon.t2i(row[gold_key])] += 1
        if arguments["--mistakes"] and (
            gold_key is not None and sys_tag == row[gold_key]
        ):
            continue
        if arguments["--mentions"] and (
            sys_tag is None
            and gold_key is None
            or gold_key is not None
            and row[gold_key] is None
        ):
            continue

        results.append(
            {
                **row,
                "scores": {
                    tag: score
                    for tag, score in zip(detector.span_types_lexicon.i2t, scores)
                },
                "sys_tag": sys_tag,
            }
        )

    if arguments["--no-overlap"]:
        try:
            results = resolve_overlap(results)
        except KeyError as e:
            raise ValueError(
                "Overlaps can only be resolved if all spans have 'start' and 'end' attributes."
            ) from e

    if arguments["--format"] == "csv":
        with smart_open(arguments["<output>"], "w", newline="") as out_stream:
            for row in results:
                for k, v in row.items():
                    if v is None:
                        row[k] = "None"
                row["mention"] = " ".join(
                    [
                        *row.pop("left_context"),
                        "|",
                        *row.pop("content"),
                        "|",
                        *row.pop("right_context"),
                    ]
                )
                row["scores"] = "|".join(
                    f"{t}={round(s, 5)}" for t, s in row["scores"].items()
                )
            start_fields = ["mention"]
            end_fields = ["scores", "sys_tag"]
            fieldnames = start_fields[:]
            for k in results[0].keys():
                if k not in start_fields and k not in end_fields:
                    fieldnames.append(k)
            fieldnames.extend(end_fields)
            writer = csv.DictWriter(
                out_stream, fieldnames=fieldnames, delimiter="\t", quotechar='"'
            )
            writer.writeheader()
            writer.writerows(results)
    elif arguments["--format"] in ("json", "prettyjson"):
        with smart_open(arguments["<output>"], "w") as out_stream:
            json.dump(
                results,
                out_stream,
                indent=4 if arguments["--format"] == "prettyjson" else None,
                ensure_ascii=False,
            )
    if arguments["--eval"]:
        class_prf = utils.PRF(confusion)
        macro = class_prf.mean(dim=0)
        print(
            {
                "class": {
                    c: {"P": P.item(), "R": R.item(), "F": F.item()}
                    for c, (P, R, F) in zip(detector.span_types_lexicon.i2t, class_prf)
                },
                "macro": {n: v.item() for n, v in zip("PRF", macro)},
                "accuracy": confusion.diag().sum().div(confusion.sum()).item(),
            }
        )


if __name__ == "__main__":
    main_entry_point()
