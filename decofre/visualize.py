#! /usr/bin/env python3
r"""visualize clusters

Usage:
   clusterize [options] <in-file> [<out-file>]

Arguments:
  <in-file>   input file (json)), `-` for standard input
  <out-file>  output file (GraphML), [default: -]

Options:
  -h, --help  Show this screen.
  --tag <t>   Tag marking coreference [default: True]
  --gold <f>  Path to gold output if it is needed
  --algo <a>  Clustering algorithm (`best` or `all`) [default: best]

Example:
  `visualie in.tsv out.json`
"""

import contextlib
import json
import signal
import sys

import typing as ty

import networkx as nx

from docopt import docopt

from decofre import __version__

# Deal with piping output in a standard-compliant way
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


# Thanks http://stackoverflow.com/a/17603000/760767
@contextlib.contextmanager
def smart_open(filename: str = None, mode: str = "r", *args, **kwargs):
    """Open files and i/o streams transparently."""
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


def best_first(
    antecedents: ty.Iterable[ty.Tuple[str, ty.Tuple[str, float, bool]]]
) -> ty.Iterable[ty.Tuple[str, str, ty.Dict[str, ty.Any]]]:
    for m, a in antecedents:
        try:
            best, weight, correct = max(a, key=lambda x: x[1])
        except ValueError:  # No antecedent
            continue
        yield (m, best, {"weight": weight, "color": "green" if correct else "red"})


def transitive(
    antecedents: ty.Iterable[ty.Tuple[str, ty.Tuple[str, float, bool]]]
) -> ty.Iterable[ty.Tuple[str, str, ty.Dict[str, ty.Any]]]:
    for m, a in antecedents:
        yield from (
            (m, m2, {"weight": weight, "color": "black" if correct else "red"})
            for m2, weight, correct in a
        )


def gold_antecedents(pairs: ty.Dict, coref_tags: ty.Collection[str]):
    for m, v in pairs.items():
        try:
            closest = min(
                (a for a in v["candidates"] if a["gold"] in coref_tags),
                key=lambda x: int(x["m_distance"]),
            )
        except ValueError:
            continue
        yield (
            m,
            closest["id"],
            {
                "weight": round(float(closest["score"]), 4),
                "color": "blue" if closest["sys"] in coref_tags else "black",
            },
        )


def main_entry_point(argv=None):
    arguments = docopt(__doc__, version=__version__, argv=argv)
    # Since there are no support for default positional arguments in
    # docopt yet. Might be useful for complex default values, too
    if arguments["<out-file>"] is None:
        arguments["<out-file>"] = "-"

    coref_tags = set([arguments["--tag"]])

    with smart_open(arguments["<in-file>"]) as in_stream:
        inpt_dict = json.load(in_stream)

    sys_antecedents = (
        (
            mention,
            (
                (
                    antecedent["id"],
                    float(antecedent["score"]),
                    antecedent["gold"] in coref_tags,
                )
                for antecedent in v["candidates"]
                if antecedent["sys"] in coref_tags
            ),
        )
        for mention, v in inpt_dict.items()
    )

    if arguments["--algo"] == "best":
        sys_links = best_first(sys_antecedents)
    elif arguments["--algo"] == "all":
        sys_links = transitive(sys_antecedents)
    else:
        raise ValueError(f"Invalid algo: arguments['--algo']")

    S = nx.DiGraph()
    S.add_nodes_from(
        (
            m,
            {
                "content": v["content"],
                "color": (
                    "black"
                    if any(a["gold"] in coref_tags for a in v["candidates"])
                    else "yellow"
                ),
            },
        )
        for m, v in inpt_dict.items()
    )

    # S.add_edges_from(gold_antecedents(inpt_dict, coref_tags))
    S.add_edges_from(sys_links)
    with smart_open(arguments["<out-file>"], "wb") as out_stream:
        nx.write_graphml_lxml(S, out_stream)

    if arguments["--gold"] is not None:
        gold_links = [
            (m, a["id"])
            for m, v in inpt_dict.items()
            for a in v["candidates"]
            if a["gold"] in coref_tags
        ]
        S = nx.DiGraph()
        S.add_nodes_from((m, {"content": v["content"]}) for m, v in inpt_dict.items())
        S.add_edges_from(gold_links)
        with smart_open(arguments["--gold"], "wb") as out_stream:
            nx.write_graphml_lxml(S, out_stream)


if __name__ == "__main__":
    sys.exit(main_entry_point())
