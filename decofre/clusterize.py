#! /usr/bin/env python3
# FIXME: not working with the new score output
r"""Get from antecedent scores to clusters

Usage:
   clusterize [options] <in-file> [<out-file>]

Arguments:
  <in-file>   input file (json)), `-` for standard input
  <out-file>  output file (json), `-` for standard output [default: -]

Options:
  -h, --help  Show this screen.
  --graph  Output in graph format rather than in cluster
  --gold <f>  Path to gold output if it is needed

Example:
  `clusterize in.json out.json`
"""

import contextlib
import signal
import sys

import typing as ty

import orjson

from docopt import docopt

from decofre import __version__

# Deal with piping output in a standard-compliant way
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


# Thanks http://stackoverflow.com/a/17603000/760767
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


def greedy_clustering(
    links: ty.Iterable[ty.Tuple[ty.Hashable, ty.Hashable]]
) -> ty.List[ty.Set]:
    r"""
    Get connected component clusters from a set of edges.

    This is basically done with depth 1 [disjoint-sets][1], which seemed a good trade-off between
    performance and readability. If performance is an issue, consider using e.g. the [incremental
    connected component algorithm][2] found in Boost Graph.

    Even better: don't rely on this and preprocess your data using a more clever clustering
    procedure, e.g. with [NetworkX][3].

    [1]: https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    [2]: http://www.boost.org/doc/libs/1_66_0/libs/graph/doc/incremental_components.html
    [3]: https://networkx.github.io/documentation/networkx-1.9.1/reference/algorithms.mst.html
    """
    # The idea is to build an element→cluster mapping. This way, when we get a a→b link, we just
    # add a to the cluster of b.
    clusters: ty.Dict[ty.Hashable, ty.List] = dict()
    # The issue with this is that it is tedious to deduplicate afterward since Python `set` can't
    # be passed a custom key.
    # So instead we use the first element we encounter in each cluster as its head, and keep two
    # mappings: element→head and head→cluster.
    heads: ty.Dict[ty.Hashable, ty.Hashable] = dict()
    for source, target in links:
        source_head = heads.setdefault(source, source)
        source_cluster = clusters.setdefault(source_head, [source_head])

        target_head = heads.get(target)
        if target_head is None:
            heads[target] = source_head
            source_cluster.append(target)
        elif (
            target_head is not source_head
        ):  # Merge `target`'s cluster into `source`'s'
            for e in clusters[target_head]:
                heads[e] = source_head
            source_cluster.extend(clusters[target_head])
            del clusters[target_head]

    return [set(c) for c in clusters.values()]


def best_first(
    mentions: ty.Dict[str, ty.Dict[str, ty.Any]],
) -> ty.Iterable[ty.Tuple[str, str]]:
    consumed = set()
    for mention_id, mention in mentions.items():
        best_id, best_score = max(
            (
                (candidate_id, float(candidate["score"]))
                for candidate_id, candidate in mention["candidates"].items()
            ),
            key=lambda x: x[1],
            default=(None, None),
        )
        if best_score is None or best_score < float(mention["anaphoricity"]):
            continue
        assert best_id is not None
        consumed.add(best_id)
        yield (mention_id, best_id)


def main_entry_point(argv=None):
    arguments = docopt(__doc__, version=__version__, argv=argv)
    # Since there are no support for default positional arguments in
    # docopt yet. Might be useful for complex default values, too
    if arguments["<out-file>"] is None:
        arguments["<out-file>"] = "-"

    with smart_open(arguments["<in-file>"], "rb") as in_stream:
        inpt_dict = orjson.loads(in_stream.read())

    links = list(best_first(inpt_dict["antecedents"]))

    if arguments["--graph"]:
        with smart_open(arguments["<out-file>"], "wb") as out_stream:
            out_stream.write(
                orjson.dumps(
                    {
                        "type": "graph",
                        "mentions": list(inpt_dict["mentions"].keys()),
                        "links": list(links),
                    },
                )
            )
    else:
        clusters = greedy_clustering(links)
        singletons = [
            [m]
            for m in inpt_dict["mentions"].keys()
            if not any(m in c for c in clusters)
        ]
        clusters = {
            str(i): c
            for i, c in enumerate((*singletons, *(sorted(c) for c in clusters)))
        }

        with smart_open(arguments["<out-file>"], "wb") as out_stream:
            out_stream.write(orjson.dumps({"type": "clusters", "clusters": clusters},))


if __name__ == "__main__":
    sys.exit(main_entry_point())
