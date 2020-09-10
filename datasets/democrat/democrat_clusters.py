#! /usr/bin/env python3
r"""Generate scorch cluster files from DEMOCRAT XML-TEI-URS files

Usage:
  democrat-clusters <in-file> [<out-file>]

Arguments:
  <in-file>   input file (ANCOR XML-TEI-URS), `-` for standard input
  <out-file>  output file (scorch JSON), `-` for standard output [default: -]

Options:
  -h, --help  Show this screen.

Example:
  `democrat-clusters input.tei output.json`
"""

__version__ = "0.0.0"

import contextlib
import json
import sys

import typing as ty

from docopt import docopt
from lxml import etree

TEI = "{http://www.tei-c.org/ns/1.0}"
XML = "{http://www.w3.org/XML/1998/namespace}"

NSMAP = {"tei": TEI[1:-1], "xml": XML[1:-1]}


def get_clusters(tree: etree._ElementTree) -> ty.Dict[str, ty.Set[str]]:
    chains_grp = tree.xpath(
        './tei:standOff/tei:annotations[@type="coreference"]/tei:annotationGrp[@type="Schema"]',
        namespaces=NSMAP,
    )[0]

    mentions = tree.xpath(
        (
            './tei:standOff/tei:annotations[@type="coreference"]'
            '/tei:annotationGrp[@subtype="MENTION"]/tei:span'
        ),
        namespaces=NSMAP,
    )

    res = dict()
    for c in chains_grp.iter(f"{TEI}link"):
        target = c.attrib["target"]
        res[c.attrib[f"{XML}id"]] = set((t[1:] for t in target.split()))

    non_sing = set().union(*res.values())
    for m in mentions:
        i = m.attrib[f"{XML}id"]
        if i not in non_sing:
            res[i] = {i}
    dupes = [
        (a_id, b_id, intersect)
        for a_id, a in res.items()
        for b_id, b in res.items()
        if b is not a
        for intersect in (a.intersection(b),)
        if intersect
    ]
    for a_id, b_id, intersect in dupes:
        print(f"Schemas {a_id} and {b_id} are not disjoints: {intersect}")
    return res


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


def main_entry_point(argv=None):
    arguments = docopt(__doc__, version=__version__, argv=argv)
    # Since there are no support for default positional arguments in
    # docopt yet. Might be useful for complex default values, too
    if arguments["<out-file>"] is None:
        arguments["<out-file>"] = "-"

    with smart_open(arguments["<in-file>"], "rb") as in_stream:
        tree = etree.parse(in_stream)

    try:
        clusters = get_clusters(tree)
    except:
        raise Exception(f'Something wrong with {arguments["<in-file>"]}')

    with smart_open(arguments["<out-file>"], "w") as out_stream:
        json.dump(
            {"type": "clusters", "clusters": {k: list(c) for k, c in clusters.items()}},
            out_stream,
        )


if __name__ == "__main__":
    sys.exit(main_entry_point())
