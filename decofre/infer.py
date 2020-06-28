"""End-to-end coreference resolution

Usage:
  infer [options] <detect-model> <coref-model> <input> [<output>]

Arguments:
  <detect-model>  The mention detection model
  <coref-model>  The coreference resolution model
  <input>  The input file (raw text in French), `-` for standard input
  <output>  The output file, `-` for standard input (default)

Options:
  -h --help  Show this screen.
  --intermediary-dir <path>  A path to a directory to use for intermediary files,
                             defaults to a self-destructing temp dir
  --lang <name>  spaCy model handle to use [default: fr_core_news_lg]
  --version   Show version.

Notes:
  Be warned that if you are using an existing directory as `--intermediary-dir`, existing files in
  it might be mercilessly overwritten, proceed with caution.
"""

import contextlib
import pathlib
import shutil
import sys
import tempfile
import typing as ty

import numpy as np
import orjson
import spacy

from docopt import docopt
from typing_extensions import TypedDict

from decofre.formats import raw_text
from decofre import detmentions, score, clusterize

from decofre import __version__


spacy.tokens.Doc.set_extension("clusters", default=None)
spacy.tokens.Span.set_extension("cluster", default=None)


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


@contextlib.contextmanager
def dir_manager(
    path: ty.Optional[ty.Union[pathlib.Path, str]] = None, cleanup=None
) -> ty.Generator[pathlib.Path, None, None]:
    """A context manager to deal with a directory, default to a self-destruct temp one."""
    if path is None:
        d_path = pathlib.Path(tempfile.mkdtemp())
        if cleanup is None:
            cleanup = True
    else:
        d_path = pathlib.Path(path).resolve()
        d_path.mkdir(parents=True, exist_ok=True)
        if cleanup is None:
            cleanup = False

    try:
        yield d_path
    finally:
        if cleanup:
            shutil.rmtree(d_path)


class AntecedentFeaturesDict(TypedDict):
    w_distance: int
    u_distance: int
    m_distance: int
    spk_agreement: bool
    overlap: bool
    token_incl: int
    token_com: int


def antecedents_from_mentions(
    mentions: ty.Iterable[ty.Dict[str, ty.Any]],
    max_candidates: int = 100,
    distance_buckets: ty.Sequence[int] = (1, 2, 3, 4, 5, 7, 15, 32, 63),
) -> ty.Dict[str, ty.Dict[str, AntecedentFeaturesDict]]:
    """Extract the antecedents dataset from an ANCOR TEI document."""

    sorted_mentions = sorted(mentions, key=lambda m: (m["start"], m["end"]))
    if len(sorted_mentions) < 2:
        return dict()

    # The first mention in a document has no antecedent candidates
    # FIXME: we keep slicing, which generates copies, which makes me uneasy

    res = dict()
    for i, mention in enumerate(sorted_mentions[1:], start=1):
        mention_content_set = set(mention["content"])
        antecedent_candidates = sorted_mentions[max(0, i - max_candidates) : i]
        antecedents: ty.Dict[str, AntecedentFeaturesDict] = dict()
        for j, candidate in enumerate(antecedent_candidates):
            candidate_content_set = set(candidate["content"])
            w_distance = int(
                np.digitize(
                    mention["start"] - candidate["end"],
                    bins=distance_buckets,
                    right=True,
                )
            )
            u_distance = int(
                np.digitize(
                    mention["sentence"] - candidate["sentence"],
                    bins=distance_buckets,
                    right=True,
                )
            )
            m_distance: int = int(
                np.digitize(
                    len(antecedent_candidates) - j - 1,
                    bins=distance_buckets,
                    right=True,
                )
            )
            spk_agreement = mention.get("speaker") == candidate.get("speaker")

            intersect = len(mention_content_set.intersection(candidate_content_set))
            token_incl_ratio = int(
                10
                * intersect
                / min(len(mention_content_set), len(candidate_content_set))
            )
            token_com_ratio = int(
                10 * intersect / len(mention_content_set.union(candidate_content_set))
            )

            overlap = mention["start"] < candidate["end"]

            antecedents[candidate["span_id"]] = {
                "w_distance": w_distance,
                "u_distance": u_distance,
                "m_distance": m_distance,
                "spk_agreement": spk_agreement,
                "overlap": overlap,
                "token_incl": token_incl_ratio,
                "token_com": token_com_ratio,
            }
        res[mention["span_id"]] = antecedents
    return res


def text_out(doc: spacy.tokens.Doc) -> str:
    mentions_spans = sorted(
        (m for i, c in doc._.clusters.items() for m in c),
        key=lambda m: (m.start_char, -m.end_char),
    )
    text = doc.text
    res = []
    open_spans: ty.List[spacy.tokens.Span] = []
    current_char = 0
    for m in mentions_spans:
        while open_spans and open_spans[-1].end_char <= m.start_char:
            span_to_close = open_spans.pop()
            res.append(text[current_char : span_to_close.end_char])
            res.append(f"][{span_to_close._.cluster}]")
            current_char = span_to_close.end_char
        if current_char < m.start_char:
            res.append(text[current_char : m.start_char])
            current_char = m.start_char
        res.append("[")
        open_spans.append(m)
    while open_spans:
        span_to_close = open_spans.pop()
        res.append(text[current_char : span_to_close.end_char])
        res.append(f"][{span_to_close._.cluster}]")
        current_char = span_to_close.end_char
    res.append(text[current_char:])
    return "".join(res)


def main_entry_point(argv=None):
    arguments = docopt(__doc__, version=f"decofre {__version__}")
    if arguments["<output>"] is None:
        arguments["<output>"] = "-"

    nlp = spacy.load(arguments["--lang"])

    with dir_manager(arguments["--intermediary-dir"]) as intermediary_dir:
        with smart_open(arguments["<input>"]) as in_stream:
            doc = nlp(in_stream.read())

        initial_doc_path = intermediary_dir / "initial_doc.spacy.bin"
        with open(initial_doc_path, "wb") as out_stream:
            out_stream.write(doc.to_bytes())

        spans = list(raw_text.spans_from_doc(doc))

        spans_path = intermediary_dir / "spans.json"
        with open(spans_path, "wb") as out_stream:
            out_stream.write(orjson.dumps(spans))

        mentions_path = intermediary_dir / "mentions.json"
        detmentions.main_entry_point(
            [
                "--mentions",
                "--no-overlap",
                arguments["<detect-model>"],
                str(spans_path),
                str(mentions_path),
            ]
        )

        with open(mentions_path, "rb") as in_stream:
            mentions_lst = orjson.loads(in_stream.read())

        antecedents = antecedents_from_mentions(mentions_lst)
        mention_dict = {m["span_id"]: m for m in mentions_lst}

        antecedents_path = intermediary_dir / "antecedents.json"
        with open(antecedents_path, "wb") as out_stream:
            out_stream.write(
                orjson.dumps({"mentions": mention_dict, "antecedents": antecedents})
            )

        coref_scores_path = intermediary_dir / "coref_scores.json"
        score.main_entry_point(
            [arguments["<coref-model>"], str(antecedents_path), str(coref_scores_path)]
        )

        clusters_path = intermediary_dir / "clusters.json"
        clusterize.main_entry_point([str(coref_scores_path), str(clusters_path)])

        with open(clusters_path, "rb") as in_stream:
            clusters = orjson.loads(in_stream.read())["clusters"]

        doc._.clusters = dict()
        for i, c in clusters.items():
            doc._.clusters[i] = []
            for m_id in c:
                mention = mention_dict[m_id]
                mention_span = doc[mention["start"] : mention["end"] + 1]
                mention_span._.cluster = i
                doc._.clusters[i].append(mention_span)

        # augmented_doc_path = intermediary_dir / "coref_doc.spacy.bin"
        # with open(augmented_doc_path, "wb") as out_stream:
        #     out_stream.write(doc.to_bytes())

    with smart_open(arguments["<output>"], "w") as out_stream:
        out_stream.write(text_out(doc))
        out_stream.write("\n")
    # displacy_visu_data = {
    #     "text": doc.text,
    #     "ents": [
    #         {
    #             "start": m.start_char,
    #             "end": m.end_char,
    #             "label": m._.cluster
    #         }
    #         for m in mention_spans_lst
    #     ],
    #     "title": "DeCOFre",
    # }

    # spacy.displacy.serve(displacy_visu_data, style="ent", manual=True)


if __name__ == "__main__":
    main_entry_point()
