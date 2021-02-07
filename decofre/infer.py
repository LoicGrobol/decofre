import contextlib
import pathlib
import shutil
import sys
import tempfile
import typing as ty

import click
import click_pathlib
import jsonlines
import numpy as np
import spacy
import ujson as json

from typing import Any, Dict, List, Literal, Optional, TextIO
from typing_extensions import TypedDict

from decofre.formats import formats
from decofre import detmentions, score, clusterize


spacy.tokens.Doc.set_extension("clusters", default=None)
spacy.tokens.Span.set_extension("cluster", default=None)
spacy.tokens.Span.set_extension("singleton", default=True)


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
        elif cleanup:
            if d_path.glob("*"):
                raise ValueError(f"{d_path} is not empty.")

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
    max_candidates: int = 128,
    distance_buckets: ty.Sequence[int] = (1, 2, 3, 4, 5, 7, 15, 32, 63),
) -> ty.Dict[str, ty.Dict[str, AntecedentFeaturesDict]]:
    """Extract an antecedent dataset from a list of detected mentions."""

    sorted_mentions = sorted(mentions, key=lambda m: (m["start"], m["end"]))
    if len(sorted_mentions) < 2:
        return dict()

    # The first mention in a document has no antecedent candidates

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
                )
            )
            m_distance: int = int(
                np.digitize(
                    len(antecedent_candidates) - j,
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


def text_out(doc: spacy.tokens.Doc, latex: bool = False) -> str:
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
            if span_to_close._.singleton:
                if latex:
                    res.append("}")
                else:
                    res.append("]")
            else:
                if latex:
                    res.append("}")
                else:
                    res.append(f"][{span_to_close._.cluster}]")
            current_char = span_to_close.end_char
        if current_char < m.start_char:
            res.append(text[current_char : m.start_char])
            current_char = m.start_char
        if latex:
            if m._.singleton:
                res.append(r"\mention{")
            else:
                res.append(f"\\mention[{m._.cluster}]{{")
        else:
            res.append("[")
        open_spans.append(m)
    while open_spans:
        span_to_close = open_spans.pop()
        res.append(text[current_char : span_to_close.end_char])
        if span_to_close._.singleton:
            if latex:
                res.append("}")
            else:
                res.append("]")
        else:
            if latex:
                res.append("}")
            else:
                res.append(f"][{span_to_close._.cluster}]")
        current_char = span_to_close.end_char
    res.append(text[current_char:])
    return "".join(res)


def mention_to_json(mention: spacy.tokens.Span) -> Dict[str, Any]:
    return {
        "text": mention.text,
        "start": mention.start_char,
        "token_start": mention.start,
        "token_end": mention.end,
        "end": mention.end_char,
        "type": "pattern",
        "label": "mention",
    }


def token_to_json(token: spacy.tokens.Token) -> Dict[str, Any]:
    return {
        "text": token.text,
        "start": token.idx,
        "end": token.idx + len(token),
        "id": token.i,
        "ws": bool(token.whitespace_),
        "disabled": False,
    }


def prodigy_out(doc: spacy.tokens.Doc) -> Dict[str, Any]:
    res = {
        "text": doc.text,
        "tokens": [token_to_json(t) for t in doc],
        "spans": [],
        "relations": [],
    }
    processed: List[spacy.tokens.Span] = []
    for c in doc._.clusters.values():
        antecedent: Optional[spacy.tokens.Span] = None
        for m in sorted(c, key=lambda m: (m.end, m.start)):
            # This because prodigy doesn't allow nested spans
            if any(
                o.start <= m.start <= o.end or o.start <= m.end <= o.end
                for o in processed
            ):
                continue
            res["spans"].append(mention_to_json(m))
            if antecedent is not None:
                res["relations"].append(
                    {
                        "head": m.start,
                        "child": antecedent.start,
                        "head_span": mention_to_json(m),
                        "child_span": mention_to_json(antecedent),
                        "label": "COREF",
                    }
                )
            antecedent = m
            processed.append(m)

    return res


def sacr_out(doc: spacy.tokens.Doc) -> str:
    res = []
    open_spans: ty.List[spacy.tokens.Span]
    sents = doc.spans.get("utterances", doc.sents)
    for sentence in sents:
        sentence_res = []
        # FIXME: this relies on having imported avp, which sets these extensions in the global space
        # we need a better mechanism
        if sentence._.speaker is not None:
            sentence_res.append(f"#speaker: {sentence._.speaker}\n\n")
        if sentence._.uid is not None:
            sentence_res.append(f"#uid: {sentence._.uid}\n\n")
        mentions_spans = sorted(
            (
                m
                for i, c in doc._.clusters.items()
                for m in c
                if sentence.start_char <= m.start_char < m.end_char <= sentence.end_char
            ),
            key=lambda m: (m.start_char, -m.end_char),
        )
        text = sentence.text
        current_char = 0
        open_spans: ty.List[spacy.tokens.Span] = []
        for m in mentions_spans:
            # TODO: stop fiddling with char indices ffs
            while open_spans and open_spans[-1].end_char <= m.start_char:
                span_to_close = open_spans.pop()
                sentence_res.append(
                    text[current_char : span_to_close.end_char - sentence.start_char]
                )
                sentence_res.append("}")
                current_char = span_to_close.end_char - sentence.start_char
            if current_char < m.start_char:
                sentence_res.append(
                    text[current_char : m.start_char - sentence.start_char]
                )
                current_char = m.start_char - sentence.start_char
            sentence_res.append(f"{{{m._.cluster} ")
            open_spans.append(m)
        while open_spans:
            span_to_close = open_spans.pop()
            sentence_res.append(
                text[current_char : span_to_close.end_char - sentence.start_char]
            )
            sentence_res.append("}")
            current_char = span_to_close.end_char - sentence.start_char
        sentence_res.append(text[current_char:])
        res.append("".join(sentence_res).strip())
    return "\n\n".join((s for s in res if s and not s.isspace()))


@click.command(help="End-to-end coreference resolution")
@click.argument(
    "detect-model",
    type=click_pathlib.Path(exists=True, dir_okay=False),
)
@click.argument(
    "coref-model",
    type=click_pathlib.Path(exists=True, dir_okay=False),
)
@click.argument(
    "input_file",
    type=click.File("r"),
)
@click.argument(
    "output_file",
    type=click.File("w", atomic=True),
    default="-",
)
@click.option(
    "--from",
    "input_format",
    type=click.Choice(formats.keys()),
    default="raw_text",
    help="The input format",
    show_default=True,
)
@click.option(
    "--intermediary-dir",
    "intermediary_dir_path",
    type=click_pathlib.Path(resolve_path=True, file_okay=False),
    help="A path to a directory to use for intermediary files, defaults to a self-destructing temp dir",
)
@click.option(
    "--lang",
    default="fr_core_news_lg",
    help="A spaCy model handle for the document.",
    show_default=True,
)
@click.option(
    "--to",
    "output_format",
    type=click.Choice(["latex", "prodigy", "sacr", "text"]),
    default="text",
    help="Output formats (experimental)",
)
def main_entry_point(
    coref_model: pathlib.Path,
    detect_model: pathlib.Path,
    input_format: str,
    input_file: TextIO,
    intermediary_dir_path: Optional[pathlib.Path],
    lang: str,
    output_file: TextIO,
    output_format: Literal["latex", "prodigy", "sacr", "text"],
):
    with dir_manager(intermediary_dir_path) as intermediary_dir:
        doc, spans = formats[input_format].get_doc_and_spans(input_file, lang)

        initial_doc_path = intermediary_dir / "initial_doc.spacy.json"
        with open(initial_doc_path, "w") as out_stream:
            json.dump(doc.to_json(), out_stream, ensure_ascii=False)

        spans_path = intermediary_dir / "spans.json"
        with open(spans_path, "w") as out_stream:
            json.dump(spans, out_stream, ensure_ascii=False)

        mentions_path = intermediary_dir / "mentions.json"
        detmentions.main_entry_point(
            [
                "--mentions",
                "--no-overlap",
                str(detect_model),
                str(spans_path),
                str(mentions_path),
            ]
        )

        with open(mentions_path, "r") as in_stream:
            mentions_lst = json.load(in_stream)

        antecedents = antecedents_from_mentions(mentions_lst)
        mention_dict = {m["span_id"]: m for m in mentions_lst}

        antecedents_path = intermediary_dir / "antecedents.json"
        with open(antecedents_path, "w") as out_stream:
            json.dump(
                {"mentions": mention_dict, "antecedents": antecedents},
                out_stream,
                ensure_ascii=False,
            )

        coref_scores_path = intermediary_dir / "coref_scores.json"
        score.main_entry_point(
            [str(coref_model), str(antecedents_path), str(coref_scores_path)]
        )

        clusters_path = intermediary_dir / "clusters.json"
        clusterize.main_entry_point([str(coref_scores_path), str(clusters_path)])

        with open(clusters_path, "r") as in_stream:
            clusters = json.load(in_stream)["clusters"]

        doc._.clusters = dict()
        for i, c in clusters.items():
            doc._.clusters[i] = []
            for m_id in c:
                mention = mention_dict[m_id]
                mention_span = doc[mention["start"] : mention["end"] + 1]
                mention_span._.cluster = i
                if len(c) > 1:
                    mention_span._.singleton = False

                doc._.clusters[i].append(mention_span)

        augmented_doc_path = intermediary_dir / "coref_doc.spacy.json"
        with open(augmented_doc_path, "w") as out_stream:
            json.dump(doc.to_json(), out_stream, ensure_ascii=False)

    if output_format == "latex":
        output_file.write(text_out(doc, latex=True))
        output_file.write("\n")
    elif output_format == "prodigy":
        output_dict = prodigy_out(doc)
        writer = jsonlines.Writer(output_file)
        writer.write(output_dict)
        writer.close()
    elif output_format == "sacr":
        output_file.write(sacr_out(doc))
    else:
        output_file.write(text_out(doc))
        output_file.write("\n")


if __name__ == "__main__":
    main_entry_point()
