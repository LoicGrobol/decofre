import contextlib
import re
import sys

import itertools as it
import typing as ty

from collections import deque
from typing import List, TextIO, Tuple

import click
import numpy as np
import ujson as json
import spacy

from typing_extensions import Literal, TypedDict


T = ty.TypeVar("T")


spacy.tokens.Span.set_extension("uid", default=None)
spacy.tokens.Span.set_extension("speaker", default=None)
spacy.tokens.Token.set_extension("speaker", default=None)
spacy.tokens.Token.set_extension("utterance", default=None)

IGNORE_TOKENS = re.compile(r"^[\[\]\(\)\.,<>]+$")
REMOVE_RE = re.compile(r"[\[\]\(\)\.,<>]")


def generate_spans_with_context(
    lst: ty.Iterable[T],
    min_width: int,
    max_width: int,
    left_context: int = 0,
    right_context: int = 0,
) -> ty.Iterable[ty.Tuple[ty.List[T], ty.List[T], ty.List[T]]]:
    """
    Return an iterator over all the spans of `#lst` with width in `[min_width, max_width`] and a
    context window.

    ## Output
    A iterable of tuples `(left_context, span, right_context)`, with the context truncated at the
    desired length
    """
    lst_iter = iter(lst)
    # First gobble as many elements as needed
    left_buffer = deque()  # type: ty.Deque[ty.Any]
    buffer = deque(it.islice(lst_iter, max_width + right_context))
    # Exit early if the iterable is not long enough
    if len(buffer) < min_width:
        return
    for nex in lst_iter:
        for i in range(min_width, max_width):
            yield (
                list(left_buffer),
                list(it.islice(buffer, 0, i)),
                list(it.islice(buffer, i, i + right_context)),
            )
        buffer.append(nex)
        left_buffer.append(buffer.popleft())
        if len(left_buffer) > left_context:
            left_buffer.popleft()

    # Empty the buffer when we have reached the end of `lst_iter`
    while buffer:
        for i in range(min_width, min(len(buffer) + 1, max_width)):
            yield (
                list(left_buffer),
                list(it.islice(buffer, 0, i)),
                list(it.islice(buffer, i, i + right_context + 1)),
            )
        left_buffer.append(buffer.popleft())
        if len(left_buffer) > left_context:
            left_buffer.popleft()


ChunkInclusionStatus = Literal["exact", "included", "outside", "incompatible"]


def span_inclusion(needle, sorted_spans) -> ChunkInclusionStatus:
    """Return a `ChunkInclusionStatus` for a span within a sorted iterable of spans."""
    needle_start, needle_end = needle[0], needle[-1]
    sorted_spans_itr = iter(sorted_spans)
    # In the following, `{ }` is `s` and `[ ]` is `needle`
    for s in sorted_spans_itr:
        s_start, s_end = s[0], s[-1]
        # We have gone past needle: `[ ] { }`
        if needle_end < s_start:
            return "outside"
        # We have not yet reached needle: `{ } [ ]`
        elif s_end < needle_start:
            continue
        # `{ [ } ]` or `{ [ ] }`
        elif s_start < needle_start:
            # `{ [ ] }`
            if needle_end <= s_end:
                return "included"
            else:
                return "incompatible"
        # At this stage we know that needle_start <= s_end <= needle_end
        # `[={ ] }` or `[={ } ]` or `[={ }=]`
        elif s_start == needle_start:
            if s_end == needle_end:
                return "exact"
            elif needle_end < s_end:
                return "included"
            else:
                return "incompatible"
        # `[ { ] }` or `[ { } ]`
        else:
            return "incompatible"
    # We have gone through all spans without finding an intersecting one
    return "outside"


class SpanFeats(TypedDict):
    content: ty.List[str]
    left_context: ty.List[str]
    right_context: ty.List[str]
    length: int
    pos: ty.List[str]
    entity_type: str
    chunk_inclusion: ChunkInclusionStatus
    sentence: int
    start: int
    end: int
    span_id: str
    speaker: ty.Optional[str]
    utterance: int


def spans_from_doc(
    doc: spacy.tokens.doc.Doc,
    min_width: int = 1,
    max_width: int = 26,
    context: ty.Tuple[int, int] = (10, 10),
    length_buckets: ty.Sequence[int] = (1, 2, 3, 4, 5, 7, 15, 32, 63),
) -> ty.Iterable[SpanFeats]:
    for sent_n, sent in enumerate(doc.sents):
        speakers_set = set(t._.speaker for t in sent if t._.speaker is not None)
        if len(speakers_set) > 1:
            raise ValueError("Inconsistent speakers")
        elif not speakers_set:
            speaker = None
        else:
            speaker = speakers_set.pop()

        utterances_set = set(t._.utterance for t in sent)
        if len(speakers_set) > 1:
            raise ValueError("Inconsistent utterances")
        else:
            utterance = utterances_set.pop()
        # FIXME: this generating lists instead of spacy spans is absurd
        context_spans = generate_spans_with_context(
            [t for t in sent if not t.is_space], min_width, max_width, *context
        )
        ent_dict = {(e[0], e[-1]): e.label_ for e in sent.ents}
        noun_chunks = sorted(sent.noun_chunks)
        for left_context, span, right_context in context_spans:
            span = doc[span[0].i : span[-1].i + 1]
            if IGNORE_TOKENS.match(span[0].text) or IGNORE_TOKENS.match(span[-1].text):
                continue
            if left_context:
                left_context = doc[left_context[0].i : left_context[-1].i + 1]
            else:
                left_context = []
            if right_context:
                right_context = doc[right_context[0].i : right_context[-1].i + 1]
            else:
                right_context = []
            span_content = [
                w for t in span if not t.is_space and (w := REMOVE_RE.sub("", t.text))
            ]
            assert span_content
            left_content = [
                w
                for t in left_context
                if not t.is_space and (w := REMOVE_RE.sub("", t.text))
            ]
            right_content = [
                w
                for t in right_context
                if not t.is_space and (w := REMOVE_RE.sub("", t.text))
            ]
            if len(left_content) < context[0]:
                left_content.insert(0, "<start>")
            if len(right_content) < context[1]:
                right_content.append("<end>")

            length = int(
                np.digitize(len(span_content), bins=length_buckets, right=True)
            )
            pos = [w.pos_ for w in (*left_context, *span, *right_context)]
            entity_type = ent_dict.get((span[0], span[-1]), "None")
            chunk_inclusion = span_inclusion(span, noun_chunks)
            yield {
                "content": span_content,
                "left_context": left_content,
                "right_context": right_content,
                "length": length,
                "pos": pos,
                "entity_type": entity_type,
                "chunk_inclusion": chunk_inclusion,
                "sentence": sent_n,
                "start": span[0].i,
                "end": span[-1].i,
                "span_id": f"{span[0].i}-{span[-1].i}",
                "speaker": speaker,
                "utterance": utterance,
            }


class AvpUtterance(TypedDict):
    speaker: str
    startTime: str
    endTime: str
    text: str
    type: str
    comment: str
    frames: str
    audio: str


def make_doc(
    model: spacy.language.Language, utterances: List[AvpUtterance]
) -> spacy.tokens.Doc:
    texts = [f'{u["text"]}\n' for u in utterances]
    doc = model("".join(texts))
    doc.spans["utterances"] = []
    char_offset = 0
    for i, (t, u) in enumerate(zip(texts, utterances)):
        u_span = doc.char_span(char_offset, char_offset + len(t))
        u_span._.speaker = u["speaker"]
        u_span._.uid = u.get("uid", f"u{i}")
        doc.spans["utterances"].append(u_span)
        for token in u_span:
            token._.speaker = u["speaker"]
            token._.utterance = i
        char_offset += len(t)
    return doc


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


@spacy.Language.component("sent_on_newlines")
def sent_on_newlines(doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
    for i, token in enumerate(doc[:-1]):
        if token.text == "\n":
            doc[i + 1].is_sent_start = True
    return doc


def get_doc_and_spans(
    document: TextIO, lang: str
) -> Tuple[spacy.tokens.doc.Doc, List[SpanFeats]]:
    nlp = spacy.load(lang)
    nlp.add_pipe("sent_on_newlines", name="sentence_segmenter", before="parser")
    utterances = [
        u for u in json.load(document) if u["text"] and not u["text"].isspace()
    ]
    doc = make_doc(nlp, utterances)
    spans = list(spans_from_doc(doc))
    return doc, spans


@click.command(help="Generate mention-detection inputs")
@click.argument(
    "input_file", type=click.File("r"),
)
@click.argument(
    "output_file", type=click.File("w", atomic=True), default="-",
)
@click.option(
    "--lang",
    default="fr_core_news_lg",
    help="A spaCy model handle for the document.",
    show_default=True,
)
def main_entry_point(input_file: TextIO, output_file: TextIO, lang: str):
    _, spans = get_doc_and_spans(input_file, lang)
    json.dump(spans, output_file, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main_entry_point())
