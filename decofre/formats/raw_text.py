"""Generate mention-detection inputs

Usage:
  raw_text [options] <input> [<output>]

Arguments:
  <input>   The input file (raw text in French), `-` for standard input
  <output>  The output file, `-` for standard input (default)

Options:
  --lang <name>  	A spaCy model hand to use [default: fr_core_news_lg]
  -h, --help  	Show this screen.
"""

import contextlib
import sys

import itertools as it
import typing as ty

from collections import deque

import docopt
import spacy
import numpy as np
import orjson

from typing_extensions import Literal, TypedDict

from decofre import __version__

T = ty.TypeVar("T")


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


def spans_from_doc(
    doc: spacy.tokens.doc.Doc,
    min_width: int = 1,
    max_width: int = 26,
    context: ty.Tuple[int, int] = (10, 10),
    length_buckets: ty.Sequence[int] = (1, 2, 3, 4, 5, 7, 15, 32, 63),
) -> ty.Iterable[SpanFeats]:
    for sent_n, sent in enumerate(doc.sents):
        # FIXME: this generating lists instead of spacy spans is absurd
        context_spans = generate_spans_with_context(
            [t for t in sent if not t.is_space], min_width, max_width, *context
        )
        ent_dict = {(e[0], e[-1]): e.label_ for e in sent.ents}
        noun_chunks = sorted(sent.noun_chunks)
        for left_context, span, right_context in context_spans:
            span = doc[span[0].i : span[-1].i + 1]
            if left_context:
                left_context = doc[left_context[0].i : left_context[-1].i + 1]
            else:
                left_context = []
            if right_context:
                right_context = doc[right_context[0].i : right_context[-1].i + 1]
            else:
                right_context = []
            span_content = [t.text for t in span if not t.is_space]
            left_content = [t.text for t in left_context if not t.is_space]
            right_content = [t.text for t in right_context if not t.is_space]
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
            }


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
    if arguments["<output>"] is None:
        arguments["<output>"] = "-"

    nlp = spacy.load(arguments["--lang"])

    with smart_open(arguments["<input>"]) as in_stream:
        doc = nlp(in_stream.read())

    with smart_open(arguments["<output>"], "wb") as out_stream:
        spans = list(spans_from_doc(doc))
        out_stream.write(orjson.dumps(spans, out_stream))


if __name__ == "__main__":
    sys.exit(main_entry_point())
