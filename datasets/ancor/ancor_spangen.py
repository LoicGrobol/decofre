#! /usr/bin/env python3
r"""Generate acceptable spans from the ANCOR corpus

Usage:
  spangen [options] <in-file> [-m <f>] [-a <f>]

Arguments:
  <in-file>  an ANCOR corpus file (in XML-TEI-URS format) `-` for standard input

Options:
  -h, --help  	Show this screen.
  -m, --mentions <f>  	Mention detection data file (see below)
  -a, --antecedents <f>  	Antecedent finding data file (see below)
  --buckets  	Mentions distance bucket boundaries (as a comma-separated list)
  --context <n>  	Context size [default: 10]
  --det-ratio <r>  	Maximum ratio of non-mention spans
  --filter <p>  	Keeping probability for non-coref pairs before the first antecedent [default: 1]
  --keep-single  	Keep spans of length 1 when downsampling
  --keep-named-entities  	Keep detected named entities when downsampling
  --keep-name-chunks  	Keep detected nominal chunks when downsampling
  --max-candidates <n>  	Maximum number of antecedent candidates [default: 100]
  --max-width <n>  	Maximum size for spans [default: 26]
  --only-first  	Keep only the first antecedent
  --only-id  	Only output mention ids instead of content
  --oversample  	Enforce det-ratio by oversampling instead of undersampling
  --seed <i>  A random seed for sampling

Example:
  `ancor_spangen data/corpora/ANCOR/tei.xml -m mentions.json -a antecedents.json`
"""

import dataclasses
import math
import contextlib
import random
import sys
import signal

import itertools as it
import typing as ty

from collections import deque

import spacy

import numpy as np
import orjson

from docopt import docopt
from loguru import logger
from lxml import etree  # nosec
from typing_extensions import TypedDict, Literal

from decofre import __version__

T = ty.TypeVar("T")

logger.remove(0)
logger.add(sys.stderr, level="INFO")

# Deal with piping output in a standard-compliant way
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

TEI = "{http://www.tei-c.org/ns/1.0}"
XML = "{http://www.w3.org/XML/1998/namespace}"

NSMAP = {"tei": TEI[1:-1], "xml": XML[1:-1]}

MENTION_TYPES = frozenset(("N", "PR"))
IGNORED_MENTION_TYPES = frozenset(("NULL",))

TOKEN_TAGS = frozenset((f"{TEI}w", f"{TEI}pc"))


class ElementNotFoundError(Exception):
    """An element is missing, optionally specify its id."""

    def __init__(self, message: str, eid: ty.Optional[str] = None):
        self.message = message
        self.eid = eid


def generate_spans_with_context(
    lst: ty.Iterable[T],
    min_width: int,
    max_width: int,
    left_context: int = 0,
    right_context: int = 0,
) -> ty.Iterable[ty.Tuple[ty.Tuple[T, ...], ty.Tuple[T, ...], ty.Tuple[T, ...]]]:
    """
    Return an iterator over all the spans of `#lst` with width in `[min_width, max_width`] and a
    context window.

    Output
    ------
    An iterable of tuples `(left_context, span, right_context)`, with the context truncated at the
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
                tuple(left_buffer),
                tuple(it.islice(buffer, 0, i)),
                tuple(it.islice(buffer, i, i + right_context)),
            )
        buffer.append(nex)
        left_buffer.append(buffer.popleft())
        if len(left_buffer) > left_context:
            left_buffer.popleft()

    # Empty the buffer when we have reached the end of `lst_iter`
    while buffer:
        for i in range(min_width, min(len(buffer) + 1, max_width)):
            yield (
                tuple(left_buffer),
                tuple(it.islice(buffer, 0, i)),
                tuple(it.islice(buffer, i, i + right_context + 1)),
            )
        left_buffer.append(buffer.popleft())
        if len(left_buffer) > left_context:
            left_buffer.popleft()


def xmlid(e: etree._Element) -> str:
    """Return the XML id of `e`, fails if does not have one."""
    return e.attrib["{http://www.w3.org/XML/1998/namespace}id"]


def target_to_id(s: str) -> str:
    """Convert an xml target string to the corresponding id string.

    FOR NOW THIS ONLY STRIPS a leading `#`
    """
    if not s.startswith("#"):
        raise ValueError(f"{s!r} is not a supported target string")
    return s[1:]


FeatureStructure = ty.Dict[str, ty.Union[str, bool, ty.Dict]]


def parse_fs(fs: etree._Element) -> FeatureStructure:
    """Parse a <tei:fs> element

    Note that this doesn't handle all the convoluted ways to specify fs in TEI
    but only the relatively simple subset we need here.
    """
    if fs.tag != f"{TEI}fs":
        raise ValueError(
            f"Attempting to parse a {fs.tag} element as a feature structure."
        )
    res = dict()
    for f in fs.iterfind(f"{TEI}f"):
        f_name = f.attrib[f"{TEI}name"]
        if len(f) == 0:
            f_value = f.text
        elif len(f) == 1:
            value_elem = f[0]
            if value_elem.tag in (f"{TEI}symbol", f"{TEI}numeric"):
                f_value = value_elem.attrib[f"{TEI}value"]
            elif value_elem.tag == f"{TEI}string":
                f_value = value_elem.text
            elif value_elem.tag == f"{TEI}binary":
                value_str = value_elem.attrib[f"{TEI}value"]
                if value_str in ("true", "1"):
                    f_value = True
                elif value_str in ("false", "0"):
                    f_value = False
                else:
                    raise ValueError(f"Invalid value for <tei:binary>: {value_str!r}.")
            elif value_elem.tag == f"{TEI}fs":
                f_value = parse_fs(value_elem)
            else:
                raise ValueError(f"Unsupported feature type: {value_elem.tag!r}")
        else:
            raise ValueError("Features with more than one children are not supported")
        res[f_name] = f_value
    return res


def get_fs(tree: etree._ElementTree) -> ty.Dict[str, FeatureStructure]:
    """Find and parse all the feature structures in `tree`.

    Return
    -------

    A dict mapping feature structures ids to their parsed contents.
    """
    fs_lst = tree.xpath("//tei:fs", namespaces=NSMAP)
    if not fs_lst:
        raise ElementNotFoundError(
            "There are no feature structure elements in this tree"
        )

    return {xmlid(fs): parse_fs(fs) for fs in fs_lst}


def targets_from_span(
    span: etree._ElementTree, getter: ty.Callable[[str], etree._Element]
) -> ty.List[etree._Element]:
    """Given a span and an {id: element} dict, return the list of the tokens in this span."""
    span_id = xmlid(span)
    target = span.get(f"{TEI}target")
    if target is not None:
        try:
            return [getter(target_to_id(i)) for i in target.split()]
        except KeyError as e:
            raise ElementNotFoundError(
                f"Element targetted by span {span_id} not found", e.args[0]
            ) from e

    start_id = target_to_id(span.attrib[f"{TEI}from"])
    end_id = target_to_id(span.attrib[f"{TEI}to"])
    try:
        start_node = getter(start_id)
    except KeyError as e:
        raise ElementNotFoundError(
            f"Span {span_id} start element not found", start_id
        ) from e
    targets = [start_node]
    if start_id != end_id:
        last_node = start_node
        siblings = iter(start_node.itersiblings(*TOKEN_TAGS))
        try:
            while xmlid(last_node) != end_id:
                last_node = next(siblings)
                targets.append(last_node)
        except StopIteration:
            raise ElementNotFoundError(f"Span {span_id} end element not found", end_id)
    return targets


@dataclasses.dataclass(eq=False)
class Mention:
    identifier: str
    span_type: str
    targets_parent: etree._Element
    speaker: ty.Optional[str]
    content: ty.List[str]
    features: FeatureStructure
    corresp: etree._Element
    targets: ty.Sequence[etree._Element]

    @classmethod
    def from_urs(
        cls,
        elt: etree._Element,
        id_getter: ty.Optional[ty.Callable[[str], ty.Optional[etree._Element]]] = None,
        fs_getter: ty.Optional[
            ty.Callable[[str], ty.Optional[FeatureStructure]]
        ] = None,
    ):
        """Build a `Mention` from an XML URS span element.

        You can pass an id getter and a feature structure getter or the document where
        they live.
        In last resort, we will try to find them in the document where `elt` resides.
        """
        if elt.tag != f"{TEI}span":
            raise ValueError(
                f"Attempting to build a mention from a {elt.tag!r} element"
            )
        document = elt.getroottree()
        if id_getter is None:
            id_store = {xmlid(elt): elt for elt in document.iter()}
            id_getter = id_store.get
        if fs_getter is None:
            fs_store = get_fs(document)
            fs_getter = fs_store.get

        targets = targets_from_span(elt, id_getter)
        elt_id = xmlid(elt)
        ana_target = elt.get(f"{TEI}ana")
        if ana_target is None:
            raise ValueError(f"Span {elt_id!r} has no `ana` attribute")
        fs = fs_getter(target_to_id(ana_target))
        if fs is None:
            raise ValueError(f"Span {elt_id!r} has no features")
        span_type = fs.get("type")
        if span_type is None:
            raise ValueError(f"Span {elt_id!r} has no `type` feature")
        elif not isinstance(span_type, str):
            raise ValueError(f"Span {elt_id!r} `type` feature is not a string")

        parents_set = set(t.getparent() for t in targets)
        if len(parents_set) > 1:
            raise ValueError(
                f"The targets of spans {elt_id!r} have more than one parent"
            )
        parent = next(iter(parents_set))
        return cls(
            identifier=elt_id,
            span_type=span_type,
            targets_parent=parent,
            speaker=parent.get(f"{TEI}who"),
            content=[t.text for t in targets],
            features=fs,
            corresp=elt,
            targets=targets,
        )


def get_mentions(tree: etree._ElementTree,) -> ty.Dict[ty.Tuple[str, str], Mention]:
    """Extract the mentions from an ANCOR-TEI document."""
    mentions = tree.xpath(
        (
            './tei:standOff/tei:annotation[@tei:type="coreference"]'
            '/tei:spanGrp[@tei:subtype="mention"]/tei:span'
        ),
        namespaces=NSMAP,
    )
    if not mentions:
        raise ValueError("`tree` has no mention spans")

    features = get_fs(tree)

    texts_lst = tree.findall(f"{TEI}text")
    if not texts_lst:
        raise ValueError(
            f"Attempting to extract mentions from a document without a text"
        )

    tokens_id_store = {
        xmlid(elt): elt for text in texts_lst for elt in text.iter(*TOKEN_TAGS)
    }

    res = dict()
    for m_elt in mentions:
        try:
            m = Mention.from_urs(m_elt, tokens_id_store.get, features.get)
        except ValueError as e:
            logger.warning(f"Skipping span {xmlid(m)}: {e}")
            continue
        if m.span_type not in MENTION_TYPES:
            if m.span_type in IGNORED_MENTION_TYPES:
                logger.debug(
                    f"Ignoring span {m.identifier!r} with mention type {m.span_type!r}"
                )
            else:
                logger.warning(
                    f"Span {m.identifier!r} has an invalid mention type ({m.span_type!r})"
                )
            continue
        res[(xmlid(m.targets[0]), xmlid(m.targets[-1]))] = m
    return res


def get_chains(tree: etree._ElementTree) -> ty.Dict[str, ty.Set[str]]:
    chains_grp_lst = tree.xpath(
        './tei:standOff/tei:annotation[@tei:type="coreference"]/tei:linkGrp[@tei:type="schema"]',
        namespaces=NSMAP,
    )
    chains_grp = chains_grp_lst[0]
    if len(chains_grp_lst) > 1:
        logger.warning(
            "There are more than one schema group in this document"
            f", only {xmlid(chains_grp)!r} will be taken into account"
        )

    res = dict()
    for c in chains_grp.iter(f"{TEI}link"):
        c_id = xmlid(c)
        target = c.get(f"{TEI}target")
        if target is None:
            raise ValueError(f"Schema {c_id!r} has no target attribute")
        res[c_id] = set((target_to_id(t) for t in target.split()))
    return res


def get_w_pos(tree: etree._ElementTree) -> ty.Dict[str, int]:
    """Return a dict mapping token nodes to their position in the tree."""
    return {xmlid(w): i for i, w in enumerate(tree.iter(*TOKEN_TAGS))}


def get_u_pos(tree: etree._ElementTree) -> ty.Dict[str, int]:
    """Return a dict mapping utterance nodes to their position in the tree."""
    return {xmlid(u): i for i, u in enumerate(tree.iter(f"{TEI}u"))}


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


MentionFeaturesDict = TypedDict(
    "MentionFeaturesDict",
    {
        "content": ty.Sequence[str],
        "left_context": ty.Sequence[str],
        "right_context": ty.Sequence[str],
        "length": int,
        "type": ty.Optional[str],
        "new": ty.Optional[str],
        "def": ty.Optional[str],
        "id": ty.Optional[str],
        "start": int,
        "end": int,
        "pos": ty.Sequence[str],
        "lemma": ty.Sequence[str],
        "morph": ty.Sequence[ty.Optional[ty.Collection[str]]],
        "entity_type": ty.Optional[str],
        "chunk_inclusion": ChunkInclusionStatus,
    },
    total=False,
)


def morph_from_tag(tag: str) -> ty.List[str]:
    """Extract morphosyntax features from spaCy tag str."""
    pos, rest = tag.split("__", maxsplit=1)
    return rest.split("|")


def spans_from_doc(
    doc: etree._ElementTree,
    min_width: int = 1,
    max_width: int = 26,
    context: ty.Tuple[int, int] = (10, 10),
    length_buckets: ty.Optional[ty.Sequence[int]] = (1, 2, 3, 4, 5, 7, 15, 32, 63),
) -> ty.Iterable[MentionFeaturesDict]:
    """
    Return all the text spans of `#doc`, with their mention type, definiteness and anaphoricity
    (for those who are not mentions, all of these are `None`)
    """
    w_pos = get_w_pos(doc)
    units = get_mentions(doc)
    nlp = spacy.load("fr_core_news_lg")
    nlp.tokenizer = nlp.tokenizer.tokens_from_list

    for utterance in doc.findall(".//tei:u", namespaces=NSMAP):
        content: ty.List[etree._Element] = list(utterance.iter(*TOKEN_TAGS))
        processed_utterance = nlp([t.text for t in content])
        ent_dict = {
            (e[0], e[-1]): e.label_
            for e in processed_utterance.ents
        }
        noun_chunks = sorted(processed_utterance.noun_chunks)
        spans = generate_spans_with_context(
            zip(content, ty.cast(ty.Iterable[spacy.tokens.Token], processed_utterance)),
            min_width,
            max_width,
            *context,
        )
        for left_context_t, span_t, right_context_t in spans:
            #  FIXME: Dirty way to split out spacy processing
            left_context, processed_left = (
                zip(*left_context_t) if left_context_t else ([], [])
            )
            right_context, processed_right = (
                zip(*right_context_t) if right_context_t else ([], [])
            )
            span, processed_span = zip(*span_t)
            start_id, end_id = xmlid(span[0]), xmlid(span[-1])
            mention = units.get((start_id, end_id))

            pos = [w.pos_ for w in (*processed_left, *processed_span, *processed_right)]
            lemma = [w.lemma_ for w in (*processed_left, *processed_span, *processed_right)]
            morph = [morph_from_tag(w.tag_) for w in (*processed_left, *processed_span, *processed_right)]
            left_context = [w.text for w in left_context]
            right_context = [w.text for w in right_context]
            if len(left_context) < context[0]:
                left_context.insert(0, "<start>")
                pos.insert(0, "<start>")
                lemma.insert(0, "<start>")
                morph.insert(0, [])
            if len(right_context) < context[1]:
                right_context.append("<end>")
                pos.append("<end>")
                lemma.append("<end>")
                morph.append([])

            content = [w.text for w in span]

            length = (
                int(np.digitize(len(content), bins=length_buckets, right=True))
                if length_buckets is not None
                else len(content)
            )
            entity_type = ent_dict.get((processed_span[0], processed_span[-1]), None)
            chunk_inclusion = span_inclusion(processed_span, noun_chunks)

            if mention is None:
                yield (
                    {
                        "content": content,
                        "left_context": left_context,
                        "right_context": right_context,
                        "length": length,
                        "type": None,
                        "new": None,
                        "def": None,
                        "id": None,
                        "start": w_pos[xmlid(span[0])],
                        "end": w_pos[xmlid(span[-1])],
                        "pos": pos,
                        "lemma": lemma,
                        "morph": morph,
                        "entity_type": entity_type,
                        "chunk_inclusion": chunk_inclusion,
                    }
                )
            else:
                yield {
                    "content": content,
                    "left_context": left_context,
                    "right_context": right_context,
                    "length": length,
                    "type": mention.span_type,
                    "new": mention.features.get("NEW", "_"),
                    "def": mention.features.get("DEF", "_"),
                    "id": mention.identifier,
                    "start": w_pos[xmlid(span[0])],
                    "end": w_pos[xmlid(span[-1])],
                    "pos": pos,
                    "lemma": lemma,
                    "morph": morph,
                    "entity_type": entity_type,
                    "chunk_inclusion": chunk_inclusion,
                }


class AntecedentFeaturesDict(TypedDict):
    """Features of antecedent pairs."""
    w_distance: int
    u_distance: int
    m_distance: int
    spk_agreement: bool
    overlap: bool
    coref: bool
    token_incl: int
    token_com: int


# TODO: give a sign to `token_incl` to specify which is the bigger span?
def antecedents_from_doc(
    doc: etree._ElementTree,
    min_width: int = 1,
    max_width: ty.Optional[int] = None,
    max_candidates: int = 100,
    distance_buckets: ty.Sequence[int] = (1, 2, 3, 4, 5, 7, 15, 32, 63),
) -> ty.Dict[str, ty.Dict[str, AntecedentFeaturesDict]]:
    """Extract the antecedents dataset from an ANCOR TEI document."""
    w_pos = get_w_pos(doc)
    u_pos = get_u_pos(doc)
    units = get_mentions(doc)
    sort_filt_units = sorted(
        (
            (start, end, mention)
            for (start, end), mention in units.items()
            if (
                min_width <= len(mention.content) < max_width
                if max_width is not None
                else (min_width <= len(mention.content))
            )
        ),
        key=lambda x: (w_pos[x[0]], w_pos[x[1]]),
    )
    if len(sort_filt_units) < 2:
        return dict()

    schemas = get_chains(doc)
    chain_from_mention = {m: c for c in schemas.values() for m in c}

    # The first mention in a document has no antecedent candidates
    # FIXME: we keep slicing, which generates copies, which makes me uneasy
    mentions = enumerate(sort_filt_units[1:], start=1)

    res = dict()
    for i, (s1, e1, mention) in mentions:
        mention_content_set = set(mention.content)
        chain = chain_from_mention.get(mention.identifier, None)
        antecedent_candidates = sort_filt_units[max(0, i - max_candidates) : i]
        antecedents: ty.Dict[str, AntecedentFeaturesDict] = dict()
        for j, (s2, e2, candidate) in enumerate(antecedent_candidates):
            candidate_content_set = set(candidate.content)
            coref = chain is not None and candidate.identifier in chain

            w_distance = int(
                np.digitize(w_pos[s1] - w_pos[e2], bins=distance_buckets, right=True)
            )
            u_distance = int(
                np.digitize(
                    u_pos[xmlid(mention.targets_parent)]
                    - u_pos[xmlid(candidate.targets_parent)],
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
            spk_agreement = mention.speaker == candidate.speaker

            intersect = len(mention_content_set.intersection(candidate_content_set))
            token_incl_ratio = int(
                10
                * intersect
                / min(len(mention_content_set), len(candidate_content_set))
            )
            token_com_ratio = int(
                10 * intersect / len(mention_content_set.union(candidate_content_set))
            )

            overlap = w_pos[s1] < w_pos[e2]

            antecedents[candidate.identifier] = {
                "w_distance": w_distance,
                "u_distance": u_distance,
                "m_distance": m_distance,
                "spk_agreement": spk_agreement,
                "overlap": overlap,
                "coref": coref,
                "token_incl": token_incl_ratio,
                "token_com": token_com_ratio,
            }
        res[mention.identifier] = antecedents
    return res


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


def main_entry_point(argv=None):
    arguments = docopt(__doc__, version=__version__, argv=argv)
    if arguments["--seed"] is not None:
        random.seed(int(arguments["--seed"]))
    if arguments["--mentions"] is None:
        arguments["--mentions"] = "-"
    if arguments["--antecedents"] is None:
        arguments["--antecedents"] = "-"
    strict_max_width = int(arguments["--max-width"])

    with smart_open(arguments["<in-file>"], "rb") as in_stream:
        tree = etree.parse(in_stream)

    spans = list(
        spans_from_doc(
            tree, context=[int(arguments["--context"])] * 2, max_width=strict_max_width
        )
    )
    all_mentions = get_mentions(tree)
    mentions = {m["id"]: m for m in spans if m["id"] is not None}
    skipped_mentions = len(all_mentions) - len(mentions)
    if skipped_mentions:
        logger.info(f"Skipping {skipped_mentions} mentions due to width limits")
    antecedents = antecedents_from_doc(
        tree, max_candidates=int(arguments["--max-candidates"])
    )

    if arguments["--det-ratio"] is not None:
        r = float(arguments["--det-ratio"])
        # FIXME: this is unbearably ugly
        mentions_i = []
        non_mentions_i = []
        for i, s in enumerate(spans):
            if s["type"] is None:
                non_mentions_i.append(i)
            else:
                mentions_i.append(i)
        if arguments["--oversample"]:
            replications = (
                math.ceil(((1 - r) / r) * len(non_mentions_i) / len(mentions_i)) - 1
            )
            spans.extend(replications * [spans[i] for i in mentions_i])
        else:
            n_non_mentions = math.ceil(len(mentions_i) * r / (1 - r))
            if n_non_mentions < len(non_mentions_i):
                sampled_non_mentions_i = random.sample(non_mentions_i, n_non_mentions)
                if arguments["--keep-single"]:
                    for i in non_mentions_i:
                        if len(spans[i]["content"]) == 1:
                            sampled_non_mentions_i.append(i)
                if arguments["--keep-named-entities"]:
                    for i in non_mentions_i:
                        if spans[i]["entity_type"] is not None:
                            sampled_non_mentions_i.append(i)
                if arguments["--keep-name-chunks"]:
                    for i in non_mentions_i:
                        if spans[i]["chunk_inclusion"] == "exact":
                            sampled_non_mentions_i.append(i)
                sampled_non_mentions_i = set(sampled_non_mentions_i)
                sampled_spans_i = sorted((*mentions_i, *sampled_non_mentions_i))
                spans = [spans[i] for i in sampled_spans_i]
            else:
                logger.warning(
                    f"Not enough non-mentions in {arguments['<in-file>']}"
                    f" to enforce non-mention ratio of {r}"
                    f" (actual ratio {len(non_mentions_i)/len(spans)})"
                )

    with smart_open(arguments["--mentions"], "wb") as out_stream:
        out_stream.write(orjson.dumps(spans))

    with smart_open(arguments["--antecedents"], "wb") as out_stream:
        out_stream.write(
            orjson.dumps({"mentions": mentions, "antecedents": antecedents, "args": dict(arguments)})
        )


if __name__ == "__main__":
    sys.exit(main_entry_point())
