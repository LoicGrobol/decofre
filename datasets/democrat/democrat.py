import pathlib
import subprocess  # nosec

from typing import Any, Dict, Optional

from loguru import logger
import tqdm

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def generate_spans_and_antecedents(
    text_path: pathlib.Path,
    annotation_path: pathlib.Path,
    spans_dir: pathlib.Path,
    antecedents_dir: pathlib.Path,
    seed: Optional[int] = None,
    ratio: Optional[float] = None,
    max_m_distance: Optional[int] = None,
    max_candidates: Optional[int] = None,
):
    spans_dir.mkdir(parents=True, exist_ok=True)
    antecedents_dir.mkdir(parents=True, exist_ok=True)
    args = [
        str(text_path),
        str(annotation_path),
        "-m",
        str(spans_dir / f"{text_path.stem}.json"),
        "-a",
        str(antecedents_dir / f"{text_path.stem}.json"),
    ]
    if seed is not None:
        args.extend(("--seed", str(seed),))
    if ratio is not None:
        args.extend(("--det-ratio", str(ratio)))
    if max_m_distance is not None:
        args.extend(("--max-m-dist", str(max_m_distance)))
    if max_candidates is not None:
        args.extend(("--max-candidates", str(max_candidates)))
    subprocess.run(
        ["python", str(SCRIPT_DIR / "democrat_spangen.py"), *args], check=True,
    )


def generate_cluster_file(annotation_path: pathlib.Path, clusters_path: pathlib.Path):
    subprocess.run(
        [
            "python",
            str(SCRIPT_DIR / "democrat_clusters.py"),
            str(annotation_path),
            str(clusters_path),
        ],
        check=True,
    )


def generate_dataset(
    source_dir: pathlib.Path,
    target_dir: pathlib.Path,
    spangen_args=Optional[Dict[str, Any]],
):
    logger.info(f"Generating training from {source_dir} to {target_dir}")
    spans_dir = target_dir / "mentions"
    antecedents_dir = target_dir / "antecedents"
    text_lst = sorted(
        f for f in source_dir.glob("*.xml") if not str(f).endswith("-urs.xml")
    )
    for text in tqdm.tqdm(text_lst, unit="documents", desc="Building datafiles"):
        annotation = source_dir / f"{text.stem}-urs.xml"
        generate_spans_and_antecedents(
            text, annotation, spans_dir, antecedents_dir, **spangen_args
        )


def generate_clusters(source_dir: pathlib.Path, target_dir: pathlib.Path):
    logger.info(f"Extracting clusters from {source_dir} to {target_dir}")
    clusters_dir = target_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True, parents=True)
    text_lst = sorted(
        f for f in source_dir.glob("*.xml") if not str(f).endswith("-urs.xml")
    )
    for text in tqdm.tqdm(text_lst, unit="documents", desc="Building cluster files"):
        annotation = source_dir / f"{text.stem}-urs.xml"
        clusters_file = clusters_dir / f"{text.stem}.json"
        generate_cluster_file(annotation, clusters_file)


def process_split(root_dir: pathlib.Path):
    subcorpora = (
        ("train", {"seed": 0, "ratio": 0.99, "max_m_distance": 128, "max_candidates": 16}),
        ("dev", {"seed": 0, "max_m_distance": 128}),
        ("test", {"seed": 0, "max_m_distance": 128}),
    )
    for subcorpus, args in subcorpora:
        generate_dataset(
            root_dir / "data" / "democrat" / subcorpus,
            root_dir / "local" / "processed" / subcorpus,
            spangen_args=args,
        )
        generate_clusters(
            root_dir / "data" / "democrat" / subcorpus,
            root_dir / "local" / "processed" / subcorpus,
        )


process_split(SCRIPT_DIR)
