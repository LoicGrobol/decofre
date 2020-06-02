import pathlib
import subprocess

import tqdm

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def generate_spans_and_antecedents(
    text_path: pathlib.Path,
    annotation_path: pathlib.Path,
    spans_dir: pathlib.Path,
    antecedents_dir: pathlib.Path,
):
    spans_dir.mkdir(parents=True, exist_ok=True)
    antecedents_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python",
            str(SCRIPT_DIR / "democrat_spangen.py"),
            str(text_path),
            str(annotation_path),
            "-m",
            str(spans_dir / f"{text_path.stem}.json"),
            "-a",
            str(antecedents_dir / f"{text_path.stem}.json"),
        ],
        check=True,
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


def generate_dataset(source_dir: pathlib.Path, target_dir: pathlib.Path):
    spans_dir = target_dir / "mentions"
    antecedents_dir = target_dir / "antecedents"
    text_lst = list((source_dir / "xml").glob("*.xml"))
    for text in tqdm.tqdm(text_lst, unit="documents", desc="Building datafiles"):
        annotation = source_dir / "urs" / f"{text.stem}-urs.xml"
        generate_spans_and_antecedents(text, annotation, spans_dir, antecedents_dir)


def generate_clusters(source_dir: pathlib.Path, target_dir: pathlib.Path):
    clusters_dir = target_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True, parents=True)
    text_lst = list((source_dir / "xml").glob("*.xml"))
    for text in tqdm.tqdm(text_lst, unit="documents", desc="Building cluster files"):
        annotation = source_dir / "urs" / f"{text.stem}-urs.xml"
        clusters_file = clusters_dir / f"{text.stem}.json"
        generate_cluster_file(annotation, clusters_file)


generate_dataset(SCRIPT_DIR / "data" / "democrat", SCRIPT_DIR / "local")
generate_clusters(SCRIPT_DIR / "data" / "democrat", SCRIPT_DIR / "local")
