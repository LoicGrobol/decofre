import json
import pathlib
import tempfile
import shutil
import time

import typing as ty

import doit
import doit.tools
import requests
import tqdm

from fouine import download

import ancor_clusters
import ancor_spangen

# Python < 3.8 shim
try:
    from gzip import BadGzipFile  # type: ignore # (silence mypy for < 3.8)
except ImportError:
    BadGzipFile = OSError


root = pathlib.Path(__file__).resolve().parent


def success_wrapper(fun: ty.Callable) -> ty.Callable:
    def aux(*args, **kwargs):
        fun(*args, **kwargs)
        return True

    return aux


def build_dataset(
    source_dir: ty.Union[str, pathlib.Path],
    target_dir: ty.Union[str, pathlib.Path],
    opts: ty.Iterable[str] = None,
):
    if opts is None:
        opts = []
    source_dir = pathlib.Path(source_dir)
    target_dir = pathlib.Path(target_dir)

    mentions_dir = pathlib.Path(target_dir) / "mentions"
    antecedents_dir = pathlib.Path(target_dir) / "antecedents"
    mentions_dir.mkdir(parents=True, exist_ok=True)
    antecedents_dir.mkdir(parents=True, exist_ok=True)

    source_files = list(source_dir.glob("*.tei"))
    pbar = tqdm.tqdm(
        source_files,
        unit="documents",
        desc="Processing",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        leave=False,
        mininterval=0.5,
    )
    for f in pbar:
        pbar.desc = f"Processing {f.stem}"
        ancor_spangen.main_entry_point(
            [
                *opts,
                "--only-id",
                str(f),
                "--mentions",
                str(mentions_dir / f"{f.stem}.json"),
                "--antecedents",
                str(antecedents_dir / f"{f.stem}.json"),
            ]
        )


def copy(source, target):
    shutil.copy(source, target)


def task_get_word2vec():
    word2vec_url = (
        "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz"
    )
    target_dir = root / "data"
    dest_name = "cc.fr.300.vec"
    return {
        "actions": [
            (doit.tools.create_folder, (target_dir,)),
            (
                success_wrapper(download),
                [],
                {
                    "source": word2vec_url,
                    "target_dir": target_dir,
                    "dest_name": dest_name,
                },
            ),
        ],
        "targets": [target_dir / dest_name],
        "uptodate": [doit.tools.run_once],
    }


def url_last_date(url):
    r = requests.head(url, allow_redirects=True)

    if r.status_code == 200:
        return r.headers.get("Last-Modified", str(time.time()))


def task_get_data_source():
    config_file = "ancor.json"
    with open(config_file) as config_stream:
        config = json.load(config_stream)

    source_url = config["source"]
    tempdir = tempfile.TemporaryDirectory()
    temppath = pathlib.Path(tempdir.name)
    target_dir = root / "data"

    def split_data():
        for name, files in config["subcorpora"].items():
            subdir = target_dir / name / "source"
            subdir.mkdir(parents=True, exist_ok=True)
            for f in files:
                copy(temppath / f, subdir / f)

    return {
        "actions": [
            (doit.tools.create_folder, (target_dir,)),
            (success_wrapper(download), (source_url, temppath, "ancor.tar.gz")),
            (shutil.unpack_archive, (str(temppath / "ancor.tar.gz"), str(temppath))),
            split_data,
            tempdir.cleanup,
        ],
        "file_dep": ["ancor.json"],
        "uptodate": [doit.tools.config_changed(url_last_date(source_url))],
    }


def task_generate_gold_clusters():
    for dataset in ("dev", "test"):
        source_dir = root / "data" / dataset / "source"
        target_dir = root / "data" / dataset / "clusters"
        yield {
            "name": dataset,
            "actions": [
                (doit.tools.create_folder, (target_dir,)),
                *(
                    (
                        ancor_clusters.main_entry_point,
                        ([str(source), str(target_dir / f"{source.stem}.json")],),
                    )
                    for source in source_dir.glob("*.tei")
                ),
            ],
            "file_dep": list(source_dir.glob("*.tei")),
        }


def task_generate_datasets():
    data_dir = root / "data"
    datasets = [
        ("train", ["--det-ratio", "0.95", "--keep-single", "--keep-named-entities", "--seed", "0"]),
        ("dev", ["--seed", "0"]),
        ("test", ["--seed", "0"]),
    ]
    for n, opts in datasets:
        source_dir = data_dir / n / "source"
        target_dir = data_dir / n
        yield {
            "name": n,
            "actions": [
                (doit.tools.create_folder, (target_dir,)),
                (build_dataset, (source_dir, target_dir, opts)),
            ],
            "file_dep": [*source_dir.glob("*.tei"), "ancor_spangen.py"],
            "targets": [target_dir / "mentions.json", target_dir / "antecedents.json"],
        }


def task_copy_fixtures():
    data_dir = root / "data" / "train"
    fixtures_dir = root / "data" / "fixtures"
    fixtures = [
        (data_dir / "mentions" / "004_-1.json", fixtures_dir / "mentions.json"),
        (data_dir / "antecedents" / "004_-1.json", fixtures_dir / "antecedents.json"),
    ]
    return {
        "actions": [
            (doit.tools.create_folder, (fixtures_dir,)),
            *((copy, (s, t)) for s, t in fixtures),
        ],
        "file_dep": [s for s, _ in fixtures],
        "targets": [t for _, t in fixtures],
    }


def task_copy_tiny_fixtures():
    data_dir = root / "data" / "train"
    fixtures_dir = root / "data" / "fixtures-tiny"
    fixtures = [
        (data_dir / "mentions" / "1AG0391.json", fixtures_dir / "mentions.json"),
        (data_dir / "antecedents" / "1AG0391.json", fixtures_dir / "antecedents.json"),
    ]
    return {
        "actions": [
            (doit.tools.create_folder, (fixtures_dir,)),
            *((copy, (s, t)) for s, t in fixtures),
        ],
        "file_dep": [s for s, _ in fixtures],
        "targets": [t for _, t in fixtures],
    }


if __name__ == "__main__":
    import doit

    doit.run(globals())
