r"""Evaluate a coreference detection at corpus level

Usage:
  eval [options] <model> <corpus-path> [<out-file>]

Arguments:
  <corpus-path>  path of the corpus directory
  <out-file>     result file path (in the format you want), `-` for standard output [default: -]

Options:
  -h, --help  Show this screen.
  --debug  Run in debug mode
  --intermediate-dir <p>  Intermediate directory path (defaults to using tempdir)
  --overwrite  Overwrite existing score files
"""

import contextlib
import pathlib
import signal
import sys
import tempfile

import typing as ty

import tqdm

import scorch.main

from docopt import docopt

from loguru import logger

from decofre import clusterize, score, utils
from decofre import __version__

# Deal with piping output in a standard-compliant way
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


# Thanks http://stackoverflow.com/a/17603000/760767
@contextlib.contextmanager
def smart_open(filename: str, mode: str = "r", *args, **kwargs):
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
def directory_manager(dir_path: str = None, *args, **kwargs):
    """Unified interface for either using a named directory or a temp one"""
    tmpdir: ty.Optional[tempfile.TemporaryDirectory]
    if dir_path is None:
        tmpdir = tempfile.TemporaryDirectory(*args, **kwargs)
        dir_path = tmpdir.name
    else:
        tmpdir = None

    try:
        yield pathlib.Path(dir_path)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def main_entry_point(argv=None):
    arguments = docopt(__doc__, version=__version__, argv=argv)
    logger.remove(0)  # Don't log directly to stderr
    if arguments["--debug"]:
        log_level = "DEBUG"
        log_fmt = (
            "[decofre] "
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>@<cyan>{line}</cyan>: "
            "<level>{message}</level>"
        )
    else:
        log_level = "INFO"
        log_fmt = (
            "[decofre] "
            "<green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            "<level>{message}</level>"
        )
    logger.add(
        utils.TqdmCompatibleStream(), level=log_level, format=log_fmt, colorize=True
    )
    # Since there are no support for default positional arguments in
    # docopt yet. Might be useful for complex default values, too
    if arguments["<out-file>"] is None:
        arguments["<out-file>"] = "-"

    corpus_dir = pathlib.Path(arguments["<corpus-path>"]).resolve()
    antecedents_dir = (corpus_dir / "antecedents").resolve()

    with directory_manager(arguments["--intermediate-dir"]) as intermediate_dir:
        score_dir = (intermediate_dir / "score").resolve()
        gold_clusters_dir = (corpus_dir / "clusters").resolve()
        sys_clusters_dir = (intermediate_dir / "clusters").resolve()
        for p in (score_dir, sys_clusters_dir):
            p.mkdir(parents=True, exist_ok=True)

        antecedents_files = sorted(antecedents_dir.glob("*.json"))
        pbar = tqdm.tqdm(
            antecedents_files,
            unit="documents",
            desc="Processing",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            leave=False,
            disable=None,
        )
        for data_file in pbar:
            stem = data_file.stem
            pbar.set_description(f"Processing {stem}")
            score_file = score_dir / f"{stem}.json"
            if not score_file.exists() or arguments["--overwrite"]:
                score.main_entry_point(
                    [arguments["<model>"], str(data_file), str(score_file)]
                )
            else:
                logger.debug(f"Skipping scoring {score_file}")

            sys_clusters_file = sys_clusters_dir / f"{stem}.json"
            if not sys_clusters_file.exists() or arguments["--overwrite"]:
                clusterize.main_entry_point([str(score_file), str(sys_clusters_file)])
            else:
                logger.debug(f"Skipping clustering {sys_clusters_file}")

        with smart_open(arguments["<out-file>"], "w") as out_stream:
            out_stream.writelines(
                scorch.main.process_dirs(gold_clusters_dir, sys_clusters_dir)
            )


if __name__ == "__main__":
    sys.exit(main_entry_point())
