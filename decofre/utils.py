import collections.abc
import json
import pathlib
import sys

from typing import Optional

import _jsonnet

import git
import schema
import torch
import torch.jit
import tqdm

import boltons.iterutils as itu


@torch.jit.script
def confusion(
    predicted: torch.Tensor, target: torch.Tensor, n: int = -1
) -> torch.Tensor:
    """Return the confusion matrix for a classification in `n` classes.

    If `n` is not given, infer one from the contents of `predicted` and `target`.
    The definition of the confusion matrix is the one from
    <https://en.wikipedia.org/w/index.php?title=Confusion_matrix&oldid=903456530>
    ```text
    Gold ->  A   B   C
       Sys
        A   aa  ab  ac
        B   ba  bb  bc
        C   ca  cb  cc
    ```
    """
    assert predicted.shape == target.shape  # nosec
    assert predicted.dtype == target.dtype == torch.long  # nosec
    if n == -1:
        n = max(predicted.max(), target.max())
    res = torch.zeros((n, n), device=target.device, dtype=torch.long)
    for i, j in zip(predicted.flatten(), target.flatten()):
        res[i, j] += 1
    return res


@torch.jit.script
def PRF(confusion_matrix: torch.Tensor) -> torch.Tensor:
    """Return the list of `$(P, R, F₁)$` scores for every class.

    ## Output
      - `output`: a `torch.Tensor` of shape `(n_classes, 3)`
    """
    tp = torch.diagonal(confusion_matrix).float()
    pos = confusion_matrix.sum(dim=1).float()
    tru = confusion_matrix.sum(dim=0).float()
    P = tp / pos
    R = tp / tru
    F = 2 * tp / (pos + tru)
    prf = torch.stack((P, R, F)).t()
    # Edge case: if any of this metric is NaN, it is because
    # it was a 0/0 case. In which case it was perfectly accurate
    prf.masked_fill_(torch.isnan(prf), 1.0)
    return prf


class TqdmCompatibleStream:
    """Dummy file-like that will write to tqdm"""

    def __init__(self, file=sys.stderr):
        self.file = file

    def write(self, x):
        if getattr(tqdm.tqdm, "_instances", []):
            # Avoid print() second call (useless \n)
            x = x.rstrip()
            if len(x) > 0:
                tqdm.tqdm.write(x, file=self.file)
        else:
            if self.file is None:
                sys.stderr.write(x)
            else:
                self.file.write(x)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)


def git_commit_hash(path=None) -> Optional[str]:
    try:
        repo = git.Repo(str(path), search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return None
    return repo.head.object.hexsha


def git_dirty(path=None) -> str:
    repo = git.Repo(str(path), search_parent_directories=True)
    return repo.is_dirty()


def merge_nested(d1, d2):
    def visit(p, k, v):
        try:
            update_v = itu.get_path(d2, p)[k]
        except (itu.PathAccessError, KeyError):
            return (k, v)
        if isinstance(update_v, collections.abc.Mapping) and isinstance(
            v, collections.abc.Mapping
        ):
            return (k, {**v, **update_v})
        return (k, update_v)

    out = itu.remap(d1, visit=visit)
    out.update({k: v for k, v in d2.items() if k not in d1})
    return out


class PathUri:
    """An (absolute) path validator for `schema`."""

    def __init__(self, must_exist: bool = False, mkdir: bool = False):
        self.must_exist = must_exist
        self.mkdir = mkdir

    def validate(self, data: str) -> pathlib.Path:
        p = pathlib.Path(data)
        if self.must_exist and not p.exists():
            raise schema.SchemaError(f"{p} must exist already")
        if self.mkdir:
            p.mkdir(exist_ok=True, parents=True)
        return p


def load_jsonnet(json_path):
    json_path = pathlib.Path(json_path).resolve()
    raw_config = json.loads(_jsonnet.evaluate_file(str(json_path)))

    def visit(p, k, v):
        # Deal with file uris. For now only `file:///absolute/path` and
        # `file:/relative/to/json_path.parent()`
        if isinstance(v, str) and v.startswith("file:"):
            fpath = v[5:]
            # Absolute path
            if fpath.startswith("///"):
                p = pathlib.Path(v[7:]).resolve()
            # Relative path
            elif fpath.startswith("/"):
                p = (json_path.parent / pathlib.Path(fpath[1:])).resolve()
            else:
                raise ValueError(f"Unsupported file uri {v}")
            return (k, str(p))
        return True

    return itu.remap(raw_config, visit=visit)


def dumpable_config(config):
    return itu.remap(
        config,
        visit=lambda p, k, v: not isinstance(v, pathlib.Path)
        and (not isinstance(v, collections.abc.Collection) or bool(v)),
    )
