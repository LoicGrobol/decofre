import pathlib
import tempfile

import typing as ty
from typing import Tuple

import ignite.engine
import torch
import torch.jit

from loguru import logger

from decofre import datatools
from decofre import libdecofre
from decofre import runners


def train_eval(
    model: torch.nn.Module,
    train_file: ty.Union[str, pathlib.Path],
    span_digitizer: ty.Callable[[ty.Mapping[str, ty.Any]], datatools.FeaturefulSpan],
    pair_feats_digitizer: ty.Callable[[ty.Mapping[str, str]], ty.Iterable[int]],
    out_dir: ty.Union[str, pathlib.Path],
    device: torch.device,
    epochs: int,
    patience: int,
    optimizer: torch.optim.Optimizer,
    dev_file: ty.Optional[ty.Union[str, pathlib.Path]] = None,
    test_file: ty.Optional[ty.Union[str, pathlib.Path]] = None,
    loss_fun: ty.Optional[ty.Callable] = None,
    num_workers: int = 0,
    debug: bool = False,
    **kwargs,
):
    "Full training loop for a model"
    if loss_fun is None:

        def loss_fun(source, target):
            return libdecofre.masked_multi_cross_entropy(
                source.to(device=device), target.to(device=device)
            )

    # TODO: allow reusing an existind db cache. See eval.py for an example
    with tempfile.TemporaryDirectory(prefix="decofre_antecedents_") as temp_dir:
        logger.info(f"Using tempdir {temp_dir}")
        engine, loader, kwargs = train_cor(
            model=model,
            train_file=train_file,
            span_digitizer=span_digitizer,
            pair_feats_digitizer=pair_feats_digitizer,
            out_dir=out_dir,
            temp_dir=temp_dir,
            device=device,
            epochs=epochs,
            patience=patience,
            dev_file=dev_file,
            optimizer=optimizer,
            loss_fun=loss_fun,
            num_workers=num_workers,
            debug=debug,
            **kwargs,
        )
        engine.run(loader, **kwargs)
    del loader, engine
    if test_file is not None:
        evaluate_cor(
            model=model,
            test_file=test_file,
            span_digitizer=span_digitizer,
            pair_feats_digitizer=pair_feats_digitizer,
            loss_fun=loss_fun,
            device=device,
            num_workers=num_workers,
        )


def train_cor(
    model: torch.nn.Module,
    train_file: ty.Union[str, pathlib.Path],
    span_digitizer: ty.Callable[[ty.Mapping[str, ty.Any]], datatools.FeaturefulSpan],
    pair_feats_digitizer: ty.Callable[[ty.Mapping[str, str]], ty.Iterable[int]],
    out_dir: ty.Union[str, pathlib.Path],
    temp_dir: ty.Union[str, pathlib.Path],
    device: torch.device,
    epochs: int,
    patience: int,
    train_batch_size: int = 8,
    dev_file: ty.Optional[ty.Union[str, pathlib.Path]] = None,
    optimizer=None,
    loss_fun: ty.Callable = libdecofre.masked_multi_cross_entropy,
    trainer_cls=runners.SinkTrainer,
    num_workers: int = 0,
    debug: bool = False,
    **kwargs,
) -> ty.Tuple[ignite.engine.Engine, ty.Iterable, ty.Dict[str, ty.Any]]:
    logger.info("Training antecedent scoring")
    model = model.to(device)
    train_set = datatools.AntecedentsDataset.from_json(
        train_file,
        span_digitizer=span_digitizer,
        pair_feats_digitizer=pair_feats_digitizer,
        cache_dir=temp_dir,
        set_name="train_cor",
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(train_set),
            batch_size=train_batch_size,
            drop_last=False,
        ),
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
    )

    if dev_file is not None:
        dev_set = datatools.AntecedentsDataset.from_json(
            dev_file,
            span_digitizer=span_digitizer,
            pair_feats_digitizer=pair_feats_digitizer,
            cache_dir=temp_dir,
            set_name="dev_cor",
        )
        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_set,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(dev_set),
                batch_size=train_batch_size,
                drop_last=False,
            ),
            collate_fn=lambda x: x[0],
            num_workers=num_workers,
        )  # type: ty.Optional[torch.data.DataLoader]
    else:
        dev_loader = None

    cor_trainer = trainer_cls(
        model,
        checkpointed_models={"cor": model},
        loss_fun=loss_fun,
        optimizer=optimizer,
        dev_loss=loss_fun,
        dev_metrics={
            "antecedent_accuracy": runners.MultiLoss(
                loss_fn=lambda x, y: _summable_antecedent_accuracy(
                    x.to(device), y.to(device)
                ),
                num_loss=3,
                output_transform=runners.extract_output,
                device=device,
                loss_names=(
                    "total_accuracy",
                    "mention_new_accuracy",
                    "anaphora_accuracy",
                ),
            ),
            "attributions": runners.MultiLoss(
                loss_fn=lambda x, y: attributions(x.to(device), y.to(device)),
                output_transform=runners.extract_output,
                averaged=False,
                device=device,
                loss_names=(
                    "true_new",
                    "false_new",
                    "correct_link",
                    "false_link",
                    "wrong_link",
                ),
            ),
        },
        save_path=out_dir,
        debug=debug,
        **kwargs,
    )

    return (
        cor_trainer,
        train_loader,
        {
            "max_epochs": epochs,
            "patience": patience,
            "dev_loader": dev_loader,
            "run_name": "antecedent_scoring",
        },
    )


def evaluate_cor(
    model: torch.nn.Module,
    test_file: ty.Union[str, pathlib.Path],
    span_digitizer: ty.Callable[[ty.Mapping[str, ty.Any]], datatools.FeaturefulSpan],
    pair_feats_digitizer: ty.Callable[[ty.Mapping[str, str]], ty.Iterable[int]],
    loss_fun: ty.Callable = libdecofre.masked_multi_cross_entropy,
    device=None,
    num_workers: int = 0,
) -> ty.Tuple[torch.FloatTensor]:
    model = model.to(device)
    with tempfile.TemporaryDirectory(prefix="decofre_antecedents_") as temp_dir:
        test_set = datatools.AntecedentsDataset.from_json(
            test_file,
            span_digitizer=span_digitizer,
            pair_feats_digitizer=pair_feats_digitizer,
            cache_dir=temp_dir,
            set_name="test",
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(test_set), batch_size=8, drop_last=False
            ),
            collate_fn=lambda x: x[0],
            num_workers=num_workers,
        )

        logger.info("Evaluating on the test set")
        model.eval()
        evaluator = runners.Evaluator(
            model,
            loss=loss_fun,
            metrics={
                "antecedent_accuracy": runners.MultiLoss(
                    loss_fn=lambda x, y: _summable_antecedent_accuracy(
                        x.to(device), y.to(device)
                    ),
                    output_transform=runners.extract_output,
                    device=device,
                    loss_names=(
                        "total_accuracy",
                        "mention_new_accuracy",
                        "anaphora_accuracy",
                    ),
                ),
                "attributions": runners.MultiLoss(
                    loss_fn=lambda x, y: attributions(x.to(device), y.to(device)),
                    output_transform=runners.extract_output,
                    averaged=False,
                    device=device,
                    loss_names=(
                        "true_new",
                        "false_new",
                        "correct_link",
                        "false_link",
                        "wrong_link",
                    ),
                ),
            },
        )
        state = evaluator.run(test_loader)
    for name, value in state.metrics.items():
        logger.info(f"{name}: {value}")
    return state.metrics["loss"]


@torch.jit.script
def antecedent_accuracy(
    scores: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = targets.to(torch.bool)
    _, highest_scores_indices = torch.max(scores, dim=-1)
    argmax_mask = torch.nn.functional.one_hot(
        highest_scores_indices, num_classes=scores.size(-1)
    ).to(torch.bool)
    # Casting to float is needed for the subsequent mean computations
    is_correct = targets.masked_select(argmax_mask).to(torch.float)
    mention_new_targets = targets.narrow(-1, 0, 1).squeeze()
    mention_new_accuracy = is_correct.masked_select(mention_new_targets).mean()
    anaphora_accuracy = is_correct.masked_select(
        mention_new_targets.logical_not()
    ).mean()
    total_accuracy = is_correct.mean()
    return total_accuracy, mention_new_accuracy, anaphora_accuracy


@torch.jit.script
def _summable_antecedent_accuracy(
    scores: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    out = torch.stack(antecedent_accuracy(scores, targets))
    return out


@torch.jit.script
def attributions(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Error analysis for antecedent scoring

    ## Inputs
    - `scores`: `(batch_size, max_antecedent_number)`-shaped float tensor.
        The first dimension might be padded with `-float("-inf")` or
        approximations (such as `1e-32`) when a mention has less than
        ` max_antecedent_number` antecedent candidates.
    - `targets`: `(batch_size, max_antecedent_number)`-shaped byte tensor
        mask with `true` for the correct antecedents.
    """
    targets = targets.to(torch.bool)
    _, highest_scores_indices = torch.max(scores, dim=-1)
    sys_new = highest_scores_indices.eq(0)
    gold_new = targets.narrow(-1, 0, 1).squeeze()
    true_new = gold_new & sys_new
    false_new = gold_new.logical_not() & sys_new
    false_link = gold_new & sys_new.logical_not()
    argmax_mask = torch.nn.functional.one_hot(
        highest_scores_indices, num_classes=scores.size(-1)
    ).to(torch.bool)
    correct = targets.masked_select(argmax_mask)
    correct_link = correct & true_new.logical_not()
    wrong_link = (correct | false_new | false_link).logical_not()

    outpt = torch.stack((true_new, false_new, correct_link, false_link, wrong_link))
    outpt = outpt.sum(dim=-1)
    return outpt
