import pathlib
import tempfile

import typing as ty

from collections import defaultdict

import ignite.contrib.handlers
import ignite.engine
import torch

from loguru import logger

from decofre import datatools
from decofre import lexicon
from decofre import libdecofre
from decofre import runners

from decofre.optimizers import DenseSparseAdamW


# FIXME: make something with loss_fun or remove it
def train_eval(
    model: torch.nn.Module,
    train_file: ty.Union[str, pathlib.Path],
    span_digitizer: ty.Callable[[ty.Mapping[str, ty.Any]], datatools.FeaturefulSpan],
    types_lex: lexicon.Lexicon,
    out_dir: ty.Union[str, pathlib.Path],
    device: ty.Union[str, torch.device],
    epochs: int,
    mention_boost: float,
    patience: int,
    dev_file: ty.Union[str, pathlib.Path] = None,
    test_file: ty.Union[str, pathlib.Path] = None,
    train_batch_size: int = 32,
    eval_batch_size: int = 128,
    loss_fun: ty.Optional[ty.Callable] = None,
    num_workers: int = 0,
    *,
    debug: bool = False,
    config: ty.Optional[ty.Dict[str, ty.Any]] = None,
    **kwargs,
):
    "Full training loop for a model"
    if loss_fun is not None:
        logger.warning("Training with a custom loss function is not supported yet")
    with tempfile.TemporaryDirectory(prefix="decofre_spans_") as temp_dir:
        logger.info(f"Using tempdir {temp_dir}")
        trainer, loader, kwargs = train_det(
            model=model,
            train_file=train_file,
            span_digitizer=span_digitizer,
            types_lex=types_lex,
            out_dir=out_dir,
            temp_dir=temp_dir,
            device=device,
            epochs=epochs,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            patience=patience,
            dev_file=dev_file,
            mention_boost=mention_boost,
            num_workers=num_workers,
            debug=debug,
            config=config,
            **kwargs,
        )
        trainer.run(loader, **kwargs)
        del loader, trainer, kwargs
    if test_file is not None:
        loss = evaluate_classif(
            model=model,
            test_file=test_file,
            span_digitizer=span_digitizer,
            classes_lex=types_lex,
            device=device,
            num_workers=num_workers,
            batch_size=eval_batch_size,
        )
        logger.info(f"Loss for test set: {loss}")


def train_det(
    model: torch.nn.Module,
    train_file: ty.Union[str, pathlib.Path],
    span_digitizer: ty.Callable[[ty.Mapping[str, ty.Any]], datatools.FeaturefulSpan],
    types_lex: lexicon.Lexicon,
    out_dir: ty.Union[str, pathlib.Path],
    temp_dir: ty.Union[str, pathlib.Path],
    device: ty.Union[str, torch.device],
    epochs: int,
    patience: int,
    mention_boost: ty.Optional[float] = None,
    dev_file: ty.Union[str, pathlib.Path] = None,
    test_file: ty.Union[str, pathlib.Path] = None,
    train_batch_size: int = 32,
    eval_batch_size: int = 128,
    trainer_cls=runners.SinkTrainer,
    *,
    num_workers: int = 0,
    debug: bool = False,
    config: ty.Optional[ty.Dict[str, ty.Any]] = None,
    **kwargs,
) -> ty.Tuple[ignite.engine.Engine, ty.Iterable, ty.Dict[str, ty.Any]]:
    logger.info("Training mention detection")
    config = defaultdict(lambda: None, config if config is not None else dict())
    device = torch.device(device)  # type: ignore
    model = model.to(device)
    train_set = datatools.SpansDataset.from_json(
        train_file,
        span_digitizer=span_digitizer,
        tags_lexicon=types_lex,
        cache_dir=temp_dir,
        set_name="train_det",
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
    dev_loader: ty.Optional[torch.utils.data.DataLoader]
    if dev_file is not None:
        dev_set = datatools.SpansDataset.from_json(
            dev_file,
            span_digitizer=span_digitizer,
            tags_lexicon=types_lex,
            cache_dir=temp_dir,
            set_name="dev_det",
        )
        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_set,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(dev_set),
                batch_size=eval_batch_size,
                drop_last=False,
            ),
            collate_fn=lambda x: x[0],
            num_workers=num_workers,
        )

    else:
        dev_set = None
        dev_loader = None

    if mention_boost is not None:
        class_weight = torch.tensor(
            [1 if c is None else mention_boost for c in types_lex.i2t],
            device=device,
            dtype=torch.float,
        )
        logger.debug(f"Training with weights {class_weight} for weighted nll_loss")

        def loss_fun(output, target):
            return libdecofre.averaged_nll_loss(
                output.to(device=device), target.to(device=device), weight=class_weight
            )

    else:
        logger.debug("Training with unweighted batch-averaged NLL loss")

        def loss_fun(output, target):
            return torch.nn.functional.nll_loss(
                output.to(device=device), target.to(device=device), reduction="mean"
            )

    # TODO: use accuracy instead ?
    def dev_loss(output, target):
        return torch.nn.functional.nll_loss(
            output.to(device=device), target.to(device=device), reduction="mean"
        )

    train_classif = runners.ClassificationMetrics(
        types_lex.i2t,
        output_transform=runners.extract_output,
        aggregates={"mentions": [t for t in types_lex.i2t if t is not None]},
    )
    dev_classif = runners.ClassificationMetrics(
        types_lex.i2t,
        output_transform=runners.extract_output,
        aggregates={"mentions": [t for t in types_lex.i2t if t is not None]},
    )
    optimizer = DenseSparseAdamW(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config["weight-decay"],
    )
    det_trainer = trainer_cls(
        model,
        checkpointed_models={"det": model},
        loss_fun=loss_fun,
        optimizer=optimizer,
        dev_loss=dev_loss,
        train_metrics={"classif": train_classif},
        dev_metrics={"classif": dev_classif},
        save_path=out_dir,
        debug=debug,
        **kwargs,
    )
    if config["lr-schedule"] == "step":
        logger.debug("Training with 'step' LR schedule, using γ=0.7")
        torch_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, len(train_loader), gamma=0.7
        )
        scheduler = ignite.contrib.handlers.create_lr_scheduler_with_warmup(
            torch_lr_scheduler,
            warmup_start_value=0.0,
            warmup_end_value=optimizer.defaults["lr"],
            warmup_duration=1000,
        )
        det_trainer.add_event_handler(ignite.engine.Events.ITERATION_STARTED, scheduler)

    return (
        det_trainer,
        train_loader,
        {
            "max_epochs": epochs,
            "patience": patience,
            "dev_loader": dev_loader,
            "run_name": "mention_detection",
        },
    )


def evaluate_classif(
    model: torch.nn.Module,
    test_file: ty.Union[str, pathlib.Path],
    span_digitizer: ty.Callable[[ty.Mapping[str, ty.Any]], datatools.FeaturefulSpan],
    classes_lex: lexicon.Lexicon,
    batch_size: int = 128,
    device=None,
    num_workers: int = 0,
) -> torch.Tensor:
    """Evaluate a classification model."""
    model = model.to(device)

    def loss_fun(output, target):
        return torch.nn.functional.nll_loss(
            output.to(device=device), target.to(device=device), reduction="mean"
        )

    with tempfile.TemporaryDirectory(prefix="decofre_") as temp_dir:
        test_set = datatools.SpansDataset.from_json(
            test_file,
            span_digitizer=span_digitizer,
            tags_lexicon=classes_lex,
            cache_dir=temp_dir,
            set_name="test",
        )
        evaluator = runners.Evaluator(
            model,
            loss=loss_fun,
            metrics={
                "classif": runners.ClassificationMetrics(
                    classes_lex.i2t,
                    output_transform=runners.extract_output,
                    aggregates={
                        "mentions": [t for t in classes_lex.i2t if t is not None]
                    },
                )
            },
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(test_set),
                batch_size=batch_size,
                drop_last=False,
            ),
            collate_fn=lambda x: x[0],
            num_workers=num_workers,
        )
        state = evaluator.run(test_loader)
    for name, value in state.metrics.items():
        logger.info(f"{name}: {value}")
    return state.metrics["loss"]
