from __future__ import annotations

import collections.abc
import datetime
import enum
import math
import pathlib
import sys
import time

import itertools as it
import typing as ty

import ignite
import tensorboardX
import torch
import tqdm

from loguru import logger

import boltons.iterutils as itu

from ignite._utils import _to_hours_mins_secs

from decofre import datatools
from decofre.optimizers import DenseSparseAdamW
from decofre.utils import PRF, confusion

T = ty.TypeVar("T")


class CustomEvents(enum.Enum):
    LOGGING = "logging_event"


def extract_output(output: ty.Mapping[str, ty.Any]):
    return (output["output"], output["target"])


class SkipBatchError(Exception):
    def __init__(self, batch):
        self.batch = batch


class ClassificationMetrics(ignite.metrics.Metric):
    conf: torch.Tensor

    def __init__(
        self,
        class_names: ty.Iterable,
        output_transform=lambda x: x,
        aggregates: ty.Optional[ty.Dict[str, ty.Collection]] = None,
        device: ty.Union[torch.device, str] = "cpu",
    ):
        self.class_names = list(class_names)
        self.n_classes = len(self.class_names)
        self.device = device
        self.aggregates = (
            {
                k: torch.tensor(
                    [self.class_names.index(c) for c in v], device=self.device
                )
                for k, v in aggregates.items()
            }
            if aggregates is not None
            else dict()
        )
        super().__init__(output_transform)

    def reset(self):
        self.conf = torch.zeros(
            [self.n_classes, self.n_classes], device=self.device, dtype=torch.int64
        )

    def aggregated_prf(self, c: str) -> ty.Tuple[float, float, float]:
        with torch.no_grad():
            class_indices = self.aggregates[c]
            cols_sum = self.conf.index_select(1, class_indices).sum(dim=1)
            tp = cols_sum.index_select(0, class_indices).sum().item()
            tru = cols_sum.sum().item()
            pos = self.conf.index_select(0, class_indices).sum().item()
            P = tp / pos if pos != 0 else 1.0
            R = tp / tru if tru != 0 else 1.0
            f_denom = pos + tru
            F = 2 * tp / (pos + tru) if f_denom != 0 else 1.0
        return (P, R, F)

    def update(self, output: ty.Tuple[torch.Tensor, torch.Tensor]):
        with torch.no_grad():
            scores, target = output
            scores = scores.to(device=self.device)
            target = target.to(device=self.device)
            predicted = scores.argmax(dim=-1)
            conf = confusion(predicted, target, self.n_classes)
            self.conf.add_(conf)

    def compute(self):
        class_prf = PRF(self.conf)
        macro = class_prf.mean(dim=0)
        return {
            "class": {
                c: {"P": P.item(), "R": R.item(), "F": F.item()}
                for c, (P, R, F) in zip(self.class_names, class_prf)
            },
            "aggregates": {
                c: {"P": P, "R": R, "F": F}
                for c in self.aggregates.keys()
                for (P, R, F) in (self.aggregated_prf(c),)
            },
            "macro": {n: v.item() for n, v in zip("PRF", macro)},
            "accuracy": self.conf.diag()
            .sum()
            .to(torch.float)
            .div(self.conf.sum())
            .item(),
        }

    def custom_layout(self, prefix):
        return {
            **{
                f"{prefix}/{c}": [
                    "Multiline",
                    [f"{prefix}/class/{c}/{name}" for name in "PR"],
                ]
                for c in [*self.class_names, *self.aggregates]
            },
            f"{prefix}/macro": [
                "Multiline",
                [f"{prefix}/macro/{name}" for name in "PR"],
            ],
        }


def add_epoch_bar(engine: ignite.engine.Engine, mininterval: float = 1):
    @engine.on(ignite.engine.Events.EPOCH_STARTED)
    def epoch_init(engine):
        if engine.state.max_epochs > 1:
            desc = f"Epoch {engine.state.epoch}/{engine.state.max_epochs}"
        else:
            desc = "Running model"
        if isinstance(engine.state.dataloader, ty.Sized):
            total = len(engine.state.dataloader)
        else:
            total = None

        engine.state.epoch_bar = tqdm.tqdm(
            desc=desc,
            initial=0,
            total=total,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            unit_scale=True,
            mininterval=mininterval,
            disable=None,
        )

    @engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def update_bars(engine):
        engine.state.epoch_bar.update()

    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def epoch_feedback(engine):
        engine.state.epoch_bar.close()


def write_metric(
    writer: tensorboardX.SummaryWriter,
    name: str,
    value: ty.Union[ty.Mapping, ty.Iterable, torch.Tensor, float, str],
    step: int,
):
    if isinstance(value, collections.abc.Mapping):
        for k, v in value.items():
            write_metric(writer, f"{name}/{k}", v, step)
    elif isinstance(value, torch.Tensor) and value.numel() == 1:
        writer.add_scalar(name, value.item(), step)
    elif isinstance(value, collections.abc.Iterable):
        for i, v in enumerate(value):
            write_metric(writer, f"{name}/{i}", v, step)
    else:
        writer.add_scalar(name, value, step)


def write_parameters_and_gradients(
    writer: tensorboardX.SummaryWriter, model: torch.nn.Module, step: int, loss=None
):
    for name, param in model.named_parameters():
        std, mean = torch.std_mean(param.data)
        writer.add_scalar(f"parameter_mean/{name}", mean, step)
        if not torch.isnan(std):
            writer.add_scalar(f"parameter_std/{name}", std, step)
        if param.grad is not None:
            if param.grad.is_sparse:
                grad_data = param.grad.data._values()
            else:
                grad_data = param.grad.data

            # skip empty gradients
            if torch.prod(torch.tensor(grad_data.shape)).item() > 0:
                writer.add_scalar(f"gradient_norm/{name}", grad_data.norm(), step)
                grad_std, grad_mean = torch.std_mean(grad_data)
                writer.add_scalar(f"gradient_mean/{name}", grad_mean, step)
                if not torch.isnan(std):
                    writer.add_scalar(f"gradient_std/{name}", grad_std, step)
                if loss is not None and loss > 0.0:
                    writer.add_scalar(
                        f"gradient_relative_norm/{name}", grad_data.norm() / loss, step
                    )


# TODO: fix docstring
# TODO: allow using multiple dev_metrics
# TODO: the error management facilities should be upstreamed to ignite, vf is ok with it
class SinkTrainer(ignite.engine.Engine):
    """A full-featured trainer

    Parameters
    ----------

    `model`
        The `#torch.nn.Module` to train
    `loss_fun`
        The loss function used to train the model
    `optimizer`
        The optimizer used to train the model (default: ADAMW)
    `dev_loss`
        The loss function used for validation set checkpoints
    `train_metrics`
        Additional training metrics to report
    `dev_metrics`
        Additional validation metrics to report
    `save_path`
        The directory to save model checkpoints to
    `checkpointed_models`
        A dict mapping model names to models that have to be saved
    `summary_interval`
        The number of iterations between training tensorboard summaries
    `debug`
        Report instantaneous per-batch loss, set default logging interval to 1 and don't try to
        recover from exceptions
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fun: ty.Callable[[ty.Iterable], torch.Tensor],
        optimizer: ty.Optional[torch.optim.Optimizer] = None,
        dev_loss: ty.Optional[ty.Callable] = None,
        train_metrics: ty.Optional[ty.Dict[str, ty.Callable]] = None,
        dev_metrics: ty.Optional[ty.Dict[str, ty.Callable]] = None,
        save_path: ty.Optional[ty.Union[str, pathlib.Path]] = None,
        checkpointed_models: ty.Optional[ty.Dict[str, torch.nn.Module]] = None,
        summary_interval: ty.Optional[int] = None,
        custom_tb_layout: ty.Optional[ty.Dict] = None,
        device: ty.Optional[torch.device] = None,
        debug: bool = False,
        load_best_after_training: bool = True,  # FIXME: temp patch
        save_calls: ty.Optional[ty.Collection[ty.Callable[[ignite.engine.State]]]] = None,
    ):
        super().__init__(SinkTrainer._train_and_store_loss)
        self.register_events(*CustomEvents)
        self.model = model
        self.loss_fun = loss_fun
        self.device = device
        self.custom_tb_layout = custom_tb_layout
        self.debug = debug
        self.save_calls = save_calls

        if summary_interval is None:
            if self.debug:
                self.summary_interval = 1
            else:
                self.summary_interval = 100
        else:
            self.summary_interval = summary_interval

        self.save_path = (
            pathlib.Path(save_path) if save_path is not None else pathlib.Path(".")
        )
        self.checkpointed_models = (
            checkpointed_models if checkpointed_models is not None else []
        )

        if optimizer is None:
            optimizer = DenseSparseAdamW(
                filter(lambda x: x.requires_grad, model.named_parameters())
            )
            logger.debug(
                f"Training the following parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}"
            )
        logger.debug(f"Optimizer defaults: {optimizer.defaults}")
        self.optimizer = optimizer

        self.train_metrics = {
            "running_avg_loss": ignite.metrics.RunningAverage(
                output_transform=lambda x: x["loss"], alpha=0.98, epoch_bound=False
            )
        }
        train_metrics = train_metrics if train_metrics is not None else {}
        self.train_metrics.update(train_metrics)

        for name, metric in self.train_metrics.items():
            metric.attach(self, name)

        self.batch_timer = ignite.handlers.Timer(average=True)
        self.batch_timer.attach(
            self,
            start=ignite.engine.Events.EPOCH_STARTED,
            resume=ignite.engine.Events.ITERATION_STARTED,
            pause=ignite.engine.Events.ITERATION_COMPLETED,
            step=ignite.engine.Events.ITERATION_COMPLETED,
        )

        self.epoch_timer = ignite.handlers.Timer(average=False)
        self.epoch_timer.attach(
            self,
            start=ignite.engine.Events.EPOCH_STARTED,
            pause=ignite.engine.Events.EPOCH_COMPLETED,
        )

        self.add_event_handler(
            ignite.engine.Events.EPOCH_STARTED, SinkTrainer._epoch_init
        )
        self.add_event_handler(
            ignite.engine.Events.EPOCH_COMPLETED, SinkTrainer._write_train_metrics
        )
        self.add_event_handler(
            ignite.engine.Events.STARTED, SinkTrainer._setup_train_bar
        )
        self.add_event_handler(
            ignite.engine.Events.COMPLETED, SinkTrainer._teardown_train_bar
        )

        dev_metrics = dev_metrics if dev_metrics is not None else dict()
        if dev_loss is not None:
            train_loss = ignite.metrics.Loss(
                loss_fun, output_transform=extract_output, batch_size=len
            )
            dev_metrics = {"train_loss": train_loss, **dev_metrics}
        else:
            dev_loss = loss_fun

        self.validator = Evaluator(self.model, dev_loss, dev_metrics, self.device)
        self.best_checkpointer = ignite.handlers.ModelCheckpoint(
            dirname=self.save_path,
            filename_prefix="checkpoint",
            score_function=(lambda t: -self.validator.state.metrics["loss"]),
            score_name="score",
            save_as_state_dict=True,
            create_dir=True,
            require_empty=False,
            n_saved=1,
        )
        self.validator.add_event_handler(
            ignite.engine.Events.COMPLETED, self._write_validation_metrics
        )

        def checkpoint_and_keep(validator, models):
            self.best_checkpointer(validator, models)
            # FIXME: this relies on there being only one save model
            self.state.best_model_path = self.best_checkpointer._saved[-1][1][0]

        self.validator.add_event_handler(
            ignite.engine.Events.COMPLETED,
            checkpoint_and_keep,
            self.checkpointed_models,
        )
        if load_best_after_training:
            self.add_event_handler(
                ignite.engine.Events.COMPLETED, SinkTrainer._load_best
            )

        self.restart_checkpointer = ignite.handlers.ModelCheckpoint(
            dirname=self.save_path,
            filename_prefix="checkpoint",
            save_interval=1,
            n_saved=1,
            save_as_state_dict=True,
            create_dir=True,
            require_empty=False,
        )
        self.add_event_handler(
            ignite.engine.Events.EPOCH_COMPLETED,
            self.restart_checkpointer,
            self.checkpointed_models,
        )

        self.add_event_handler(
            ignite.engine.Events.ITERATION_COMPLETED, SinkTrainer._update_bars
        )
        self.add_event_handler(
            ignite.engine.Events.EPOCH_COMPLETED, SinkTrainer._epoch_feedback
        )
        if debug:
            self.add_event_handler(
                ignite.engine.Events.ITERATION_COMPLETED,
                SinkTrainer._update_epoch_bar_desc,
            )

        self.add_event_handler(CustomEvents.LOGGING, SinkTrainer._train_log_tensorboard)
        self.add_event_handler(CustomEvents.LOGGING, SinkTrainer._write_train_metrics)

    def _run_once_on_dataset(self):
        start_time = time.time()

        try:
            for batch in self.state.dataloader:
                try:
                    if self.should_terminate or self.should_terminate_single_epoch:
                        self.should_terminate_single_epoch = False
                        break
                    self.state.batch = batch
                    self.state.iteration += 1
                    self._fire_event(ignite.engine.Events.ITERATION_STARTED)
                    self.state.output = self._process_function(self, batch)
                    self._fire_event(ignite.engine.Events.ITERATION_COMPLETED)
                    if self.state.iteration % self.summary_interval == 0:
                        self.fire_event(CustomEvents.LOGGING)

                except BaseException as e:
                    self._handle_exception(e)
        except BaseException as e:
            self._logger.error(
                "Current epoch is terminating due to exception: %s.", str(e)
            )
            raise e

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)
        return hours, mins, secs

    # TODO: rewrite this with the new detachable event handler API
    def run(
        self,
        train_loader: ty.Iterable,
        max_epochs: int,
        patience: ty.Optional[int] = None,
        dev_loader: ty.Optional[torch.utils.data.DataLoader] = None,
        run_name: ty.Optional[str] = None,
    ):
        """Train on a dataset

        ## Parameters
        - `train_loader`: loader of training samples
        - `max_epochs`: the maximum number of pass over the dataset
        - `dev_loader`: loader of samples used for feedback during training,
          disabled if `None` (default)
        - `patience`: the number of epochs to wait for an improvement
          in `dev_loss`
          before stopping training prematurely
        - `run_name`: the name of the run in tensorboard
        """

        old_mode = self.model.training
        old_device = next(self.model.parameters()).device
        self.model.to(self.device)
        logger.debug(f"Training on {next(self.model.parameters()).device}")

        if patience is not None:
            stopper = ignite.handlers.EarlyStopping(
                patience=patience,
                score_function=(lambda t: -t.state.metrics["dev_loss"]),
                trainer=self,
            )
        else:
            stopper = None

        add_epoch_bar(self, mininterval=0.1 if self.debug else 2.0)

        if run_name is None:
            run_name = f"run{datetime.datetime.now().isoformat(timespec='seconds')}"

        self.state = ignite.engine.State(
            dataloader=train_loader, epoch=0, max_epochs=max_epochs, metrics={}
        )
        self.state.tb_writer = tensorboardX.SummaryWriter(
            logdir=str(self.save_path / "tensorboard" / run_name)
        )
        if self.custom_tb_layout is not None:
            self.state.tb_writer.add_custom_scalars(self.custom_tb_layout)

        try:
            self._logger.info(
                "Engine run starting with max_epochs={}.".format(max_epochs)
            )
            start_time = time.time()
            self._fire_event(ignite.engine.Events.STARTED)
            while self.state.epoch < max_epochs and not self.should_terminate:
                self.state.epoch += 1
                self._fire_event(ignite.engine.Events.EPOCH_STARTED)
                hours, mins, secs = self._run_once_on_dataset()
                self._logger.info(
                    "Epoch[%s] Complete. Time taken: %02d:%02d:%02d",
                    self.state.epoch,
                    hours,
                    mins,
                    secs,
                )
                self._fire_event(ignite.engine.Events.EPOCH_COMPLETED)
                if dev_loader is not None:
                    self._dev_eval(dev_loader)
                    if stopper is not None:
                        stopper(self)
                if self.should_terminate:
                    break

            self._fire_event(ignite.engine.Events.COMPLETED)
            time_taken = time.time() - start_time
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self._logger.info(
                "Engine run complete. Time taken %02d:%02d:%02d" % (hours, mins, secs)
            )

        except BaseException as e:
            self._logger.error(
                "Engine run is terminating due to exception: %s.", str(e)
            )
            raise e
        self.state.train_bar.close
        self.optimizer.zero_grad()
        # Retore the model in its previous state
        self.model.train(old_mode)
        self.model.to(old_device)
        return self.state

    def _train_and_store_loss(self, batch):
        inpt, target = batch
        inpt = datatools.move(inpt, self.device)
        self.optimizer.zero_grad()
        try:
            target = datatools.move(target, self.device)
            output = self.model(inpt)
            batch_loss = self.loss_fun(output, target)
            batch_loss.backward()
            self.optimizer.step()
        except RuntimeError as e:
            if self.debug:
                raise e
            else:
                logger.warning(
                    f"Runtime Error, let's hope it's CUDA OOM and skip this batch : {e}"
                )
                torch.cuda.empty_cache()
                raise SkipBatchError(batch) from e
        return {
            "loss": batch_loss.item(),
            "target": target.detach(),
            "output": output.detach(),
        }

    def _handle_exception(self, e):
        if (
            isinstance(e, KeyboardInterrupt)
            and (self.state.iteration > 1)
            and not self.should_terminate
        ):
            try:
                # In interactive context, CTRL-C asks to save an intermediate model instead of
                # halting alltogether. A second CTRL-C stops all
                if sys.stdout.isatty() and self.save_calls is not None:
                    print()
                    while True:
                        ans = input(
                            "Save an intermediate step (SIGINT to abort all) [Yn] "
                        )
                        if not ans or "yes".startswith(ans.lower()):
                            for c in self.save_calls:
                                c(self.state)
                            break
                        elif not ans or "no".startswith(ans.lower()):
                            return
                        print(f"{ans!r} is not a valid answer")
                    return
            except KeyboardInterrupt:
                pass
            self.terminate()
            logger.warning(
                "KeyboardInterrupt caught. Saving checkpointed models before exiting"
            )
            self.restart_checkpointer(self, self.checkpointed_models)
            raise e
        elif isinstance(e, SkipBatchError):
            log_path = self.save_path / "skipped_batch.log"
            if self.debug:
                logger.debug(f"Logging skipped batch to {log_path}")
                with open(log_path, "a") as errlog:
                    cause = e.__cause__
                    errlog.write(f"## {type(cause).__name__}\n{cause}\n")
                    errlog.write(str(e.batch))
                    errlog.write("\n")
                raise e
        else:
            if ignite.engine.Events.EXCEPTION_RAISED in self._event_handlers:
                self._fire_event(ignite.engine.Events.EXCEPTION_RAISED, e)
            else:
                self._logger.error(
                    "Current run is terminating due to exception: %s.", str(e)
                )
                raise e

    def _epoch_init(self):
        self.model.train()
        self.state.epoch_iteration = 0
        self.state.epoch_loss = torch.zeros(1, device=self.device)

    def _update_bars(self):
        self.state.train_bar.update()

    def _update_epoch_bar_desc(self):
        avg_loss = self.state.metrics["running_avg_loss"]
        self.state.epoch_bar.set_postfix(
            loss=(f'{self.state.output["loss"]:.5f}|' f"{avg_loss:.5f}")
        )

    def _train_log_tensorboard(self):
        write_parameters_and_gradients(
            self.state.tb_writer,
            self.model,
            self.state.iteration,
            self.state.output["loss"],
        )
        lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.state.tb_writer.add_scalar(
            "train/learning_rate", math.fsum(lrs) / len(lrs), self.state.iteration
        )

    def _epoch_feedback(self):
        self.state.train_bar.clear()
        avg_loss = self.state.metrics["running_avg_loss"]
        logger.info(
            f"Running average train set loss for epoch {self.state.epoch}: "
            f"{avg_loss:.5f}"
        )
        # FIXME: this only measures the average batch processing time but it does not take the
        # dataloading time into account (see the definition of `batch_timer`)
        elapsed = datetime.timedelta(seconds=math.ceil(self.epoch_timer.value()))
        batch_average = self.batch_timer.value()
        if batch_average > 1:
            logger.info(
                f"Epoch {self.state.epoch} duration: "
                f"{elapsed} ({self.batch_timer.value():.2f} s/batch)"
            )
        else:
            logger.info(
                f"Epoch {self.state.epoch} duration: "
                f"{elapsed} ({1/self.batch_timer.value():.2f} batch/s)"
            )

    def _dev_eval(self, dev_loader: torch.utils.data.DataLoader):
        validator_state = self.validator.run(dev_loader)

        for name, value in validator_state.metrics.items():
            logger.info(f"Dev {name} for epoch {self.state.epoch}: {value}")
            write_metric(
                self.state.tb_writer, f"validation/{name}", value, self.state.iteration
            )

        self.state.metrics["dev_loss"] = validator_state.metrics["loss"]

    def _write_train_metrics(self):
        for name, value in self.state.metrics.items():
            write_metric(
                self.state.tb_writer, f"train/{name}", value, self.state.iteration
            )

    def _write_validation_metrics(self, validator):
        for name, value in validator.state.metrics.items():
            write_metric(
                self.state.tb_writer, f"validation/{name}", value, self.state.iteration
            )

    def _load_best(self):
        logger.debug(
            f"Loading the dev-best parameters from {self.state.best_model_path}"
        )
        best_state_dict = torch.load(self.state.best_model_path)
        self.model.load_state_dict(best_state_dict)

    def _setup_train_bar(self):
        self.state.train_bar = tqdm.tqdm(
            total=len(self.state.dataloader) * self.state.max_epochs,
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
            leave=False,
            unit_scale=1 / len(self.state.dataloader),
            mininterval=1.0,
            bar_format="{l_bar}{bar}| {n:.02f}/{total:.00f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            disable=None,
        )

    def _teardown_train_bar(self):
        self.state.train_bar.close()


class PilotableSinkTrainer(SinkTrainer):
    """A `SinkTraine` that can be ran step by step."""

    def _run_once_on_dataset(self):
        start_time = time.time()

        try:
            for batch in self.state.dataloader:
                try:
                    if self.should_terminate or self.should_terminate_single_epoch:
                        self.should_terminate_single_epoch = False
                        break
                    self.state.batch = batch
                    self.state.iteration += 1
                    self._fire_event(ignite.engine.Events.ITERATION_STARTED)
                    self.state.output = self._process_function(self, batch)
                    self._fire_event(ignite.engine.Events.ITERATION_COMPLETED)
                    if self.state.iteration % self.summary_interval == 0:
                        self.fire_event(CustomEvents.LOGGING)
                    yield

                except BaseException as e:
                    self._handle_exception(e)
        except BaseException as e:
            self._logger.error(
                "Current run is terminating due to exception: %s.", str(e)
            )
            raise e

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)
        return hours, mins, secs

    def run(
        self,
        train_loader: ty.Iterable,
        max_epochs: int,
        patience: ty.Optional[int] = None,
        dev_loader: ty.Optional[torch.utils.data.DataLoader] = None,
        run_name: ty.Optional[str] = None,
    ):
        """Train on a dataset

        ## Parameters
        - `train_loader`: loader of training samples
        - `max_epochs`: the maximum number of pass over the dataset
        - `dev_loader`: loader of samples used for feedback during training,
          disabled if `None` (default)
        - `patience`: the number of epochs to wait for an improvement
          in `dev_loss`
          before stopping training prematurely
        - `run_name`: the name of the run in tensorboard
        """

        old_mode = self.model.training
        old_device = next(self.model.parameters()).device
        self.model.to(self.device)

        if patience is not None:
            stopper = ignite.handlers.EarlyStopping(
                patience=patience,
                score_function=(lambda t: -t.state.metrics["dev_loss"]),
                trainer=self,
            )
        else:
            stopper = None

        add_epoch_bar(self, mininterval=0.1 if self.debug else 2.0)

        if run_name is None:
            run_name = f"run{datetime.datetime.now().isoformat(timespec='seconds')}"

        self.state = ignite.engine.State(
            dataloader=train_loader, epoch=0, max_epochs=max_epochs, metrics={}
        )
        self.state.tb_writer = tensorboardX.SummaryWriter(
            logdir=str(self.save_path / "tensorboard" / run_name)
        )
        if self.custom_tb_layout is not None:
            self.state.tb_writer.add_custom_scalars(self.custom_tb_layout)

        try:
            self._logger.info(
                "Engine run starting with max_epochs={}.".format(max_epochs)
            )
            start_time = time.time()
            self._fire_event(ignite.engine.Events.STARTED)
            while self.state.epoch < max_epochs and not self.should_terminate:
                self.state.epoch += 1
                self._fire_event(ignite.engine.Events.EPOCH_STARTED)
                hours, mins, secs = yield from self._run_once_on_dataset()
                logger.debug(f"End epoch {self.state.epoch} for {run_name}")
                self._logger.info(
                    "Epoch[%s] Complete. Time taken: %02d:%02d:%02d",
                    self.state.epoch,
                    hours,
                    mins,
                    secs,
                )
                self._fire_event(ignite.engine.Events.EPOCH_COMPLETED)
                if dev_loader is not None:
                    self._dev_eval(dev_loader)
                    if stopper is not None:
                        stopper(self)
                if self.should_terminate:
                    logger.debug(f"Forcibly terminating {run_name}")
                    break

            self._fire_event(ignite.engine.Events.COMPLETED)
            time_taken = time.time() - start_time
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self._logger.info(
                "Engine run complete. Time taken %02d:%02d:%02d" % (hours, mins, secs)
            )

        except BaseException as e:
            self._logger.error(
                "Engine run is terminating due to exception: %s.", str(e)
            )
            self._handle_exception(e)
        self.state.train_bar.close
        self.optimizer.zero_grad()
        # Retore the model in its previous state
        self.model.train(old_mode)
        self.model.to(old_device)
        return self.state


def run_multi(
    tasks: ty.Mapping[
        str, ty.Tuple[PilotableSinkTrainer, ty.Iterable, ty.Dict[str, ty.Any]]
    ],
    steps_per_task: ty.Optional[ty.Mapping[str, int]] = None,
) -> ty.Dict[str, ignite.engine.State]:
    """Run several `PilotableSinkTrainer`s in parallel

    This works by cycling through the trainers, runnning one (or more,, see
    #Arguments) step of each. When one trainers ends, all are forcibly stopped
    by setting `should_terminate` to `True`.

    ## Arguments
      - `task` A mapping from tasks names to (trainer, dataset, kwargs) tuples
      - `steps_per_task` A mapping from task names to the number of steps
        to run on the corresponding tasks before switching to the next one,
        defaults to 1 for every task

    ## Return
      - `states`: A mapping from task names to the final trainer state.
    """
    if steps_per_task is None:
        steps_per_task = {t: 1 for t in tasks}
    runs = {t: r.run(l, **kwargs) for t, (r, l, kwargs) in tasks.items()}
    while True:
        try:
            for (
                task,
                run,
            ) in runs.items():  # The order should be deterministic as of python 3.7
                for _ in range(steps_per_task[task]):
                    next(run)
        except StopIteration as e:
            logger.debug(f"{task!r} run ended, terminating the other tasks")
            states = {task: e.value}
            for t, (r, l, k) in tasks.items():
                if t == task:
                    continue
                logger.debug(f"Terminating {t!r}")
                r.should_terminate = True
                try:
                    next(runs[t])
                except StopIteration as e:
                    states[t] = e.value
            break
    return states


class Evaluator(ignite.engine.Engine):
    """A custom evaluating engine.

    No device is enforced for the source and targets, so the loss and metrics
    must ensure that they are moved on their preferred devices.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: ty.Callable,
        metrics: ty.Optional[ty.Mapping[str, ignite.metrics.Metric]] = None,
        device: ty.Optional[torch.device] = None,
    ):
        super().__init__(Evaluator._inference)
        self.model = model
        self.device = device
        self.loss = ignite.metrics.Loss(
            loss, output_transform=extract_output, batch_size=len
        )
        self.metrics = {"loss": self.loss}
        if metrics is not None:
            self.metrics.update(metrics)
        for name, metric in self.metrics.items():
            metric.attach(self, name)
        add_epoch_bar(self)

    def _run_once_on_dataset(self):
        start_time = time.time()
        try:
            for batch in self.state.dataloader:
                try:
                    if self.should_terminate or self.should_terminate_single_epoch:
                        self.should_terminate_single_epoch = False
                        break
                    self.state.batch = batch
                    self.state.iteration += 1
                    self._fire_event(ignite.engine.Events.ITERATION_STARTED)
                    self.state.output = self._process_function(self, batch)
                    self._fire_event(ignite.engine.Events.ITERATION_COMPLETED)

                except BaseException as e:
                    self._handle_exception(e)
        except BaseException as e:
            self._logger.error(
                "Current run is terminating due to exception: %s.", str(e)
            )
            raise e
        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)
        return hours, mins, secs

    def _inference(self, batch):
        cpu_inpt, cpu_target = batch
        try:
            inpt = datatools.move(cpu_inpt, self.device)
            with torch.no_grad():
                output = self.model(inpt)
        except RuntimeError as e:
            logger.error(f"RuntimeError ({e}) with batch\n{batch}")
            raise e
        return {"target": cpu_target, "output": output}

    def _handle_exception(self, e):
        if (
            isinstance(e, KeyboardInterrupt)
            and (self.state.iteration > 1)
            and not self.should_terminate
        ):
            self.terminate()
            logger.warning("KeyboardInterrupt caught")
            raise e
        elif isinstance(e, SkipBatchError):
            cause = e.__cause__
            logger.debug(f"Skipped batch due to {type(cause).__name__}: {cause}")
        else:
            if ignite.engine.Events.EXCEPTION_RAISED in self._event_handlers:
                self._fire_event(ignite.engine.Events.EXCEPTION_RAISED, e)
            else:
                self._logger.error(
                    "Current run is terminating due to exception: %s.", str(e)
                )
                raise e

    def run(self, dataloader, max_epochs=1):
        old_mode = self.model.training
        self.model.eval()
        old_device = next(self.model.parameters()).device
        self.model.to(self.device)
        super().run(dataloader, max_epochs)
        self.model.to(old_device)
        self.model.train(old_mode)
        return self.state


def run_model(
    model: torch.nn.Module,
    data: ty.Iterable[ty.Union[T, ty.Sequence[T]]],
    prepare_batch: ty.Optional[ty.Callable[[ty.Sequence[T]], ty.Any]] = None,
    batch_size: ty.Optional[int] = None,
    join: ty.Union[
        str, ty.Callable[[ty.Iterable[ty.Union[T, ty.Sequence[T]]]], ty.Any]
    ] = "cat",
    data_len: ty.Optional[int] = None,
):
    """Run a model on a dataset.

    ## Parameters
    - `model`: the model to run
    - `data`: an iterable over either samples or batch of samples
    - `prepare_batch`: if set, this is applied to all the batches before they are fed to `model`
    - `batch_size`: is set, the samples coming from `data` will be grouped in lists of at most `batch_size`
    - `join`: this will be applied to the list of all batch outputs.
    """
    device = next(model.parameters()).device
    if batch_size is not None:
        if data_len is None and isinstance(data, ty.Sized):
            data_len = (len(data) - 1) // batch_size + 1
        data = itu.chunked_iter(data, batch_size)

    if prepare_batch is not None:
        data = map(prepare_batch, ty.cast(ty.Iterable[ty.Sequence[T]], data))

    outpt = []
    pbar = tqdm.tqdm(
        data,
        total=data_len,
        unit="batch",
        desc="Running model",
        mininterval=2,
        unit_scale=True,
        dynamic_ncols=True,
        leave=False,
        disable=None,
    )
    for d in pbar:
        r = model(datatools.move(d, device))
        outpt.append(r)

    if join == "cat":
        return torch.cat(outpt, dim=0)
    elif join == "chain":
        return it.chain.from_iterable(outpt)
    else:
        return ty.cast(
            ty.Callable[[ty.Iterable[ty.Union[T, ty.Sequence[T]]]], ty.Any], join
        )(outpt)


class MultiLoss(ignite.metrics.Metric):
    """like `#ignite.metrics.Loss`, but accepts tensor-valued loss functions."""

    def __init__(
        self,
        loss_fn: ty.Callable[..., torch.Tensor],
        num_loss: ty.Optional[int] = None,
        output_transform=lambda x: x,
        batch_size: ty.Callable[[torch.Tensor], int] = lambda x: x.shape[0],
        device: ty.Optional[torch.device] = None,
        averaged: bool = True,
        loss_names: ty.Optional[ty.Sequence[str]] = None,
    ):
        self._loss_fn = loss_fn
        if num_loss is not None:
            self._num_loss = num_loss
        else:
            if loss_names is None:
                raise ValueError("Either `num_loss` or `loss_names` must be specified")
            self._num_loss = len(loss_names)
        self._batch_size = batch_size
        self.device = device
        self.averaged = averaged
        self.loss_names = tuple(loss_names) if loss_names is not None else None
        super().__init__(output_transform)

    def reset(self):
        self._sum = torch.zeros(self._num_loss, device=self.device)
        self._num_examples = torch.zeros(
            self._num_loss, dtype=torch.long, device=self.device
        )

    # TODO: figure out a way to unit test this
    def update(self, output):
        with torch.no_grad():
            if len(output) == 2:
                y_pred, y = output
                kwargs = {}
            else:
                y_pred, y, kwargs = output
            loss = self._loss_fn(y_pred, y, **kwargs).to(dtype=torch.float)
            nans = torch.isnan(loss)
            loss = torch.where(
                nans, torch.tensor(0.0, dtype=torch.float, device=self.device), loss
            )
            N = self._batch_size(y)
            counts = torch.where(
                nans,
                torch.tensor(0, dtype=torch.long, device=self.device),
                torch.tensor(N, dtype=torch.long, device=self.device),
            )
            if self.averaged:
                self._sum.addcmul_(counts.to(dtype=torch.float), loss)
            else:
                self._sum.add_(loss)
            self._num_examples.add_(counts)

    def compute(self):
        final_loss = self._sum / self._num_examples.to(dtype=torch.float)
        if self.loss_names is not None:
            return dict(zip(self.loss_names, final_loss.tolist()))
        return final_loss
