import tempfile

from collections import defaultdict

import torch
import ignite.engine

from loguru import logger

from decofre import libdecofre, runners

# FIXME: change this to use torch.optim.AdamW as soon as
# the support is the same
from decofre.optimizers import DenseSparseAdamW
from decofre.tasks import antecedent_scoring, mention_detection


# FIXME: generate separate dev-best models
# FIXME: early stopping on **both** scores
# TODO: better progress display
def train(det, cor, config, out_dir, device, num_workers: int = 0, debug: bool = False):
    det_config = defaultdict(lambda: None, config["mention-detection"])
    cor_config = defaultdict(lambda: None, config["antecedent-scoring"])
    cor_optimizer = DenseSparseAdamW(
        filter(lambda x: x.requires_grad, cor.model.parameters()),
        lr=cor_config["lr"],
        weight_decay=cor_config["weight-decay"],
    )
    with tempfile.TemporaryDirectory(prefix="decofre_") as temp_dir:
        logger.info(f"Using tempdir {temp_dir}")
        det_setting = mention_detection.train_det(
            model=det.model,
            train_file=det_config["train-file"],
            span_digitizer=det.digitize_span,
            types_lex=det.span_types_lexicon,
            out_dir=out_dir,
            temp_dir=temp_dir,
            epochs=det_config["epochs"],
            train_batch_size=det_config["train-batch-size"],
            eval_batch_size=det_config["eval-batch-size"],
            device=device,
            mention_boost=det_config["mention-boost"],
            dev_file=det_config["dev-file"],
            patience=det_config["patience"],
            trainer_cls=runners.PilotableSinkTrainer,
            num_workers=num_workers,
            debug=debug,
            config=config["mention-detection"],
            save_calls=[
                lambda s: det.save(
                    out_dir / f"detector-interrupt-{s.epoch}-{s.iteration}.model"
                )
            ],
        )
        det_setting[0].add_event_handler(
            ignite.engine.Events.COMPLETED,
            lambda e, *args, **kwargs: det.save(out_dir / "detector.model"),
        )
        cor_setting = antecedent_scoring.train_cor(
            model=cor.model,
            train_file=cor_config["train-file"],
            span_digitizer=cor.digitize_span,
            pair_feats_digitizer=cor.get_pair_feats,
            out_dir=out_dir,
            temp_dir=temp_dir,
            epochs=cor_config["epochs"],
            train_batch_size=cor_config["train-batch-size"],
            device=device,
            dev_file=cor_config["dev-file"],
            patience=cor_config["patience"],
            optimizer=cor_optimizer,
            trainer_cls=runners.PilotableSinkTrainer,
            debug=debug,
            save_calls=[
                lambda s: det.save(
                    out_dir / f"coref-interrupt-{s.epoch}-{s.iteration}.model"
                )
            ],
        )
        cor_setting[0].add_event_handler(
            ignite.engine.Events.COMPLETED,
            lambda e, *args, **kwargs: cor.save(out_dir / "coref.model"),
        )

        try:
            states = runners.run_multi(
                {"detection": det_setting, "antecedent": cor_setting},
                steps_per_task=config["training-scheme"]["steps"],
            )
        except BaseException as e:
            logger.warning("Exception caught: saving models before exiting")
            det.save(out_dir / "detector-interrupt.model")
            cor.save(out_dir / "coref-interrupt.model")
            raise e

    logger.info("Loading best detection model and evaluating on test")
    det_best_state_dict = torch.load(states["detection"].best_model_path)
    det.model.load_state_dict(det_best_state_dict)
    det_loss = mention_detection.evaluate_classif(
        model=det.model,
        test_file=det_config["test-file"],
        span_digitizer=det.digitize_span,
        classes_lex=det.span_types_lexicon,
        loss_fun=torch.nn.functional.nll_loss,
        device=device,
    )
    logger.info(f"Det loss for test set: {det_loss}")

    logger.info("Loading best coreference model and evaluating on test")
    cor_best_state_dict = torch.load(states["antecedent"].best_model_path)
    cor.model.load_state_dict(cor_best_state_dict)
    cor_loss = antecedent_scoring.evaluate_cor(
        model=cor.model,
        test_file=cor_config["test-file"],
        span_digitizer=cor.digitize_span,
        pair_feats_digitizer=cor.get_pair_feats,
        loss_fun=libdecofre.masked_multi_cross_entropy,
        device=device,
    )
    logger.info(f"Cor loss for test set: {cor_loss}")
