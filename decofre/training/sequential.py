from collections import defaultdict

from loguru import logger

# FIXME: change this to use torch.optim.AdamW as soon as
# the support is the same
from decofre.optimizers import DenseSparseAdamW

from decofre.tasks import antecedent_scoring, mention_detection


def train(det, cor, config, out_dir, device, num_workers: int = 0, debug: bool = False):
    if config.get("mention-detection", {}).get("epochs", 0) > 0:
        det_config = defaultdict(lambda: None, config["mention-detection"])
        logger.debug(
            f"Training the following parameters: {[n for n, p in det.model.named_parameters() if p.requires_grad]}"
        )
        try:
            mention_detection.train_eval(
                model=det.model,
                train_file=det_config["train-file"],
                span_digitizer=det.digitize_span,
                types_lex=det.span_types_lexicon,
                out_dir=out_dir,
                epochs=det_config["epochs"],
                train_batch_size=det_config["train-batch-size"],
                eval_batch_size=det_config["eval-batch-size"],
                device=device,
                mention_boost=det_config["mention-boost"],
                dev_file=det_config["dev-file"],
                test_file=det_config["test-file"],
                patience=det_config["patience"],
                debug=debug,
                config=config["mention-detection"],
                save_calls=[
                    lambda s: det.save(
                        out_dir / f"detector-interrupt-{s.epoch}-{s.iteration}.model"
                    )
                ],
            )
            det.save(out_dir / "detector.model")
        except BaseException as e:
            if not debug:
                logger.warning("Exception caught: saving models before exiting")
                det.save(out_dir / "detector-interrupt.model")
            raise e

    if config.get("antecedent-scoring", {}).get("epochs", 0) > 0:
        cor_config = defaultdict(lambda: None, config["antecedent-scoring"])
        cor_optim = DenseSparseAdamW(
            filter(lambda x: x.requires_grad, cor.model.parameters()),
            lr=cor_config["lr"],
            weight_decay=cor_config["weight-decay"],
        )
        logger.debug(
            f"Training the following parameters: {[n for n, p in cor.model.named_parameters() if p.requires_grad]}"
        )
        try:
            antecedent_scoring.train_eval(
                model=cor.model,
                train_file=cor_config["train-file"],
                span_digitizer=cor.digitize_span,
                pair_feats_digitizer=cor.get_pair_feats,
                out_dir=out_dir,
                epochs=cor_config["epochs"],
                device=device,
                dev_file=cor_config["dev-file"],
                test_file=cor_config["test-file"],
                patience=cor_config["patience"],
                optimizer=cor_optim,
                debug=debug,
                save_calls=[
                    lambda s: det.save(
                        out_dir / f"coref-interrupt-{s.epoch}-{s.iteration}.model"
                    )
                ],
            )
            cor.save(out_dir / "coref.model")
        except BaseException as e:
            logger.warning("Exception caught: saving models before exiting")
            cor.save(out_dir / "coref-interrupt.model")
            raise e
