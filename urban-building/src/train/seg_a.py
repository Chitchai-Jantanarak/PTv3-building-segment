from pathlib import Path

import torch
from omegaconf import DictConfig

from src.core.utils import get_logger, load_pretrained_encoder, set_seed
from src.datasets import build_dataloader
from src.losses import focal_loss
from src.models.seg_heads import SegAModel, generate_pseudo_labels
from src.train._base import build_optimizer, build_scheduler, train_loop


def train_seg_a(cfg: DictConfig) -> None:
    logger = get_logger("SEGA")
    logger.info("Starting Seg-A training")

    set_seed(cfg.run.seed)

    model = SegAModel(cfg)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    mae_ckpt = cfg.task.get("pretrained_encoder", None)
    if mae_ckpt:
        n_loaded = load_pretrained_encoder(model, mae_ckpt)
        logger.info(f"Loaded {n_loaded} encoder weight tensors from MAE: {mae_ckpt}")
    else:
        logger.warning("No pretrained_encoder specified — training from scratch!")

    logger.info(f"Encoder frozen: {cfg.task.freeze_encoder}")
    logger.info(f"RGB injection: {cfg.task.get('use_rgb', False)}")

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    alpha = None
    weight_method = cfg.task.loss.get("weight_method", None)
    if weight_method:
        dataset = train_loader.dataset
        if hasattr(dataset, "get_class_weights"):
            alpha = dataset.get_class_weights(method=weight_method)
            logger.info(f"Class weights ({weight_method}): {alpha.tolist()}")

            building_boost = cfg.task.loss.get("building_boost", 1.0)
            if building_boost != 1.0:
                building_cls = cfg.task.loss.get("building_class", 2)
                alpha[building_cls] *= building_boost
                alpha = alpha / alpha.mean()
                logger.info(
                    f"Building class {building_cls} weight boosted {building_boost}x → "
                    f"final weights: {alpha.tolist()}"
                )
        else:
            logger.warning("Dataset does not support get_class_weights(), training without class weights")

    gamma = cfg.task.loss.get("gamma", 2.0)

    def criterion(model, batch, device):
        feat = batch["points"].to(device)
        coord = batch["coords"].to(device)
        batch_idx = batch["batch"].to(device)

        rgb = batch.get("rgb")
        if rgb is not None:
            rgb = rgb.to(device)

        if "labels" in batch and batch["labels"] is not None:
            labels = batch["labels"].to(device)
        else:
            rel_z = feat[:, 3]
            labels = generate_pseudo_labels(coord, rel_z)

        output = model(feat, coord, batch_idx, rgb=rgb)
        logits = output["logits"]
        n_classes = logits.shape[-1]

        ignore_index = -100
        labels = labels.clone()
        invalid = (labels < 0) | (labels >= n_classes)
        labels[invalid] = ignore_index

        _alpha = alpha.to(device) if alpha is not None else None
        loss = focal_loss(logits, labels, alpha=_alpha, gamma=gamma, ignore_index=ignore_index)
        return loss

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    result = train_loop(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        logger=logger,
    )
    
    # Post-training evaluation
    from src.eval import run_evaluation

    device = torch.device(cfg.run.device)
    out_dir = Path(cfg.paths.ckpt_root) / cfg.task.name
    run_evaluation(
        task="seg_a",
        model=result.model,
        val_loader=val_loader,
        device=device,
        out_dir=out_dir,
        train_losses=result.train_losses,
        val_losses=result.val_losses,
        cfg=cfg,
    )

    logger.info("Seg-A training complete")


train = train_seg_a
