# src/train/seg_a.py
from omegaconf import DictConfig

from src.core.utils import get_logger, load_pretrained_encoder, set_seed
from src.datasets import build_dataloader
from src.losses import focal_loss
from src.models.seg_heads import SegAModel, generate_pseudo_labels
from src.train._base import build_optimizer, build_scheduler, train_loop


def seg_a_criterion(model, batch, device):
    feat = batch["points"].to(device)
    coord = batch["coords"].to(device)
    batch_idx = batch["batch"].to(device)

    if "labels" in batch and batch["labels"] is not None:
        labels = batch["labels"].to(device)
    else:
        rel_z = feat[:, 3]
        labels = generate_pseudo_labels(coord, rel_z)

    output = model(feat, coord, batch_idx)
    logits = output["logits"]

    loss = focal_loss(logits, labels, gamma=2.0, ignore_index=-1)
    return loss


def train_seg_a(cfg: DictConfig) -> None:
    logger = get_logger("SEGA")
    logger.info("Starting Seg-A training")

    set_seed(cfg.run.seed)

    model = SegAModel(cfg)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load MAE pretrained encoder weights
    mae_ckpt = cfg.task.get("pretrained_encoder", None)
    if mae_ckpt:
        n_loaded = load_pretrained_encoder(model, mae_ckpt)
        logger.info(f"Loaded {n_loaded} encoder weight tensors from MAE: {mae_ckpt}")
    else:
        logger.warning("No pretrained_encoder specified — training from scratch!")

    logger.info(f"Encoder frozen: {cfg.task.freeze_encoder}")

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    model = train_loop(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=seg_a_criterion,
        logger=logger,
    )

    logger.info("Seg-A training complete")


train = train_seg_a
