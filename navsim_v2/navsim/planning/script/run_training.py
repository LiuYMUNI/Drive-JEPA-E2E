from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    synthetic_scenes_path = Path(cfg.synthetic_scenes_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        synthetic_scenes_path=synthetic_scenes_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        synthetic_scenes_path=synthetic_scenes_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    print("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True, drop_last=True)
    print(f"Num training samples: {len(train_data)}")
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False, drop_last=True)
    print(f"Num val samples: {len(val_data)}")

    print('Building swanlab logger')
    log_config = {
        'agent': OmegaConf.to_container(cfg.agent, resolve=True),
        'trainer': OmegaConf.to_container(cfg.trainer, resolve=True)
    }
    wandb_logger = WandbLogger(name=cfg.experiment_name, save_dir=cfg.output_dir + '/wandb', project='navsim')
    wandb_logger.log_hyperparams(log_config)

    pdm_supervision = getattr(getattr(agent, "_config", None), "pdm_supervision", True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.output_dir + "/checkpoints/",
        save_top_k=1,
        monitor='val/score_epoch' if pdm_supervision else 'val/loss_epoch',
        filename='best_{epoch}',
        mode="max" if pdm_supervision else "min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = agent.get_training_callbacks() + [checkpoint_cb, lr_monitor]

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, logger=wandb_logger, callbacks=callbacks)

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        # ckpt_path='/home/linhan/yinlin/projects/navsim_workspace/exp/train_ipad_vjepa_v2_topk/2025.12.03.02.55.01/checkpoints/best_epoch=19.ckpt',
        # ckpt_path='/home/linhan/yinlin/projects/navsim_workspace/exp/train_ipad_vjepa_v2_baseline_2/2025.12.02.21.33.08/checkpoints/best_epoch=13.ckpt'
    )


if __name__ == "__main__":
    main()
