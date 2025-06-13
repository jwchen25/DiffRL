import os
from pathlib import Path
from typing import List, Literal
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import numpy as np
from numpy.typing import NDArray
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.suite.base import ModelSuite
from models.diffcsp.diffusion import DiffCSPModule
from models.diffcsp.sample import DiffCSPSampler
from models.diffcsp.finetune import DiffCSPDataset


AVA_MODEL_NAME = Literal[
    "diffcsp",
]


class DiffCSPSuite(ModelSuite):
    def __init__(
        self,
        model_name: AVA_MODEL_NAME,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        model_path: str | None = None,
        config_overrides: list[str] = [],
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            model_path=model_path,
            config_overrides=config_overrides,
            device=device,
            **kwargs,
        )

    def load_model(self):
        model_path = Path(os.path.abspath(self.model_path))
        with initialize_config_dir(str(model_path), version_base="1.1"):
            cfg = compose(config_name="hparams")
            cfg.model._target_ = "models.diffcsp.diffusion.DiffCSPModule"
            model = hydra.utils.instantiate(
                cfg.model,
                optim=cfg.optim,
                _recursive_=False,
            )
            ckpts = list(model_path.glob("*.ckpt"))
            if len(ckpts) > 0:
                ckpt = None
                for ck in ckpts:
                    if "last" in ck.parts[-1]:
                        ckpt = str(ck)
                if ckpt is None:
                    ckpt_epochs = np.array(
                        [
                            int(ckpt.parts[-1].split("-")[0].split("=")[1])
                            for ckpt in ckpts
                            if "last" not in ckpt.parts[-1]
                        ]
                    )
                    ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            hparams = os.path.join(str(model_path), "hparams.yaml")
            model = model.load_from_checkpoint(
                ckpt, hparams_file=hparams, strict=False,
            )
            try:
                model.lattice_scaler = torch.load(
                    os.path.join(model_path, "lattice_scaler.pt")
                )
                model.scaler = torch.load(
                    os.path.join(model_path, "prop_scaler.pt")
                )
            except:
                pass

        model.config = cfg
        return model

    def get_sampler(self):
        sampler = DiffCSPSampler(
            batch_size=self.sample_cfg.batch_size,
            num_batches=self.sample_cfg.num_batches,
        )
        return sampler

    def get_dataloader(
        self,
        samples: List[Data],
        rewards: NDArray | None,
        batch_size: int | None = None,
        shuffle: bool = True,
    ):
        if batch_size is None:
            batch_size = self.finetune_cfg.batch_size
        dataset = DiffCSPDataset(samples, rewards)
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
        )
        return dataloader

    def save_model(
        self,
        model: DiffCSPModule,
        save_dir: str,
    ):
        os.makedirs(save_dir, exist_ok=True)
        cfg = model.config
        ckpt_dict = {
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True)
        }
        torch.save(ckpt_dict, os.path.join(save_dir, "last.ckpt"))
        OmegaConf.save(cfg, os.path.join(save_dir, "hparams.yaml"))
