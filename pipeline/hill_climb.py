import os
import time
from pathlib import Path
import logging
from typing import Union, Tuple, List, Dict
import numpy as np
import torch
from omegaconf import DictConfig

from pipeline.base import ReinL
from pipeline.filters import invalid_filter
from pipeline.utils.save import save_structures
from pipeline.utils.logger import Logger
from models.suite.base import ModelSuite
from rewards.reward import Reward


class HillClimb(ReinL):
    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        topk_ratio: float,
        save_dir: str,
        save_freq: int,
        device: str = None,
        logger: Logger = None,
        replay: bool = False,
        replay_args: Dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            rl_epoch=rl_epoch,
            model_suite=model_suite,
            reward=reward,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            save_dir=save_dir,
            save_freq=save_freq,
            device=device,
            logger=logger,
            replay=replay,
            replay_args=replay_args,
            **kwargs,
        )
        assert topk_ratio > 0.0 and topk_ratio < 1.0
        self.topk_ratio = topk_ratio
        self.load_model()

    def load_model(self):
        self.agent = self.model_suite.load_model()
        for param in self.agent.parameters():
            param.requires_grad = True
        self.agent.to(self.device)

    def sample_step(self):
        sample_data, sample_struc = self.sampler.generate(
            model=self.agent, **self.sample_cfg,
        )
        # Filter invalid samples
        sample_data, sample_struc = invalid_filter(sample_data, sample_struc)

        # save all generated valid structures
        valid_xyz_path = save_structures(
            structures=sample_struc,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}_valid.extxyz',
        )

        # MLIP relaxation
        if self.sample_cfg.get('mlip_opt'):
            mlip_opt = self.sample_cfg.mlip_opt
            sample_struc, energies = mlip_opt(sample_struc, valid_xyz_path)
        else:
            energies = None

        # Filter bad samples by selected metrics
        if self.sample_cfg.get('filter'):
            filter = self.sample_cfg.filter
            sample_data, sample_struc, metrics = filter(
                sample_data, sample_struc, energies,
            )
            logging.info(f'Number of filtered samples: {len(sample_struc)}')
            log_str = [f'{k}: {v:.6f}' for k, v in metrics.items()]
            logging.info(', '.join(log_str))

        # max sample size to score/reward
        if self.sample_cfg.get('max_num'):
            max_num = self.sample_cfg.max_num
            if len(sample_struc) > max_num:
                sample_data = sample_data[:max_num]
                sample_struc = sample_struc[:max_num]

        # save structures for evaluation
        eval_xyz_path = save_structures(
            structures=sample_struc,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}_eval.extxyz',
        )

        return sample_data, sample_struc, eval_xyz_path

    def ft_step(self, data_list):
        cfg = self.finetune_cfg
        loader = self.model_suite.get_dataloader(
            samples=data_list,
            rewards=None,
            batch_size=cfg.batch_size,
        )

        # model = model.to(args.device)
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=cfg.lr)

        for epoch in range(cfg.epochs):
            logging.info(f"Epoch {epoch} starts:")
            self.agent.train()
            loss_all = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss, loss_dict = self.agent.ft_step(batch)
                loss.backward()
                loss_all += loss.item() * len(batch)
                optimizer.step()
                log_str = [f'{k}: {v:.4f}' for k, v in loss_dict.items()]
                logging.info(', '.join(log_str))
            loss_all /= len(loader.dataset)
            logging.info(f"Epoch {epoch} loss: {loss_all}")

    def rl_step(self):
        logging.info(f'*****   LOOP {self.step} START   *****')
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        logging.info('SAMPLE:')
        sample_list, sample_struc, xyz_path = self.sample_step()

        # sample scoring, remove failed samples, ranking and get top k samples
        logging.info('SCORE:')
        sample_list, sample_struc, rewards, prop_dict = self.reward_step(
            sample_list, sample_struc, xyz_path, f'step_{self.step:0>4d}',
        )

        log_dict = {f'{k} mean': v.mean() for k, v in prop_dict.items()}
        log_dict.update({f'{k} std': v.std() for k, v in prop_dict.items()})
        log_dict.update({'reward mean': rewards.mean(), 'reward std': rewards.std()})

        # long-term memory
        self.ltm.extend(sample_struc, rewards, self.step)
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, 'long_term_memory.csv'))
        logging.info(
            f'{len(self.ltm)} crystals generated so far, ' +
            f'{len(self.ltm.unique_comps)} unique components.' +
            f'  Burden: {metrics[0]}, Div. Ratio: {metrics[1]}.'
        )
        log_dict.update(
            {
                'crystal_num': len(self.ltm),
                'unique_comps': len(self.ltm.unique_comps),
                'burden': metrics[0],
                'div_ratio': metrics[1],
                'cost': self.cost,
            }
        )
        if self.logger is not None:
            self.logger.log(log_dict, step=self.step)

        # topk data points
        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: int(self.sample_cfg.batch_size * self.topk_ratio)]
        sample_topk = [sample_list[_i] for _i in topk_idx]
        reward_topk = rewards[topk_idx]

        # finetuning
        logging.info('FINETUNE:')
        self.ft_step(sample_topk)

        logging.info(f'*****   LOOP {self.step} FINISH   *****\n\n')

    def run_rl(self):
        logging.info('*****   RL START   *****')
        start_time = time.time()

        for step in range(self.rl_epoch):
            self.step = step
            self.rl_step()
            # Save the agent weights every few iterations
            if (step + 1) % self.save_freq == 0:
                ckpt_dir = os.path.join(self.models_dir, f'loop_{step:0>4d}')
                self.model_suite.save_model(self.agent, ckpt_dir)
        # If the entire training finishes, clean up
        ckpt_dir = os.path.join(self.models_dir, 'final')
        self.model_suite.save_model(self.agent, ckpt_dir)

        logging.info('*****   RL END   *****')
        end_time = time.time()
        logging.info('Total time taken: {} s.'.format(int(end_time - start_time)))
