import os
import sys
import argparse
import logging
from tqdm import tqdm
import warnings
from copy import deepcopy
from typing import Any, List, Dict, Optional, Union, Callable
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import Filter
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import BFGS, FIRE
from ase.optimize.optimize import Optimizer

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import sevenn.util as util
import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.sequential import AtomGraphSequential
import sevenn.train.dataload as dataload
from sevenn.train.dataload import _set_atoms_y, unlabeled_atoms_to_graph
from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn.train.modal_dataset import SevenNetMultiModalDataset


class SevennDataset(SevenNetGraphDataset):

    def __init__(
        self,
        cutoff: float,
        root: Optional[str] = None,
        files: Optional[Union[str, List[Any]]] = None,
        process_num_cores: int = 1,
        processed_name: str = 'graph.pt',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        use_data_weight: bool = False,
        log: bool = True,
        force_reload: bool = False,
        drop_info: bool = True,
        **process_kwargs,
    ):
        super().__init__(
            cutoff,
            root,
            files,
            process_num_cores,
            processed_name,
            transform,
            pre_transform,
            pre_filter,
            use_data_weight,
            log,
            force_reload,
            drop_info,
            **process_kwargs,
        )

    @staticmethod
    def atoms_to_graph_list(
        atoms_list: List[Atoms], cutoff: float,
    ) -> List[AtomGraphData]:
        num_cores = os.cpu_count()
        new_atoms_list = []
        for atoms in atoms_list:
            _atoms = deepcopy(atoms)
            _atoms.calc = None
            new_atoms_list.append(_atoms)
        new_atoms_list = _set_atoms_y(new_atoms_list)
        graph_list = dataload.graph_build(
            new_atoms_list,
            cutoff,
            num_cores,
            transfer_info=True,
            allow_unlabeled=True,
        )
        return graph_list


class DummyBatchCalculator(Calculator):
    def __init__(self):
        super().__init__()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        pass

    def get_potential_energy(self, atoms=None):
        return atoms.info["total_energy"]

    def get_forces(self, atoms=None):
        return atoms.arrays["forces"]

    def get_stress(self, atoms=None):
        return units.GPa * atoms.info["stress"]


class BatchRelaxer(object):
    """BatchRelaxer is a class for batch structural relaxation.
    It is more efficient than Relaxer when relaxing a large number of structures."""

    SUPPORTED_OPTIMIZERS = {"BFGS": BFGS, "FIRE": FIRE}
    SUPPORTED_FILTERS = {
        "EXPCELLFILTER": ExpCellFilter,
        "FRECHETCELLFILTER": FrechetCellFilter,
    }

    def __init__(
        self,
        potential: str,
        modal: Optional[str] = None,
        optimizer: Union[str, type[Optimizer]] = "FIRE",
        filter: Union[type[Filter], str, None] = "FRECHETCELLFILTER",
        fmax: float = 0.02,
        max_steps: int = 800,
        max_natoms_per_batch: int = 512,
        device: Union[torch.device, str] = 'auto',
    ):
        if isinstance(device, str):
            if device == 'auto':
                self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        mlip = util.load_checkpoint(potential)
        self.potential = mlip.build_model('e3nn')
        self.cutoff = mlip.config[KEY.CUTOFF]
        self.type_map = mlip.config[KEY.TYPE_MAP]
        self.mlip_config = mlip.config
        self.potential.to(self.device)
        self.potential.set_is_batch_data(True)
        self.potential.eval()

        # if modal:
        #     if self.potential.modal_map is None:
        #         raise ValueError('Modality given, but model has no modal_map')
        #     if modal not in self.potential.modal_map:
        #         _modals = list(self.potential.modal_map.keys())
        #         raise ValueError(f'Unknown modal {modal} (not in {_modals})')

        self.modal = None
        if isinstance(self.potential, AtomGraphSequential):
            modal_map = self.potential.modal_map
            if modal_map:
                modal_ava = list(modal_map.keys())
                if not modal:
                    raise ValueError(f'modal argument missing (avail: {modal_ava})')
                elif modal not in modal_ava:
                    raise ValueError(f'unknown modal {modal} (not in {modal_ava})')
                self.modal = modal
            elif not self.potential.modal_map and modal:
                warnings.warn(f'modal={modal} is ignored as model has no modal_map')

        if isinstance(optimizer, str):
            if optimizer.upper() not in self.SUPPORTED_OPTIMIZERS:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            self.optimizer = self.SUPPORTED_OPTIMIZERS[optimizer.upper()]
        elif issubclass(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        if isinstance(filter, str):
            if filter.upper() not in self.SUPPORTED_FILTERS:
                raise ValueError(f"Unsupported filter: {filter}")
            self.filter = self.SUPPORTED_FILTERS[filter.upper()]
        elif filter is None or issubclass(filter, Filter):
            self.filter = filter
        else:
            raise ValueError(f"Unsupported filter: {filter}")
        self.fmax = fmax
        self.max_steps = max_steps
        self.max_natoms_per_batch = max_natoms_per_batch
        self.optimizer_instances: List[Optimizer] = []
        self.is_active_instance: List[bool] = []
        self.finished = False
        self.total_converged = 0
        self.trajectories: Dict[int, List[Atoms]] = {}

    def insert(self, atoms: Atoms):
        atoms.set_calculator(DummyBatchCalculator())
        optimizer_instance = self.optimizer(
            self.filter(atoms) if self.filter else atoms
        )
        optimizer_instance.fmax = self.fmax
        self.optimizer_instances.append(optimizer_instance)
        self.is_active_instance.append(True)

    def loader_from_atoms(self, atoms_list: List[Atoms]):
        dataset = SevennDataset.atoms_to_graph_list(
            atoms_list, self.cutoff
        )

        if self.modal:
            dataset = SevenNetMultiModalDataset({self.modal: dataset})

        loader = DataLoader(dataset, len(dataset), shuffle=False)
        batch = next(iter(loader))
        batch = batch.to(self.device)
        return batch

    def batch_from_atoms(self, atoms_list: List[Atoms]):
        data_list = []
        for atoms in atoms_list:
            data = AtomGraphData.from_numpy_dict(
                unlabeled_atoms_to_graph(atoms, self.cutoff)
            )
            if self.modal:
                data[KEY.DATA_MODALITY] = self.modal
            data_list.append(data)
        batch = Batch.from_data_list(data_list).to(self.device)
        return batch

    def step_batch(self):
        atoms_list = []
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                atoms_list.append(opt.atoms.atoms)

        batch = self.batch_from_atoms(atoms_list)

        output_list = []
        output = self.potential(batch)
        output_list.extend(util.to_atom_graph_list(output))

        counter = 0
        self.finished = True
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                # Set the properties so the dummy calculator can
                # return them within the optimizer step
                opt.atoms.info["total_energy"] = output_list[counter][KEY.PRED_TOTAL_ENERGY].detach().cpu().item()
                opt.atoms.arrays["forces"] = output_list[counter][KEY.PRED_FORCE].detach().cpu().numpy()
                # print(output_list[counter][KEY.PRED_STRESS][0])
                stress = np.array(
                    (-output_list[counter][KEY.PRED_STRESS][0])
                    .detach()
                    .cpu()
                    .numpy()[[0, 1, 2, 4, 5, 3]]  # as voigt notation
                )
                opt.atoms.info["stress"] = stress

                try:
                    self.trajectories[opt.atoms.info["structure_index"]].append(
                        opt.atoms.copy()
                    )
                except KeyError:
                    self.trajectories[opt.atoms.info["structure_index"]] = [
                        opt.atoms.copy()
                    ]

                opt.step()
                if opt.converged() or opt.Nsteps > self.max_steps:
                    self.is_active_instance[idx] = False
                    self.total_converged += 1
                    if self.total_converged % 100 == 0:
                        logging.info(f"Relaxed {self.total_converged} structures.")
                else:
                    self.finished = False
                counter += 1

        # remove inactive instances
        self.optimizer_instances = [
            opt
            for opt, active in zip(self.optimizer_instances, self.is_active_instance)
            if active
        ]
        self.is_active_instance = [True] * len(self.optimizer_instances)

    def relax(
        self,
        atoms_list: List[Atoms],
    ) -> Dict[int, List[Atoms]]:
        self.trajectories = {}
        self.tqdmcounter = tqdm(total=len(atoms_list), file=sys.stdout)
        pointer = 0
        atoms_list_ = []
        for i in range(len(atoms_list)):
            atoms_list_.append(atoms_list[i].copy())
            atoms_list_[i].info["structure_index"] = i

        while (
            pointer < len(atoms_list) or not self.finished
        ):  # While there are unfinished instances or atoms left to insert
            while pointer < len(atoms_list) and (
                sum([len(opt.atoms) for opt in self.optimizer_instances])
                + len(atoms_list[pointer])
                <= self.max_natoms_per_batch
            ):
                # While there are enough n_atoms slots in the
                # batch and we have not reached the end of the list.
                self.insert(
                    atoms_list_[pointer]
                )  # Insert new structure to fire instances
                self.tqdmcounter.update(1)
                pointer += 1
            self.step_batch()
        self.tqdmcounter.close()

        return self.trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str, help="input path of extxyz file")
    parser.add_argument("out_path", type=str, help="output path of extxyz file")
    parser.add_argument(
        '--max_natoms',
        type=int,
        default=512,
        help='max_natoms_per_batch',
    )
    args = parser.parse_args()

    from ase.io import read, write
    atoms_list = read(args.in_path, index=":")
    batch_relaxer = BatchRelaxer(
        potential='7net-mf-ompa',
        modal="mpa",
        max_natoms_per_batch=args.max_natoms,
    )
    relaxation_trajectories = batch_relaxer.relax(atoms_list)
    relaxed_atoms = [t[-1] for t in relaxation_trajectories.values()]
    write(args.out_path, relaxed_atoms)
