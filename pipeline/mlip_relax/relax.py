import io
import os
import contextlib
import subprocess
from typing import List, Literal, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import torch

from mattergen.evaluation.utils.relaxation import relax_structures


def get_device(device: str | None = None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    return torch.device(device)


class MLIPRelaxer:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        silent: bool = True,
    ) -> None:
        self.model_name = model_name
        self.silent = silent
        self.device = get_device(device)

    def __call__(
        self,
        structures: list[Structure],
        xyz_path: str,
    ):

        if self.silent:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                relaxed_structures, energies = self.relax(
                    structures, xyz_path,
                )
        else:
            relaxed_structures, energies = self.relax(
                structures, xyz_path,
            )

        return relaxed_structures, energies

    def mattersim_opt(
        self,
        structures: list[Structure],
        xyz_path: str | None = None,
    ):
        relaxed_structures, energies = relax_structures(
            structures,
            device=self.device,
            potential_load_path="MatterSim-v1.0.0-5M.pth",
        )

        return relaxed_structures, energies

    def sevenn_opt(
        self,
        structures: list[Structure],
        xyz_path: str,
    ):
        assert isinstance(xyz_path, str) and os.path.isfile(xyz_path)
        directory, filename = os.path.split(xyz_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_opt{ext}"
        xyz_opt_path =  os.path.join(directory, new_filename)
        process = subprocess.run(
            [
                'conda', 'run', '-n', 'sevenn',
                'python', 'pipeline/mlip_relax/sevenn.py',
                xyz_path, xyz_opt_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output = process.stdout.decode().strip()

        from ase.io import read
        relaxed_atoms = read(xyz_opt_path, index=":")
        energies = np.array([a.info["total_energy"] for a in relaxed_atoms])
        relaxed_structures = [AseAtomsAdaptor.get_structure(a) for a in relaxed_atoms]

        return relaxed_structures, energies

    def relax(
        self,
        structures: list[Structure],
        xyz_path: str,
    ):
        if self.model_name == 'MatterSim':
            relaxed_structures, energies = self.mattersim_opt(structures, xyz_path)
        elif self.model_name == 'SevenNet':
            relaxed_structures, energies = self.sevenn_opt(structures, xyz_path)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return relaxed_structures, energies
