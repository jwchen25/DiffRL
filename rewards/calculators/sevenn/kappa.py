import os
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from moyopy import MoyoDataset
from moyopy.interface import MoyoAdapter
from pymatviz.enums import Key
from sevenn.calculator import SevenNetCalculator
from matbench_discovery.phonons import check_imaginary_freqs
from matbench_discovery.phonons import thermal_conductivity as ltc

from utils import get_supercell_parameters

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

model_name = "sevennet"
model_variant = "sevennet-mf-ompa"  # choose 7net model variant to eval
device = "cuda" if torch.cuda.is_available() else "cpu"
calculator_kwargs = {
    "sevennet-0": {"model": "7net-0"},
    "sevennet-l3i5": {"model": "7net-l3i5"},
    "sevennet-mf-ompa": {"model": "7net-mf-ompa", "modal": "mpa"},
}[model_variant]
calculator_kwargs["device"] = device

# attempt to down load checkpoint from online. took several minutes
calc = SevenNetCalculator(**calculator_kwargs)

# Relaxation parameters. These params are for reproducing 7net-mf-ompa.
ase_optimizer: Literal["FIRE", "LBFGS", "BFGS"] = "FIRE"
max_steps = 300
force_max = 1e-4
symprec = 1e-5
enforce_relax_symm = True
conductivity_broken_symm = False
prog_bar = False
save_forces = False  # Save force sets to file
temperatures = [300]  # Temperatures to calculate conductivity at in Kelvin
displacement_distance = 0.03  # Displacement distance for phono3py
task_type = "LTC"  # lattice thermal conductivity


def get_supercell_matrix(
    structure: Structure,
    min_length: float = 20.0,
    max_length: float = None,
    prefer_90_degrees: bool = True,
    allow_orthorhombic: bool = False,
    **kwargs,
) -> list[list[float]]:
    """
    Determine supercell size with given min_length and max_length.

    Parameters
    ----------
    structure: Structure Object
        Input structure that will be used to determine supercell
    min_length: float
        minimum length of cell in Angstrom
    max_length: float
        maximum length of cell in Angstrom
    prefer_90_degrees: bool
        if True, the algorithm will try to find a cell with 90 degree angles first
    allow_orthorhombic: bool
        if True, orthorhombic supercells are allowed
    **kwargs:
        Additional parameters that can be set.
    """
    kwargs.setdefault("force_diagonal", False)
    common_kwds = dict(
        min_length=min_length,
        max_length=max_length,
        min_atoms=kwargs.get("min_atoms"),
        max_atoms=kwargs.get("max_atoms"),
        step_size=kwargs.get("step_size", 0.1),
        force_diagonal=kwargs["force_diagonal"],
    )
    if not prefer_90_degrees:
        transformation = CubicSupercellTransformation(
            **common_kwds,
            force_90_degrees=False,
            allow_orthorhombic=allow_orthorhombic,
        )
        transformation.apply_transformation(structure=structure)
    else:
        try:
            common_kwds.update({"max_atoms": kwargs.get("max_atoms", 1200)})
            transformation = CubicSupercellTransformation(
                **common_kwds,
                force_90_degrees=True,
                angle_tolerance=kwargs.get("angle_tolerance", 1e-2),
                allow_orthorhombic=allow_orthorhombic,
            )
            transformation.apply_transformation(structure=structure)

        except AttributeError:
            transformation = CubicSupercellTransformation(
                **common_kwds,
                force_90_degrees=False,
                allow_orthorhombic=allow_orthorhombic,
            )
            transformation.apply_transformation(structure=structure)
    # matrix from pymatgen has to be transposed
    return transformation.transformation_matrix.transpose().tolist()


def calc_kappa(atoms):

    struc = AseAtomsAdaptor.get_structure(atoms)
    analyzer = SpacegroupAnalyzer(struc, symprec=0.1)
    symmetrized_structure = analyzer.get_refined_structure()
    primitive_structure = symmetrized_structure.get_primitive_structure()
    atoms = AseAtomsAdaptor.get_atoms(primitive_structure)
    print(analyzer.get_space_group_symbol())

    # Set up the relaxation and force set calculation
    optim_cls: Callable[..., Optimizer] = {"FIRE": FIRE, "LBFGS": LBFGS}[ase_optimizer]
    spg_num = MoyoDataset(MoyoAdapter.from_atoms(atoms)).number

    # Initialize relax_dict to avoid "possibly unbound" errors
    relax_dict = {
        "max_stress": None,
        "reached_max_steps": False,
        "broken_symmetry": False,
    }

    try:
        atoms.calc = calc
        if max_steps > 0:
            if enforce_relax_symm:
                atoms.set_constraint(FixSymmetry(atoms))
                # Use standard mask for no-tilt constraint
                filtered_atoms = FrechetCellFilter(atoms, mask=[True] * 3 + [False] * 3)
            else:
                filtered_atoms = FrechetCellFilter(atoms)

            optimizer = optim_cls(filtered_atoms, logfile="/dev/null")
            optimizer.run(fmax=force_max, steps=max_steps)
            reached_max_steps = optimizer.nsteps >= max_steps

            max_stress = atoms.get_stress().reshape((2, 3), order="C").max(axis=1)
            atoms.calc = None
            atoms.constraints = None

            # Check if symmetry was broken during relaxation
            relaxed_spg = MoyoDataset(MoyoAdapter.from_atoms(atoms)).number
            broken_symmetry = spg_num != relaxed_spg
            relax_dict = {
                "max_stress": max_stress,
                "reached_max_steps": reached_max_steps,
                "relaxed_space_group_number": relaxed_spg,
                "broken_symmetry": broken_symmetry,
            }

    except Exception as exc:
        print(1)
        return None

    # Calculation of force sets
    try:
        # struc = AseAtomsAdaptor.get_structure(atoms)
        # fc3_supercell = get_supercell_matrix(struc, min_length=8)
        # fc2_supercell = get_supercell_matrix(struc, min_length=15)
        # print(fc3_supercell)
        # print(fc2_supercell)
        # fc3_supercell, q_point_mesh = get_supercell_parameters(atoms)
        # print(fc3_supercell)
        # print(q_point_mesh)
        fc2_supercell = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        fc3_supercell = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        q_point_mesh = np.array([16, 16, 16])
        # Initialize phono3py with the relaxed structure
        ph3 = ltc.init_phono3py(
            atoms,
            fc2_supercell=fc2_supercell,
            fc3_supercell=fc3_supercell,
            q_point_mesh=q_point_mesh,
            displacement_distance=displacement_distance,
            symprec=symprec,
        )

        # Calculate force constants and frequencies
        ph3, fc2_set, freqs = ltc.get_fc2_and_freqs(
            ph3, calculator=calc, pbar_kwargs={"leave": False, "disable": not prog_bar}
        )

        # Check for imaginary frequencies
        # has_imaginary_freqs = check_imaginary_freqs(freqs)

        # If conductivity condition is met, calculate fc3
        ltc_condition = True

        if ltc_condition:  # Calculate third-order force constants
            fc3_set = ltc.calculate_fc3_set(
                ph3,
                calculator=calc,
                pbar_kwargs={"leave": False, "disable": not prog_bar},
            )
            ph3.produce_fc3(symmetrize_fc3r=True)
        else:
            fc3_set = []

        if not ltc_condition:
            print(2)
            return None

    except Exception as exc:
        print(3)
        return None

    try:  # Calculate thermal conductivity
        ph3, kappa_dict, _ = ltc.calculate_conductivity(ph3, temperatures=temperatures)
    except Exception as exc:
        print(4)
        return None

    kappa_tot_rta = np.array(kappa_dict['kappa_tot_rta'])
    k_tot = kappa_tot_rta[0][:3].mean()
    return k_tot


if __name__ == "__main__":
    atoms_list = read('/home/jwchen/work/code/MatterRL/exp_res/matgen_dft_bandgap2/samples/step_0000_eval.extxyz', index=':')
    # atoms_list = read('/home/jwchen/work/code/MatterRL/rewards/calculators/sevenn/matbench-discovery/data/phonons/2024-11-09-phononDB-PBE-103-structures.extxyz', index=':')
    k_tot = calc_kappa(atoms_list[6])
    print(f'k_tot={k_tot}')
