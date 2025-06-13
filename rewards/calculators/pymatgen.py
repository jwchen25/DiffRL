import os
import subprocess
import numpy as np
from typing import Union, Tuple, List
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.analysis.hhi import HHIModel

from rewards.calculators.base import Calculator


def data2struc(data):
    frac_coords = data.frac_coords.numpy()
    atom_types = data.atom_types.numpy()
    lengths = data.lengths[0].tolist()
    angles = data.angles[0].tolist()
    lattice = Lattice.from_parameters(*(lengths + angles))
    struc = Structure(
        lattice=lattice,
        species=atom_types,
        coords=frac_coords,
        coords_are_cartesian=False,
    )

    return struc


def calc_density(struc_list: List[Structure])-> np.ndarray[float]:
    density = np.array(
        [struc.density for struc in struc_list]
    )
    return density


def calc_hhi(struc_list: List[Structure])-> np.ndarray[float]:
    calc = HHIModel()
    hhi = np.array(
        [calc.get_hhi_reserve(s.composition) for s in struc_list]
    )
    hhi = np.where(hhi == None, 10000.0, hhi).astype(float)
    return hhi


class PyMatGen(Calculator):
    def __init__(
        self,
        root_dir: str,
        task: str = 'density',
    ) -> None:
        super().__init__(root_dir, task)

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp'
    ) -> np.ndarray[float]:

        # struc_list = [data2struc(data) for data in samples[0]]
        struc_list = samples[0]
        out_path = os.path.join(self.root_dir, f'{label}.txt')
        out_path = os.path.abspath(out_path)

        if self.task == 'density':
            results = calc_density(struc_list)
        elif self.task == 'hhi':
            results = calc_hhi(struc_list)
        else:
            raise ValueError(
                f"{self.task} is unknown task for PyMatGen calculator!"
            )
        
        # save results
        np.savetxt(out_path, results, fmt='%.8f')

        return results
