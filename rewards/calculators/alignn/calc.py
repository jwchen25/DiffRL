import io
import os
import contextlib
import numpy as np
from typing import Tuple, List
from pymatgen.core.structure import Structure
from jarvis.core.atoms import pmg_to_atoms

from rewards.calculators.base import Calculator
from rewards.calculators.alignn.prediction import get_multiple_predictions


class ALIGNN(Calculator):
    def __init__(
        self,
        root_dir: str,
        task: str = 'band_gap',
        device: str | None = None,
        silent: bool = True
    ) -> None:
        super().__init__(root_dir, task)
        self.device = device
        self.silent = silent

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp'
    ) -> np.ndarray[float]:

        struc_list = samples[0]
        atoms_list = [pmg_to_atoms(struc) for struc in struc_list]
        out_path = os.path.join(self.root_dir, f'{label}.txt')
        out_path = os.path.abspath(out_path)

        if self.task == 'band_gap':
            model_name = 'mp_gappbe_alignn'
        elif self.task == 'formation_energy':
            model_name = 'mp_e_form_alignn'
        elif self.task == 'bulk_modulus':
            model_name = 'jv_bulk_modulus_kv_alignn'
        elif self.task == 'shear_modulus':
            model_name = 'jv_shear_modulus_gv_alignn'
        elif self.task == 'magnetic_density':
            model_name = 'jv_magmom_oszicar_alignn'
        else:
            raise ValueError(
                f"{self.task} is unknown task for ALIGNN calculator!"
            )

        if self.silent:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                results = get_multiple_predictions(
                    atoms_array=atoms_list,
                    model_name=model_name,
                    device=self.device,
                )
        else:
            results = get_multiple_predictions(
                atoms_array=atoms_list,
                model_name=model_name,
                device=self.device,
            )
        results = np.array(results)

        if self.task == 'band_gap':
            results[results < 0.0] = 0.0

        if self.task == 'magnetic_density':
            volumes = np.array([s.volume for s in struc_list])
            fu = np.array(
                [s.composition.get_reduced_composition_and_factor()[1] for s in struc_list]
            )
            results = results * fu / volumes
            results[results < 0.0] = 0.0

        np.savetxt(out_path, results, fmt="%.6f")

        return results
