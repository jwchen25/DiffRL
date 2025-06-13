import os
import logging
import subprocess
import numpy as np
from typing import Union, Tuple, List

from rewards.calculators.base import Calculator


class ALIGNN(Calculator):
    def __init__(
        self,
        root_dir: str,
        task: str = 'band_gap',
        env_name: str = 'alignn',
    ) -> None:
        super().__init__(root_dir, task)
        self.env_name = env_name

    def calc(
        self,
        samples: Tuple[list, str],
        label: str = 'tmp'
    ) -> np.ndarray[float]:

        pkl_path = samples[1]
        out_path = os.path.join(self.root_dir, f'{label}.txt')

        # Absolute path
        pkl_path = os.path.abspath(pkl_path)
        out_path = os.path.abspath(out_path)

        if self.task == 'band_gap':
            model_name = 'mp_gappbe_alignn'
        elif self.task == 'formation_energy':
            model_name = 'mp_e_form_alignn'
        else:
            raise ValueError(
                f"{self.task} is unknown task for ALIGNN calculator!"
            )

        process = subprocess.run(
            [
                'conda', 'run', '-n', self.env_name,
                'python', 'rewards/calculators/alignn/script.py',
                '--file_format', 'pkl',
                '--model_name', model_name,
                '--file_path', pkl_path,
                '--save_path', out_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # proc_error = process.stderr.decode()
        # if proc_error != "":
        #     logging.error(proc_error)
        # print(proc_error)

        assert os.path.isfile(out_path)
        with open(out_path, "r") as file:
            lines = file.readlines()
        results = [float(line.strip()) for line in lines]
        results = np.array(results)
        if self.task == 'band_gap':
            results[results < 0.0] = 0.0

        return results
