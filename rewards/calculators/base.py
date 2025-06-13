import os
import time
import pickle
import argparse
import subprocess
import numpy as np
from ase import Atoms
import torch
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset


class Calculator():
    def __init__(
        self,
        root_dir: str,
        task: str,
    ) -> None:
        self.root_dir = root_dir
        self.task = task
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def calc(self):
        NotImplementedError
