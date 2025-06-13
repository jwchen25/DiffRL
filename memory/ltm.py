from typing import Tuple, List
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


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


class LongTimeMem:
    """
    Long-term memory class which stores all generated crystals so far.
        1. Crystals (data, pymatgen.Structure, compositions)
        2. Reward
    """

    def __init__(
        self,
        # df_tol: int = 10,
        # df_buff: int = 20,
    ) -> None:
        # self.df_tol = df_tol
        # self.df_buff = df_buff
        # assert self.df_tol < self.df_buff

        # Stores all generated crystals and rewards
        self.memory = pd.DataFrame(
            columns=["struc", "comp", "ele_comb", "reward", "RL_step"]
        )
        self.unique_comps = []

    def extend(self, strucs: List[Structure], rewards: np.ndarray[float], step: int) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
        steps = [step for s in strucs]
        ele_comb = []
        for s in strucs:
            elements = set(str(e) for e in s.species)
            comb = tuple(sorted(elements))
            ele_comb.append(comb)

        # cs_list, pg_list, sg_list = [], [], []
        # for struc in strucs:
        #     analyzer = SpacegroupAnalyzer(struc)
        #     # Get the crystal system
        #     crystal_system = analyzer.get_crystal_system()
        #     # Get the point group
        #     point_group = analyzer.get_point_group_symbol()
        #     # Get the space group symbol and number
        #     space_group = analyzer.get_space_group_symbol()
        #     cs_list.append(crystal_system)
        #     pg_list.append(point_group)
        #     sg_list.append(space_group)

        df_sample = pd.DataFrame.from_dict({
            "struc": strucs,
            "comp": comps,
            "ele_comb": ele_comb,
            "reward": rewards,
            "RL_step": steps,
        })
        if len(self.memory) > 0:
            self.memory = pd.concat([self.memory, df_sample])
        else:
            self.memory = df_sample
        self.unique_comps = self.memory["comp"].unique()

    def div_filter(
        self,
        strucs: List[Structure],
        rewards: np.ndarray[float],
        tol: int = 10,
        buff: int = 20,
        method: str ="composition",
        **kwargs,
    ) -> Tuple[np.ndarray[float], int, int]:
        # ref: Augmented Hill-Climb, https://doi.org/10.1186/s13321-022-00646-z
        # tol = tolerance, buff = buffer, occ = occurrences
        assert tol < buff
        comps = [s.composition.reduced_formula for s in strucs]
        ele_comb = []
        for s in strucs:
            elements = set(str(e) for e in s.species)
            comb = tuple(sorted(elements))
            ele_comb.append(comb)

        if method == "composition":
            key = "comp"
            values = comps
        elif method == "element_comb":
            key = "ele_comb"
            values = ele_comb

        new_rewards = []
        penalty_idx = []
        tol_n = 0
        buff_n = 0
        for i, v in enumerate(values):
            occ = self.memory[key].value_counts().get(v, 0)
            if occ <= tol:
                new_rewards.append(rewards[i])
            elif occ > tol and occ < buff:
                new_rewards.append(
                    rewards[i] * (buff - occ) / (buff - tol)
                )
                tol_n += 1
            else:
                new_rewards.append(0.0)
                penalty_idx.append(i)
                buff_n += 1

        return np.array(new_rewards), penalty_idx, tol_n, buff_n

    def calc_metrics(
        self,
        thred: float,
        budget: int = 3000,
        num_candidate: int = 100,
    ) -> Tuple[float, float]:
        # Burden metric
        _df = self.memory.sort_values("reward", ascending=False)
        unique_df = _df.drop_duplicates(subset=["comp"])
        candidates = (unique_df["reward"] > thred).sum()
        calc_cost = len(self.memory)
        if candidates >= num_candidate:
            burden = calc_cost / candidates
        else:
            burden = None

        # Diversity ratio
        num_uni_comp = len(self.unique_comps)
        if calc_cost <= budget:
            div_ratio = num_uni_comp / calc_cost
        else:
            div_ratio = None

        return burden, div_ratio

    def get_baseline(self, step: int, prev: int = 3):
        baseline = self.memory[self.memory["RL_step"] > step - prev]["reward"].mean()
        return baseline

    def deduplicate(self, df: pd.DataFrame, method="composition") -> pd.DataFrame:
        """
        Removes duplicate crystals based on different methods like composition,
        StructureMatcher, symmetry (crystal system, space group, etc.)
        Keep only non-zero rewards crystals.
        """
        if method == "composition":
            _df = df.sort_values("reward", ascending=False)
            unique_df = _df.drop_duplicates(subset=["comp"])
            return unique_df

    def sample(self) -> Tuple[List[Data], np.ndarray[float]]:
        sample_size = min(len(self.memory), self.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            data = sampled["data"].values.tolist()
            rewards = sampled["reward"].values
            return data, rewards
        else:
            return [], []

    def save(self, save_path: str):
        df = self.memory.copy()
        strucs = df["struc"].values
        cif_list = [s.to(fmt="cif") for s in strucs]
        df["cif"] = cif_list
        df.to_csv(save_path, index=False, quoting=1)

    def __len__(self) -> int:
        return len(self.memory)
