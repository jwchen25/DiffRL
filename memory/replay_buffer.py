"""
Some code is based on the implementation from https://github.com/MolecularAI/Reinvent.
"""
from typing import Tuple, List
# from typing import List, Literal, Optional, Dict, Any
import numpy as np
import pandas as pd
from copy import deepcopy
import ase
from torch_geometric.data import Data
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class ReplayBuffer:
    """
    Replay buffer class which stores the top K highest reward crystals generated so far.
        1. Crystals (data, pymatgen.Structure, compositions)
        2. Reward
    """

    def __init__(
        self,
        buffer_size: int = 100,
        sample_size: int = 8,
        reward_cutoff: float = 0.0,
    ) -> None:
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.reward_cutoff = reward_cutoff
        # Stores the top N highest reward crystal generated so far
        self.buffer = pd.DataFrame(
            columns=["data", "struc", "comp", "ele_comb", "reward"]
        )

    def extend(
        self,
        data: list,
        strucs: List[Structure],
        rewards: np.ndarray[float],
    ) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
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

        df_sam = pd.DataFrame.from_dict({
            "data": data,
            "struc": strucs,
            "comp": comps,
            "ele_comb": ele_comb,
            "reward": rewards
        })
        if len(self.buffer) > 0:
            df_all = pd.concat([self.buffer, df_sam])
        else:
            df_all = df_sam
        unique_df = self.deduplicate(df_all)
        sorted_df = unique_df.sort_values("reward", ascending=False)
        self.buffer = sorted_df.head(self.buffer_size)
        # reward cutoff
        self.buffer = self.buffer.loc[self.buffer["reward"] > self.reward_cutoff]

    def deduplicate(self, df: pd.DataFrame, method="composition") -> pd.DataFrame:
        """
        Removes duplicate crystals based on different methods like composition,
        StructureMatcher, symmetry (crystal system, space group, etc.)
        Keep only non-zero rewards crystals.
        """
        _df = df.sort_values("reward", ascending=False)
        if method == "composition":
            unique_df = _df.drop_duplicates(subset=["comp"])
        elif method == "element_comb":
            unique_df = _df.drop_duplicates(subset=["ele_comb"])

        return unique_df

    def sample(self) -> Tuple[List[Data], np.ndarray[float]]:
        sample_size = min(len(self.buffer), self.sample_size)
        if sample_size > 0:
            sampled = self.buffer.sample(sample_size)
            data = sampled["data"].values.tolist()
            rewards = sampled["reward"].values
            return data, rewards
        else:
            return [], []

    def memory_purge(self, strucs: List[Structure]) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
        self.buffer = self.buffer[~self.buffer["comp"].isin(comps)]

    def __len__(self) -> int:
        return len(self.buffer)

    # def selective_memory_purge(self, smiles: np.ndarray[str], rewards: np.ndarray[float]) -> None:
    #     """
    #     Augmented Memory's key operation to prevent mode collapse and promote diversity:
    #     Purges the memory of SMILES that have penalized rewards (0.0) *before* executing Augmented Memory updates.
    #     Intuitively, this operation prevents penalized SMILES from directing the Agent's chemical space navigation.

    #     # NOTE: Consider a MPO objective task using a product aggregator. If one of the OracleComponent's reward is 0,
    #     #       then the aggregated reward may be 0. But other OracleComponents may have a non-zero reward. We do not
    #     #       want to purge the memory of these scaffolds. This is already handled because 0 reward SMILES are not
    #     #       added to the memory in the first place. Selective Memory Purge *only* removes scaffolds that are
    #     #       penalized by the Diversity Filter.
    #     """
    #     zero_reward_indices = np.where(rewards == 0.)[0]
    #     if len(zero_reward_indices) > 0:
    #         smiles_to_purge = smiles[zero_reward_indices]
    #         scaffolds_to_purge = [get_bemis_murcko_scaffold(smiles) for smiles in smiles_to_purge]
    #         purged_memory = deepcopy(self.buffer)
    #         purged_memory["scaffolds"] = purged_memory["smiles"].apply(get_bemis_murcko_scaffold)
    #         purged_memory = purged_memory.loc[~purged_memory["scaffolds"].isin(scaffolds_to_purge)]
    #         purged_memory.drop("scaffolds", axis=1, inplace=True)
    #         self.buffer = purged_memory
    #     else:
    #         # If no scaffolds are penalized, do nothing
    #         return

    # def prepopulate_buffer(self, oracle: Oracle) -> Oracle:
    #     """
    #     Seeds the replay buffer with a set of SMILES.
    #     Useful if there are known high-reward molecules to pre-populate the Replay Buffer with.

    #     Oracle is returned here because seeding updates the Oracle's history with the seeded SMILES.

    #     NOTE: With more SMILES to seed with, the generative experiment will become more like
    #           transfer learning rather than reinforcement learning (at the start). Continuing
    #           the run will more and more leverage reinforcement learning to find other diverse
    #           solutions. Therefore, while seeding will quick-start the Agent's learning, there
    #           are implications on the diversity of the solutions found.
    #     """
    #     if len(self.parameters.smiles) > 0:
    #         canonical_smiles = canonicalize_smiles_batch(self.parameters.smiles)
    #         mols = [Chem.MolFromSmiles(s) for s in canonical_smiles]
    #         mols = [mol for mol in mols if mol is not None]

    #         oracle_components_df = pd.DataFrame()
    #         rewards = np.empty((len(oracle.oracle), len(mols)))
    #         for idx, oracle_component in enumerate(oracle.oracle):
    #             raw_property_values, component_rewards = oracle_component.calculate_reward(
    #                 mols, oracle_calls=0)
    #             oracle_components_df[f"{oracle_component.name}_raw_values"] = raw_property_values
    #             oracle_components_df[f"{oracle_component.name}_reward"] = component_rewards
    #             rewards[idx] = component_rewards

    #         aggregated_rewards = oracle.aggregator(rewards, oracle.oracle_weights)

    #         # Add the SMILES to the Replay Buffer
    #         self.add(
    #             smiles=self.parameters.smiles,  # add the original SMILES
    #             rewards=aggregated_rewards)

    #         oracle.update_oracle_history(smiles=self.parameters.smiles,
    #                                      scaffolds=np.vectorize(get_bemis_murcko_scaffold)(
    #                                          self.parameters.smiles),
    #                                      rewards=aggregated_rewards,
    #                                      penalized_rewards=aggregated_rewards,
    #                                      oracle_components_df=oracle_components_df)
    #         # Update the Oracle Cache with the canonical SMILES
    #         oracle.update_oracle_cache(canonical_smiles, rewards)

    #     return oracle
