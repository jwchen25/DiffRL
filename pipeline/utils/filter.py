import numpy as np
from p_tqdm import p_map
from collections import Counter
from multiprocessing.pool import Pool
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

from .eval import smact_validity, structure_validity


class Crystal(object):

    def __init__(self, data):
        self.frac_coords = data.frac_coords.numpy()
        self.atom_types = data.atom_types.numpy()
        self.lengths = data.lengths[0].numpy()
        self.angles = data.angles[0].numpy()
        if len(self.atom_types.shape) > 1:
            self.atom_types = np.argmax(self.atom_types, axis=-1) + 1

        self.get_structure()
        self.get_composition()
        self.get_validity()
        # self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        if (
            np.isnan(self.lengths).any()
            or np.isnan(self.angles).any()
            or np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = "nan_value"
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())
                    ),
                    species=self.atom_types,
                    coords=self.frac_coords,
                    coords_are_cartesian=False,
                )
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = "unrealistically_small_lattice"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [
            (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
        ]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        if max(self.lengths.tolist()) > 25.0:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    # def get_fingerprints(self):
    #     elem_counter = Counter(self.atom_types)
    #     comp = Composition(elem_counter)
    #     self.comp_fp = CompFP.featurize(comp)
    #     try:
    #         site_fps = [CrystalNNFP.featurize(
    #             self.structure, i) for i in range(len(self.structure))]
    #     except Exception:
    #         # counts crystal as invalid if fingerprint cannot be constructed.
    #         self.valid = False
    #         self.comp_fp = None
    #         self.struct_fp = None
    #         return
    #     self.struct_fp = np.array(site_fps).mean(axis=0)


def get_valid(data):
    c = Crystal(data)
    return c.valid


def invalid_filter(data_list):
    # pool = Pool(processes=24)
    # validity = pool.map(get_valid, data_list)
    crystal_list = p_map(lambda x: Crystal(x), data_list)
    filtered_list = []
    valid_bool = []
    valid_idx = []
    for i, crystal in enumerate(crystal_list):
        if crystal.valid:
            filtered_list.append(data_list[i])
            valid_idx.append(i)
        valid_bool.append(crystal.valid)

    return filtered_list, valid_idx
