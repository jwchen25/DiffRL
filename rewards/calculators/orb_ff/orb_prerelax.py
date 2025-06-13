import pickle
import argparse
import multiprocessing as mp
from ase.optimize import BFGS
import torch
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator


def atoms_opt(atoms):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    orbff = pretrained.orb_v1(device=device) # or choose another model using ORB_PRETRAINED_MODELS[model_name]()
    calc = ORBCalculator(orbff, device=device)
    atoms.set_calculator(calc)
    dyn = BFGS(atoms)
    dyn.run(fmax=0.05, steps=200)
    energy = atoms.get_potential_energy()

    return atoms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Atomistic Line Graph Neural Network Pretrained Models"
    )

    parser.add_argument(
        "--file_path",
        default="test.pkl",
        help="Path to file.",
    )

    parser.add_argument(
        "--save_path",
        default="./atoms_opt.pkl",
        help="Path to save results.",
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='num worker',
    )

    args = parser.parse_args()
    with open(args.file_path, 'rb') as f_pkl:
        atoms_list = pickle.load(f_pkl)

    pool = mp.Pool(args.workers)
    opt_atoms_list = list(pool.imap(atoms_opt, atoms_list))
    pool.close()

    with open(args.save_path, "wb") as f:
        pickle.dump(opt_atoms_list, f)
