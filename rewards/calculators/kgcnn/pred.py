import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import json

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras as ks

from matbench.bench import MatbenchBenchmark
from graphlist import GraphList, HDFGraphList
from kgcnn.literature.coGN import make_model, model_default, model_default_nested
from kgcnn.crystal.preprocessor import KNNUnitCell, KNNAsymmetricUnitCell, CrystalPreprocessor, VoronoiAsymmetricUnitCell
from kgcnn.graph.methods import get_angle_indices
from processing import MatbenchDataset


model = make_model(**model_cfg)

def get_graphlist(
    crystals: Iterable[Structure],
    preprocessor: CrystalPreprocessor,
    out_file: Path,
    additional_graph_attributes=[],
    processes=8,
    batch_size=500,
):