import torch
import queue
import os
import random
import warnings
import time
import json
import argparse
from glob import glob
import pandas as pd

from modules.preprocess import preprocessing
from modules.trainer import trainer
from modules.utils import (
    get_optimizer,
    get_criterion,
    get_lr_scheduler,
)
from modules.audio import (
    FilterBankConfig,
    MelSpectrogramConfig,
    MfccConfig,
    SpectrogramConfig,
)
from modules.model import build_model
from modules.vocab import KoreanSpeechVocabulary
from modules.data import split_dataset, collate_fn
from modules.utils import Optimizer
from modules.metrics import get_metric
from modules.inference import single_infer


from torch.utils.data import DataLoader

import nova
from nova import DATASET_PATH

print("yj code start!")
print(os.getcwd())

# data_dir = os.path.join(DATASET_PATH)
# print(data_dir)
# print(os.listdir(data_dir))

# label_path = os.path.join(DATASET_PATH, 'train','train_label')
# filename, fileExtension = os.path.splitext(label_path)
# print(filename, fileExtension)
