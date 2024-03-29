import torch
import queue
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

import os
print("yj code start!")
print(os.getcwd())

# with open('Jeno.txt', 'w') as file:
#     # 파일에 내용을 쓰기
#     file.write('Hello, I am Jeno')
label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
transcript_df = pd.read_csv(label_path)
#print(transcript_df)
#transcript_df.to_csv(os.path.join(os.getcwd(),'transcript_yj.csv'))
transcript_df['count_str1'] = transcript_df['text'].str.count('아') 
# print(count_str)
transcript_df['count_str2'] = transcript_df['text'].str.count('롱')
transcript_df['count_str3'] = transcript_df['text'].str.count('/un')
transcript_df['count_str4'] = transcript_df['text'].str.count('\?') #79481
transcript_df['count_str5'] = transcript_df['text'].str.count('%')
# print(count_str)
print(transcript_df['count_str1'].sum()) #370443 -> 110602
print(transcript_df['count_str2'].sum()) #967 -> 254
print(transcript_df['count_str3'].sum())
print(transcript_df['count_str4'].sum())
print(transcript_df['count_str5'].sum())
# data_dir = os.path.join(DATASET_PATH)
# print(data_dir)
# print(os.listdir(data_dir))

# label_path = os.path.join(DATASET_PATH, 'train','train_label')
# filename, fileExtension = os.path.splitext(label_path)
# print(filename, fileExtension)
