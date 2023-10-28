#class for conformer
import torch
import torch.nn as nn
from omegaconf import DictConfig
from astropy.modeling import ParameterError

from kospeech.models.conformer import Conformer
from kospeech.vocabs import Vocabulary
from kospeech.models.las import EncoderRNN
from kospeech.decode.ensemble import (
    BasicEnsemble,
    WeightedEnsemble,
)
from kospeech.models import (
    ListenAttendSpell,
    DeepSpeech2,
    SpeechTransformer,
    Jasper,
    RNNTransducer,
)
