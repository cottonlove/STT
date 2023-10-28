# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from omegaconf import DictConfig
from astropy.modeling import ParameterError

from kospeech.models.conformer import Conformer
from kospeech.vocabs import Vocabulary
from kospeech.models.las import EncoderRNN
# from kospeech.decode.ensemble import (
#     BasicEnsemble,
#     WeightedEnsemble,
# )
from kospeech.models import (
    ListenAttendSpell,
    DeepSpeech2,
    SpeechTransformer,
    Jasper,
    RNNTransducer,
)


def build_model(
        config: DictConfig,
        vocab: Vocabulary,
        device: torch.device,
) -> nn.DataParallel:
    """ Various model dispatcher function. """
    if config.audio.transform_method.lower() == 'spect':
        if config.audio.feature_extract_by == 'kaldi':
            input_size = 257
        else:
            input_size = (config.audio.frame_length << 3) + 1
    else:
        input_size = config.audio.n_mels

    model = build_conformer(
        num_classes=len(vocab),
        input_size=input_size,
        encoder_dim=config.model.encoder_dim,
        decoder_dim=config.model.decoder_dim,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        decoder_rnn_type=config.model.decoder_rnn_type,
        num_attention_heads=config.model.num_attention_heads,
        feed_forward_expansion_factor=config.model.feed_forward_expansion_factor,
        conv_expansion_factor=config.model.conv_expansion_factor,
        input_dropout_p=config.model.input_dropout_p,
        feed_forward_dropout_p=config.model.feed_forward_dropout_p,
        attention_dropout_p=config.model.attention_dropout_p,
        conv_dropout_p=config.model.conv_dropout_p,
        decoder_dropout_p=config.model.decoder_dropout_p,
        conv_kernel_size=config.model.conv_kernel_size,
        half_step_residual=config.model.half_step_residual,
        device=device,
        decoder=config.model.decoder,
    )

    print(model)

    return model


def build_conformer(
        num_classes: int,
        input_size: int,
        encoder_dim: int,
        decoder_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_rnn_type: str,
        num_attention_heads: int,
        feed_forward_expansion_factor: int,
        conv_expansion_factor: int,
        input_dropout_p: float,
        feed_forward_dropout_p: float,
        attention_dropout_p: float,
        conv_dropout_p: float,
        decoder_dropout_p: float,
        conv_kernel_size: int,
        half_step_residual: bool,
        device: torch.device,
        decoder: str,
) -> nn.DataParallel:
    if input_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if feed_forward_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if attention_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if conv_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_size < 0:
        raise ParameterError("input_size should be greater than 0")
    assert conv_expansion_factor == 2, "currently, conformer conv expansion factor only supports 2"

    return nn.DataParallel(Conformer(
        num_classes=num_classes,
        input_dim=input_size,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        decoder_rnn_type=decoder_rnn_type,
        num_attention_heads=num_attention_heads,
        feed_forward_expansion_factor=feed_forward_expansion_factor,
        conv_expansion_factor=conv_expansion_factor,
        input_dropout_p=input_dropout_p,
        feed_forward_dropout_p=feed_forward_dropout_p,
        attention_dropout_p=attention_dropout_p,
        conv_dropout_p=conv_dropout_p,
        decoder_dropout_p=decoder_dropout_p,
        conv_kernel_size=conv_kernel_size,
        half_step_residual=half_step_residual,
        device=device,
        decoder=decoder,
    )).to(device)


def load_test_model(config: DictConfig, device: torch.device):
    model = torch.load(config.model_path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model.module.decoder.device = device
        model.module.encoder.device = device

    else:
        model.encoder.device = device
        model.decoder.device = device

    return model


def load_language_model(path: str, device: torch.device):
    model = torch.load(path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model = model.module

    model.device = device

    return model


# def build_ensemble(model_paths: list, method: str, device: torch.device):
#     models = list()

#     for model_path in model_paths:
#         models.append(torch.load(model_path, map_location=lambda storage, loc: storage))

#     if method == 'basic':
#         ensemble = BasicEnsemble(models).to(device)
#     elif method == 'weight':
#         ensemble = WeightedEnsemble(models).to(device)
#     else:
#         raise ValueError("Unsupported ensemble method : {0}".format(method))

#     return ensemble