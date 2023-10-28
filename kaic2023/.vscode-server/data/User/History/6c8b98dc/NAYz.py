import os
import numpy as np
import torchaudio

import torch
import torch.nn as nn
from torch import Tensor

from modules.vocab import KoreanSpeechVocabulary
from modules.data import load_audio
from modules.model import DeepSpeech2


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)
    return torch.FloatTensor(feature).transpose(0, 1)

def revise(sentence):
    words = sentence[0].split()
    result = []
    for word in words:
        tmp = ''    
        for t in word:
            if not tmp:
                tmp += t
            elif tmp[-1]!= t:
                tmp += t
        if tmp == '스로':
            tmp = '스스로'
        result.append(tmp)
    return ' '.join(result)

# pipe-line 추가 시 여기를 수정해줘야한다!
def single_infer(model, audio_path):
    
    print("single_infer")
    device = 'cuda'
    print('model is ', model)
    feature = parse_audio(audio_path, del_silence=True)
    input_length = torch.LongTensor([len(feature)])

    #load vocabulary (여기서도 vocabulary 바꿔주기 labels.csv대신 만든 csvfile로 바꾸기!)
    vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'yj_labels.csv'), output_unit='character')
    #vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')

    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    model.device = device
    y_hats = model.recognize(feature.unsqueeze(0).to(device), input_length)
    sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
    print('predicted sentece is ', sentence)
    #post-processing 들어갈 부분
    #senetence = revise(sentence)
    return sentence