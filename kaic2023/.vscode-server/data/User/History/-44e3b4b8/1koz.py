import torch
import queue
import os
import random
import warnings
import time
import json
import argparse
from glob import glob

from modules.preprocess import preprocessing
from modules.preprocess import * #yj
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
from modules.data import split_and_cross_validate, collate_fn #for cross validation

from torch.utils.data import DataLoader

import nova
from nova import DATASET_PATH

#여길 고쳐서 여러개의 checkpoints 가능해짐
def bind_model(model, optimizer=None):
    
    #최종 model.pt 파일 저장하기
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    #model params 불러오기
    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)

    nova.bind(save=save, load=load, infer=infer)  # 'nova.bind' function must be called at the end.


def inference(path, model, **kwargs):
    
    model.eval()

    results = []
    for i in glob(os.path.join(path, '*')):
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer(model, i)[0]
            }
        )
    return sorted(results, key=lambda x: x['filename'])

def sample_inference(paths, transcripts, model, vocab):
    recover = False
    if model.training:
        recover = True
    model.eval()

    results = []
    dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data')
    for index in range(len(paths)):
        path = paths[index]
        transcript = transcripts[index]
        labels = [int(label) for label in transcript.split()]
        labels_np = np.ndarray(labels)
        #only one
        for i in glob(os.path.join(dataset_path, path)):
            results.append(
                {
                    'filename': i.split('/')[-1],
                    'text': single_infer(model, i)[0],
                    'transcript': vocab.label_to_string(labels_np)
                }
            )
    if recover:
        model.train()
    return results


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # DONOTCHANGE
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)


    # Parameters 
    args.add_argument('--use_cuda', type=bool, default=True)
    args.add_argument('--seed', type=int, default=777)
    args.add_argument('--num_epochs', type=int, default=5)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--save_result_every', type=int, default=10)
    args.add_argument('--checkpoint_every', type=int, default=1)
    args.add_argument('--print_every', type=int, default=50)
    args.add_argument('--dataset', type=str, default='kspon')
    args.add_argument('--output_unit', type=str, default='character')
    args.add_argument('--num_workers', type=int, default=16)
    args.add_argument('--num_threads', type=int, default=16)
    args.add_argument('--init_lr', type=float, default=1e-06)
    args.add_argument('--final_lr', type=float, default=1e-06)
    args.add_argument('--peak_lr', type=float, default=1e-04)
    args.add_argument('--init_lr_scale', type=float, default=1e-02)
    args.add_argument('--final_lr_scale', type=float, default=5e-02)
    args.add_argument('--max_grad_norm', type=int, default=400)
    args.add_argument('--warmup_steps', type=int, default=1000)
    args.add_argument('--weight_decay', type=float, default=1e-05)
    args.add_argument('--reduction', type=str, default='mean')
    args.add_argument('--optimizer', type=str, default='adam')
    args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler')
    args.add_argument('--total_steps', type=int, default=200000)

    args.add_argument('--architecture', type=str, default='deepspeech2')
    args.add_argument('--use_bidirectional', type=bool, default=True)
    args.add_argument('--dropout', type=float, default=3e-01)
    args.add_argument('--num_encoder_layers', type=int, default=3)
    args.add_argument('--hidden_dim', type=int, default=1024)
    args.add_argument('--rnn_type', type=str, default='gru')
    args.add_argument('--max_len', type=int, default=400)
    args.add_argument('--activation', type=str, default='hardtanh')
    args.add_argument('--teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--teacher_forcing_step', type=float, default=0.0)
    args.add_argument('--min_teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--joint_ctc_attention', type=bool, default=True) #False -> True

    args.add_argument('--audio_extension', type=str, default='wav')
    args.add_argument('--transform_method', type=str, default='fbank')
    args.add_argument('--feature_extract_by', type=str, default='kaldi')
    args.add_argument('--sample_rate', type=int, default=16000)
    args.add_argument('--frame_length', type=int, default=20)
    args.add_argument('--frame_shift', type=int, default=10)
    args.add_argument('--n_mels', type=int, default=80)
    args.add_argument('--freq_mask_para', type=int, default=18)
    args.add_argument('--time_mask_num', type=int, default=4)
    args.add_argument('--freq_mask_num', type=int, default=2)
    args.add_argument('--normalize', type=bool, default=True)
    args.add_argument('--del_silence', type=bool, default=True)
    args.add_argument('--spec_augment', type=bool, default=True)
    args.add_argument('--input_reverse', type=bool, default=True) #False -> True

    config = args.parse_args()
    warnings.filterwarnings('ignore')
    
    #print("os.getcwd",os.getcwd())

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = 'cuda' if config.use_cuda == True else 'cpu'
    if hasattr(config, "num_threads") and int(config.num_threads) > 0:
        torch.set_num_threads(config.num_threads)

    #여기에 vocab 다르게 해주자!
    transcripts_dest = os.path.join(DATASET_PATH, 'train', 'train_data')
    transcript_df = pd.read_csv(transcripts_dest) #이미 하나로 모은 듯! transcript_df: []
    # # yj. 문장부호 등 다 없애주기 (전처리)
    transcript_df['text'] = transcript_df['text'].map(lambda x:rule(x))  #sentence_filter(x, 'phonetic')
    # # labels2.csv를 새로 만들기 (Vocab)
    label_dest =os.path.join(os.getcwd(), 'yj_labels.csv')
    generate_character_labels(transcript_df, label_dest) #-> 이건 main에서 하자!

    #labels.csv 대신 다른 거 쓰기
    vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'yj_labels.csv'), output_unit='character')
    #vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')
    model = build_model(config, vocab, device)
    optimizer = get_optimizer(model, config)
    bind_model(model, optimizer=optimizer)
    metric = get_metric(metric_name='CER', vocab=vocab)

    if config.pause:
        nova.paused(scope=locals())

    if config.mode == 'train':

        #Load DATASET
        config.dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data')
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
        
        #preprocessing
        #preprocessing(label_path, os.getcwd())
        preprocessing(transcript_df, os.getcwd())

        #TRAIN/VAL DATASET SPLIT -> 여기도 바꿔주기 (cross validation & transcript.txt -> 다른 걸로!)
        train_dataset, valid_dataset = split_and_cross_validate(config, os.path.join(os.getcwd(), 'yj_transcripts.txt'), vocab)
        #train_datasets, valid_datasets = split_and_cross_validate(config, os.path.join(os.getcwd(), 'transcripts.txt'), vocab)
        lr_scheduler = get_lr_scheduler(config, optimizer, len(train_dataset))
        optimizer = Optimizer(optimizer, lr_scheduler, int(len(train_dataset)*config.num_epochs), config.max_grad_norm)
        criterion = get_criterion(config, vocab)

        num_epochs = config.num_epochs
        num_workers = config.num_workers

        train_begin_time = time.time()

        for epoch in range(num_epochs):
            print('[INFO] Epoch %d start' % epoch)
            for fold_num in range(5):
                print('[INFO] Fold_No %d start' % fold_num)
                # train

                train_loader = DataLoader(
                    train_dataset[fold_num],
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=config.num_workers
                )

                model, train_loss, train_cer = trainer(
                    'train',
                    config,
                    train_loader,
                    optimizer,
                    model,
                    criterion,
                    metric,
                    train_begin_time,
                    device
                )
                print('[INFO] Epoch %d Folds %d (Training) Loss %0.4f CER %0.4f' % (epoch,fold_num, train_loss, train_cer))
                #print('[INFO] Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

                # valid

                valid_loader = DataLoader(
                    valid_dataset[fold_num],
                    #valid_dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=config.num_workers
                )

                model, valid_loss, valid_cer = trainer(
                    'valid',
                    config,
                    valid_loader,
                    optimizer,
                    model,
                    criterion,
                    metric,
                    train_begin_time,
                    device
                )
                print('[INFO] Epoch %d Folds %d (Validation) Loss %0.4f  CER %0.4f' % (epoch,fold_num, valid_loss, valid_cer))
                #print('[INFO] Epoch %d (Validation) Loss %0.4f  CER %0.4f' % (epoch, valid_loss, valid_cer))

                nova.report(
                    summary=True,
                    epoch=epoch,
                    train_loss=train_loss,
                    train_cer=train_cer,
                    val_loss=valid_loss,
                    val_cer=valid_cer
                )
                
                print(f'[INFO] fold {fold_num} is done')
            # checkpoint 저장
            if epoch % config.checkpoint_every == 0:
                nova.save(epoch)
            torch.cuda.empty_cache()
            print(f'[INFO] epoch {epoch} is done')
        print('[INFO] train process is done')
