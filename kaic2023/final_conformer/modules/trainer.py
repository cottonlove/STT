# import torch
# import numpy as np
# import math
# from dataclasses import dataclass
# import time
# from nova import DATASET_PATH

#mode: 'train' 
#main.py에서 call 되는 것!

import torch
import numpy as np
import math
from dataclasses import dataclass
import time
from modules.vocab import KoreanSpeechVocabulary
from nova import DATASET_PATH
from glob import glob
import os
from modules.inference import single_infer


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
        #labels_np = np.ndarray(labels)
        if len(labels) > 15:
            continue
        else:
            labels_np = np.array(labels)
        #only one
        for i in glob(os.path.join(dataset_path, path)):
            #print(i)
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

def trainer(mode, config, dataloader, optimizer, model, criterion, metric, train_begin_time, device, vocab):

    log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] {mode} Start')
    epoch_begin_time = time.time()
    cnt = 0
    for inputs, targets, input_lengths, target_lengths in dataloader:
        begin_time = time.time()

        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        model = model.to(device)

        outputs, output_lengths = model(inputs, input_lengths, targets, target_lengths)


        loss = criterion(
            outputs.transpose(0, 1),
            targets[:, 1:],
            tuple(output_lengths),
            tuple(target_lengths)
        )

        y_hats = outputs.max(-1)[1]

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(model)

        total_num += int(input_lengths.sum())
        epoch_loss_total += loss.item()

        torch.cuda.empty_cache()

        if cnt % config.print_every == 0:

            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0
            train_elapsed = (current_time - train_begin_time) / 3600.0
            cer = metric(targets[:, 1:], y_hats)
            print(log_format.format(
                cnt, len(dataloader), loss,
                cer, elapsed, epoch_elapsed, train_elapsed,
                optimizer.get_lr(),
            ))
            if cnt % 1000 == 0:
                sample_audio_paths = dataloader.dataset.audio_paths[:15]
                sample_labels = dataloader.dataset.transcripts[:15]
                sample_result = sample_inference(sample_audio_paths, sample_labels, model, vocab)
                print(sample_result)
        cnt += 1
    return model, epoch_loss_total/len(dataloader), metric(targets[:, 1:], y_hats)



# Original
# def trainer(mode, config, dataloader, optimizer, model, criterion, metric, train_begin_time, device):

#     log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
#                               "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
#     total_num = 0
#     epoch_loss_total = 0.
#     print(f'[INFO] {mode} Start')
#     epoch_begin_time = time.time()
#     cnt = 0
#     for inputs, targets, input_lengths, target_lengths in dataloader:
#         begin_time = time.time()

#         optimizer.zero_grad()
#         inputs = inputs.to(device)
#         targets = targets.to(device)
#         input_lengths = input_lengths.to(device)
#         target_lengths = torch.as_tensor(target_lengths).to(device)
#         model = model.to(device)

#         outputs, output_lengths = model(inputs, input_lengths)

#         loss = criterion(
#             outputs.transpose(0, 1),
#             targets[:, 1:],
#             tuple(output_lengths),
#             tuple(target_lengths)
#         )

#         y_hats = outputs.max(-1)[1]

#         if mode == 'train':
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step(model)

#         total_num += int(input_lengths.sum())
#         epoch_loss_total += loss.item()

#         torch.cuda.empty_cache()

#         if cnt % config.print_every == 0:

#             current_time = time.time()
#             elapsed = current_time - begin_time
#             epoch_elapsed = (current_time - epoch_begin_time) / 60.0
#             train_elapsed = (current_time - train_begin_time) / 3600.0
#             cer = metric(targets[:, 1:], y_hats)
#             print(log_format.format(
#                 cnt, len(dataloader), loss,
#                 cer, elapsed, epoch_elapsed, train_elapsed,
#                 optimizer.get_lr(),
#             ))
#         cnt += 1
#     return model, epoch_loss_total/len(dataloader), metric(targets[:, 1:], y_hats)
