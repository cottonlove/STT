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
# from nova import DATASET_PATH
from glob import glob
import os
# from modules.inference import single_infer

# from main import config
import wandb
import random

# Original
# TODO (wandb로 train loss랑 validation loss, accuracy 기록하기)
def trainer(mode, config, dataloader, optimizer, model, criterion, metric, train_begin_time, device):

    log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] {mode} Start')
    epoch_begin_time = time.time()
    cnt = 0
    if mode == "train":
        for inputs, targets, input_lengths, target_lengths in dataloader:
            begin_time = time.time()
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = torch.as_tensor(target_lengths).to(device)
            model = model.to(device)

            outputs, output_lengths = model(inputs, input_lengths)

            loss = criterion(
                outputs.transpose(0, 1),
                targets[:, 1:],
                tuple(output_lengths),
                tuple(target_lengths)
            )

            y_hats = outputs.max(-1)[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(model)
            total_num += int(input_lengths.sum())
            epoch_loss_total += loss.item()

            torch.cuda.empty_cache()
            
            wandb.log({"Training loss/step": loss})
            cer = metric(targets[:, 1:], y_hats)
            wandb.log({"Training CER/step": cer})
            
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

            cnt += 1
        wandb.log({"Training CER/epoch": metric(targets[:, 1:],y_hats)}, step = cnt)  
        wandb.log({"Training loss/epoch": epoch_loss_total/len(dataloader)}, step = cnt)
    else: #Validation
        for inputs, targets, input_lengths, target_lengths in dataloader:
            begin_time = time.time()

            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = torch.as_tensor(target_lengths).to(device)
            model = model.to(device)

            outputs, output_lengths = model(inputs, input_lengths)

            loss = criterion(
                outputs.transpose(0, 1),
                targets[:, 1:],
                tuple(output_lengths),
                tuple(target_lengths)
            )

            y_hats = outputs.max(-1)[1]
            total_num += int(input_lengths.sum())
            epoch_loss_total += loss.item()

            torch.cuda.empty_cache()
            wandb.log({"Validation loss/step": loss})
            cer = metric(targets[:, 1:], y_hats)
            wandb.log({"Validation CER/step": cer})
            
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
            cnt += 1
        wandb.log({"Validation CER/epoch": metric(targets[:, 1:],y_hats)}, step = cnt)  
        wandb.log({"Validation loss/epoch": epoch_loss_total/len(dataloader)}, step = cnt)
    return model, epoch_loss_total/len(dataloader), metric(targets[:, 1:], y_hats)