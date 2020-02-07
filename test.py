"""
    This script is for evaluating the performance of vcii model
"""
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable


from evaluate import run_eval
from util import get_models, init_lstm, set_train, set_eval

def load_eval_model

def test():
    print('Start evaluation...')

    set_eval(nets)

    eval_loaders = get_eval_loaders()
    for eval_name, eval_loader in eval_loaders.items():
        eval_begin = time.time()
        eval_loss, mssim, psnr = run_eval(nets, eval_loader, args,
            output_suffix='iter%d' % train_iter)

        print('Evaluation @iter %d done in %d secs' % (
            train_iter, time.time() - eval_begin))
        print('%s Loss   : ' % eval_name
                + '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
        print('%s MS-SSIM: ' % eval_name
                + '\t'.join(['%.5f' % el for el in mssim.tolist()]))
        print('%s PSNR   : ' % eval_name
                + '\t'.join(['%.5f' % el for el in psnr.tolist()]))




