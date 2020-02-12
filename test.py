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


from dataset import get_loader
from evaluate import run_eval
from util import get_models, init_lstm, set_train, set_eval, load_model

from test_options import parser

def get_eval_loaders(args):
  # We can extend this dict to evaluate on multiple datasets.
  eval_loaders = {
    'TVL': get_loader(
        is_train=False,
        root=args.eval, mv_dir=args.eval_mv,
        args=args),
  }
  return eval_loaders

def test():

    args = parser.parse_args()
    print(args)
    print('Start evaluation...')
    # Load model
    nets, solver, milestones, scheduler = load_model(args)

    # Get params from checkpoint
    names = ['encoder', 'binarizer', 'decoder', 'unet']

    if args.load_model_name:
        print('Loading %s@iter %d' % (args.load_model_name,
                                    args.load_iter))

        index = args.load_model_name, args.load_iter
        train_iter = args.load_iter
        scheduler.last_epoch = train_iter - 1
    else:
        print("please specify the model and iterration for evaluation")

    for net_idx, net in enumerate(nets):
        if net is not None:
            name = names[net_idx]
            checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(
                args.model_dir, args.save_model_name, 
                name, index)

            print('Loading %s from %s...' % (name, checkpoint_path))
            net.load_state_dict(torch.load(checkpoint_path))

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




