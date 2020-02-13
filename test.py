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
    encoder, binarizer, decoder, unet = get_models(
        args=args, v_compress=args.v_compress, 
        bits=args.bits,
        encoder_fuse_level=args.encoder_fuse_level,
        decoder_fuse_level=args.decoder_fuse_level)

    nets = [encoder, binarizer, decoder]
    if unet is not None:
        nets.append(unet)

    # Using GPUS
    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    if len(gpus) > 1:
        print("Using GPUs {}.".format(gpus))
        for net in nets:
            net = nn.DataParallel(net, device_ids=gpus)

    # Get params from checkpoint
    names = ['encoder', 'binarizer', 'decoder', 'unet']

    if args.load_model_name:
        print('Loading %s@iter %d' % (args.load_model_name,
                                    args.load_iter))

        index = args.load_iter
        train_iter = args.load_iter
    else:
        print("please specify the model and iterration for evaluation")
        exit(1)

    for net_idx, net in enumerate(nets):
        if net is not None:
            # print(">>target net:")
            # print(net)
            name = names[net_idx]
            checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(
                args.model_dir, args.load_model_name, 
                name, index)

            print('Loading %s from %s...' % (name, checkpoint_path))
            loaded_net = torch.load(checkpoint_path)
            # print(">>loaded:")
            # print(loaded_net)
            net.load_state_dict(loaded_net)

    set_eval(nets)

    eval_loaders = get_eval_loaders(args)
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




if __name__ == "__main__":
    test()
