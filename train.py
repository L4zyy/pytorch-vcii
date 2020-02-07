"""
    Modified version of training code for vcii ECCV2018
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
from train_options import parser
from training_oprations import save, resume
from util import get_models, init_lstm, set_train, set_eval, load_model
from util import prepare_inputs, forward_ctx



def train():
    
    # Using the original argument parser
    args = parser.parse_args()
    print(args)


    # Load training data
    train_loader = get_loader(
        is_train=True,
        root=args.train, mv_dir=args.train_mv, 
        args=args
    )

    # Load model
    nets, solver, milestones, scheduler = load_model(args)

    # Check if resume training
    train_iter = 0
    just_resumed = False
    if args.load_model_name:
        print('Loading %s@iter %d' % (args.load_model_name,
                                    args.load_iter))

        nets = resume(args, nets, args.load_iter)
        train_iter = args.load_iter
        scheduler.last_epoch = train_iter - 1
        just_resumed = True


    # Start training
    while True:

        for batch, (crops, ctx_frames, _) in enumerate(train_loader):
            scheduler.step()
            train_iter += 1

            if train_iter > args.max_train_iters:
                break

            batch_t0 = time.time()

            solver.zero_grad()

            # Init LSTM states.
            (encoder_h_1, encoder_h_2, encoder_h_3,
            decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_lstm(
                batch_size=(crops[0].size(0) * args.num_crops), height=crops[0].size(2),
                width=crops[0].size(3), args=args)

            # Forward U-net.
            if len(nets) == 4:
                unet = nets[3]
            else:
                unet = None

            if args.v_compress:
                unet_output1, unet_output2 = forward_ctx(unet, ctx_frames)
            else:
                unet_output1 = Variable(torch.zeros(args.batch_size,)).cuda()
                unet_output2 = Variable(torch.zeros(args.batch_size,)).cuda()

            res, frame1, frame2, warped_unet_output1, warped_unet_output2 = prepare_inputs(
                crops, args, unet_output1, unet_output2)

            losses = []

            bp_t0 = time.time()
            _, _, height, width = res.size()

            out_img = torch.zeros(1, 3, height, width).cuda() + 0.5

            for _ in range(args.iterations):
                if args.v_compress and args.stack:
                    encoder_input = torch.cat([frame1, res, frame2], dim=1)
                else:
                    encoder_input = res

                # Encode.
                encoder = nets[0]
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    encoder_input, encoder_h_1, encoder_h_2, encoder_h_3,
                    warped_unet_output1, warped_unet_output2)

                # Binarize.
                binarizer = nets[1]
                codes = binarizer(encoded)

                # Decode.
                decoder = nets[2]
                (output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = decoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,
                    warped_unet_output1, warped_unet_output2)

                # loss function
                res = res - output
                out_img = out_img + output.data
                losses.append(res.abs().mean())

            bp_t1 = time.time()

            loss = sum(losses) / args.iterations
            loss.backward()

            for net in [encoder, binarizer, decoder, unet]:
                if net is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)

            solver.step()

            batch_t1 = time.time()

            print(
                '[TRAIN] Iter[{}]; LR: {}; Loss: {:.6f}; Backprop: {:.4f} sec; Batch: {:.4f} sec'.
                format(train_iter, 
                    scheduler.get_lr()[0], 
                    loss.item(),
                    bp_t1 - bp_t0, 
                    batch_t1 - batch_t0))

            if train_iter % 100 == 0:
                print('Loss at each step:')
                print(('{:.4f} ' * args.iterations +
                    '\n').format(* [l.data[0] for l in losses]))

            if train_iter % args.checkpoint_iters == 0:
                save(args, nets, train_iter)

        if train_iter > args.max_train_iters:
            print('Training done.')
            break


if __name__ == "__main__":
    train()