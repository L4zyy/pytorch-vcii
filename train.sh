#!/bin/bash
if (( $# != 1 )); then
    echo "Usage: ./train.sh [0-2], e.g. ./train.sh 2"
    exit
fi
hier=$1

modeldir=model

# train="data/train"
# eval="data/eval"
# train_mv="data/train_mv"
# eval_mv="data/eval_mv"
# train="of/train"
# eval="of/eval"
# train_mv="of/train_mv"
# eval_mv="of/eval_mv"
train="/mnt/nfs/scratch1/zhiyilai/vcii/data/train"
eval="/mnt/nfs/scratch1/zhiyilai/vcii/data/val"
train_mv="/mnt/nfs/scratch1/zhiyilai/vcii/data/train_of"
eval_mv="/mnt/nfs/scratch1/zhiyilai/vcii/data/val_of"

if [[ ${hier} == "0" ]]; then
  distance1=6
  distance2=6
  bits=16
  encoder_fuse_level=1
  decoder_fuse_level=1
elif [[ ${hier} == "1" ]]; then
  distance1=3
  distance2=3
  bits=16
  encoder_fuse_level=2
  decoder_fuse_level=3
elif [[ ${hier} == "2" ]]; then
  distance1=1
  distance2=2
  bits=48
  encoder_fuse_level=1
  decoder_fuse_level=1
else
  echo "Usage: ./train.sh [0-2], e.g. ./train.sh 2"
  exit
fi

# Warning: with --save-out-img, output images are stored
# each time we run evaluation. This can take a lot of space
# when using a big evaluation dataset.
# (for the demo data it's okay.)


python -u train.py \
  --train ${train} \
  --train-mv ${train_mv} \
  --encoder-fuse-level ${encoder_fuse_level} \
  --decoder-fuse-level ${decoder_fuse_level} \
  --v-compress --warp --stack --fuse-encoder \
  --bits ${bits} \
  --distance1 ${distance1} --distance2 ${distance2} \
  --save-model-name of-48 \
  --load-model-name of-48 \
  --load-iter 100001 \
  --max-train-iters 150001
