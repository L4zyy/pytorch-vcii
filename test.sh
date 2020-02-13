#!/bin/bash
if (( $# != 3 )); then
    echo "Usage: ./test.sh [0-2] [checkpoint name] [checkpoint iteration], e.g. ./test.sh 2 demo 101"
    exit
fi
hier=$1
ckpt_name=$2
ckpt_iter=$3

modeldir=model

eval="data/eval"
eval_mv="data/eval_mv"

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
  bits=8
  encoder_fuse_level=1
  decoder_fuse_level=1
else
  echo "Usage: ./test.sh [0-2], e.g. ./test.sh 2 demo 101"
  exit
fi

# Warning: with --save-out-img, output images are stored
# each time we run evaluation. This can take a lot of space
# when using a big evaluation dataset.
# (for the demo data it's okay.)


python -u test.py \
  --eval ${eval} \
  --eval-mv ${eval_mv} \
  --encoder-fuse-level ${encoder_fuse_level} \
  --decoder-fuse-level ${decoder_fuse_level} \
  --v-compress --warp --stack --fuse-encoder \
  --bits ${bits} \
  --distance1 ${distance1} --distance2 ${distance2} \
  --load-model-name ${ckpt_name} \
  --load-iter ${ckpt_iter} \
  --save-out-img
