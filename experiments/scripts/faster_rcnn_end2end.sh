#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh  [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 iei /mnt/data/train/ /mnt/data/test/ \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

EXPERIMENT_NAME=$1
PROTOTXT_DIR=$2
WEIGHTS=$3
TRAIN_PATH=$4
TEST_PATH=$5
MAX_ITERS=$6
GPU_ID=$7

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:6:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB="${EXPERIMENT_NAME}_train"
TEST_IMDB="${EXPERIMENT_NAME}_test"

LOG="experiments/logs/faster_rcnn_end2end_${EXPERIMENT_NAME}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --solver ${PROTOTXT_DIR}/solver.prototxt \
  --weights ${WEIGHTS} \
  --imdb ${TRAIN_IMDB} \
  --imdb_path ${TRAIN_PATH} \
  --iters ${MAX_ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --gpu ${GPU_ID} \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --def ${PROTOTXT_DIR}/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --imdb_path ${TEST_PATH} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --gpu ${GPU_ID} \
  ${EXTRA_ARGS}
