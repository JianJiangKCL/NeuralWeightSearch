#!/bin/bash

cd ..

SEEDS=(
 1993
 1994
 1995
)

#SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[0]}
SAVE_PATH='results/cifar_experiments/sub_vgg_adam'

GPU_ID=0
DATASET_PATH='/jmain02/home/J2AD019/exk04/jxj51-exk04/dataset/cifar100_org'

CKPT='ckpt/subimagenet_vgg16/lr0.01_nemb512_epoch160/checkpoints/KP_resume.pt'

CONFIG='configs/splitcifar.yaml'
for TASK_ID in `seq 0 19`;
  do
    if [ "$TASK_ID" == "0" ]
    then
      PRETRAINED_CLASS=1000
      FINETUNE=${CKPT}
      USE_RECON_CODES=0

    else
      echo "prvious task_id: $((TASK_ID-1))"
      PRETRAINED_CLASS=5
      FINETUNE='previous_task'
      USE_RECON_CODES=1


    fi
   CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG} --dataset_path ${DATASET_PATH} --results_dir $SAVE_PATH --end_class 5 --pretrained_end_class ${PRETRAINED_CLASS}  --finetune ${FINETUNE} --task_id $TASK_ID --seed $SEED  --use_recon_codes ${USE_RECON_CODES} --wandb_mode offline --disable_tqdm -t optimizer=adam -t name=adam_vgg_ciar -t arch=vgg16 #-t epoch=30

done