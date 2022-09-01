#!/bin/bash

cd ..
NUM_CLASSES=(
    200
    196
    102
    195
    250
)
DATASETS_name=(
    'cubs_cropped'
    'stanford_cars_cropped'
    'flowers'
    'wikiart'
    'sketches'
)

GPU_ID=0

DATASET_PATH='/jmain02/home/J2AD019/exk04/jxj51-exk04/dataset/KM_dataset'

SEEDS=(
    1993
    1994
    1995
)
SAVE_PATH='results/cub2sketches_experiments/sub_mv2_adam'
CKPT='ckpt/subimagenet_mobilenetv2/lr0.01_nemb512_epoch160/checkpoints/KP_resume.pt'
CONFIG='configs/cub2sketches.yaml'

#SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[0]}
for TASK_ID in `seq 0 4`;
  do
    if [ "$TASK_ID" == "0" ]
    then
      PRETRAINED_CLASS=1000
      FINETUNE=${CKPT}
      USE_RECON_CODES=0

    else
      echo "prvious task_id: $((TASK_ID-1))"
      PRETRAINED_CLASS=${NUM_CLASSES[$((TASK_ID -1))]}
      FINETUNE='previous_task'
      USE_RECON_CODES=1

    fi
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG} --dataset ${DATASETS_name[$((TASK_ID))]} --dataset_path ${DATASET_PATH}/${DATASETS_name[$((TASK_ID))]} --results_dir $SAVE_PATH --end_class ${NUM_CLASSES[$((TASK_ID))]} --pretrained_end_class ${PRETRAINED_CLASS}  --finetune ${FINETUNE} --task_id $TASK_ID --seed $SEED  --use_recon_codes ${USE_RECON_CODES} --wandb_mode offline  -t optimizer=adam -t name=cub_mv2_sub_adam  -t arch=mobilenetv2 --disable_tqdm #-t epoch=30

done