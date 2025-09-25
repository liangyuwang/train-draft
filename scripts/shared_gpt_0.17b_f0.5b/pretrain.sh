#!/bin/bash
# ================================
# Torch Distributed Training Script
# ================================

NUM_NODES=1
NUM_GPUS=8
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
B=8
USE_MUON=1
if [ $USE_MUON -eq 1 ]; then
  unset CUBLAS_WORKSPACE_CONFIG
fi

DISTRIBUTED_ARGS="\
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
"

TRAINING_ARGS="\
  --seed 1337 \
  --dataset_path ../data/fineweb-edu-sample-10BT/ \
  --log_dir ./log \
  --tokenizer_name gpt2 \
  --total_batch_size 2097152 \
  --B $B \
  --T 4096 \
  --shift 1 \
  --max_lr 2e-3 \
  --min_lr 3e-5 \
  --weight_decay 0.1 \
  --grad_clip_value 1.0 \
  --warmup_steps 2000 \
  --max_epochs 1 \
  --do_save \
  --save_every_steps 500 \
  --use_compile \
"
if [ $USE_MUON -eq 1 ]; then
  TRAINING_ARGS="$TRAINING_ARGS --use_muon"
fi

MODEL_ARGS="\
  --block_size 4096 \
  --vocab_size 151936 \
  --num_layer 20 \
  --num_attention_heads 32 \
  --num_key_value_heads 4 \
  --hidden_size 1024 \
  --intermediate_size 4096 \
  --dropout 0.0 \
  --tied_lm_head \
  --use_moe_ratio 0.0 \
  --use_shared_layers \
  --shared_layers 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
"

torchrun $DISTRIBUTED_ARGS train.py $TRAINING_ARGS $MODEL_ARGS
