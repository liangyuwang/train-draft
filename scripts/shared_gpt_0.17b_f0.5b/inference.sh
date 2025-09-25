#!/bin/bash
# ================================
# Torch Distributed Inference Script
# ================================

NUM_NODES=1
NUM_GPUS=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500

DISTRIBUTED_ARGS="\
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
"

INFERENCE_ARGS="\
  --seed 1337 \
  --tokenizer_name gpt2 \
  --ckpt ./log/checkpoint_009440_model.pt \
  --prompt 'Once upon a time' \
  --max_new_tokens 50 \
  --temperature 1.0 \
  --use_compile \
"

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

torchrun $DISTRIBUTED_ARGS inference.py $INFERENCE_ARGS $MODEL_ARGS
