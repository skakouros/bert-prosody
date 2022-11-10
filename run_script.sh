#!/bin/bash
#
#
# Run script for prominence language model training
#
###################################################

# define vars
experiment_dir='./experiments'
expriment_name='test_exp'

# activate environment
# conda deactivate
# conda activate prosody

# run
python3 main.py \
    --model BertUncased \
    --train_set train_360 \
    --split_dir libritts/with_pos \
    --batch_size 32 \
    --epochs 2 \
    --save_path results_bert_small.txt \
    --log_every 50 \
    --learning_rate 0.00005 \
    --weight_decay 0 \
#    --gpu 0 \
    --fraction_of_train_data 1 \
    --optimizer adam \
    --seed 1234 \
    --nclasses 2 \
    --use_pos False \
    --exp_name $experiment_dir/$experiment_name
