#!/bin/sh
env="IA"
scenario="ia_simple" 
num_harvester=3
num_transporter=1
algo="mappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ./onpolicy/scripts/train/eval_ia.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_harvester ${num_harvester} --num_transporter ${num_transporter} --seed ${seed} \
    --n_rollout_threads 128 --episode_length 500 --num_env_steps 200000000 \
    --ppo_epoch 10 --use_wandb --hidden_size 1024 --layer_N 2 --entropy_coef 0.02\
    --harvester_max_v 3.0 3.5 4.0 --cap_harvester 60.0 80.0 100.0 --transporter_max_v 10.0 --cap_transporter 200.0 --field_length 300.0 \
    --d_range 5.0 --dt 0.5
done