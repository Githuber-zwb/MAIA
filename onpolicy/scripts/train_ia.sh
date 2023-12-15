#!/bin/sh
env="IA"
scenario="ia_simple" 
num_harvester=3
num_transporter=2
algo="mappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_ia.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_harvester ${num_harvester} -num_transporter ${num_transporter} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 500 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_wandb 
    # --world_width 6.0 --world_length 30.0 --working_width 1.0 --headland_width 3.0 --ridge_width 0.3 --harvester_max_v 5.0 --transporter_max_v 10.0
done