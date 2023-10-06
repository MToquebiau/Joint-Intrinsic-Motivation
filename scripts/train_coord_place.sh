#!/bin/sh
n_run=1
env="scenarios/coordinated_placement.py"
model_name="qmix_JIM"
sce_conf_path="configs/2a_pol.json"
n_frames=10000000
n_explo_frames=8000000
episode_length=100
frames_per_update=100
eval_every=100000
eval_scenar_file="eval_scenarios/hard_corners_24.json"
init_explo_rate=0.3
epsilon_decay_fn="linear"
intrinsic_reward_mode="central" # "local" for LIM
intrinsic_reward_algo="e2snoveld" # "none" for QMIX vanilla
int_reward_coeff=2.0 # 4.0 for LIM
int_reward_decay_fn="constant"
gamma=0.99
int_rew_enc_dim=64 # 36 for LIM 
int_rew_hidden_dim=512 # 256 for LIM
scale_fac=0.5
int_rew_lr=0.0001
cuda_device="cuda:0"

source venv/bin/activate

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python train_qmix.py\
    --env_path ${env}\
    --model_name ${model_name}\
    --sce_conf_path ${sce_conf_path}\
    --seed ${seed} \
    --n_frames ${n_frames}\
    --cuda_device ${cuda_device}\
    --gamma ${gamma}\
    --episode_length ${episode_length}\
    --frames_per_update ${frames_per_update} \
    --init_explo_rate ${init_explo_rate}\
    --n_explo_frames ${n_explo_frames}\
    --use_per\
    --intrinsic_reward_mode ${intrinsic_reward_mode}\
    --intrinsic_reward_algo ${intrinsic_reward_algo}\
    --int_reward_coeff ${int_reward_coeff}\
    --int_reward_decay_fn ${int_reward_decay_fn}\
    --scale_fac ${scale_fac}\
    --int_rew_lr ${int_rew_lr}\
    --int_rew_enc_dim ${int_rew_enc_dim}\
    --int_rew_hidden_dim ${int_rew_hidden_dim}\
    --eval_every ${eval_every}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
