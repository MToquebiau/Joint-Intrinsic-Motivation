#!/bin/sh
n_run=1
env="scenarios/rel_overgen.py"
model_name="qmix_4a_40_09_JIM"
sce_conf_path="configs/2a_pol.json"
n_frames=12000000
n_explo_frames=10000000
episode_length=40
frames_per_update=40
eval_every=100000
eval_scenar_file="eval_scenarios/hard_corners_24.json"
init_explo_rate=0.1
epsilon_decay_fn="linear"
intrinsic_reward_mode="central" # "local" for LIM
intrinsic_reward_algo="e2snoveld" # "none" for QMIX vanilla
int_reward_coeff=1.0
int_reward_decay_fn="constant"
gamma=0.99
int_rew_enc_dim=64 # 32 for LIM
int_rew_hidden_dim=256 # 64 for LIM
scale_fac=0.5 # def 0.5
int_rew_lr=0.0002 # 0.0001 for LIM
state_dim=40
optimal_diffusion_coeff=0.9
suboptimal_diffusion_coeff=0.08
ro_n_agents=4
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
    --eval_every ${eval_every}\
    --state_dim ${state_dim}\
    --optimal_diffusion_coeff ${optimal_diffusion_coeff}\
    --suboptimal_diffusion_coeff ${suboptimal_diffusion_coeff}\
    --ro_n_agents ${ro_n_agents}"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
