#!/bin/sh
n_run=1
env="scenarios/rel_overgen.py"
model_name="qmix_jim"
n_frames=10000000
n_explo_frames=9000000
episode_length=100 # def 100
frames_per_update=100
init_explo_rate=0.3
epsilon_decay_fn="linear"
intrinsic_reward_mode="central"
intrinsic_reward_algo="e2snoveld"
int_reward_coeff=1.0
int_reward_decay_fn="constant"
gamma=0.99
int_rew_enc_dim=64
int_rew_hidden_dim=512 
scale_fac=0.5 # def 0.5
int_rew_lr=0.0001 # def 0.0001
state_dim=40
optimal_diffusion_coeff=40
cuda_device="cuda:3"

for n in $(seq 1 $n_run)
do
    printf "Run ${n}/${n_run}\n"
    seed=$RANDOM
    comm="python train_qmix.py --env_path ${env} --model_name ${model_name} --seed ${seed} \
--n_frames ${n_frames} --cuda_device ${cuda_device} --gamma ${gamma} --frames_per_update ${frames_per_update} \
--init_explo_rate ${init_explo_rate} --n_explo_frames ${n_explo_frames} --use_per \
--intrinsic_reward_mode ${intrinsic_reward_mode} --intrinsic_reward_algo ${intrinsic_reward_algo} \
--int_reward_coeff ${int_reward_coeff} --int_reward_decay_fn ${int_reward_decay_fn} \
--scale_fac ${scale_fac} --int_rew_lr ${int_rew_lr} --int_rew_enc_dim ${int_rew_enc_dim} --int_rew_hidden_dim ${int_rew_hidden_dim} \
--state_dim ${state_dim} --optimal_diffusion_coeff ${optimal_diffusion_coeff} --save_visited_states"
    printf "Starting training with command:\n${comm}\n\nSEED IS ${seed}\n"
    eval $comm
    printf "DONE\n\n"
done
