# Joint-Intrinsic-Motivation
Code for paper "Joint Intrinsic Motivation for Coordinated Exploration in Multi-Agent Reinforcement Learning".

To train the model, run the script '/scripts/qmix_train.h'. To run the three variants in the paper (QMIX, QMIX+LIM, QMIX+JIM), change the parameters ``intrinsic_reward_mode`` and ``intrinsic_reward_algo``:
* QMIX: ``intrinsic_reward_algo="none"``
* QMIX+LIM: ``intrinsic_reward_algo="e2snoveld"`` and ``intrinsic_reward_mode="local"``
* QMIX+JIM: ``intrinsic_reward_algo="e2snoveld"`` and ``intrinsic_reward_mode="central"``

In the experiments shown in the paper, we used different hyperparameters for the architecture of the intrinsic reward module in JIM and LIM:
* QMIX+LIM: ``int_rew_enc_dim=32`` and ``int_rew_hidden_dim=256``
* QMIX+JIM: ``int_rew_enc_dim=64`` and ``int_rew_hidden_dim=512``

This is because the input of the intrinsic reward module changes between LIM and JIM: LIM takes the local observations and JIM takes the joint-observation. Thus, we scale the number of neurons to fit the inputs and to have similar counts of parameters between the two versions. 
