# Joint-Intrinsic-Motivation
Code for paper "Leveraging Coordination with Joint Intrinsic Motivation in Multi-Agent Deep Reinforcement Learning".

## Getting started:

With python 3.8 and pip, in a virtual environment:
- Install our version multi-agent particle environment: `cd` into the "multiagent-particle-envs" directory and run `pip install -e .`.
- Install the requirements: `cd` into the main directory and run `pip install -r requirements.txt`.

## Running experiments

To train the model, run the command `bash /scripts/qmix_train.sh`. To run the three variants in the paper (QMIX, QMIX+LIM, QMIX+JIM), change the parameters ``intrinsic_reward_mode`` and ``intrinsic_reward_algo``:
* QMIX: ``intrinsic_reward_algo="none"``
* QMIX+LIM: ``intrinsic_reward_algo="e2snoveld"`` and ``intrinsic_reward_mode="local"``
* QMIX+JIM: ``intrinsic_reward_algo="e2snoveld"`` and ``intrinsic_reward_mode="central"``

Change the `env` parameter to choose which task you want to run training on. The different tasks are available in the "scenario" folder.

Check the technical appendix for hyperparameters used in our experiments.

