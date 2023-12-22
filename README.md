# Joint-Intrinsic-Motivation

## Getting started:

With python 3.8 and pip, in a virtual environment:
- Install our version multi-agent particle environment: `cd` into the "multiagent-particle-envs" directory and run `pip install -e .`.
- Install the requirements: `cd` into the main directory and run `pip install -r requirements.txt`.

## Running experiments

To train the model, run the command `bash /scripts/train_coop_push.sh`. To run the three variants in the paper (QMIX, QMIX+LIM, QMIX+JIM), change the parameters ``intrinsic_reward_mode`` and ``intrinsic_reward_algo``:
* QMIX: ``intrinsic_reward_algo="none"``
* QMIX+LIM: ``intrinsic_reward_algo="e2snoveld"`` and ``intrinsic_reward_mode="local"``
* QMIX+JIM: ``intrinsic_reward_algo="e2snoveld"`` and ``intrinsic_reward_mode="central"``

In the `scripts/` directory are multiple scripts that allow to run all the experiments shown in the paper.

Check the technical appendix for hyperparameters used in our experiments.

## Citation

If you use this work, please cite cthe following paper:

```
@inproceedings{JIM2024,
  title={Joint Intrinsic Motivation for Coordinated Exploration in Multi-agent Deep Reinforcement Learning},
  author={Toquebiau, Maxime and Benamar, Faïz and Bredeche, Nicolas and Jun, Jae Yun},
  booktitle={Proceedings of the 23rd Conference on Autonomous Agents and Multiagent Systems},
  year={2024}
}
```
