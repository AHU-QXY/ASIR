# ASIR: Addressing Out-of-Distribution States for Offline Reinforcement Learning with Interference Robustness

Implementation of the ASIR algorithm.

## Environment
Paper results were collected with [MuJoCo 210](https://mujoco.org/) (and [mujoco-py 2.1.2.14](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.23.1](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/Farama-Foundation/D4RL). Networks are trained using [PyTorch 2.1.1](https://github.com/pytorch/pytorch) and [Python 3.9](https://www.python.org/).


## Usage


### Offline RL Training

Use the following command to train offline RL on D4RL, including Gym locomotion and Antmaze tasks, and save the models.
```
python main.py --env halfcheetah-medium-v2 --lam 0.25 --alpha 0.05 --max_weight 2 --save_model
python main.py --env antmaze-umaze-v2 --lam 0.25 --alpha 0.5 --max_weight 40 --no_normalize --save_model
```

For all AntMaze tasks except antmaze-umaze-diverse, you must add the `--no_normalize` flag.
### Logging
To monitor and save your experiment runs, you can use either TensorBoard or Weights & Biases (W&B).

 **To log with TensorBoard:**
  Append the `--use_tensorboard` flag to your command when running the script.

**To log with Weights & Biases (W&B):**
Append the `--use_wandb` fflag to your command when running the script.
