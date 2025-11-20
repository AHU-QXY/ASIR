import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import utils
import inversemodel
import wandb
from torch.utils.tensorboard import SummaryWriter
import ASIR
import random
import time
def eval_policy(policy, env_name, seed, mean, std, eval_episodes=100,seed_offset=100):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)
	eval_env.action_space.seed(seed + seed_offset)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score

if __name__ == "__main__":
	new_directory = "/home/qxy/ASIR"
	os.chdir(new_directory)
	

	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--env", default="walker2d-expert-v2")   
	parser.add_argument("--seed", default=7, type=int)  
	parser.add_argument("--eval_freq", default=None, type=int)      
	parser.add_argument("--eval_episodes", default=None, type=int)
	parser.add_argument("--device", default="cuda:0")
	parser.add_argument("--max_timesteps", default=1e6, type=int)  
	parser.add_argument("--batch_size", default=256, type=int)     
	parser.add_argument("--tau", default=0.005)                   
	parser.add_argument("--no_normalize", action="store_true")         
	parser.add_argument("--lam", default=0.25, type=float)         
	parser.add_argument("--alpha", default=0.1, type=float)          
	parser.add_argument("--max_weight", default=3, type=float)
	parser.add_argument("--save_model", action="store_true")
	parser.add_argument("--use_tensorboard", action="store_true")
	parser.add_argument("--use_wandb", action="store_true")
	args = parser.parse_args()
	print("---------------------------------------")
	print(f"Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.use_tensorboard:
		writer = SummaryWriter(log_dir=f'{args.env}_{time.strftime("%Y-%m-%d_%H-%M-%S")}_{args.nu}_{args.max_weight}-{args.seed}')
		writer.add_text("config", str(vars(args)))
	if args.use_wandb:
		wan = utils.use_wandb(args)
	env = gym.make(args.env)
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	
	inverse_model = inversemodel.inverse(state_dim, action_dim, max_action).to(args.device)
	inverse_model.load_state_dict(torch.load('./inverse/inverse_%s.pt'%(args.env)))
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim,args.device)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
	
	
	if not args.no_normalize:				
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1


	if 'antmaze' in args.env:
		replay_buffer.reward = np.where(replay_buffer.reward == 1.0, 0.0, -1.0)
		antmaze = True
		args.eval_episodes = 100 if args.eval_episodes is None else args.eval_episodes
		args.eval_freq = 50000 if args.eval_freq is None else args.eval_freq
		expectile = 0.9 # follow IQL
		temp = 10.0 
	else:
		antmaze = False
		args.eval_episodes = 10 if args.eval_episodes is None else args.eval_episodes
		args.eval_freq = 20000 if args.eval_freq is None else args.eval_freq
		expectile = 0.7 # follow IQL
		temp = 3.0 
	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"replay_buffer": replay_buffer,
		"inverse":inverse_model,
		"tau":args.tau,
		"temp": temp,
		"expectile": expectile,
		"lam": args.lam,
		"alpha": args.alpha,
		"max_weight": args.max_weight,
 		"device": args.device,
	}

	policy = ASIR.ASIR(**kwargs)
	for t in range(int(args.max_timesteps)):
		policy.train_offline(args.batch_size)
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			score = eval_policy(policy, args.env, args.seed,mean=mean,std=std, eval_episodes= args.eval_episodes)
			if args.use_tensorboard:
			
				writer.add_scalar('score', score, t+1)
			if args.use_wandb:
				wandb.log({"score": score, "timestep": t+1})
	writer.close()
	if args.save_model:
		policy.save("model",args.env)
	