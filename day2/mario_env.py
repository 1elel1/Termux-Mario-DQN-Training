from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import pyglet
pyglet.compat_platform = 'linux'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--times", type=int, default=50000)
parser.add_argument("--batch_num", type=int, default=4)
args = parser.parse_args()

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
dqn_l_dir = os.path.dirname(current_dir)
sys.path.append(dqn_l_dir)
from day1.eps_greedy import select_action
from day1.replay_buffer import ReplayBuffer, Replay

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

import numpy as np
from model import Model
model = Model(env.action_space.n)

state, info = env.reset()
replays = ReplayBuffer(1000)
for step in range(args.times):
	for replay_times in range(4):
		action = select_action(model, state)#epsilon逻辑
		total_reward = 0
		old_state = state
		for i in range(8):
			env.render()
			state, reward, done, truncated, info = env.step(action)
			total_reward += reward
			if done:
				break
		if done:
			state, info = env.reset()
		print("Step:", step, " Reward:", reward, " X coordinate:", info['x_pos'])
		replays.add(Replay(old_state, action, total_reward, state, done)) 
	model.train(replays.sample(args.batch_num))#tderror逻辑和双网络逻辑
	if step % 500 == 0: 							model.target_model.load_state_dict(model.model.state_dict())
env.close()

