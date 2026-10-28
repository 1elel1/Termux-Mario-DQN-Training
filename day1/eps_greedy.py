import random
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", type=float, default=0.05)
args = parser.parse_args()

def act(q_values, eps):
	q_values = q_values.squeeze(0)
	num_actions = q_values.shape[0]
	if random.random() < 1-eps:
		return q_values.argmax().item()
	else: 
		return random.randrange(num_actions)

def select_action(model, state):
	return act(model.predict(state), args.epsilon)
