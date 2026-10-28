from collections import deque
import random

class ReplayBuffer:
	def __init__(self, capacity):
		self.replays = deque()
		self.buffer_size = capacity


	def add(self, replay):
		self.replays.append(replay)
		if len(self.replays) > self.buffer_size:
			self.replays.pop()


	def sample(self, batch_size):
		return random.sample(self.replays, batch_size)
		

class Replay:
	def __init__(self, state, action, reward, next_state, done):
		self.state = state
		self.action = action
		self.next_state = next_state
		self.done = done
		self.reward = reward