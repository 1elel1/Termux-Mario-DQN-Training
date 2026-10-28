import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Model:
    def __init__(self, env_action_space_n):
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240*256*3, 128),
            nn.ReLU(),
            nn.Linear(128, env_action_space_n)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.target_model = copy.deepcopy(self.model)

    def train(self, replays):
        states = torch.tensor(np.stack([t.state for t in replays])).float()/255
        next_states = torch.tensor(np.stack([t.next_state for t in replays])).float()/255
        actions = torch.tensor([t.action for t in replays])
        rewards = torch.tensor([t.reward for t in replays])
        done = torch.tensor([t.done for t in replays])

        prediction = self.model(states).gather(1,actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1).values
            target = rewards + 0.9 * (1-done.float()) * next_q
        loss = self.criterion(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f"Loss: {loss.item():.4f}")

    def predict(self, state):
        return self.model(torch.tensor(state.copy()).float().unsqueeze(0)/255)
