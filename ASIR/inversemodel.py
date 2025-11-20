from torch import nn
from  torch.nn import functional as F
import torch


class inverse(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super().__init__()
		self.max_action = max_action
		# Q1 architecture
		self.l1 = nn.Linear(state_dim * 2, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, action_dim)

	def forward(self, state, next_state):
		sa = torch.cat([state, next_state], 1)

		out = F.relu(self.l1(sa))
		out = F.relu(self.l2(out))
		out = F.relu(self.l3(out))
		out = F.tanh(self.l4(out)) * self.max_action
		return out