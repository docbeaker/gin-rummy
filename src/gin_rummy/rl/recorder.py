from torch import Tensor
from torch.utils.data import Dataset


class GameplayDataset(Dataset):
    def __init__(self):
        """
        TODO: will likely need to record hands as well
        """
        self.states = []
        self.actions = []
        self.rewards = []

    def append(self, state: Tensor, action: int, reward: float = None):
        self.states.append(state)
        self.actions.append(action)
        if reward:
            self.rewards.append(reward)

    def apply_game_reward(self, reward: float):
        while len(self.rewards) < len(self.states):
            self.rewards.append(reward)

    def clear_unrewarded(self):
        nr = len(self.rewards)
        self.states = self.states[:nr]
        self.actions = self.actions[:nr]

    def __getitem__(self, i: int):
        return self.states[i], self.actions[i], self.rewards[i]

    def __len__(self):
        return len(self.states)
