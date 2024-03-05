from torch import Tensor
from torch.utils.data import Dataset

from numpy.typing import ArrayLike


class GameplayDataset(Dataset):
    def __init__(self):
        self.player_hand = []
        self.opponent_hand = []
        self.actions = []
        self.win_label = []

    def record_hands_and_action(self, player_hand: ArrayLike, opponent_hand: ArrayLike, action: int):
        self.player_hand.append(Tensor(player_hand))
        self.opponent_hand.append(Tensor(opponent_hand))
        self.actions.append(action)

    def record_win_label(self, label: float):
        while len(self.win_label) < len(self.player_hand):
            self.win_label.append(float(label))

    def clear_unlabelebd(self):
        nr = len(self.win_label)
        for a in ["player_hand", "opponent_hand", "actions"]:
            setattr(self, a, getattr(self, a)[:nr])

    def __getitem__(self, i: int):
        return self.player_hand[i], self.opponent_hand[i], self.actions[i], self.win_label[i]

    def __len__(self):
        return len(self.player_hand)
