from torch import Tensor
from torch.utils.data import Dataset

from numpy.typing import ArrayLike


class GameplayDataset(Dataset):
    def __init__(self):
        self.state = []
        self.actions = []
        self.v_next = []
        self.win_label = []

    def record_state(self, state: ArrayLike, action: int, v_next: float):
        # Shift the recorded values by one so each state is associated with
        # the value prediction for the next state (effectively, throw away first)
        if len(self.v_next) == len(self.actions) - 1:
            self.v_next.append(v_next)
        self.state.append(Tensor(state))
        self.actions.append(action)

    def record_win(self, label: float):
        # This is effectively discounted reward from the end state, with discount factor gamma = 1
        while len(self.win_label) < len(self.state):
            self.win_label.append(float(label))
        nv, na = len(self.v_next), len(self.actions)
        assert na - 1 == nv, f"actions (n = {na}) and values (n = {nv}) should be off by one!"
        # Set v_next to be rT (terminal reward) to handle last state
        self.v_next.append(label)

    def clear_unlabeled(self):
        # compute existing length for which we have labels, and truncate everything to that
        nr = len(self.win_label)
        for a in ["state", "actions", "v_next"]:
            setattr(self, a, getattr(self, a)[:nr])

    def __getitem__(self, i: int):
        return self.state[i], self.actions[i], self.v_next[i], self.win_label[i]

    def __len__(self):
        return len(self.state)
