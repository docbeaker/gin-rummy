import numpy as np
from scipy.signal import convolve2d
from numpy.typing import ArrayLike

from torch import nn, Tensor, no_grad
from torch.distributions import Categorical

from ..gameplay.game_manager import BasePlayer
from ..rl.recorder import GameplayDataset


class CardPointPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        # build kernel
        self.points_kernel = np.zeros((7, 7))
        self.points_kernel[:, 3] = 1.5
        self.points_kernel[3] = [0, 1, 2, 1, 2, 1, 0]

    def compute_state(self, game: "GameManager"):
        return None

    def _choose_card_to_discard(self, state) -> int:
        """
        Run convolution, and then use argmin to return index in flattened array
        """
        hand_pts = convolve2d(
            self.hand_matrix,
            self.points_kernel,
            mode="same"
        )
        hand_pts = hand_pts + (1 - self.hand_matrix) * 10 * hand_pts.max()
        return hand_pts.argmin()


class PointsFilterNN(nn.Module):
    def __init__(self, temperature: float=1.0, init_kernel: ArrayLike=None):
        super().__init__()
        self.c2d = nn.Conv2d(
            1, 1, (7, 7), padding="same", bias=False
        )
        self.t = temperature
        self.activation = nn.LogSoftmax(dim=-1)
        if init_kernel is not None:
            self.c2d.weight.data = Tensor(
                np.expand_dims(init_kernel, (0, 1))
            )

    def forward(self, state: Tensor):
        B, ns, nv = state.size()
        x = -self.c2d(state.unsqueeze(1)).squeeze()  # put in one channel
        x = x.masked_fill(state == 0, float("-inf")).view(B, ns * nv)
        return self.activation(x / self.t)


class CardPointNNPlayer(CardPointPlayer):
    def __init__(self, initialized: bool = False):
        super().__init__()
        self.model = PointsFilterNN(
            init_kernel=self.points_kernel if initialized else None
        )

    def _choose_card_to_discard(self, state) -> int:
        hm_tensor = Tensor(self.hand_matrix)
        with no_grad():
            logits = self.model(hm_tensor.unsqueeze(0))  # No batch here, so make a batch dimension
        m = Categorical(nn.functional.softmax(logits.squeeze(), dim=-1))
        a_idx = int(m.sample())
        if self.dataset is not None:
            self.dataset.append(hm_tensor, a_idx)
        return a_idx
