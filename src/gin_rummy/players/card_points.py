import numpy as np
from scipy.signal import convolve2d
from typing import Tuple, Optional

from torch import nn, Tensor, no_grad, ones, tensor
from torch.distributions import Categorical

from ..gameplay.game_manager import BasePlayer


class CardPointPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        # build kernel
        self.points_kernel = np.zeros((7, 7))
        self.points_kernel[:, 3] = 1.5
        self.points_kernel[3] = [0, 1, 2, 1, 2, 1, 0]

    def _choose_card_to_discard(self, discard_top: int) -> int:
        """
        Run convolution, and then use argmin to return index in flattened array

        Does not use discard top
        """
        hand_pts = convolve2d(
            self.hand_matrix,
            self.points_kernel,
            mode="same"
        )
        hand_pts = hand_pts + (1 - self.hand_matrix) * 10 * hand_pts.max()
        return hand_pts.argmin()


class PointsConvolutionNN(nn.Module):
    init_scale = 0.0
    init_bias = 0.0

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.t = temperature
        self.c2d = nn.Conv2d(
            1, 1, (7, 7), padding="same", bias=False
        )
        self.play_activation = nn.LogSoftmax(dim=-1)
        self.win_activation = nn.LogSigmoid()
        self.scale = nn.Parameter(tensor(self.init_scale))
        self.bias = nn.Parameter(tensor(self.init_bias))

        nn.init.normal_(self.c2d.weight, std=0.1)
        with no_grad():
            self.c2d.weight.data[..., 3, :] += ones(7)
            self.c2d.weight.data[..., 3] += ones(7)
            self.c2d.weight.data[..., 3, 3] -= 1

    def forward(self, player_hand: Tensor, opponent_hand: Tensor = None) -> Tuple[Tensor, Optional[Tensor]]:
        B, ns, nv = player_hand.size()
        x = self.c2d(player_hand.unsqueeze(1)).squeeze()  # add dummy "channel" for convolution
        loga = x.masked_fill(player_hand == 0, float("inf")).view(B, ns * nv)
        loga = self.play_activation(-loga / self.t)
        if opponent_hand is not None:
            y = self.c2d(opponent_hand.unsqueeze(1)).squeeze()
            logw = self.scale * (x.sum(dim=(-1, -2)) - y.sum(dim=(-1, -2))) + self.bias
            logw = self.win_activation(logw)
        else:
            logw = None
        return loga, logw


class CardPointNNPlayer(CardPointPlayer):
    def __init__(self):
        super().__init__()
        self.model = PointsConvolutionNN()
        # For debugging: initialize the model so that it matches the manual implementation
        if False:
            self.model.c2d.weight.data = Tensor(
                np.expand_dims(self.points_kernel, (0, 1))
            )

    def _choose_card_to_discard(self, discard_top: int) -> int:
        hm_tensor = Tensor(self.hand_matrix)
        with no_grad():
            logits, _ = self.model(hm_tensor.unsqueeze(0))  # No batch here, so make a batch dimension
        m = Categorical(logits=logits.squeeze())  # squeeze to remove batch
        a_idx = int(m.sample())
        return a_idx
