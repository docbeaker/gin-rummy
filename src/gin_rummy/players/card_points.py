import numpy as np
from scipy.signal import convolve2d
from typing import Tuple, Optional

from torch import nn, Tensor, no_grad, ones, cat
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
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.t = temperature
        self.c2d = nn.Conv2d(
            1, 1, (7, 7), padding="same", bias=False
        )
        self.play_activation = nn.LogSoftmax(dim=-1)
        self.win_activation = nn.LogSigmoid()

        self.cross_init_(self.c2d.weight)

        # Critic definition
        self.critic_conv = nn.Conv2d(
            1, 8, kernel_size=(5, 5), padding="same", bias=True
        )
        nn.init.normal_(self.critic_conv.weight, std=0.02)
        nn.init.zeros_(self.critic_conv.bias)
        # TODO: would dropout be helpful?
        self.fc = nn.Linear(16, 1)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    @staticmethod
    def cross_init_(param, std: float = 0.1):
        nn.init.normal_(param, std=std)
        with no_grad():
            param.data[..., 3, :] += ones(7)
            param.data[..., 3] += ones(7)
            param.data[..., 3, 3] -= 1

    def _critic_forward(self, x: Tensor):
        B, ns, nv = x.size()
        cx = self.critic_conv(x.unsqueeze(1)).view(B, 8, ns * nv)
        # max pooling across channels
        return cx.max(dim=-1)[0]

    def forward(self, player_hand: Tensor, opponent_hand: Tensor = None) -> Tuple[Tensor, Optional[Tensor]]:
        B, ns, nv = player_hand.size()
        x = self.c2d(player_hand.unsqueeze(1)).squeeze()  # add dummy "channel" for convolution
        x = x.masked_fill(player_hand == 0, float("inf")).view(B, ns * nv)
        if opponent_hand is not None:
            pc = self._critic_forward(player_hand)
            oc = self._critic_forward(opponent_hand)
            s = self.fc(cat((pc, oc), dim=-1))
            s = self.win_activation(s.squeeze())
        else:
            s = None
        return self.play_activation(-x / self.t), s


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
