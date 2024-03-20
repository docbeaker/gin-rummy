from torch import nn, Tensor, cat, no_grad, ones, tensor

from typing import Tuple, Optional


__all__ = [
    "ConvActorScoreCriticNN",
    "JointConvolutionNN"
]

default = "ConvActorScoreCriticNN"


class PolicyNetwork(nn.Module):
    """
    Policy Networks forward calls should return a tuple of
    - probability of discarding each card in hard, and
    - predicted win probability of the game.
    """
    @property
    def config(self):
        return dict(network=self.__class__.__name__, **self._config)

    @property
    def _config(self):
        return {}

    @staticmethod
    def cross_init_(
        param,
        scale: float = 1.0,
        std: float = 0.1,
        row_idx: int = 3,
        col_idx: int = 3,
        center_offset: float = -1.0,
    ):
        nn.init.normal_(param, std=std)
        with no_grad():
            if row_idx is not None:
                param.data[..., 3, :] += scale * ones(7)
            if col_idx is not None:
                param.data[..., 3] += scale * ones(7)
            param.data[..., 3, 3] += center_offset


class ConvActorScoreCriticNN(PolicyNetwork):
    def __init__(self, temperature: float = 1.0, channels: int = 32):
        super().__init__()
        self.t = temperature

        self.play_activation = nn.LogSoftmax(dim=-1)
        self.win_activation = nn.LogSigmoid()

        self.c2d = nn.Conv2d(
            1, 1, (7, 7), padding="same", bias=False
        )
        self.cross_init_(self.c2d.weight)

        self.nc = channels
        self.critic_conv = nn.Conv2d(
            1, self.nc, kernel_size=(7, 7), padding="same", bias=True
        )
        self.fc = nn.Linear(self.nc, 1, bias=True)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    @property
    def _config(self):
        return dict(channels=self.nc)

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        b, ns, nv = state.size()
        mask = state == 0

        state = state.unsqueeze(1)  # add dummy "channel" for convolution
        x = self.c2d(state).squeeze()
        x = x.masked_fill(mask, float("inf")).view(b, ns * nv)

        yp = state + self.critic_conv(state)  # here, take "residual" approach
        yp = yp.view(b, self.nc, ns * nv).max(dim=-1)[0]
        v = self.win_activation(self.fc(yp).squeeze(dim=-1))

        return self.play_activation(-x / self.t), v


class JointConvolutionNN(PolicyNetwork):
    def __init__(self, channels: int = 16, temperature: float = 1.0):
        super().__init__()
        self.t = temperature
        self.channels = channels
        self.play_activation = nn.LogSoftmax(dim=-1)
        self.win_activation = nn.LogSigmoid()

        self.c2d = nn.Conv2d(
            1, channels, (7, 7), padding="same", bias=True
        )

        # Play network: takes channel scores to single output
        self.playn = nn.Sequential(
            nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        # Value network: takes value in each channel to single output
        self.scoren = nn.Sequential(
            nn.Linear(channels, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    @property
    def _config(self):
        return dict(channels=self.channels)

    def _convolve(self, x: Tensor) -> Tensor:
        b, ns, nv = x.size()
        x = x.unsqueeze(1)
        return (x + self.c2d(x)).view(b, self.channels, ns * nv)

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        b, ns, nv = state.size()
        x = self._convolve(state)
        v = self.scoren(x.max(dim=-1)[0])
        card_scores = self.playn(x.transpose(-2, -1)).squeeze()  # transpose to use channel scores per card
        card_scores = card_scores.masked_fill(state.view(b, ns * nv) == 0, float("inf"))
        return self.play_activation(-card_scores / self.t), v
