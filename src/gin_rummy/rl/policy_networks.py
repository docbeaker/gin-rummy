from torch import nn, Tensor, cat, no_grad, ones, tensor

from typing import Tuple, Optional


__all__ = [
    "SimpleConvolutionNN",
    "ConvActorScoreCriticNN",
    "JointConvolutionNN"
]

default = "SimpleConvolutionNN"


class PolicyNetwork(nn.Module):
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


class SimpleConvolutionNN(PolicyNetwork):
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
        self.nc = 32
        self.critic_conv = nn.Conv2d(
            1, self.nc, kernel_size=(7, 7), padding="same", bias=True
        )
        nn.init.normal_(self.critic_conv.weight, std=0.02)
        nn.init.zeros_(self.critic_conv.bias)
        # TODO: would dropout be helpful?
        self.fc = nn.Linear(2 * self.nc, 2 * self.nc)
        nn.init.normal_(self.fc.weight, std=0.02)
        self.fc_out = nn.Linear(2 * self.nc, 1)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def _critic_forward(self, x: Tensor):
        B, ns, nv = x.size()
        cx = (
                x.unsqueeze(1) + self.critic_conv(x.unsqueeze(1))
        ).view(B, self.nc, ns * nv)
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
            s = self.fc_out(nn.functional.relu(s))
            s = self.win_activation(s.squeeze())
        else:
            s = None
        return self.play_activation(-x / self.t), s


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
        self.fc = nn.Linear(self.nc, 1, bias=False)
        nn.init.normal_(self.fc.weight, std=0.02)
        self.bias = nn.Parameter(tensor(0.0))

    @property
    def _config(self):
        return dict(channels=self.nc)

    def forward(self, player_hand: Tensor, opponent_hand: Tensor = None) -> Tuple[Tensor, Optional[Tensor]]:
        b, ns, nv = player_hand.size()
        mask = player_hand == 0
        player_hand = player_hand.unsqueeze(1)  # add dummy "channel" for convolution
        x = self.c2d(player_hand).squeeze()
        x = x.masked_fill(mask, float("inf")).view(b, ns * nv)

        if opponent_hand is not None:
            yp = player_hand + self.critic_conv(player_hand)
            yp = yp.view(b, self.nc, ns * nv).max(dim=-1)[0]
            v = self.win_activation(self.bias + self.fc(yp).squeeze())
        else:
            v = None

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
        self.bias = nn.Parameter(tensor(0.0))

    @property
    def _config(self):
        return dict(channels=self.channels)

    def _convolve(self, x: Tensor) -> Tensor:
        b, ns, nv = x.size()
        x = x.unsqueeze(1)
        return (x + self.c2d(x)).view(b, self.channels, ns * nv)

    def forward(self, player_hand: Tensor, opponent_hand: Tensor = None) -> Tuple[Tensor, Optional[Tensor]]:
        b, ns, nv = player_hand.size()
        x = self._convolve(player_hand)
        if opponent_hand is not None:
            y = self._convolve(opponent_hand)
            v = self.bias + self.scoren(x.max(dim=-1)[0]) - self.scoren(y.max(dim=-1)[0])
            v = self.win_activation(v).squeeze()
        else:
            v = None
        card_scores = self.playn(x.transpose(-2, -1)).squeeze()  # transpose to use channel scores per card
        card_scores = card_scores.masked_fill(player_hand.view(b, ns * nv) == 0, float("inf"))
        return self.play_activation(-card_scores / self.t), v
