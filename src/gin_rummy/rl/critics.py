from torch import nn, no_grad, Tensor


class MLPCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(52, 128),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.LogSigmoid()
        )

    def forward(self, player_hand: Tensor, opponent_hand: Tensor):
        B, ns, nv = player_hand.size()
        x = player_hand - opponent_hand
        x = x.view(B, ns * nv)
        x = self.layers(x).squeeze()
        return x
