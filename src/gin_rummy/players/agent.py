from torch import Tensor, no_grad, load, exp
from torch.distributions import Categorical
from pathlib import Path
from typing import Tuple
from numpy.typing import ArrayLike

from ..gameplay.game_manager import BasePlayer
from ..rl import policy_networks


class RummyAgent(BasePlayer):
    def __init__(self, network: str = policy_networks.default, **network_kwargs):
        super().__init__()
        self.model = getattr(policy_networks, network)(**network_kwargs)

    def set_temperature(self, temperature: float):
        self.model.t = temperature

    def load_model(self, model_fp: Path):
        self.model.load_state_dict(load(model_fp)["model"])

    def _choose_card_to_discard(self, state: ArrayLike) -> Tuple[int, float]:
        hm_tensor = Tensor(state)
        with no_grad():
            logits, log_pwin = self.model(hm_tensor.unsqueeze(0))  # No batch here, so make a batch dimension
        m = Categorical(logits=logits.squeeze())  # squeeze to remove batch
        a_idx = int(m.sample())
        return a_idx, exp(log_pwin).squeeze().item()
