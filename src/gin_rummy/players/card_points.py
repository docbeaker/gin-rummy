import numpy as np
from scipy.signal import convolve2d
from numpy.typing import ArrayLike

from ..gameplay.game_manager import BasePlayer


class CardPointPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        # build kernel
        self.points_kernel = np.zeros((7, 7))
        self.points_kernel[:, 3] = 1.5
        self.points_kernel[3] = [0, 1, 2, 1, 2, 1, 0]

    def compute_state(self, game: "GameManager"):
        return None

    def _compute_card_points(self, state) -> ArrayLike:
        return convolve2d(
            self.hand_matrix,
            self.points_kernel,
            mode="same"
        )

    def _choose_card_to_discard(self, state) -> int:
        """
        Run convolution, and then use argmin to return index in flattened array
        """
        hand_pts = self._compute_card_points(state)
        hand_pts = hand_pts + (1 - self.hand_matrix) * 10 * hand_pts.max()
        return hand_pts.argmin()
