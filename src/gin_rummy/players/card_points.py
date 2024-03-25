import numpy as np
from scipy.signal import convolve2d
from typing import Tuple
from numpy.typing import ArrayLike

from ..gameplay.game_manager import BasePlayer


class CardPointPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        # build kernel
        self.points_kernel = np.zeros((7, 7))
        self.points_kernel[:, 3] = 1.5
        self.points_kernel[3] = [0, 1, 2, 1, 2, 1, 0]

    def _choose_card_to_discard(self, state: ArrayLike) -> Tuple[int, float]:
        """
        Run convolution, and then use argmin to return index in flattened array

        Does not use discard top
        """
        hand_pts = convolve2d(
            state,
            self.points_kernel,
            mode="same"
        )
        hand_pts = hand_pts + (1 - self.hand_matrix) * 10 * hand_pts.max()
        return int(hand_pts.argmin()), 0.0
