from numpy.random import random

from gin_rummy.gameplay.game_manager import BasePlayer


class SameKindPlayer(BasePlayer):
    def _choose_card_to_discard(self, state) -> int:
        matching_kinds = self.hand_matrix.sum(axis=0)
        pts = self.hand_matrix * matching_kinds + 5 * (1 - self.hand_matrix)
        pts = pts + 0.05 * random(size=self.hand_matrix.shape)
        return pts.argmin()