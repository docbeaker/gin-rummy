from numpy import where
from numpy.random import choice
from gin_rummy.gameplay.game_manager import BasePlayer, NV


class TestPlayer(BasePlayer):
    def _choose_card_to_discard(self) -> int:
        sidx, vidx = where(self.hand_matrix)
        card_idx = choice(len(sidx))
        return NV * sidx[card_idx] + vidx[card_idx]


class TestWinConditions:
    straight4 = [x + 13 for x in range(2, 6)]
    kind4 = [8 + 13 * i for i in range(4)]

    def test_winning_combos(self):
        combos = {
            "4kind&3kind": [3 + 13, 3 + 26, 3 + 39] + self.kind4,
            "4straight&3kind": self.straight4 + [8, 8 + 26, 8 + 39],
            "4kind&3straight": self.kind4 + self.straight4[:3],
            "4straight&3straight": self.straight4 + [x + 39 for x in range(9, 12)],
            "overlapping": [6 + 39, 7 + 39, 9 + 39] + self.kind4,
        }

        for _combo_name, _combo in combos.items():
            player = TestPlayer()
            for c in _combo:
                player.accept_card(c)
            assert player.hand_matrix.sum() == 7, "wrong number of cards"
            assert player.check_for_victory(), f"failed for {_combo_name}"

    def test_nonwinning_combos(self):
        combos = {
            "4and2-or-3and3": self.kind4 + [7, 9, 39],
            "two3s": self.kind4[1:] + [4] + [5 + 13 * i for i in range(3)]
        }

        for _combo_name, _combo in combos.items():
            player = TestPlayer()
            for c in _combo:
                player.accept_card(c)
            assert player.hand_matrix.sum() == 7, "wrong number of cards"
            assert not player.check_for_victory(), f"failed for {_combo_name}"
