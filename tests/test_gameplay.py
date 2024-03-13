from numpy import where
from numpy.random import choice
from gin_rummy.gameplay.game_manager import BasePlayer, NV, GameManager


class RandomPlayer(BasePlayer):
    def _choose_card_to_discard(self, state) -> int:
        sidx, vidx = where(self.hand_matrix)
        card_idx = choice(len(sidx))
        return NV * sidx[card_idx] + vidx[card_idx]


class TestWinConditions:
    straight4 = ["3D", "4D", "5D", "6D"]
    kind4 = [f"9C", "9D", "9H", "9S"]

    def test_winning_combos(self):
        combos = [
            ["JS", "JC", "JH"] + self.kind4,
            self.straight4 + ["10C", "10H", "10D"],
            self.kind4 + self.straight4[:3],
            self.straight4 + ["JS", "QS", "KS"],
            ["6S", "7S", "8S"] + self.kind4,
        ]

        for _combo in combos:
            player = RandomPlayer()
            for c in _combo:
                player.accept_card(c)
            assert player.hand_matrix.sum() == 7, "wrong number of cards"
            assert player.check_for_victory(), f"failed for {player.get_hand(human=True)}"

    def test_nonwinning_combos(self):
        combos = [
            self.kind4 + ["8C", "10C", "QC"],
            self.kind4[1:] + ["10C", "10H", "10D", "JS"],
            ["2C", "3C", "7D", "8H", "JH", "AS", "2S"]
        ]

        for _combo in combos:
            player = RandomPlayer()
            for c in _combo:
                player.accept_card(c)
            assert player.hand_matrix.sum() == 7, "wrong number of cards"
            assert not player.check_for_victory(), f"failed for {player.get_hand(human=True)}"

    def test_discard_logic(self):
        p1 = RandomPlayer()
        p2 = RandomPlayer()

        gm = GameManager(max_reshuffles=0)
        gm.play_game(p1, p2)

        # Check that the discard is right
        state = gm.get_3d_state(p1, p2)
        discarded_cards = set(gm.discard_pile)
        for idx, is_discarded in enumerate(state[-1].flatten()):
            if is_discarded:
                assert idx in discarded_cards
            else:
                assert idx not in discarded_cards
