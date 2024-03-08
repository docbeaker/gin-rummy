from itertools import product

from gin_rummy.gameplay.game_manager import GameManager
from gin_rummy import players


class TestPlayers:
    def test_player_combos(self):
        player_klass = [p for p in players.__all__ if p != "HumanPlayer"]
        for p1, p2 in product(player_klass, player_klass):
            _ = GameManager().play_game(
                getattr(players, p1)(),
                getattr(players, p2)(),
            )
