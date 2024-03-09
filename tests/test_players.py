from itertools import product

from gin_rummy.gameplay.game_manager import GameManager
from gin_rummy import players
from gin_rummy.rl import policy_networks


class TestPlayers:
    def test_player_combos(self):
        player_klass = [p for p in players.__all__ if p != "HumanPlayer"]
        for p1, p2 in product(player_klass, player_klass):
            _ = GameManager().play_game(
                getattr(players, p1)(),
                getattr(players, p2)(),
            )

        for pn in policy_networks.__all__:
            _ = GameManager().play_game(
                players.RummyAgent(network=pn),
                players.SameKindPlayer()
            )
