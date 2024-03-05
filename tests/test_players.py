from itertools import product

from gin_rummy.gameplay.game_manager import GameManager
from gin_rummy.players.simple import SameKindPlayer
from gin_rummy.players.card_points import CardPointPlayer, CardPointNNPlayer


class TestPlayers:
    def test_player_combos(self):
        player_klass = [SameKindPlayer, CardPointPlayer, CardPointNNPlayer]
        for p1, p2 in product(player_klass, player_klass):
            _ = GameManager().play_game(p1(), p2())
