from gin_rummy.players.card_points import CardPointPlayer
from gin_rummy.players.human import HumanPlayer
from gin_rummy.players.simple import SameKindPlayer
from gin_rummy.gameplay.game_manager import GameManager


if __name__ == "__main__":
    GameManager().play_game(
        CardPointPlayer(),
        SameKindPlayer(),
        verbose=2
    )
