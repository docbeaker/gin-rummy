from gin_rummy.gameplay.game_manager import GameManager
from gin_rummy.players.card_points import CardPointPlayer, CardPointNNPlayer
from gin_rummy.players.human import HumanPlayer


if __name__ == "__main__":
    GameManager().play_game(
        CardPointNNPlayer(initialized=False),
        CardPointPlayer(),
        verbose=True
    )