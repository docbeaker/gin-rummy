from gin_rummy.gameplay.game_manager import GameManager
from gin_rummy.players.card_points import CardPointPlayer


if __name__ == "__main__":
    GameManager().play_game(CardPointPlayer(), CardPointPlayer())