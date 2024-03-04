from typing import Union, List
from tqdm import tqdm
from random import choice

from gin_rummy.gameplay.game_manager import GameManager, BasePlayer


def run_playouts(
    n_games: int,
    player: BasePlayer,
    opponent_pool: Union[List[BasePlayer], BasePlayer]
):
    player_wins, valid_games = 0, 0
    gm = GameManager()

    starting_turn = choice([0, 1])
    for _ in tqdm(range(n_games)):
        opponent = opponent_pool if isinstance(opponent_pool, BasePlayer) else choice(opponent_pool)
        winner = gm.play_game(
            player,
            opponent,
            turn=starting_turn,
            verbose=0,
        )
        if winner < 0:
            # It's a draw, so forget this game
            player.dataset.clear_unrewarded()
        else:
            player_wins += 1 - winner
            valid_games += 1
            # Record win for all moves in play
            if player.dataset is not None:
                player.dataset.apply_game_reward(1.0 - winner)

        starting_turn = 1 - starting_turn

    return player_wins, valid_games
