from typing import Union, List
from tqdm import tqdm
from random import choice
from typing import Tuple

from gin_rummy.gameplay.game_manager import GameManager, BasePlayer
from gin_rummy.rl.recorder import GameplayDataset


def run_playouts(
    n_games: int,
    player: BasePlayer,
    opponent_pool: Union[List[BasePlayer], BasePlayer]
) -> Tuple[int, int, GameplayDataset]:
    player_wins, valid_games = 0, 0
    gm = GameManager(record=True)

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
            # NOTE: could do 0.5 reward? But not informative, so perhaps just clear
            gm.dataset.clear_unlabelebd()
        else:
            player_wins += 1 - winner
            valid_games += 1
            gm.dataset.record_win_label(1 - winner)

        starting_turn = 1 - starting_turn

    return player_wins, valid_games, gm.dataset
