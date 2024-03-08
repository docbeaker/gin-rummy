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

    if isinstance(opponent_pool, BasePlayer):
        opponent_pool = [opponent_pool]
    for p in [player] + opponent_pool:
        if hasattr(p, "model"):
            p.model.eval()

    starting_turn = choice([0, 1])
    for _ in tqdm(range(n_games)):
        opponent = choice(opponent_pool)

        winner = gm.play_game(
            player,
            opponent,
            turn=starting_turn,
            verbose=0,
        )
        if winner >= 0:
            player_wins += 1 - winner
            valid_games += 1

        starting_turn = 1 - starting_turn

    return player_wins, valid_games, gm.dataset
