import click

from torch import load
from pathlib import Path

from gin_rummy import players
from gin_rummy.gameplay.game_manager import GameManager
from gin_rummy.gameplay.playouts import run_playouts
from gin_rummy.rl.policy_networks import default


def create_player(player_name_or_path: str):
    if Path(player_name_or_path).exists():
        config = load(player_name_or_path).get("config", dict(network=default))
        player = players.RummyAgent(**config)
        player.load_model(Path(player_name_or_path))
    else:
        assert hasattr(players, player_name_or_path), f"{player_name_or_path} is not a valid player"
        player = getattr(players, player_name_or_path)()
    return player


@click.command()
@click.argument("opponent", type=str)
@click.option("--player", default=players.HumanPlayer.__name__, required=False, type=str)
@click.option("--go-first/--go-second", is_flag=True)
@click.option("--games", type=int, default=1)
def play_game(
    opponent: str,
    player: str = None,
    go_first: bool = None,
    games: int = 1,
):
    if games <= 0:
        print(f"You want to play {games} games?")
        return

    play = create_player(player)
    opp = create_player(opponent)

    if games > 1:
        if opp.requires_input or play.requires_input:
            print("Humans should play games one at a time")
            return
        nwin, nvalid, _ = run_playouts(games, play, opp)
        print(
            f"Win = {nwin / games:.3f}, Draw = {1 - nvalid / games:.3f}, Loss = {(nvalid - nwin) / games:.3f}"
        )
        return

    if go_first is None:
        turn = None
    elif go_first:
        turn = 0
    else:
        turn = 1

    _ = GameManager().play_game(
        play,
        opp,
        turn=turn,
        verbose=2
    )


if __name__ == "__main__":
    play_game()
