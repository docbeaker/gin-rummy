import click

from pathlib import Path

from gin_rummy import players
from gin_rummy.gameplay.game_manager import GameManager, BasePlayer
from gin_rummy.gameplay.playouts import run_playouts


@click.command()
@click.argument("opponent", type=str)
@click.option("--opponent-model", type=Path)
@click.option("--player", default=players.HumanPlayer.__name__, required=False, type=str)
@click.option("--player-model", type=Path)
@click.option("--go-first/--go-second", is_flag=True)
@click.option("--games", type=int, default=1)
def play_game(
    opponent: str,
    opponent_model: Path = None,
    player: str = None,
    player_model: Path = None,
    go_first: bool = None,
    games: int = 1,
):
    assert games > 0, "you want to play <= 0 games?"

    opp = getattr(players, opponent)()
    if opponent_model:
        opp.load_model(opponent_model)
    play = getattr(players, player)()
    if player_model:
        play.load_model(player_model)

    if games > 1:
        assert not (opp.requires_input or play.requires_input), "Humans should play games one at at time"
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
