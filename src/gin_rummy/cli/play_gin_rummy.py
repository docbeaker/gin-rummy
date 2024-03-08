import click

from pathlib import Path

from gin_rummy import players
from gin_rummy.gameplay.game_manager import GameManager
from gin_rummy.gameplay.playouts import run_playouts


def create_player(player_name: str, model_path: Path = None):
    player = getattr(players, player_name)()
    if model_path:
        player.load_model(model_path)
    return player


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

    play = create_player(player, model_path=player_model)
    opp = create_player(opponent, model_path=opponent_model)

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
