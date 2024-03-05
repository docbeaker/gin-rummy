import click

from gin_rummy import players
from gin_rummy.gameplay.game_manager import GameManager


@click.command()
@click.argument("opponent", type=str)
@click.option("--player", default=players.HumanPlayer.__name__, required=False, type=str)
@click.option("--go-first/--go-second", is_flag=True)
def play_game(opponent: str, player: str = None, go_first: bool = None):
    if go_first is None:
        turn = None
    elif go_first:
        turn = 0
    else:
        turn = 1

    opp = getattr(players, opponent)()
    play = getattr(players, player)()

    _ = GameManager().play_game(
        play,
        opp,
        turn=turn,
        verbose=2
    )


if __name__ == "__main__":
    play_game()
