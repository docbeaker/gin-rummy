import click

from pathlib import Path

from gin_rummy.rl.training import train_agent
from gin_rummy.players.card_points import CardPointNNPlayer, CardPointPlayer


@click.command()
@click.argument("output")
@click.option("--games", type=int, default=500)
@click.option("--epochs", type=int, default=10)
@click.option("--critic", type=str, required=False)
@click.option("--workers", type=int, default=4)
def main(output: Path, games: int = 500, epochs: int = 10, critic: str = None, workers: int = 4):
    agent = CardPointNNPlayer()
    opponent_pool = CardPointPlayer()

    agent = train_agent(
        epochs,
        games,
        agent,
        opponent_pool,
        critic_type=critic,
        output=Path(output, "ckpt.pt"),
        always_save_checkpoint=False,
        num_workers=workers
    )


if __name__ == "__main__":
    main()
