import click

from pathlib import Path

from gin_rummy.rl.training import train_agent
from gin_rummy.players import CardPointPlayer, RummyAgent


@click.command()
@click.argument("network", type=str)
@click.argument("output", type=Path)
@click.option("--games", type=int, default=500)
@click.option("--epochs", type=int, default=10)
@click.option("--ignore-critic", is_flag=True, default=False)
@click.option("--lr", type=float, default=0.1)
@click.option("--lr-warmup-steps", type=int, default=0)
@click.option("--unscaled-rewards", is_flag=True, default=False)
@click.option("--workers", type=int, default=4)
def main(
    network: str,
    output: Path,
    games: int = 500,
    epochs: int = 10,
    ignore_critic: bool = False,
    lr: float = 0.1,
    lr_warmup_steps: int = 0,
    unscaled_rewards: bool = False,
    workers: int = 4
):
    agent = RummyAgent(network=network)
    opponent_pool = CardPointPlayer()

    agent = train_agent(
        epochs,
        games,
        agent,
        opponent_pool,
        ignore_critic=ignore_critic,
        lr=lr,
        lr_warmup_steps=lr_warmup_steps,
        unscaled_rewards=unscaled_rewards,
        output=output,
        always_save_checkpoint=False,
        num_workers=workers
    )


if __name__ == "__main__":
    main()
