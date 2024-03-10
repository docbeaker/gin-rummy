import click

from pathlib import Path

from gin_rummy.rl.training import train_agent, RF
from gin_rummy.players import CardPointPlayer, RummyAgent


@click.command()
@click.argument("network", type=str)
@click.argument("output", type=Path)
@click.option("--games", type=int, default=500)
@click.option("--epochs", type=int, default=10)
@click.option("--algorithm", type=str, default=RF)
@click.option("--ignore-critic", is_flag=True, default=False)
@click.option("--lr", type=float, default=0.1)
@click.option("--lr-warmup-steps", type=int, default=0)
@click.option("--no-minibatch-scaling", is_flag=True, default=False)
@click.option("--minibatch-size", type=int, default=512)
@click.option("--max-minibatches", type=int, default=8)
@click.option("--epsilon", type=float, default=0.2)
@click.option("--workers", type=int, default=4)
def main(
    network: str,
    output: Path,
    games: int = 500,
    epochs: int = 10,
    algorithm: str = RF,
    ignore_critic: bool = False,
    lr: float = 0.1,
    lr_warmup_steps: int = 0,
    no_minibatch_scaling: bool = False,
    minibatch_size: int = 512,
    max_minibatches: int = 8,
    epsilon: float = 0.2,
    workers: int = 4
):
    agent = RummyAgent(network=network)
    opponent_pool = CardPointPlayer()

    agent = train_agent(
        epochs,
        games,
        agent,
        opponent_pool,
        algorithm=algorithm.upper(),
        ignore_critic=ignore_critic,
        lr=lr,
        lr_warmup_steps=lr_warmup_steps,
        scale_minibatch=not no_minibatch_scaling,
        minibatch_size=minibatch_size,
        max_minibatches=max_minibatches,
        epsilon=epsilon,
        output=output,
        always_save_checkpoint=False,
        num_workers=workers
    )


if __name__ == "__main__":
    main()
