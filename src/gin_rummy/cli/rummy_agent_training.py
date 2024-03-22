import click

from pathlib import Path

from gin_rummy.rl.training import train_agent, RF, PPO
from gin_rummy.players import CardPointPlayer, RummyAgent
from gin_rummy.rl.policy_networks import __all__ as pno, default as default_policy


@click.command()
@click.argument("output", type=Path)
@click.option("--policy", type=click.Choice(pno), default=default_policy)
@click.option("--games", type=int, required=True)
@click.option("--steps", type=int, required=True)
@click.option("--algorithm", type=click.Choice([RF, PPO]), default=RF)
@click.option("--gael", type=float, default=1.0)
@click.option("--lr", type=float, default=0.1)
@click.option("--lr-warmup-steps", type=int, default=0)
@click.option("--no-minibatch-scaling", is_flag=True, default=False)
@click.option("--minibatch-size", type=int, default=0)
@click.option("--k-epochs", type=int, default=1)
@click.option("--epsilon", type=float, default=0.2)
@click.option("--workers", type=int, default=4)
def main(
    output: Path,
    policy: str = default_policy,
    games: int = 500,
    steps: int = 10,
    algorithm: str = RF,
    gael: float = 1.0,
    lr: float = 0.1,
    lr_warmup_steps: int = 0,
    no_minibatch_scaling: bool = False,
    minibatch_size: int = 0,
    k_epochs: int = 1,
    epsilon: float = 0.2,
    workers: int = 4
):
    agent = RummyAgent(network=policy)
    opponent_pool = CardPointPlayer()

    agent = train_agent(
        steps,
        games,
        agent,
        opponent_pool,
        algorithm=algorithm.upper(),
        gael=gael,
        lr=lr,
        lr_warmup_steps=lr_warmup_steps,
        scale_minibatch=not no_minibatch_scaling,
        minibatch_size=minibatch_size,
        k_epochs_per_step=k_epochs,
        epsilon=epsilon,
        output=output,
        always_save_checkpoint=False,
        num_workers=workers
    )


if __name__ == "__main__":
    main()
