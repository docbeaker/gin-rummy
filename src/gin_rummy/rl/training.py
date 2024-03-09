"""
Useful references:

https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
"""

import torch
from torch.utils.data import DataLoader
from time import time
from pathlib import Path

from typing import List, Union

from gin_rummy.gameplay.game_manager import BasePlayer
from gin_rummy.gameplay.playouts import run_playouts
from gin_rummy.players.agent import RummyAgent


def save_checkpoint(
    output: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    **kwargs
):
    state = {
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict(),
    }
    for k, v in kwargs.items():
        state[k] = v
    torch.save(state, output)


def get_lr(step: int, lr: float, lr_warmup_steps: int):
    if step >= lr_warmup_steps:
        return lr
    return step * lr / lr_warmup_steps


def train_agent(
    n_epochs: int,
    n_games: int,
    player: RummyAgent,
    opponent_pool: Union[List[BasePlayer], BasePlayer],
    ignore_critic: bool = False,
    lr: float = 0.1,
    lr_warmup_steps: int = 0,
    unscaled_rewards: bool = False,
    output: Path = None,
    always_save_checkpoint: bool = False,
    num_workers: int = 4,
) -> BasePlayer:
    assert hasattr(player, "model"), "player must be using a model for training!"

    optimizer = torch.optim.Adam(player.model.parameters(), lr=0.1)
    critic_loss_function = torch.nn.BCEWithLogitsLoss()

    if output:
        output.mkdir(exist_ok=True, parents=True)

    win_rates = []
    best_win_rate = 0.0
    for e in range(n_epochs + 1):
        st = time()
        n_wins, n_played, dataset = run_playouts(n_games, player, opponent_pool)
        if not n_played:
            print(f"Epoch {e} failed to generate any valid games!")
            continue
        print(
            f"Epoch {e}: agent win percentage = {100 * n_wins / n_played: .1f}% "
            f"({n_played} valid games)"
        )

        # Check if we should be saving a model
        wr = n_wins / n_played
        win_rates.append(wr)
        if output and e and ((wr > best_win_rate) or always_save_checkpoint):
            save_checkpoint(Path(output, "ckpt.pt"), player.model, optimizer, win_rates=win_rates)
            print("Model checkpoint saved!")
        best_win_rate = max(wr, best_win_rate)

        if e == n_epochs:
            break

        elr = get_lr(e + 1, lr, lr_warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = elr

        dl = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=True,
            num_workers=num_workers,
        )

        # Basic REINFORCE algorithm to start
        player.model.train()
        for state, ostate, action, reward in dl:
            optimizer.zero_grad()

            logp, logr = player.model(state, None if ignore_critic else ostate)
            if ignore_critic:
                critic_loss = 0.0
            else:
                critic_loss = critic_loss_function(logr, reward)
                with torch.no_grad():
                    reward = reward - torch.exp(logr)

            # TODO: make configurable
            if not unscaled_rewards:
                reward = (reward - reward.mean()) / (reward.std() + 1E-8)
            logp_a = logp.gather(-1, action.unsqueeze(-1)).squeeze()
            actor_loss = -(reward * logp_a).mean()

            loss = actor_loss + critic_loss
            loss.backward()
            optimizer.step()

        print(f"Epoch completed in {time() - st:.2f}s")

    if output:
        save_checkpoint(
            Path(output, f"model.pt"),
            player.model,
            optimizer,
            win_rates=win_rates
        )

    return player
