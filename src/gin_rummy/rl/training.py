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
from gin_rummy.rl import critics


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


def train_agent(
    n_epochs: int,
    n_games: int,
    player: BasePlayer,
    opponent_pool: Union[List[BasePlayer], BasePlayer],
    critic_type: str = None,
    output: Path = None,
    always_save_checkpoint: bool = False,
) -> BasePlayer:
    assert hasattr(player, "model"), "player must be using a model for training!"

    model_params = [
        {"params": player.model.parameters()}
    ]
    if critic_type:
        critic = getattr(critics, critic_type)()
        model_params.append(
            {"params": critic.parameters(), "lr": 0.01}
        )
    else:
        critic = None
    critic_loss_function = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model_params, lr=0.1)

    if output:
        output.parent.mkdir(exist_ok=True, parents=True)

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
            add_kwargs = dict(win_rates=win_rates)
            if critic is not None:
                add_kwargs["critic_model"] = critic.state_dict()
            save_checkpoint(output, player.model, optimizer, **add_kwargs)
            print("Model checkpoint saved!")
        best_win_rate = max(wr, best_win_rate)

        if e == n_epochs:
            break

        dl = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=True,
            num_workers=8
        )

        # Basic REINFORCE algorithm to start
        for state, ostate, action, reward in dl:
            if critic is not None:
                rpred = critic(state, ostate)
                critic_loss = critic_loss_function(rpred, reward)
                with torch.no_grad():
                    reward = reward - torch.exp(rpred)
            else:
                critic_loss = 0.0

            # TODO: make configurable
            scaled_reward = (reward - reward.mean()) / (reward.std() + 1E-8)

            # Compute policy probabilities
            optimizer.zero_grad()
            logp = player.model(state)
            logp_a = logp.gather(-1, action.unsqueeze(-1)).squeeze()
            actor_loss = -(scaled_reward * logp_a).mean()

            loss = actor_loss + critic_loss
            loss.backward()
            optimizer.step()

        print(f"Epoch completed in {time() - st:.2f}s")

    return player
