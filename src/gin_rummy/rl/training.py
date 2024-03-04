import torch
from torch.utils.data import DataLoader
from time import time
from pathlib import Path

from typing import List, Union

from gin_rummy.gameplay.game_manager import BasePlayer
from gin_rummy.gameplay.playouts import run_playouts


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
    output: Path = None,
    always_save_checkpoint: bool = False
) -> BasePlayer:
    assert hasattr(player, "model"), "player must be using a model for training!"
    optimizer = torch.optim.Adam(
        player.model.parameters(),
        lr=1E-1
    )

    output.parent.mkdir(exist_ok=True, parents=True)

    win_rates = []
    best_win_rate = 0.0
    for e in range(n_epochs + 1):
        st = time()
        player.initialize_dataset()
        n_wins, n_played = run_playouts(n_games, player, opponent_pool)
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
        if e and ((wr > best_win_rate) or always_save_checkpoint):
            save_checkpoint(output, player.model, optimizer, win_rates=win_rates)
            print("Model checkpoint saved!")
        best_win_rate = max(wr, best_win_rate)

        if e == n_epochs:
            break

        dl = DataLoader(
            player.dataset,
            batch_size=len(player.dataset),
            shuffle=True,
            num_workers=8
        )

        # Basic REINFORCE algorithm to start
        for state, action, reward in dl:
            scaled_reward = (reward - reward.mean()) / (reward.std() + 1E-8)

            optimizer.zero_grad()
            logp = player.model(state)
            logp_a = logp.gather(-1, action.unsqueeze(-1)).squeeze()

            loss = -(scaled_reward * logp_a).mean()

            loss.backward()
            optimizer.step()

        print(f"Epoch completed in {time() - st:.2f}s")

    return player
