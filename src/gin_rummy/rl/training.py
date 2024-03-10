"""
Useful references:

https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
https://karpathy.github.io/2016/05/31/rl/
"""

import torch

from torch.utils.data import DataLoader
from time import time
from pathlib import Path
from typing import List, Union
from copy import deepcopy

from gin_rummy.gameplay.game_manager import BasePlayer
from gin_rummy.gameplay.playouts import run_playouts
from gin_rummy.players.agent import RummyAgent
from gin_rummy.rl.policy_networks import PolicyNetwork


RF = "REINFORCE"
PPO = "PPO"


def save_checkpoint(
    output: Path,
    model: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    **kwargs
):
    state = {
        "config": model.config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
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
    algorithm: str = RF,
    ignore_critic: bool = False,
    lr: float = 0.1,
    lr_warmup_steps: int = 0,
    scale_minibatch: bool = True,
    minibatch_size: int = 512,
    max_minibatches: int = 8,
    epsilon: float = 0.2,
    output: Path = None,
    always_save_checkpoint: bool = False,
    num_workers: int = 4,
) -> BasePlayer:
    assert hasattr(player, "model"), "player must be using a model for training!"
    assert algorithm in {"REINFORCE", "PPO"}, f"unknown RL training algorithm {algorithm}"

    optimizer = torch.optim.Adam(player.model.parameters(), lr=0.1)
    critic_loss_function = torch.nn.BCEWithLogitsLoss()

    if output:
        output.mkdir(exist_ok=True, parents=True)

    if algorithm == PPO:
        ref_model = deepcopy(player.model)
        ref_model.eval()
    else:
        ref_model = None

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

        if algorithm == "REINFORCE":
            minibatch_size, max_minibatches = len(dataset), 1

        dl = DataLoader(
            dataset,
            batch_size=minibatch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # Run REINFORCE on entire dataset, or PPO on a maximum number of minibatches
        player.model.train()
        minibatches = 0
        for state, ostate, action, reward in dl:
            optimizer.zero_grad()

            logp, logr = player.model(state, None if ignore_critic else ostate)
            if ignore_critic:
                critic_loss = 0.0
            else:
                critic_loss = critic_loss_function(logr, reward)
                with torch.no_grad():
                    reward = reward - torch.exp(logr)

            if scale_minibatch:
                reward = (reward - reward.mean()) / (reward.std() + 1E-8)
            logp_a = logp.gather(-1, action.unsqueeze(-1)).squeeze()

            if algorithm == RF:
                actor_loss = -(reward * logp_a).mean()
            elif algorithm == PPO:
                logp_ref, _ = ref_model(state)
                logp_a_ref = logp_ref.gather(-1, action.unsqueeze(-1)).squeeze()
                prob_ratio = torch.exp(logp_a - logp_a_ref)
                clamped_prob_ratio = torch.clamp(prob_ratio, 1 - epsilon, 1 + epsilon)
                actor_loss = -torch.minimum(prob_ratio * reward, clamped_prob_ratio * reward).mean()
            else:
                raise NotImplementedError

            loss = actor_loss + critic_loss
            loss.backward()
            optimizer.step()

            minibatches += 1
            if minibatches >= max_minibatches:
                break

        if ref_model is not None:
            ref_model.load_state_dict(player.model.state_dict())

        print(f"Epoch completed in {time() - st:.2f}s")

    if output:
        save_checkpoint(
            Path(output, f"model.pt"),
            player.model,
            optimizer,
            win_rates=win_rates
        )

    return player
