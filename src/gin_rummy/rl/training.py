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
    n_steps: int,
    n_games: int,
    player: RummyAgent,
    opponent_pool: Union[List[BasePlayer], BasePlayer],
    algorithm: str = RF,
    gael: float = 1,
    lr: float = 0.1,
    lr_warmup_steps: int = 0,
    k_epochs_per_step: int = 1,
    scale_minibatch: bool = True,
    minibatch_size: int = 0,
    epsilon: float = 0.2,
    output: Path = None,
    log_steps: int = 1,
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
    for t in range(n_steps + 1):
        st = time()
        n_wins, n_played, dataset = run_playouts(n_games, player, opponent_pool, progress_bar=log_steps == 1)
        if not n_played:
            print(f"Step {t} failed to generate any valid games!")
            continue
        if (t % log_steps == 0) or (t == n_steps):
            print(
                f"Step {t}: agent win percentage = {100 * n_wins / n_played: .1f}% "
                f"({n_played} valid games)"
            )

        # Check if we should be saving a model
        wr = n_wins / n_played
        win_rates.append(wr)
        if output and t and ((wr > best_win_rate) or always_save_checkpoint):
            save_checkpoint(Path(output, "ckpt.pt"), player.model, optimizer, win_rates=win_rates)
            print(f"Model checkpoint saved at step {t}!")
        best_win_rate = max(wr, best_win_rate)

        if t == n_steps:
            break

        elr = get_lr(t + 1, lr, lr_warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = elr

        dl = DataLoader(
            dataset,
            batch_size=minibatch_size if minibatch_size else len(dataset),
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        player.model.train()
        for _ in range(k_epochs_per_step):
            for state, action, vt_plus_1, win in dl:
                optimizer.zero_grad()

                logp, logvt = player.model(state)
                critic_loss = critic_loss_function(logvt, win)
                with torch.no_grad():
                    vt = torch.exp(logvt)

                if gael > 0.999:
                    # TODO: add value network warmup steps?
                    # Use full reward, high variance approach
                    reward = win - vt
                elif gael < 0.001:
                    # Use predicted reward, high bias approach (biased by value network)
                    reward = vt_plus_1 - vt
                else:
                    raise NotImplementedError("intermediate values of GAE-lambda not supported")

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

        if ref_model is not None:
            ref_model.load_state_dict(player.model.state_dict())

        if t % log_steps == 0:
            print(f"Step completed in {time() - st:.2f}s")

    if output:
        save_checkpoint(
            Path(output, f"model.pt"),
            player.model,
            optimizer,
            win_rates=win_rates
        )

    return player
