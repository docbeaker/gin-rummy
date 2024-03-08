from torch import load, no_grad, exp
from torch.utils.data import DataLoader

from gin_rummy.players.card_points import CardPointPlayer, CardPointNNPlayer
from gin_rummy.gameplay.game_manager import GameManager


if __name__ == "__main__":
    trained_agent = CardPointNNPlayer()
    trained_agent.model.load_state_dict(
        load("/Users/jkearney/data/gin-rummy-models/v0/actor-bigcc-lr0.03/ckpt.pt")["model"]
    )
    alt_agent = CardPointPlayer()

    gm = GameManager(record=True)
    turn = gm.play_game(trained_agent, alt_agent, verbose=1)
    gm.dataset.record_win_label(turn)

    dl = DataLoader(gm.dataset, batch_size=len(gm.dataset), shuffle=True)

    for ph, oh, _, _ in dl:
        with no_grad():
            loga, logw = trained_agent.model(ph, oh)

        print(exp(logw).min(), exp(logw).max())
        break
