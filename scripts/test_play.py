from gin_rummy.players.card_points import CardPointPlayer, CardPointNNPlayer
from gin_rummy.rl.training import train_agent
from gin_rummy.gameplay.playouts import run_playouts


if __name__ == "__main__":
    agent = CardPointNNPlayer(initialized=True)
    print(agent.model.c2d.weight)

    agent = train_agent(
        10,
        500,
        agent,
        CardPointPlayer(),
    )

    print(agent.model.c2d.weight)

    agent.set_temperature(1E-3)
    n_wins, n_valid = run_playouts(200, agent, CardPointPlayer())
    print(n_wins / n_valid)
