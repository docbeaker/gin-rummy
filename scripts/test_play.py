from gin_rummy.players.card_points import CardPointPlayer, CardPointNNPlayer
from gin_rummy.rl.training import train_agent


if __name__ == "__main__":
    agent = CardPointNNPlayer(initialized=True)
    print(agent.model.c2d.weight)

    agent = train_agent(
        10,
        64,
        agent,
        CardPointPlayer(),
    )

    print(agent.model.c2d.weight)
