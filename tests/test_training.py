from gin_rummy.rl.training import train_agent
from gin_rummy.players.card_points import CardPointPlayer, CardPointNNPlayer


class TestRLTraining:
    def test_training_loop(self):
        _ = train_agent(
            2, 10, CardPointNNPlayer(), CardPointPlayer()
        )

    def test_training_with_critic(self):
        _ = train_agent(
            2, 10, CardPointNNPlayer(), CardPointPlayer(), critic_type="MLPCritic"
        )
