from torch import isclose, tensor

from gin_rummy.rl.training import train_agent
from gin_rummy.players.card_points import CardPointPlayer, CardPointNNPlayer, PointsConvolutionNN


class TestRLTraining:
    def test_training_loop(self):
        agent = train_agent(
            2, 10, CardPointNNPlayer(), CardPointPlayer()
        )
        assert not isclose(agent.model.scale, tensor(PointsConvolutionNN.init_scale))
        assert not isclose(agent.model.bias, tensor(PointsConvolutionNN.init_bias))

    def test_training_no_critic(self):
        agent = train_agent(
            2, 10, CardPointNNPlayer(), CardPointPlayer(), ignore_critic=True
        )
        assert isclose(agent.model.scale, tensor(PointsConvolutionNN.init_scale))
        assert isclose(agent.model.bias, tensor(PointsConvolutionNN.init_bias))
