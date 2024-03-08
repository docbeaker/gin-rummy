from torch import isclose

from gin_rummy.rl.training import train_agent
from gin_rummy.players import CardPointPlayer, RummyAgent


class TestRLTraining:
    def test_training_loop(self):
        agent = RummyAgent()
        original_weights = agent.model.c2d.weight.detach().clone()
        agent = train_agent(
            2, 10, agent, CardPointPlayer()
        )
        assert not isclose(agent.model.c2d.weight, original_weights).all()

    def test_training_no_critic(self):
        agent = RummyAgent()
        original_critic_weights = agent.model.fc.weight.detach().clone()
        agent = train_agent(
            2, 10, agent, CardPointPlayer(), ignore_critic=True
        )
        assert isclose(agent.model.fc.weight, original_critic_weights).all()

    def test_training_joint(self):
        agent = RummyAgent(network="JointConvolutionNN")
        original_weights = agent.model.c2d.weight.detach().clone()
        agent = train_agent(
            2, 10, agent, CardPointPlayer()
        )
        assert not isclose(agent.model.c2d.weight, original_weights).all()
