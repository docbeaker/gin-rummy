import numpy as np

from random import shuffle, randint
from abc import ABC, abstractmethod
from typing import Tuple
from numpy.typing import ArrayLike
from scipy.signal import convolve2d


NS = 4
NV = 13


class CardFormatter:
    _card_map = {
        0: "A",
        10: "J",
        11: "Q",
        12: "K"
    }
    _suits = [
        chr(x) for x in [9827, 9830, 9829, 9824][:NS]
    ]

    @classmethod
    def to_card(cls, suit_idx: int, value_idx: int = None) -> str:
        if value_idx is None:
            suit_idx, value_idx = suit_idx // NV, suit_idx % NV
        return f"{cls._card_map.get(value_idx, value_idx + 1)}{cls._suits[suit_idx]}"


class BasePlayer(ABC):
    def __init__(self):
        self.hand_matrix = np.zeros((NS, NV), dtype=bool)
        self.straight_kernel = np.arange(1, 8).reshape((1, -1))
        self.straight_kernel[0,4:] = 0  # for evaluation

    def accept_card(self, card: int):
        self.hand_matrix[card // NV, card % NV] = True

    @abstractmethod
    def _choose_card_to_discard(self) -> int:
        pass

    def _get_straight_scores(self) -> ArrayLike:
        return convolve2d(
            self.hand_matrix,
            self.straight_kernel,
            mode="same",
        )

    def check_for_victory(self) -> bool:
        """
        Victory check is a bit specific for 7 card variant, could perhaps
        be generalized by building all threes and fours as in
        https://markfasciano.medium.com/computer-gin-rummy-27-years-later-ad3c2325315c
        """
        straights = self._get_straight_scores()
        ofakinds = self.hand_matrix.sum(axis=0)

        k4_idx = np.where(ofakinds == 4)[0]
        s4_row_idx, s4_col_idx = np.where(straights == 10)

        # Check that we have at least four of a kind
        if k4_idx.size + s4_row_idx.size == 0:
            return False

        # If 4 of a kind and 3 of a kind, win
        if k4_idx.size and (ofakinds == 3).any():
            return True

        # Remove 4 of a kind, check if straight still there, replace
        for _idx in k4_idx:
            self.hand_matrix[:, _idx] = False
            win = (self._get_straight_scores() == 9).any()
            self.hand_matrix[:, _idx] = True
            if win:
                return True

        # Remove each straight, check
        for i, j in zip(s4_row_idx, s4_col_idx):
            self.hand_matrix[i, j:j+4] = False
            win = (self.hand_matrix.sum(axis=0) == 3).any() or (self._get_straight_scores() == 9).any()
            self.hand_matrix[i, j:j+4] = True
            if win:
                return True

        return False

    def discard_card(self) -> Tuple[int, bool]:
        """
        Return a tuple of card to add to discard pile and bool indicating whether gin
        """
        discard_card = self._choose_card_to_discard()
        self.hand_matrix[discard_card // NV, discard_card % NV] = False
        return discard_card, self.check_for_victory()

    def get_hand(self, human: bool=False):
        """
        TODO: organize into groups using winning combos?
        """
        _hand = [(sidx, cidx) for sidx, cidx in zip(*np.where(self.hand_matrix))]
        if human:
            _hand = [CardFormatter.to_card(*tup) for tup in _hand]
        return ",".join(_hand)


class GameManager:
    HAND_SIZE = 7

    def __init__(self):
        self.deck, self.discard_pile = list(range(NS * NV)), None
        self.shuffle()

    def shuffle(self):
        shuffle(self.deck)
        self.discard_pile = []

    def deal(self, player_1: BasePlayer, player_2: BasePlayer):
        assert not self.discard_pile, "trying to deal when discard pile exists!"
        for _ in range(self.HAND_SIZE):
            for player_ in [player_1, player_2]:
                player_.accept_card(self.deck.pop())
        self.discard_pile.append(self.deck.pop())

    def process_play(self, player: BasePlayer) -> bool:
        """
        Return whether it is the end of the game
        """
        if not self.deck:
            self.deck = self.discard_pile
            self.shuffle()

        # If model wants to discard the discard, have it take top of deck instead
        discard_top = self.discard_pile[-1]
        player.accept_card(self.discard_pile.pop())
        discard_card, _ = player.discard_card()
        if discard_card == discard_top:
            self.discard_pile.append(discard_card)  # put that card back and try again
            player.accept_card(self.deck.pop())
            discard_card, gin = player.discard_card()
            if gin:
                return True
        self.discard_pile.append(discard_card)
        return False

    def play_game(self, player_1: BasePlayer, player_2: BasePlayer, turn: int = None):
        self.deal(player_1, player_2)

        if turn is None:
            turn = randint(0, 1)
        players = [player_1, player_2]

        end_of_game = False
        turn = 1 - turn  # swap the turn initially, as we always swap before play
        while not end_of_game:
            turn = 1 - turn  # swap the turn
            print(
                f"{player_1.get_hand(human=True)}\t"
                f"{player_2.get_hand(human=True)}\t"
                f"Discard: {CardFormatter.to_card(self.discard_pile[-1])}"
            )
            end_of_game = self.process_play(players[turn])  # play the turn

        print(f"Game concluded! Player {turn + 1} wins!")
        for idx in [turn, 1 - turn]:
            print(f">> Player {idx + 1} hand: {players[idx].get_hand(human=True)}")
