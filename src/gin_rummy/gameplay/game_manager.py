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
    _suits_char = {
        s: i for i, s in enumerate(["C", "D", "H", "S"])
    }
    _val_map = {
        k: i for i, k in _card_map.items()
    }

    @classmethod
    def to_card(cls, suit_idx: int, value_idx: int = None) -> str:
        if value_idx is None:
            suit_idx, value_idx = suit_idx // NV, suit_idx % NV
        return f"{cls._card_map.get(value_idx, value_idx + 1)}{cls._suits[suit_idx]}"

    @classmethod
    def to_index(cls, card: str) -> int:
        _suit_idx = cls._suits_char.get(card[-1], len(cls._suits_char))
        if card[:-1] in cls._val_map:
            _val_idx = cls._val_map[card[:-1]]
        else:
            try:
                _val_idx = int(card[:-1]) - 1
            except ValueError:
                _val_idx = -1
        if _val_idx < 0 or _val_idx > 12:
            _val_idx = NS * NV
        _idx = _suit_idx * NV + _val_idx
        if _idx < 0 or _idx >= NV * NS:
            return -1
        return _idx


class BasePlayer(ABC):
    def __init__(self):
        self.hand_matrix = np.zeros((NS, NV), dtype=bool)
        self.straight_kernel = np.arange(1, 8).reshape((1, -1))
        self.straight_kernel[0,4:] = 0  # for evaluation

    @property
    def requires_input(self):
        return False

    def accept_card(self, card: int):
        self.hand_matrix[card // NV, card % NV] = True

    def compute_state(self, game: "GameManager"):
        return None

    @abstractmethod
    def _choose_card_to_discard(self, state) -> int:
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

    def discard_card(self, state) -> Tuple[int, bool]:
        """
        Return a tuple of card to add to discard pile and bool indicating whether gin
        """
        discard_card = self._choose_card_to_discard(state)
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
        state = player.compute_state(self)
        discard_top = self.discard_pile[-1]
        player.accept_card(self.discard_pile.pop())
        discard_card, gin = player.discard_card(state)
        if discard_card == discard_top:
            assert not gin, "should not be declaring gin when returning discard card!"
            self.discard_pile.append(discard_card)  # put that card back and try again
            player.accept_card(self.deck.pop())
            discard_card, gin = player.discard_card(state)
        self.discard_pile.append(discard_card)
        return gin

    def print_obfuscated(self, player: BasePlayer):
        print(
            f"Player hand: {player.get_hand(human=True):<30}"
            f"Discard: {CardFormatter.to_card(self.discard_pile[-1])}"
        )

    def print_full(self, player_1: BasePlayer, player_2: BasePlayer):
        print(
            f"{player_1.get_hand(human=True)}\t"
            f"{player_2.get_hand(human=True)}\t"
            f"Discard: {CardFormatter.to_card(self.discard_pile[-1])}"
        )

    def play_game(
        self,
        player_1: BasePlayer,
        player_2: BasePlayer,
        turn: int = None,
        verbose: bool = False,
    ):
        self.deal(player_1, player_2)

        if turn is None:
            turn = randint(0, 1)
        players = [player_1, player_2]

        # Do not print if a human is playing
        verbose = verbose and not (player_1.requires_input or player_2.requires_input)

        end_of_game = False
        turn = 1 - turn  # swap the turn initially, as we always swap before play
        while not end_of_game:
            turn = 1 - turn  # swap the turn
            if verbose:
                self.print_full(*players)
            if players[turn].requires_input:
                self.print_obfuscated(players[turn])
            end_of_game = self.process_play(players[turn])  # play the turn

        print(f"Game concluded! Player {turn + 1} wins!")
        for idx in [turn, 1 - turn]:
            print(f">> Player {idx + 1} hand: {players[idx].get_hand(human=True)}")
