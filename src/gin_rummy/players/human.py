from numpy import where

from gin_rummy.gameplay.game_manager import GameManager, BasePlayer, CardFormatter, NV


class HumanPlayer(BasePlayer):
    @property
    def requires_input(self):
        return True

    def compute_state(self, game: GameManager):
        return game.discard_pile[-1]

    def _choose_card_to_discard(self, discard_top: int) -> int:
        """
        State used by this player is just the top discard card
        """

        # First case: checking if we want the discard card. If not,
        # we will just return it as is
        if self.hand_matrix.flatten()[discard_top]:
            take = None
            while take not in {"y", "n"}:
                take = input("Take discarded card? [y/n] ")
                if take == "n":
                    return discard_top
                if take == "debug":
                    import pdb; pdb.set_trace()

        idx = None
        while idx is None:
            to_discard = input(
                f"Choose a card to discard from {self.get_hand(human=True)}: "
            )

            try:
                idx = int(to_discard)
                # Support both positive and negative indexing
                if abs(idx) >= self.hand_matrix.sum():
                    idx = -1
                else:
                    card_idxes = where(self.hand_matrix)
                    idx = card_idxes[0][idx] * NV + card_idxes[1][idx]
            except ValueError:
                idx = CardFormatter.to_index(to_discard)

            if (idx < 0) or not self.hand_matrix.flatten()[idx] or (idx == discard_top):
                print(f"Invalid selection {to_discard}! Please try again...")
                idx = None
                continue

            if input(f"Discard {CardFormatter.to_card(idx // NV, idx % NV)}: ") != "y":
                idx = None
                continue

        return idx

