from itertools import product
from gin_rummy.gameplay.game_manager import CardFormatter


class TestInverseFormatting:
    def test_card_to_idx(self):
        values = [str(x) for x in range(2, 11)] + ["A", "J", "Q", "K"]
        suits = ["C", "D", "H", "S"]

        for v, s in product(values, suits):
            idx = CardFormatter.to_index(v + s)
            assert 0 <= idx < 52, f"converting {v+s} yielded invalid index {idx}"
