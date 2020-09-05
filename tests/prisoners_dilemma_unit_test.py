import unittest
from unittest.mock import MagicMock

from prisoners_dilemma_sim import PrisonersDilemma


def always_defect_factory():
    always_defect = MagicMock()
    always_defect.next = MagicMock(return_value='d')
    return always_defect


def always_coop_factory():
    always_coop = MagicMock()
    always_coop.next = MagicMock(return_value='c')
    return always_coop


class PrisonersDilemmaUnitTests(unittest.TestCase):
    cost_matrix = [
        [(1, 1), (10, 0)],
        [(0, 10), (5, 5)]
    ]

    def test_run_tournament_simulation_2_prisoners(self):
        prisoners = [always_coop_factory(), always_defect_factory()]
        exp_sentences = {prisoners[0]: 10, prisoners[1]: 0}

        prisoners_dilemma = PrisonersDilemma(prisoners, self.cost_matrix)
        for sims in range(1, 5):
            with self.subTest(i=sims - 1):
                actual_sentences = prisoners_dilemma.run_tournament_simulations(simulations=sims)
                self.assertDictEqual(exp_sentences, actual_sentences, msg="Unexpected sentences")

    def test_run_tournament_simulation_3_prisoners(self):
        prisoners = [always_coop_factory() for _ in range(3)]
        exp_sentences = {prisoners[0]: 2, prisoners[1]: 2, prisoners[2]: 2}

        prisoners_dilemma = PrisonersDilemma(prisoners, self.cost_matrix)
        for sims in range(1, 5):
            with self.subTest(i=sims - 1):
                actual_sentences = prisoners_dilemma.run_tournament_simulations(simulations=sims)
                self.assertDictEqual(exp_sentences, actual_sentences, msg="Unexpected sentences")


if __name__ == '__main__':
    unittest.main()
