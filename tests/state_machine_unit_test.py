import unittest

from prisoners_dilemma_sim import StateMachine


class StateMachineUnitTests(unittest.TestCase):
    state_matrix_1 = [(1, 0, 0, 0)]

    state_matrix_2 = [
        (1, 1, 0, 1),
        (0, 1, 0, 0)
    ]

    state_matrix_3 = [
        ('a', 'b', 1, 2),
        ('a', 'a', 0, 2),
        ('b', 'a', 0, 1)]

    input_domains = [
        [0, 1],
        ['c', 'd'],
        ['c', 'd']
    ]

    state_matrices = [state_matrix_1, state_matrix_2, state_matrix_3]

    def setUp(self):
        self.state_machines = list(map(
            lambda t: StateMachine.from_matrix(*t),
            zip(self.state_matrices, self.input_domains)
        ))

    def test_correct_output(self):
        inputs = [
            [0, 0, 1, 1, 0, 1],
            ['c', 'd', 'd', 'd', 'c', 'c', 'c'],
            ['c', 'd', 'c', 'd', 'd', 'c', 'd', 'c', 'd']
        ]
        exp_outputs = [
            [1, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1],
            ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b', 'b']
        ]

        for i in range(3):
            fsm = self.state_machines[i]
            actual_output = []
            for input_ in inputs[i]:
                actual_output.append(fsm.next(input_))
            with self.subTest(i=i):
                self.assertListEqual(exp_outputs[i], actual_output, msg="Unexpected output")

    def test_output_matrix(self):
        fsm = self.state_machines[0]
        exp_output_matrix = [[1, 0]]
        self.assertListEqual(exp_output_matrix, fsm._output_matrix, msg="Unexpected output matrix")

    def test_add_state(self):
        exp_transition_matrix = [[0, 0]]
        exp_output_matrix = [[1, 0]]
        fsm = StateMachine(self.state_matrix_1)
        fsm.add_state(*self.state_matrix_1[0])
        with self.subTest(msg="Transition matrix"):
            self.assertListEqual(exp_transition_matrix, fsm._transition_matrix, msg="Unexpected transition matrix")
        with self.subTest(msg="Output matrix"):
            self.assertListEqual(exp_output_matrix, fsm._output_matrix, msg="Unexpected output matrix")

    def test_output_if_input_none(self):
        input_ = None
        fsm = self.state_machines[1]
        exp_output = 1
        fsm.output_if_none(exp_output)
        actual_output = fsm.next(input_)
        self.assertEqual(exp_output, actual_output, msg="Unexpected output when input is None ")

    def test_default_constructor_creates_empty_fsm(self):
        input_domain, exp_output_matrix, exp_transition_matrix = [], [], []
        fsm = StateMachine(input_domain)
        with self.subTest(msg="output_matrix_test"):
            self.assertListEqual(exp_output_matrix, fsm._output_matrix)
        with self.subTest(msg="transition_matrix_test"):
            self.assertListEqual(exp_transition_matrix, fsm._transition_matrix)

if __name__ == '__main__':
    unittest.main()
