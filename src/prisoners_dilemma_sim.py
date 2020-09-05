from functools import partial
from typing import Any, List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np


class StateMachine:

    def __init__(self, input_domain: List[Any], name=None, output_if_none=None):
        self._transition_matrix = []
        self._output_matrix = []
        self._current_state = 0
        self._input_to_int = {}
        self.name = name
        self._none_output = output_if_none
        for i in range(len(input_domain)):
            self._input_to_int[input_domain[i]] = i

    @classmethod
    def from_matrix(cls, matrix: List[Tuple[Any, Any, int, int]], input_domain: List[Any], name=None,
                    output_if_none=None) -> 'StateMachine':
        machine = cls(input_domain, name, output_if_none=output_if_none)
        for t in matrix:
            machine.add_state(*t)
        return machine

    @classmethod
    def random_machine(cls, states: int, output_domain: List[Any], input_domain: List[Any], name=None) \
            -> 'StateMachine':
        matrix = []
        for _ in range(states):
            state = (
                np.random.choice(output_domain),
                np.random.choice(output_domain),
                np.random.choice(range(states)),
                np.random.choice(range(states))
            )
            matrix.append(state)
        machine = cls.from_matrix(
            matrix=matrix, input_domain=input_domain, name=name, output_if_none=np.random.choice(output_domain)
        )
        return machine

    def __str__(self):
        return self.name

    def next(self, input_: Any) -> Any:
        if input_ is None:
            return self._none_output
        elif input_ not in self._input_to_int.keys():
            raise Exception("Input outside input domain")
        i = self._input_to_int.get(input_)
        output = self._output_matrix[self._current_state][i]
        self._current_state = self._transition_matrix[self._current_state][i]
        return output

    def add_state(self, output_0: Any, output_1: Any, next_state_0: int, next_state_1: int) -> None:
        self._transition_matrix.append([next_state_0, next_state_1])
        self._output_matrix.append([output_0, output_1])

    def output_if_none(self, output):
        self._none_output = output


class PrisonersDilemma:
    _choice_to_int = {'c': 0, 'd': 1}

    def __init__(self, prisoners: List[StateMachine], cost_matrix: List[List[Tuple[int, int]]]):
        self.cost_matrix = cost_matrix
        self.prisoners = prisoners

    def run_tournament_simulations(self, simulations: int = 20) -> Dict[StateMachine, int]:
        """Runs a round-robin tournament <code>simulation</code> time(s) and returns the average sentence of each
            simulation for every prisoner."""
        sentences_acc = {}
        for p in self.prisoners:
            sentences_acc[p] = 0
        for sim in range(simulations):
            sentences = self._run_tournament()
            for p in sentences.keys():
                sentences_acc[p] += sentences[p]
        for p in sentences_acc:
            sentences_acc[p] /= simulations
        return sentences_acc

    def _run_tournament(self) -> Dict[StateMachine, int]:
        prisoners = self.prisoners
        sentences = {}
        for p in prisoners:
            sentences[p] = 0
        for i in range(len(prisoners) - 1):
            prisoner_a = prisoners[i]
            choice_a = None
            choice_b = None
            for j in range(i + 1, len(prisoners)):
                prisoner_b = prisoners[j]
                choice_a, choice_b = prisoner_a.next(choice_b), prisoner_b.next(choice_a)
                sentence_a, sentence_b = self._sentence(choice_a, choice_b)
                sentences[prisoner_a] += sentence_a
                sentences[prisoner_b] += sentence_b
        return sentences

    def _sentence(self, choice_a: chr, choice_b: chr):
        choice_a = self._choice_to_int.get(choice_a)
        choice_b = self._choice_to_int.get(choice_b)
        return self.cost_matrix[choice_a][choice_b]


class RandomPrisoner(StateMachine):
    def __init__(self, name=None):
        self.name = name

    def next(self, input_):
        return np.random.choice(['c', 'd'])


class EPPrisoner(StateMachine):
    def __init__(self, input_domain: List[Any], output_domain: List[Any], pop_size: int = 4, name: str = None,
                 beta: float = 1, gamma: float = 0):
        super().__init__(input_domain, name=name)
        self._population: List[StateMachine] = []
        self._history: List[int] = []
        self._output_to_int = {}
        self._int_to_output = {}
        self._pop_size: int = pop_size

        for _ in range(pop_size):
            self._population.append(self.random_machine(
                states=8, input_domain=input_domain, output_domain=output_domain
            ))
        i = 0
        for el in output_domain:
            self._output_to_int[el] = i
            self._int_to_output[i] = el
            i += 1
        self.beta = beta
        self.gamma = gamma

    def _mutate(self) -> None:
        pass

    def next(self, input_: Any) -> Any:
        _population = self._population
        self._mutate()

        # Select pop_size individuals of lowest cost
        _population = sorted(_population, key=self._cost)
        _population = _population[:self._pop_size]
        for machine in _population:
            machine._current_state = 0

        # Return value of fittest individual
        best_machine = _population[0]
        for entry in self._history:
            best_machine.next(entry)
        next_move = best_machine.next(input_)
        self._history.append(input_)
        best_machine._current_state = 0
        return next_move

    def _cost(self):
        pass


input_domain = ['c', 'd']
from_matrix_ = partial(StateMachine.from_matrix, input_domain=input_domain)
random_machine_ = partial(StateMachine.random_machine, output_domain=input_domain, input_domain=input_domain)

''' Specify strategies here '''
prisoners = [
    from_matrix_([('d', 'd', 0, 0)], name='always_defect', output_if_none='d'),
    from_matrix_([('c', 'c', 0, 0)], name='always_coop', output_if_none='c'),
    random_machine_(4, name="random_const"),
    RandomPrisoner(name="random_output"),
    from_matrix_([('c', 'd', 0, 0)], name="tit_for_tat", output_if_none='c'),
    from_matrix_([
        ('c', 'd', 0, 1),
        ('d', 'd', 1, 1)
    ], name="grim", output_if_none='c'),
    from_matrix_([
        ('c', 'd', 0, 1),
        ('d', 'd', 2, 1),
        ('d', 'd', 0, 1)
    ], name="punish", output_if_none='c'),
    from_matrix_([
        ('c', 'd', 1, 0),
        ('c', 'd', 2, 0),
        ('d', 'd', 2, 0),
        ('d', 'c', 3, 0)
    ], name="betray", output_if_none='c')
]

cost_matrix = [
    [(1, 1), (10, 0)],
    [(0, 10), (5, 5)]
]
prisoners_dilemma = PrisonersDilemma(prisoners, cost_matrix)
sentences = prisoners_dilemma.run_tournament_simulations(simulations=1000)
min_ = min(sentences.values())
for p in prisoners:
    sentences[p] /= min_
prisoners, sentences = sorted(prisoners, key=sentences.get), sorted(sentences.values())

plt.figure(figsize=(14, 4))
plt.xlabel("Prisoner strategies")
plt.ylabel("Relative cost (#years in prison)")
plt.bar(list(map(lambda p: p.name, prisoners)), sentences, width=0.2)
plt.grid(axis='y', linestyle='dashed')
#plt.show()