from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional
import numpy as np


class GreedyStochastic(BestFirstSearch):
    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 T_init: float = 1.0, N: int = 5, T_scale_factor: float = 0.95):
        # GreedyStochastic is a graph search algorithm. Hence, we use close set.
        super(GreedyStochastic, self).__init__(use_close=True)
        self.heuristic_function_type = heuristic_function_type
        self.T = T_init
        self.N = N
        self.T_scale_factor = T_scale_factor
        self.solver_name = 'GreedyStochastic (h={heuristic_name})'.format(
            heuristic_name=heuristic_function_type.heuristic_name)

    def _init_solver(self, problem: GraphProblem):
        super(GreedyStochastic, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        """
        Called by solve_problem() in the implementation of `BestFirstSearch`
        whenever creating a new successor node.
        This method is responsible for adding this just-created successor
        node into the `self.open` priority queue, and may check the existence
        of another node representing the same state in `self.close`.
        Note - Manor (23/11) -  assuming that 'stochastic greedy best', in it's essence is greedy-best. and therefore
                                manege the open and close Queues like A* algorithm.
        """
        if self.open.has_state(successor_node.state):
            already_found_node_with_same_state_in_open = self.open.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state_in_open.expanding_priority <= successor_node.expanding_priority:
                return
            self.open.extract_node(already_found_node_with_same_state_in_open)

        if self.close.has_state(successor_node.state):
            already_found_node_with_same_state_in_close = self.close.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state_in_close.expanding_priority <= successor_node.expanding_priority:
                return
            self.close.remove_node(already_found_node_with_same_state_in_close)

        self.open.push_node(successor_node)

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        Remember: `GreedyStochastic` is greedy.
        """
        assert (search_node.cost is not None)

        return self.heuristic_function.estimate(search_node.state)

    def _extract_next_search_node_to_expand(self) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
        using the stochastic method to choose out of the N
        best items from open.

        Use `np.random.choice(...)` whenever you need to randomly choose
        an item from an array of items given a probabilities array `p`.
        You can read the documentation of `np.random.choice(...)` and
        see usage examples by searching it in Google.

        Notice: You might want to pop min(N, len(open) items from the
                `open` priority queue, and then choose an item out
                of these popped items. The other items have to be
                pushed again into that queue.
        """

        if self.open.is_empty():
            return None

        optional_nodes_2_expand = list()
        optional_nodes_2_expand_heuristics = list()
        optional_nodes_2_expand_probabilities = list()

        if self.N < len(self.open):
            number_of_nodes_2_expand = self.N
        else:
            number_of_nodes_2_expand = len(self.open)

        # add best N nodes in open Queue to optional_nodes_2_expand
        for i in range(number_of_nodes_2_expand):
            node2add = self.open.pop_next_node()
            optional_nodes_2_expand.append(node2add)

        # get their expanding_priority in order to calculate probability
        for optional_node in optional_nodes_2_expand:
            hueristic2add = optional_node.expanding_priority
            optional_nodes_2_expand_heuristics.append(hueristic2add)

        # calculate probabilities for each of the optional nodes
        sum = 0
        alpha = min(optional_nodes_2_expand_heuristics)

        if alpha == 0:
            return optional_nodes_2_expand[optional_nodes_2_expand_heuristics.index(alpha)]

        for heuristic in optional_nodes_2_expand_heuristics:
            # calculate numerator only. later will divide by the sum
            numerator = (heuristic / alpha) ** (-1 / self.T)
            optional_nodes_2_expand_probabilities.append(numerator)

            sum += numerator


        self.T = self.T * self.T_scale_factor

        # divided by the denominator
        optional_nodes_2_expand_probabilities = [x/sum for x in optional_nodes_2_expand_probabilities]

        # pick one node with respect to the probabilities, and remove it from the list
        node_to_expand = np.random.choice(optional_nodes_2_expand, p=optional_nodes_2_expand_probabilities)

        # push the rest back to the open Queue
        for node in optional_nodes_2_expand:
            if node != node_to_expand:
                self.open.push_node(node)

        if self.use_close:
            self.close.add_node(node_to_expand)

        return node_to_expand








