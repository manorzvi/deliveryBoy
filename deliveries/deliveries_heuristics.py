from framework.graph_search import *
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem
from .strict_deliveries_problem import StrictDeliveriesState, StrictDeliveriesProblem
from .deliveries_problem_input import DeliveriesProblemInput
from framework.ways import *

from framework.consts import Consts

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from typing import Set, Dict, FrozenSet

roads = load_map_from_csv(Consts.get_data_file_path("tlv.csv"))


class MaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MaxAirDist'

    def estimate(self, state: GraphProblemState) -> float:
        """
        Calculates the maximum among air distances between the location
         represented by `state` and the locations of the waiting deliveries.
        """
        assert isinstance(self.problem, RelaxedDeliveriesProblem)
        assert isinstance(state, RelaxedDeliveriesState)
        if self.problem.is_goal(state):
            return 0
        air_dist_list = list()
        optional_next_junctions = set(self.problem.drop_points)
        for already_dropped in state.dropped_so_far:
            optional_next_junctions.remove(already_dropped)
        for optional_next_junction in optional_next_junctions:
            air_dist_list.append(state.current_location.calc_air_distance_from(optional_next_junction))
        return max(air_dist_list)


class MSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MSTAirDist'

    def __init__(self, problem: GraphProblem):
        super(MSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, RelaxedDeliveriesProblem)
        self._junctions_distances_cache: Dict[FrozenSet[Junction], float] = dict()

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, RelaxedDeliveriesProblem)
        assert isinstance(state, RelaxedDeliveriesState)

        remained_drop_points = set(self.problem.drop_points - state.dropped_so_far)
        remained_drop_points.add(state.current_location)
        return self._calculate_junctions_air_dist_mst_weight(remained_drop_points)

    def _get_distance_between_junctions(self, junction1: Junction, junction2: Junction):
        junctions_pair = frozenset({junction1, junction2})
        if junctions_pair in self._junctions_distances_cache:
            return self._junctions_distances_cache[junctions_pair]
        dist = junction1.calc_air_distance_from(junction2)
        self._junctions_distances_cache[junctions_pair] = dist
        return dist

    def _calculate_junctions_air_dist_mst_weight(self, junctions: Set[Junction]) -> float:
        nr_junctions = len(junctions)
        idx_to_junction = {idx: junction for idx, junction in enumerate(junctions)}
        distances_matrix = np.zeros((nr_junctions, nr_junctions), dtype=np.float)
        for j1_idx in range(nr_junctions):
            for j2_idx in range(nr_junctions):
                if j1_idx == j2_idx:
                    continue
                dist = self._get_distance_between_junctions(idx_to_junction[j1_idx], idx_to_junction[j2_idx])
                distances_matrix[j1_idx, j2_idx] = dist
                distances_matrix[j2_idx, j1_idx] = dist
        return mst(distances_matrix).sum()


class RelaxedDeliveriesHeuristic(HeuristicFunction):
    heuristic_name = 'RelaxedProb'

    def estimate(self, state: GraphProblemState) -> float:
        """
        Solve the appropriate relaxed problem in order to
         evaluate the distance to the goal.
        """

        assert isinstance(self.problem, StrictDeliveriesProblem)
        assert isinstance(state, StrictDeliveriesState)

        small_delivery = DeliveriesProblemInput(input_name=state.__str__(),
                                                start_point=state.current_location,
                                                drop_points=self.problem.drop_points.difference(state.dropped_so_far),

                                                gas_stations=self.problem.gas_stations,

                                                #gas_tank_capacity=state.fuel,
                                                gas_tank_capacity=self.problem.gas_tank_capacity,
                                                #gas_tank_init_fuel=self.problem.gas_tank_capacity,
                                                gas_tank_init_fuel=state.fuel)
        small_deliveries_prob = RelaxedDeliveriesProblem(small_delivery)

        a_star_mst_air_dist_heuristic = AStar(MSTAirDistHeuristic)
        a_star_mst_air_dist_heuristic_res = a_star_mst_air_dist_heuristic.solve_problem(small_deliveries_prob)

        if a_star_mst_air_dist_heuristic_res.final_search_node:
            return a_star_mst_air_dist_heuristic_res.final_search_node.cost
        else:
            return np.inf


