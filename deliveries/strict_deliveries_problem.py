from framework.graph_search import *
from framework.ways import *
from .map_problem import MapProblem
from .deliveries_problem_input import DeliveriesProblemInput
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem

from typing import Set, FrozenSet, Optional, Iterator, Tuple, Union


class StrictDeliveriesState(RelaxedDeliveriesState):
    """
    An instance of this class represents a state of the strict
     deliveries problem.
    This state is basically similar to the state of the relaxed
     problem. Hence, this class inherits from `RelaxedDeliveriesState`.

    TODO:
        If you believe you need to modify the state for the strict
         problem in some sense, please go ahead and do so.
    """
    pass


class StrictDeliveriesProblem(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem, self).__init__(problem_input)
        self.initial_state = StrictDeliveriesState(
            problem_input.start_point, frozenset(), problem_input.gas_tank_init_fuel)
        self.inner_problem_solver = inner_problem_solver
        self.roads = roads
        self.use_cache = use_cache
        self._init_cache()

    def _init_cache(self):
        self._cache = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key, val):
        if self.use_cache:
            self._cache[key] = val

    def _get_from_cache(self, key):
        if not self.use_cache:
            return None
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return self._cache.get(key)

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state_to_expand, StrictDeliveriesState)

        optional_next_junctions = self.drop_points.difference(state_to_expand.dropped_so_far)

        for optional_next_junction in optional_next_junctions:

            # check if we've already found the optimal path from origin junction to destination junction
            # if so: it will be in cache
            # else : apply inner A* in order to find optimal path from origin junction to each one of his successors
            # use a tuple of (origin junction ID , destination junction ID), to index the cache dictionary.
                # (we want to save every combination of origin-destination optimal path in cache)
            cache_access_forNextState = self._get_from_cache((state_to_expand.current_location.index,
                                                              optional_next_junction.index))
            # return None if not in cache
            if cache_access_forNextState:
                dist = cache_access_forNextState.final_search_node.cost
            else:
                # define map problem from origin to destination
                map_prob = MapProblem(self.roads,
                                      state_to_expand.current_location.index,
                                      optional_next_junction.index)
                a_star_mst_w05 = self.inner_problem_solver
                mst_res = a_star_mst_w05.solve_problem(map_prob)
                dist = mst_res.final_search_node.cost
                # insert the optimal solution into cache
                self._insert_to_cache((state_to_expand.current_location.index,
                                       optional_next_junction.index),
                                      mst_res)

            # if we have enough fuel to go to that state
            if state_to_expand.fuel >= dist:
                next_state = StrictDeliveriesState(current_location=optional_next_junction,
                                                   dropped_so_far=state_to_expand.dropped_so_far.union
                                                                  (frozenset([optional_next_junction])),
                                                   fuel=state_to_expand.fuel - dist)
                operator_cost = dist
                yield next_state, operator_cost

        for gas_station in self.gas_stations:
            cache_access_forGas = self._get_from_cache((state_to_expand.current_location.index,
                                                        gas_station.index))
            if cache_access_forGas:
                gasdist = cache_access_forGas.final_search_node.cost
            else:
                # define map problem from origin to destination
                map_prob = MapProblem(self.roads,
                                      state_to_expand.current_location.index,
                                      gas_station.index)
                a_star_mst_w05 = self.inner_problem_solver
                mst_res = a_star_mst_w05.solve_problem(map_prob)
                gasdist = mst_res.final_search_node.cost
                # insert the optimal solution into cache
                self._insert_to_cache((state_to_expand.current_location.index,
                                       gas_station.index),
                                      mst_res)

            if state_to_expand.fuel >= gasdist:
                next_state = StrictDeliveriesState(current_location=gas_station,
                                                   dropped_so_far=state_to_expand.dropped_so_far,
                                                   fuel=self.gas_tank_capacity)
                operator_cost = gasdist
                yield next_state, operator_cost

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        """
        assert isinstance(state, StrictDeliveriesState)
        # inherit from RelaxedDeliveriesState, therefor might not be needed
        return state.dropped_so_far == self.drop_points


