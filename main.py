from framework import *
from deliveries import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union

# Load the map
roads = load_map_from_csv(Consts.get_data_file_path("tlv.csv"))

# Make `np.random` behave deterministic.
Consts.set_seed()


# --------------------------------------------------------------------
# -------------------------- Map Problem -----------------------------
# --------------------------------------------------------------------

def plot_distance_and_expanded_wrt_weight_figure(
        weights: Union[np.ndarray, List[float]],
        total_distance: Union[np.ndarray, List[float]],
        total_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    """
    assert len(weights) == len(total_distance) == len(total_expanded)
    fig, ax1 = plt.subplots()
    ax1.plot(weights, total_distance, 'b-')

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('distance traveled', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()
    ax2.plot(weights, total_expanded, 'r-')
    ax2.set_ylabel('states expanded', color='r')
    ax2.tick_params('y', colors='r')

    # Plot the graph
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem):
    weights = np.linspace(0.5, 1, num=20)
    total_distance = list()
    total_expanded = list()
    for weight in weights.tolist():
        a_star = AStar(heuristic_type, weight)
        a_star_res = a_star.solve_problem(problem)
        total_distance.append(a_star_res.final_search_node.cost)
        total_expanded.append(a_star_res.nr_expanded_states)

    plot_distance_and_expanded_wrt_weight_figure(weights, total_distance, total_expanded)


def map_problem():
    print()
    print('Solve the map problem.')

    # Ex.8
    map_prob = MapProblem(roads, 54, 549)
    uc = UniformCost()
    res = uc.solve_problem(map_prob)
    print(res)

    # Ex.10
    a_star_null_heuristic = AStar(NullHeuristic)
    a_star_null_heuristic_res = a_star_null_heuristic.solve_problem(map_prob)
    print(a_star_null_heuristic_res)

    # Ex.11
    a_star_air_dist_heuristic = AStar(AirDistHeuristic)
    a_star_air_dist_heuristic_res = a_star_air_dist_heuristic.solve_problem(map_prob)
    print(a_star_air_dist_heuristic_res)

    # Ex.12
    run_astar_for_weights_in_range(AirDistHeuristic, map_prob)

# --------------------------------------------------------------------
# ----------------------- Deliveries Problem -------------------------
# --------------------------------------------------------------------


def relaxed_deliveries_problem():

    print()
    print('Solve the relaxed deliveries problem.')

    big_delivery = DeliveriesProblemInput.load_from_file('big_delivery.in', roads)
    big_deliveries_prob = RelaxedDeliveriesProblem(big_delivery)

    # Ex.16
    a_star_max_air_dist_heuristic = AStar(MaxAirDistHeuristic)
    a_star_max_air_dist_heuristic_res = a_star_max_air_dist_heuristic.solve_problem(big_deliveries_prob)
    print(a_star_max_air_dist_heuristic_res)

    # Ex.17
    a_star_mst_air_dist_heuristic = AStar(MSTAirDistHeuristic)
    a_star_mst_air_dist_heuristic_res = a_star_mst_air_dist_heuristic.solve_problem(big_deliveries_prob)
    print(a_star_mst_air_dist_heuristic_res)

    # Ex.18
    run_astar_for_weights_in_range(MSTAirDistHeuristic, big_deliveries_prob)

    # Ex.24
    # TODO:
    # 1. Run the stochastic greedy algorithm for 100 times.
    #    For each run, store the cost of the found solution.
    #    Store these costs in a list.
    # 2. The "Anytime Greedy Stochastic Algorithm" runs the greedy
    #    greedy stochastic for N times, and after each iteration
    #    stores the best solution found so far. It means that after
    #    iteration #i, the cost of the solution found by the anytime
    #    algorithm is the MINIMUM among the costs of the solutions
    #    found in iterations {1,...,i}. Calculate the costs of the
    #    anytime algorithm wrt the #iteration and store them in a list.
    # 3. Calculate and store the cost of the solution received by
    #    the A* algorithm (with w=0.5).
    # 4. Calculate and store the cost of the solution received by
    #    the deterministic greedy algorithm (A* with w=1).
    # 5. Plot a figure with the costs (y-axis) wrt the #iteration
    #    (x-axis). Of course that the costs of A*, and deterministic
    #    greedy are not dependent with the iteration number, so
    #    these two should be represented by horizontal lines.

    # first two graphs are of deterministic algorithms, therefore, would issue the same result each time.
    a_star_mst_w05 = AStar(MSTAirDistHeuristic, 0.5)
    a_star_mst_w05_res = a_star_mst_w05.solve_problem(big_deliveries_prob)
    print(a_star_mst_w05_res)
    a_star_mst_w05_res_tot_dist = a_star_mst_w05_res.final_search_node.cost
    a_star_mst_w05_res_tot_dist_duplicate = np.full(100, a_star_mst_w05_res_tot_dist)
    plt.plot(np.arange(1, 101), a_star_mst_w05_res_tot_dist_duplicate, label="A* MST heuristic weight=0.5")

    a_star_mst_w10 = AStar(MSTAirDistHeuristic, 1)
    a_star_mst_w10_res = a_star_mst_w10.solve_problem(big_deliveries_prob)
    print(a_star_mst_w10_res)
    a_star_mst_w10_res_tot_dist = a_star_mst_w10_res.final_search_node.cost
    a_star_mst_w10_res_tot_dist_duplicate = np.full(100, a_star_mst_w10_res_tot_dist)
    plt.plot(np.arange(1, 101), a_star_mst_w10_res_tot_dist_duplicate, label="A* MST heuristic weight=1.0")

    # run greedy stochastic K times
    K = 100
    greedy_stochastic_mst_res_dist = list()
    anytime_res_distances = list()

    for k in range(K):
        # greedy stochastic solver
        greedy_stochastic_mst = GreedyStochastic(MSTAirDistHeuristic)
        # algorithm result
        greedy_stochastic_mst_res = greedy_stochastic_mst.solve_problem(big_deliveries_prob)
        print(greedy_stochastic_mst_res)
        # append to list the algorithm result (distance)
        greedy_stochastic_mst_res_dist.append(greedy_stochastic_mst_res.final_search_node.cost)

    # each element at anytime's list hold the best solution up to that point
    anytime_res_distances.append(greedy_stochastic_mst_res_dist[0])
    for k in range(1, K):
        anytime_res_distances.append(min(greedy_stochastic_mst_res_dist[:k+1]))

    plt.plot(np.arange(1, K+1), greedy_stochastic_mst_res_dist, label="Greedy Stochastic")

    plt.plot(np.arange(1, K+1), anytime_res_distances, label="anytime algorithm")

    plt.xlabel('iterations')
    plt.ylabel('distance traveled')
    plt.title("distance traveled as function of Anytime Algorithm iterations")
    plt.grid()
    plt.legend()
    plt.show()


def strict_deliveries_problem():
    print()
    print('Solve the strict deliveries problem.')

    small_delivery = DeliveriesProblemInput.load_from_file('small_delivery.in', roads)
    small_deliveries_strict_problem = StrictDeliveriesProblem(
        small_delivery, roads, inner_problem_solver=AStar(AirDistHeuristic))

    a_star = AStar(MSTAirDistHeuristic)
    res = a_star.solve_problem(small_deliveries_strict_problem)
    print(res)

    # Ex.26
    # with `MSTAirDistHeuristic` and `small_deliveries_strict_problem`.
    run_astar_for_weights_in_range(MSTAirDistHeuristic, small_deliveries_strict_problem)

    # Ex.28
    # solve the `small_deliveries_strict_problem` with it and print the results (as before).
    a_star_relaxed_heuristic = AStar(RelaxedDeliveriesHeuristic)
    a_star_relaxed_heuristic_res = a_star_relaxed_heuristic.solve_problem(small_deliveries_strict_problem)
    print(a_star_relaxed_heuristic_res)


def main():
    map_problem()
    relaxed_deliveries_problem()
    strict_deliveries_problem()


if __name__ == '__main__':
    exit(main())
