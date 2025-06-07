#
# This file is part of pyperplan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#
"""Greedy best-first search with expansion counting."""

import heapq
import logging

from . import a_star, searchspace


def gbfs_search(
    task,
    heuristic,
    max_expansions=None,
    max_evaluations=None,
    use_relaxed_plan=False,
):
    """Search for a plan using greedy best-first search.

    Returns a tuple ``(plan, expansions, evaluations)`` where ``plan`` is the
    list of operators leading to the goal or ``None`` if no plan was found,
    ``expansions`` is the number of expanded nodes and ``evaluations`` is the
    number of heuristic evaluations.  The search stops after ``max_expansions``
    node expansions or ``max_evaluations`` node evaluations if given.
    """

    make_open_entry = a_star.ordered_node_greedy_best_first
    open_list = []
    state_cost = {task.initial_state: 0}
    node_tiebreaker = 0

    root = searchspace.make_root_node(task.initial_state)
    init_h = heuristic(root)
    evaluations = 1
    heapq.heappush(open_list, make_open_entry(root, init_h, node_tiebreaker))
    logging.info("Initial h value: %f" % init_h)

    besth = float("inf")
    expansions = 0

    while open_list:
        if max_expansions is not None and expansions >= max_expansions:
            break
        if max_evaluations is not None and evaluations >= max_evaluations:
            break
        (f, h, _tie, pop_node) = heapq.heappop(open_list)
        if h < besth:
            besth = h
            logging.debug(
                "Found new best h: %d after %d expansions" % (besth, expansions)
            )

        pop_state = pop_node.state
        if state_cost[pop_state] == pop_node.g:
            expansions += 1
            if task.goal_reached(pop_state):
                logging.info("Goal reached. Start extraction of solution.")
                logging.info("%d Nodes expanded" % expansions)
                return pop_node.extract_solution(), expansions, evaluations
            rplan = None
            if use_relaxed_plan:
                (rh, rplan) = heuristic.calc_h_with_plan(
                    searchspace.make_root_node(pop_state)
                )
                logging.debug("relaxed plan %s " % rplan)

            for op, succ_state in task.get_successor_states(pop_state):
                if max_evaluations is not None and evaluations >= max_evaluations:
                    break
                if use_relaxed_plan and rplan and op.name not in rplan:
                    continue
                succ_node = searchspace.make_child_node(pop_node, op, succ_state)
                h_val = heuristic(succ_node)
                evaluations += 1
                if max_evaluations is not None and evaluations >= max_evaluations:
                    # evaluation limit reached after computing heuristic
                    break
                if h_val == float("inf"):
                    continue
                old_succ_g = state_cost.get(succ_state, float("inf"))
                if succ_node.g < old_succ_g:
                    node_tiebreaker += 1
                    heapq.heappush(
                        open_list, make_open_entry(succ_node, h_val, node_tiebreaker)
                    )
                    state_cost[succ_state] = succ_node.g

        if max_evaluations is not None and evaluations >= max_evaluations:
            break

    logging.info("No operators left. Task unsolvable or limit reached.")
    logging.info("%d Nodes expanded" % expansions)
    return None, expansions, evaluations
