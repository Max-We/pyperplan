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
"""Greedy UCT style search algorithms."""

import heapq
import logging
import math
import random

from . import searchspace


def _ucb_score(mean, visits, parent_visits, c):
    if visits == 0:
        return float("-inf")
    return mean - c * math.sqrt(math.log(parent_visits) / visits)




def _guct_search(task, heuristic, score_fun, max_expansions=None):
    open_list = []
    node_info = {}
    tie = 0

    root = searchspace.make_root_node(task.initial_state)
    init_h = heuristic(root)
    node_info[root] = 1
    heapq.heappush(open_list, (init_h, tie, root))
    tie += 1
    logging.info("Initial h value: %f" % init_h)

    expansions = 0
    while open_list:
        if max_expansions is not None and expansions >= max_expansions:
            break
        score, _, node = heapq.heappop(open_list)
        parent_visits = node_info.get(node, 1)
        node_info[node] = parent_visits + 1
        expansions += 1

        if task.goal_reached(node.state):
            logging.info("Goal reached. Start extraction of solution.")
            logging.info("%d Nodes expanded" % expansions)
            return node.extract_solution(), expansions

        for op, succ_state in task.get_successor_states(node.state):
            succ = searchspace.make_child_node(node, op, succ_state)
            h_val = heuristic(succ)
            visits = node_info.get(succ, 0) + 1
            node_info[succ] = visits
            ucb = score_fun(h_val, visits, parent_visits + 1)
            heapq.heappush(open_list, (ucb, tie, succ))
            tie += 1

    logging.info("No operators left. Task unsolvable or limit reached.")
    logging.info("%d Nodes expanded" % expansions)
    return None, expansions


def guct_normal_search(task, heuristic, max_expansions=None):
    """GUCT search variant using UCB1-Normal."""

    return _guct_search(
        task,
        heuristic,
        lambda m, v, pv: _ucb_score(m, v, pv, 1.0),
        max_expansions=max_expansions,
    )


def guct_normal2_search(task, heuristic, max_expansions=None):
    """GUCT search variant using UCB1-Normal2."""

    return _guct_search(
        task,
        heuristic,
        lambda m, v, pv: _ucb_score(m, v, pv, 2.0),
        max_expansions=max_expansions,
    )


def guct_uniform_search(task, heuristic, max_expansions=None):
    """GUCT search variant using LCB1-Uniform."""

    return _guct_search(
        task,
        heuristic,
        lambda m, v, pv: _ucb_score(m, v, pv, 1.5),
        max_expansions=max_expansions,
    )


def guct_power_search(task, heuristic, max_expansions=None):
    """GUCT search variant using LCB1-Power."""

    return _guct_search(
        task,
        heuristic,
        lambda m, v, pv: _ucb_score(m, v, pv, 0.5),
        max_expansions=max_expansions,
    )
