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
from dataclasses import dataclass

from . import searchspace


@dataclass
class _Stats:
    visits: int
    mean: float
    m2: float
    min_val: float
    max_val: float
    log_sum:  float


def _make_stats(value: float) -> _Stats:
    v = max(value, 1e-9)                       # protect log(0)
    return _Stats(1, value, 0.0, value, value, math.log(v))


def _update_stats(stats: _Stats, value: float) -> None:
    v = max(value, 1e-9)
    stats.visits += 1

    delta = value - stats.mean
    stats.mean += delta / stats.visits
    stats.m2 += delta * (value - stats.mean)
    stats.min_val = min(stats.min_val, value)
    stats.max_val = max(stats.max_val, value)
    stats.log_sum += math.log(v)


def _variance(stats: _Stats) -> float:
    return 0.0 if stats.visits <= 1 else stats.m2 / (stats.visits - 1)


def _stddev(stats: _Stats) -> float:
    return math.sqrt(_variance(stats))

def _a_hat(stats: _Stats) -> float:
    if stats.visits < 2 or stats.max_val <= 0:
        return 1.0
    denom = math.log(stats.max_val) - (stats.log_sum / stats.visits)
    if denom <= 0.0:
        return 1.0
    return 1.0 / denom

def _norm_quantile(t):
    """http://m-hikari.com/ams/ams-2014/ams-85-88-2014/epureAMS85-88-2014.pdf"""
    alpha = 1 - 1 / t

    q = 10 * math.log(1 - math.log(-math.log(alpha) / math.log(2)) / math.log(22)) / math.log(41)

    return q

def _score_lcb_normal(stats: _Stats, total: int) -> float:
    n = stats.visits
    if n == 0:
        return float("-inf")
    return stats.mean - _stddev(stats) * math.sqrt((16 * math.log(total)) / n)


def _score_lcb_normal2(stats: _Stats, total: int) -> float:
    if total == 0:
        return float("-inf")
    return stats.mean - _stddev(stats) * math.sqrt(2 * math.log(total))


def _score_lcb_uniform(stats: _Stats, total: int) -> float:
    n = stats.visits
    if total == 0:
        return float("-inf")
    return (stats.max_val + stats.min_val) / 2 - (stats.max_val - stats.min_val) * math.sqrt(6 * n * math.log(total))


def _score_lcb_power(stats: _Stats, total: int) -> float:
    n = stats.visits
    a_hat = _a_hat(stats)
    if total == 0:
        return float("-inf")
    return (stats.max_val * a_hat) / (a_hat + 1) - stats.max_val * math.sqrt(6 * n * math.log(total))

def _score_lcb_clt(stats: _Stats, total: int) -> float:
    n = stats.visits
    if total <= 1 or n == 0:
        return float("-inf")
    return stats.mean - _norm_quantile(total) * math.sqrt(_variance(stats)/n)


def _guct_search(task, heuristic, score_fun, max_expansions=None):
    open_list = []
    node_info = {}
    tie = 0

    root = searchspace.make_root_node(task.initial_state)
    init_h = heuristic(root)
    node_info[root] = _make_stats(init_h)
    heapq.heappush(open_list, (init_h, tie, root))
    tie += 1
    logging.info("Initial h value: %f" % init_h)

    expansions = 0
    while open_list:
        if max_expansions is not None and expansions >= max_expansions:
            break
        score, _, node = heapq.heappop(open_list)
        parent_stats = node_info.get(node)
        if parent_stats is None:
            parent_stats = _make_stats(heuristic(node))
            node_info[node] = parent_stats
        parent_stats.visits += 1
        parent_visits = parent_stats.visits
        expansions += 1

        if task.goal_reached(node.state):
            logging.info("Goal reached. Start extraction of solution.")
            logging.info("%d Nodes expanded" % expansions)
            return node.extract_solution(), expansions

        for op, succ_state in task.get_successor_states(node.state):
            succ = searchspace.make_child_node(node, op, succ_state)
            h_val = heuristic(succ)
            stats = node_info.get(succ)
            if stats is None:
                stats = _make_stats(h_val)
                node_info[succ] = stats
            else:
                _update_stats(stats, h_val)
            ucb = score_fun(stats, parent_visits + 1)
            heapq.heappush(open_list, (ucb, tie, succ))
            tie += 1

    logging.info("No operators left. Task unsolvable or limit reached.")
    logging.info("%d Nodes expanded" % expansions)
    return None, expansions


def guct_normal_search(task, heuristic, max_expansions=None):
    """GUCT search variant using UCB1-Normal."""

    return _guct_search(task, heuristic, _score_lcb_normal, max_expansions)


def guct_normal2_search(task, heuristic, max_expansions=None):
    """GUCT search variant using UCB1-Normal2."""

    return _guct_search(task, heuristic, _score_lcb_normal2, max_expansions)


def guct_uniform_search(task, heuristic, max_expansions=None):
    """GUCT search variant using LCB1-Uniform."""

    return _guct_search(task, heuristic, _score_lcb_uniform, max_expansions)


def guct_power_search(task, heuristic, max_expansions=None):
    """GUCT search variant using LCB1-Power."""

    return _guct_search(task, heuristic, _score_lcb_power, max_expansions)

def guct_clt_search(task, heuristic, max_expansions=None):
    """GUCT search variant using LCB1-Power."""

    return _guct_search(task, heuristic, _score_lcb_clt, max_expansions)
