from pyperplan.search import gbfs_search
from pyperplan.tests import dummy_task


def test_gbfs_at_goal():
    task = dummy_task.get_search_space_at_goal()
    heuristic = dummy_task.DummyHeuristic(task)
    plan, expansions, evaluations = gbfs_search(task, heuristic)
    assert plan == []
    assert expansions >= 1
    assert evaluations >= expansions


def test_gbfs_three_step():
    task = dummy_task.get_simple_search_space()
    heuristic = dummy_task.DummyHeuristic(task)
    plan, expansions, evaluations = gbfs_search(task, heuristic)
    assert plan is not None and len(plan) == 3
    assert expansions >= len(plan)
    assert evaluations >= expansions


def test_gbfs_no_solution():
    task = dummy_task.get_search_space_no_solution()
    heuristic = dummy_task.DummyHeuristic(task)
    plan, expansions, evaluations = gbfs_search(task, heuristic)
    assert plan is None
    assert expansions > 0
    assert evaluations >= expansions
