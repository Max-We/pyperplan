from pyperplan.search import (
    guct_normal2_search,
    guct_normal_search,
    guct_power_search,
    guct_uniform_search,
)
from pyperplan.tests import dummy_task


def _run(search_fun, task):
    heuristic = dummy_task.DummyHeuristic(task)
    return search_fun(task, heuristic, max_expansions=50)


def test_guct_at_goal():
    task = dummy_task.get_search_space_at_goal()
    plan, expansions = _run(guct_normal_search, task)
    assert plan == []
    assert expansions >= 1
    plan2, expansions2 = _run(guct_normal2_search, task)
    assert plan2 == []
    assert expansions2 >= 1
    plan3, expansions3 = _run(guct_power_search, task)
    assert plan3 == []
    assert expansions3 >= 1
    plan4, expansions4 = _run(guct_uniform_search, task)
    assert plan4 == []
    assert expansions4 >= 1


def test_guct_three_step():
    task = dummy_task.get_simple_search_space()
    plan, exp = _run(guct_normal_search, task)
    assert plan is not None and len(plan) == 3
    assert exp >= len(plan)
    plan2, exp2 = _run(guct_normal2_search, task)
    assert plan2 is not None and len(plan2) == 3
    assert exp2 >= len(plan2)
    plan3, exp3 = _run(guct_power_search, task)
    assert plan3 is not None and len(plan3) == 3
    assert exp3 >= len(plan3)
    plan4, exp4 = _run(guct_uniform_search, task)
    assert plan4 is not None and len(plan4) == 3
    assert exp4 >= len(plan4)


def test_guct_no_solution():
    task = dummy_task.get_search_space_no_solution()
    plan, exp = _run(guct_normal_search, task)
    assert plan is None
    assert exp == 50
    plan2, exp2 = _run(guct_normal2_search, task)
    assert plan2 is None
    assert exp2 == 50
    plan3, exp3 = _run(guct_power_search, task)
    assert plan3 is None
    assert exp3 == 50
    plan4, exp4 = _run(guct_uniform_search, task)
    assert plan4 is None
    assert exp4 == 50
