from pyperplan.search import guct_normal_search, guct_normal2_search
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


def test_guct_three_step():
    task = dummy_task.get_simple_search_space()
    plan, exp = _run(guct_normal_search, task)
    assert plan is not None and len(plan) == 3
    assert exp >= len(plan)
    plan2, exp2 = _run(guct_normal2_search, task)
    assert plan2 is not None and len(plan2) == 3
    assert exp2 >= len(plan2)


def test_guct_no_solution():
    task = dummy_task.get_search_space_no_solution()
    plan, exp = _run(guct_normal_search, task)
    assert plan is None
    assert exp == 50
    plan2, exp2 = _run(guct_normal2_search, task)
    assert plan2 is None
    assert exp2 == 50
