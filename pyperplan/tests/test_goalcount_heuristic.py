from pyperplan.heuristics.goalcount import GoalCountHeuristic
from pyperplan.search import searchspace
from pyperplan.task import Operator, Task


def _get_simple_task():
    op1 = Operator("op1", {"var1"}, {"var2"}, set())
    op2 = Operator("op2", {"var1"}, set(), set())
    op3 = Operator("op3", {"var2"}, {"var1"}, set())
    init = frozenset(["var1"])
    goals = frozenset(["var1", "var2"])
    return Task("task1", {"var1", "var2", "var3"}, init, goals, [op1, op2, op3])


def test_goalcount_start():
    task = _get_simple_task()
    hc = GoalCountHeuristic(task)
    node = searchspace.make_root_node(task.initial_state)
    assert hc(node) == 1


def test_goalcount_goal():
    task = _get_simple_task()
    hc = GoalCountHeuristic(task)
    node = searchspace.make_root_node(task.goals)
    assert hc(node) == 0
