"""Evaluation script for agile satisficing comparisons."""

import glob
import json
import multiprocessing as mp
import os
import resource
import sys

import matplotlib.pyplot as plt


# Ensure the local pyperplan package is importable when running this
# script directly from the ``dev`` directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pyperplan import heuristics, planner, search


HEURISTICS = {
    "h-ff": heuristics.hFFHeuristic,
    "h-add": heuristics.hAddHeuristic,
    "h-max": heuristics.hMaxHeuristic,
    "h-gc": heuristics.GoalCountHeuristic,
}

SEARCHES = {
    "gbfs": search.gbfs_search,
    "guct-normal": search.guct_normal_search,
    "guct-normal2": search.guct_normal2_search,
}

MAX_GROUND_TIME = 300  # seconds
MAX_GROUND_MEMORY = 2 * 1024**3  # bytes
MAX_EXPANSIONS = 10000


def _ground_worker(domain_file, problem_file, queue):
    resource.setrlimit(resource.RLIMIT_AS, (MAX_GROUND_MEMORY, MAX_GROUND_MEMORY))
    problem = planner._parse(domain_file, problem_file)
    task = planner._ground(problem)
    queue.put(task)


def ground_problem(domain_file, problem_file):
    queue = mp.Queue(1)
    proc = mp.Process(target=_ground_worker, args=(domain_file, problem_file, queue))
    proc.start()
    proc.join(MAX_GROUND_TIME)
    if proc.is_alive():
        proc.terminate()
        return None
    if queue.empty():
        return None
    return queue.get()


def run_configuration(task, search_fun, heuristic_cls):
    heuristic = heuristic_cls(task)
    plan, expansions = search_fun(task, heuristic, max_expansions=MAX_EXPANSIONS)
    solved = plan is not None
    return solved, expansions


def evaluate():
    results = {}
    for hname, hcls in HEURISTICS.items():
        for sname, sfun in SEARCHES.items():
            key = f"{hname}-{sname}"
            results[key] = []

    benchmark_dirs = [d for d in glob.glob("benchmarks/*") if os.path.isdir(d)]
    for bdir in benchmark_dirs:
        problems = sorted(glob.glob(os.path.join(bdir, "task*.pddl")))
        for prob in problems:
            domain = planner.find_domain(prob)
            task = ground_problem(domain, prob)
            if task is None:
                continue
            for hname, hcls in HEURISTICS.items():
                for sname, sfun in SEARCHES.items():
                    solved, exp = run_configuration(task, sfun, hcls)
                    results[f"{hname}-{sname}"].append(
                        exp if solved else MAX_EXPANSIONS
                    )
    with open("evaluation_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    plot_results(results)


def plot_results(results):
    limits = range(1, MAX_EXPANSIONS + 1)
    for name, runs in results.items():
        solved_counts = []
        for lim in limits:
            solved_counts.append(sum(1 for r in runs if r <= lim))
        plt.plot(list(limits), solved_counts, label=name)
    plt.xlabel("Node evaluations")
    plt.ylabel("Solved instances")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_plot.png")


if __name__ == "__main__":
    evaluate()
