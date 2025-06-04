"""Evaluation script for agile satisficing comparisons."""

import glob
import json
import multiprocessing as mp
import os
import resource
import sys
from queue import Empty # For mp.Queue.get() timeout

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

MAX_GROUND_TIME = 60  # seconds
MAX_GROUND_MEMORY = 2 * 1024**3  # bytes
MAX_EXPANSIONS = 1000
RESULTS_FILE = "evaluation_results.json"


def _ground_worker(domain_file, problem_file, queue):
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_GROUND_MEMORY, MAX_GROUND_MEMORY))
    except Exception as e:
        print(f"Warning: Could not set memory rlimit for {problem_file} in worker: {e}", file=sys.stderr)

    try:
        problem = planner._parse(domain_file, problem_file)
        if problem:
            task = planner._ground(problem)
            queue.put(task)
        else:
            print(f"  Parsing failed for {problem_file}, putting None on queue.", file=sys.stderr)
            queue.put(None)
    except Exception as e:
        print(f"  Exception in _ground_worker for {problem_file}: {e}", file=sys.stderr)
        queue.put(None)


def ground_problem(domain_file, problem_file):
    queue = mp.Queue(1)
    proc = mp.Process(target=_ground_worker, args=(domain_file, problem_file, queue))
    proc.start()

    task = None
    timed_out = False

    proc.join(MAX_GROUND_TIME)

    if proc.is_alive():
        timed_out = True
        print(f"  Process for {problem_file} grounding timed out after {MAX_GROUND_TIME}s. Terminating.")
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            print(f"  Warning: Process for {problem_file} did not terminate gracefully after SIGTERM and 5s wait.", file=sys.stderr)
    else:
        exit_code = proc.exitcode
        if exit_code == 0:
            try:
                task = queue.get(block=True, timeout=2)
            except Empty:
                print(f"  Process for {problem_file} finished but queue was empty (timeout on get).", file=sys.stderr)
                task = None
            except Exception as e:
                print(f"  Error getting from queue for {problem_file}: {e}", file=sys.stderr)
                task = None
        else:
            print(f"  Process for {problem_file} grounding exited with code {exit_code}.", file=sys.stderr)
            task = None

    queue.close()
    queue.join_thread()

    if not timed_out and task is not None:
        return task, False # Mimics original successful return where second element is explicitly False
    return task, timed_out # Covers timeout cases and task is None cases


def run_configuration(task, search_fun, heuristic_cls):
    heuristic = heuristic_cls(task)
    plan, expansions = search_fun(task, heuristic, max_expansions=MAX_EXPANSIONS)
    solved = plan is not None
    return solved, expansions


def evaluate():
    results = {f"{h}-{s}": [] for h in HEURISTICS for s in SEARCHES}

    def dump_results():
        with open(RESULTS_FILE, "w") as fh:
            json.dump(results, fh, indent=2)

    benchmark_dirs = [d for d in glob.glob("../benchmarks/*") if os.path.isdir(d)]
    # benchmark_dirs = benchmark_dirs[1:]
    for bdir in benchmark_dirs:
        problems = sorted(glob.glob(os.path.join(bdir, "task*.pddl")))
        for prob in problems:
            print(f"Solving {prob}...")
            domain = planner.find_domain(prob)
            task, timed_out = ground_problem(domain, prob)
            if task is None:
                if timed_out:
                    print("  Grounding timed out")
                else:
                    print("  Grounding failed")
                break
            for hname, hcls in HEURISTICS.items():
                for sname, sfun in SEARCHES.items():
                    print(f"  {hname} with {sname}")
                    solved, exp = run_configuration(task, sfun, hcls)
                    results[f"{hname}-{sname}"].append(
                        exp if solved else MAX_EXPANSIONS
                    )
                    dump_results()
            plot_results(results)

    print("Evaluation finished.")


def plot_results(results):
    limits = range(1, MAX_EXPANSIONS)
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
