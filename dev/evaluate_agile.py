"""Evaluation script for agile satisficing comparisons."""

import glob
import json
import multiprocessing as mp
import os
import resource
import sys
from queue import Empty # For mp.Queue.get() timeout
import logging
import pickle
import tempfile # For temporary file creation
import uuid # For unique filenames

import matplotlib.pyplot as plt


# Ensure the local pyperplan package is importable when running this
# script directly from the ``dev`` directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pyperplan import heuristics, planner, search


HEURISTICS = {
    "h-ff": heuristics.hFFHeuristic,
    # "h-add": heuristics.hAddHeuristic,
    # "h-max": heuristics.hMaxHeuristic,
    # "h-gc": heuristics.GoalCountHeuristic,
}

SEARCHES = {
    "gbfs": search.gbfs_search,
    "guct": search.guct_search,
    "guct-normal": search.guct_normal_search,
    "guct-normal2": search.guct_normal2_search,
    "guct-power": search.guct_power_search,
    "guct-uniform": search.guct_uniform_search,
    "guct-clt": search.guct_clt_search,
}

MAX_GROUND_TIME = 300  # seconds
MAX_GROUND_MEMORY = 2 * 1024**3  # bytes
MAX_EXPANSIONS = 10000
RESULTS_FILE = "evaluation_results.json"
TEMP_DIR = tempfile.mkdtemp(prefix="pyperplan_eval_") # Create a dedicated temp directory


def _ground_worker(domain_file, problem_file, queue):
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_GROUND_MEMORY, MAX_GROUND_MEMORY))
    except Exception as e:
        logging.warning(f"Worker for {problem_file}: Could not set memory rlimit: {e}")

    output_to_queue = None
    try:
        problem = planner._parse(domain_file, problem_file)
        if problem:
            task = planner._ground(problem)
            if task:
                logging.info(f"Worker for {problem_file}: Compiled task successfully.")
                # Save task to a temporary file
                temp_task_file = os.path.join(TEMP_DIR, f"task_{uuid.uuid4()}.pkl")
                with open(temp_task_file, "wb") as f:
                    pickle.dump(task, f)
                output_to_queue = {"type": "file", "path": temp_task_file}
                logging.info(f"Worker for {problem_file}: Saved task to {temp_task_file}. Size: {os.path.getsize(temp_task_file)} bytes.")
            else: # Grounding returned None
                 logging.error(f"Worker for {problem_file}: Grounding returned None.")
                 output_to_queue = {"type": "error", "message": "Grounding returned None"}
        else:
            logging.error(f"Worker for {problem_file}: Parsing failed.")
            output_to_queue = {"type": "error", "message": "Parsing failed"}
    except MemoryError as me:
        logging.error(f"Worker for {problem_file}: MemoryError during grounding: {me}")
        output_to_queue = {"type": "error", "message": f"MemoryError: {me}"}
    except Exception as e:
        logging.exception(f"Worker for {problem_file}: Exception during grounding:")
        output_to_queue = {"type": "error", "message": f"Exception: {e}"}
    finally:
        if output_to_queue:
            logging.info(f"Worker for {problem_file}: Attempting to put result on queue: {output_to_queue}")
            queue.put(output_to_queue)
            logging.info(f"Worker for {problem_file}: Successfully put result on queue. Exiting worker.")
        else: # Should not happen if logic is correct
            logging.error(f"Worker for {problem_file}: No output generated for queue. Putting generic error.")
            queue.put({"type": "error", "message": "Worker finished without specific output."})
            logging.info(f"Worker for {problem_file}: Put generic error on queue. Exiting worker.")


def ground_problem(domain_file, problem_file):
    queue = mp.Queue(1)
    proc = mp.Process(target=_ground_worker, args=(domain_file, problem_file, queue))
    proc.start()
    logging.info(f"Main process: Worker process for {problem_file} started (PID: {proc.pid}).")

    task = None
    timed_out = False
    received_data = None # To store what we get from queue

    logging.info(f"Main process: About to call proc.join() for {problem_file}.")
    proc.join(MAX_GROUND_TIME)
    logging.info(f"Main process: proc.join() returned for {problem_file}.")

    if proc.is_alive():
        timed_out = True
        logging.warning(f"Main process: Worker for {problem_file} grounding timed out after {MAX_GROUND_TIME}s. Terminating.")
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            logging.warning(f"Main process: Worker for {problem_file} did not terminate gracefully after SIGTERM. Killing.")
            proc.kill()
            proc.join(timeout=5)
            if proc.is_alive():
                logging.error(f"Main process: Worker for {problem_file} could not be killed.")
    else:
        exit_code = proc.exitcode
        logging.info(f"Main process: Worker for {problem_file} finished with exit code {exit_code}.")
        if exit_code == 0:
            try:
                logging.info(f"Main process: Attempting queue.get() for {problem_file}.")
                received_data = queue.get(block=True, timeout=10) # Increased timeout
                logging.info(f"Main process: Received from queue for {problem_file}: {received_data}")

                if received_data and isinstance(received_data, dict):
                    if received_data.get("type") == "file":
                        task_file_path = received_data.get("path")
                        if task_file_path and os.path.exists(task_file_path):
                            logging.info(f"Main process: Loading task from file {task_file_path}.")
                            with open(task_file_path, "rb") as f:
                                task = pickle.load(f)
                            logging.info(f"Main process: Successfully loaded task from {task_file_path}. Type: {type(task)}.")
                            try:
                                os.remove(task_file_path)
                                logging.info(f"Main process: Removed temporary task file {task_file_path}.")
                            except Exception as e_remove:
                                logging.warning(f"Main process: Could not remove temp task file {task_file_path}: {e_remove}")
                        else:
                            logging.error(f"Main process: Task file path {task_file_path} not found or invalid.")
                            task = None
                    elif received_data.get("type") == "error":
                        logging.error(f"Main process: Worker for {problem_file} reported error: {received_data.get('message')}")
                        task = None
                    else:
                        logging.error(f"Main process: Received unknown data structure from queue for {problem_file}: {received_data}")
                        task = None
                else:
                    logging.error(f"Main process: Received unexpected or no data from queue for {problem_file}: {received_data}")
                    task = None

            except Empty:
                logging.warning(f"Main process: Worker for {problem_file} finished (exit code 0) but queue was empty on get.")
                task = None
            except MemoryError as me_load:
                logging.error(f"Main process: MemoryError loading task from file for {problem_file}: {me_load}")
                task = None # Task remains None
                # If task_file_path was set, try to remove it
                if received_data and received_data.get("type") == "file":
                    task_file_path_on_error = received_data.get("path")
                    if task_file_path_on_error and os.path.exists(task_file_path_on_error):
                        try:
                            os.remove(task_file_path_on_error)
                            logging.info(f"Main process: Removed temporary task file {task_file_path_on_error} after MemoryError.")
                        except Exception as e_remove_err:
                            logging.warning(f"Main process: Could not remove temp task file {task_file_path_on_error} after MemoryError: {e_remove_err}")
            except Exception as e_get:
                logging.error(f"Main process: Error processing item from queue for {problem_file}: {e_get}")
                task = None
        else:
            logging.error(f"Main process: Worker for {problem_file} grounding failed or exited with code {exit_code}.")
            task = None

    try:
        queue.close()
        queue.join_thread()
    except Exception as q_close_e:
        logging.warning(f"Main process: Exception closing queue for {problem_file}: {q_close_e}")

    if task is not None and not timed_out: # Task is successfully loaded
        return task, False
    return task, timed_out # task is None or timed_out


def run_configuration(task, search_fun, heuristic_cls):
    heuristic = heuristic_cls(task)
    plan, expansions = search_fun(task, heuristic, max_expansions=MAX_EXPANSIONS)
    solved = plan is not None
    return solved, expansions


def evaluate():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    logging.info(f"Temporary directory for tasks: {TEMP_DIR}")

    results = {f"{h}-{s}": [] for h in HEURISTICS for s in SEARCHES}

    def dump_results():
        with open(RESULTS_FILE, "w") as fh:
            json.dump(results, fh, indent=2)

    benchmark_dirs = [d for d in glob.glob("../benchmarks/*") if os.path.isdir(d)]
    if not benchmark_dirs:
        benchmark_dirs = [d for d in glob.glob("benchmarks/*") if os.path.isdir(d)]

    for bdir in benchmark_dirs:
        problems = sorted(glob.glob(os.path.join(bdir, "task*.pddl")))
        for prob in problems:
            logging.info(f"Main process: Solving {prob} in directory {bdir}...")
            domain_file_path = planner.find_domain(prob)
            logging.info(f"Main process: Found domain {domain_file_path} for {prob}.")

            task, timed_out = ground_problem(domain_file_path, prob)

            logging.info(f"Main process: Returned from ground_problem for {prob}. Task is None: {task is None}, Timed out: {timed_out}")

            if task is None:
                if timed_out:
                    logging.warning(f"Main process: Grounding timed out for {prob}. Skipping this problem.")
                else:
                    logging.error(f"Main process: Grounding failed or produced no task for {prob}. Skipping this problem.")
                continue

            logging.info(f"Main process: Successfully received task for {prob}. Proceeding to run configurations.")
            for hname, hcls in HEURISTICS.items():
                for sname, sfun in SEARCHES.items():
                    logging.info(f"Main process:  Attempting config: {hname} with {sname} for problem {prob}")
                    try:
                        solved, exp = run_configuration(task, sfun, hcls)
                        results[f"{hname}-{sname}"].append(
                            exp if solved else MAX_EXPANSIONS
                        )
                    except MemoryError as me:
                        logging.exception(f"Main process:  MemoryError during run_configuration for {prob} with {hname}-{sname}: {me}")
                        results[f"{hname}-{sname}"].append(MAX_EXPANSIONS + 1)
                    except Exception as e:
                        logging.exception(f"Main process:  Exception during run_configuration for {prob} with {hname}-{sname}: {e}")
                        results[f"{hname}-{sname}"].append(MAX_EXPANSIONS + 1)
                    dump_results()
            plot_results(results)

    logging.info("Main process: Evaluation finished.")
    # Clean up the temp directory at the very end
    try:
        import shutil
        shutil.rmtree(TEMP_DIR)
        logging.info(f"Removed temporary directory: {TEMP_DIR}")
    except Exception as e_rm_tempdir:
        logging.warning(f"Could not remove temporary directory {TEMP_DIR}: {e_rm_tempdir}")


def plot_results(results):
    plt.figure()
    limits = range(1, MAX_EXPANSIONS)
    for name, runs in results.items():
        # if "h-ff" not in name: # for debugging
        #     continue
        solved_counts = []
        for lim in limits:
            solved_counts.append(sum(1 for r in runs if isinstance(r, (int, float)) and r <= lim and r < MAX_EXPANSIONS))
        plt.plot(list(limits), solved_counts, label=name)
    plt.xlabel("Node evaluations")
    plt.ylabel("Solved instances")
    plt.legend(fontsize='small')
    plt.title("Planning Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("evaluation_plot.png")
    plt.close()


if __name__ == "__main__":
    evaluate()