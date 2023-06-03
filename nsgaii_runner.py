# EdgeSimPy components
from edge_sim_py import *

# EdgeSimPy extensions
from simulator.simulator_extensions import *

# Helper methods
from simulator.helper_methods import *

# Maintenance strategies
from simulator.strategies import *

# Pymoo components
from pymoo.util.display import Display
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

# Python libraries
import json
import pstats
import argparse
import numpy as np
import cProfile as profile


PERFORMANCE_DEBUGGING = True


def prepare_edgesimpy_simulation(executing_nsgaii_runner=True, parameters: dict = {}):
    simulator = Simulator(
        tick_duration=1,
        tick_unit="seconds",
        stopping_criterion=maintenance_stopping_criterion,
        resource_management_algorithm=eval("nsgaii"),
        dump_interval=float("inf"),
        user_defined_functions=[immobile],
    )
    simulator.executing_nsgaii_runner = executing_nsgaii_runner

    # Loading the dataset
    simulator.initialize(input_file=parameters["dataset"])

    return simulator


def reset_simulated_environment():
    NetworkFlow._object_count = 0
    NetworkFlow._instances = []

    simulator = Simulator(
        tick_duration=1,
        tick_unit="seconds",
        stopping_criterion=maintenance_stopping_criterion,
        resource_management_algorithm=eval("nsgaii"),
        dump_interval=float("inf"),
        user_defined_functions=[immobile],
    )
    simulator.executing_nsgaii_runner = True

    simulator.topology = ComponentManager.topology
    simulator.initialize_agent(agent=ComponentManager.topology)

    for component_instance in ComponentManager.components:
        component_builder = globals()[f"{component_instance.__class__.__name__}Builder"]
        component_builder.create_attributes(component_instance=component_instance, attributes_metadata=component_instance.attributes)
        component_builder.create_relationships(component_instance=component_instance)

        if hasattr(component_instance, "model") and hasattr(component_instance, "unique_id"):
            simulator.initialize_agent(agent=component_instance)

    return simulator


def run_edgesimpy_simulation(simulator):
    # Executing the simulation
    simulator.run_model()
    simulation_metrics = {
        "maintenance_time": int(simulator.model_metrics["overall"]["maintenance_time"]),
        "delay_sla_violations": int(simulator.model_metrics["overall"]["overall_delay_sla_violations"]),
        "penalty": int(simulator.model_metrics["overall"]["penalty"]),
    }
    return simulation_metrics


class MyDisplay(Display):
    """Creates a visualization on how the genetic algorithm is evolving throughout the generations."""

    def _do(self, problem: object, evaluator: object, algorithm: object):
        """Defines the way information about the genetic algorithm is printed after each generation.

        Args:
            problem (object): Instance of the problem being solved.
            evaluator (object): Object that makes modifications before calling the problem's evaluate function.
            algorithm (object): Algorithm being executed.
        """
        super()._do(problem, evaluator, algorithm)

        # Aggregating fitness values
        maintenance_time = int(np.min(algorithm.pop.get("F")[:, 0]))
        delay_sla_violations = int(np.min(algorithm.pop.get("F")[:, 1]))

        # Aggregating penalties
        penalty = int(np.min(algorithm.pop.get("CV")[:, 0]))

        self.output.append("TIME", maintenance_time)
        self.output.append("SLAV", delay_sla_violations)
        self.output.append("PENA", penalty)


class PlacementProblem(Problem):
    """Describes the application placement as an optimization problem."""

    def __init__(self, **kwargs):
        """Initializes the problem instance."""
        self.maintenance_batches = kwargs.get("maintenance_batches")
        self.number_of_services = kwargs.get("number_of_services")
        self.number_of_edge_servers = kwargs.get("number_of_edge_servers")
        self.parameters = kwargs.get("parameters")

        self.chromosome_size = int(self.number_of_services * self.maintenance_batches)

        self.simulator = prepare_edgesimpy_simulation(parameters=self.parameters)

        super().__init__(
            n_var=self.chromosome_size,
            n_obj=2,
            n_constr=1,
            xl=1,
            xu=self.number_of_edge_servers,
            type_var=int,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates solutions according to the problem objectives.

        Args:
            x (list): Solution or set of solutions that solve the problem.
            out (dict): Output of the evaluation function.
        """
        output = [self.get_fitness_score_and_constraints(solution=solution) for solution in x]

        out["F"] = np.array([item[0] for item in output])
        out["G"] = np.array([item[1] for item in output])

    def get_fitness_score_and_constraints(self, solution) -> tuple:
        """Calculates the fitness score and penalties of a solution based on the problem definition.

        Args:
            solution (list): Solution that solves the problem.

        Returns:
            tuple: Output of the evaluation function containing the fitness scores of the solution and its penalties.
        """
        # Parsing the tested solution to send it to the EdgeSimPy runner
        formatted_solution = [
            list(solution[i : i + self.number_of_services]) for i in range(0, len(solution), self.number_of_services)
        ]
        self.simulator.resource_management_algorithm_parameters["solution"] = formatted_solution

        # Applying the maintenance plan
        simulation_metrics = run_edgesimpy_simulation(simulator=self.simulator)
        self.simulator = reset_simulated_environment()

        # Aggregating fitness values
        maintenance_time = simulation_metrics["maintenance_time"]
        delay_sla_violations = simulation_metrics["delay_sla_violations"]
        fitness = (maintenance_time, delay_sla_violations)

        # print(f"solution: {formatted_solution}. Fitness: {fitness}")

        # Aggregating penalties
        penalty = simulation_metrics["penalty"]

        return (fitness, penalty)


def nsgaii_runner(parameters: dict = {}) -> list:
    seed_value = parameters["seed_value"]
    dataset = parameters["dataset"]

    pop_size = parameters["pop_size"]
    cross_prob = parameters["cross_prob"]
    mut_prob = parameters["mut_prob"]
    n_gen = parameters["n_gen"]
    maintenance_batches = parameters["maintenance_batches"]

    with open(f"{os.getcwd()}/{dataset}", "r", encoding="UTF-8") as read_file:
        data = json.load(read_file)
        number_of_services = len(data["Service"])
        number_of_edge_servers = len(data["EdgeServer"])

    # Defining genetic algorithm's attributes
    method = NSGA2(
        pop_size=pop_size,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_ux", prob=cross_prob),
        mutation=get_mutation("int_pm", prob=mut_prob),
        eliminate_duplicates=False,
    )

    # Running the genetic algorithm
    problem = PlacementProblem(
        parameters=parameters,
        number_of_services=number_of_services,
        number_of_edge_servers=number_of_edge_servers,
        maintenance_batches=maintenance_batches,
    )
    res = minimize(
        problem,
        method,
        termination=("n_gen", n_gen),
        seed=seed_value,
        verbose=True,
        display=MyDisplay(),
    )

    return

    # Parsing the NSGA-II output
    solutions = []
    for i in range(len(res.X)):
        solution = {
            "Maintenance Plan": res.X[i].tolist(),
            "Maintenance Time": res.F[i][0],
            "SLA Violations": res.F[i][1],
            "Penalty": res.CV[i][0].tolist(),
        }
        solutions.append(solution)

    # Gathering min and max values for each objective in the fitness function
    min_maintenance_time = min([solution["Maintenance Time"] for solution in solutions])
    max_maintenance_time = max([solution["Maintenance Time"] for solution in solutions])
    min_sla_violations = min([solution["SLA Violations"] for solution in solutions])
    max_sla_violations = max([solution["SLA Violations"] for solution in solutions])

    # Selecting the best maintenance plan found by the NSGA-II algorithm
    solutions = sorted(
        solutions,
        key=lambda s: (
            s["Penalty"],
            min_max_norm(
                x=s["Maintenance Time"],
                minimum=min_maintenance_time,
                maximum=max_maintenance_time,
            )
            + min_max_norm(
                x=s["SLA Violations"],
                minimum=min_sla_violations,
                maximum=max_sla_violations,
            ),
        ),
    )

    print("=== SOLUTIONS FOUND:")
    for solution in solutions:
        print(f"\t{solution}")

    best_solution = solutions[0]["Maintenance Plan"]

    parameters = {
        "algorithm": "nsgaii",
        "dataset": dataset,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "cross_prob": cross_prob,
        "mut_prob": mut_prob,
        "solution": [best_solution[i : i + number_of_services] for i in range(0, len(best_solution), number_of_services)],
    }

    simulator = Simulator(
        tick_duration=1,
        tick_unit="seconds",
        stopping_criterion=maintenance_stopping_criterion,
        resource_management_algorithm=eval("nsgaii"),
        dump_interval=float("inf"),
        resource_management_algorithm_parameters=parameters,
        user_defined_functions=[immobile],
    )
    # Loading the dataset
    simulator.initialize(input_file=parameters["dataset"])

    # Executing the simulation
    simulator.executing_nsgaii_runner = False
    simulator.run_model()

    print("\n")
    print("============================")
    print("============================")
    print("==== SIMULATION RESULTS ====")
    print("============================")
    print("============================")

    verbose_metrics_to_hide = [
        "wait_times",
        "pulling_times",
        "state_migration_times",
        "sizes_of_cached_layers",
        "sizes_of_uncached_layers",
        "migration_times",
    ]

    print("=== OVERALL ===")
    overall_metrics = simulator.model_metrics["overall"]
    for key, value in overall_metrics.items():
        if key in verbose_metrics_to_hide:
            continue
        print(f"{key}: {value}")

    print("\n=== PER BATCH ===")
    per_batch_metrics = simulator.model_metrics["per_batch"]
    for key, value in per_batch_metrics.items():
        print(f"{key}: {value}")


def main(parameters: dict = {}):
    load_edgesimpy_extensions()
    nsgaii_runner(parameters=parameters)


if __name__ == "__main__":
    # Parsing named arguments from the command line
    parser = argparse.ArgumentParser()

    # Generic arguments
    parser.add_argument("--seed", "-s", help="Seed value for EdgeSimPy", default="1")
    parser.add_argument("--dataset", "-d", help="Dataset file")

    # NSGA-II arguments
    parser.add_argument("--pop_size", "-p", help="Population size", default="0")
    parser.add_argument("--n_gen", "-g", help="Number of generations", default="0")
    parser.add_argument("--cross_prob", "-c", help="Crossover probability (0.0 to 1.0)", default="1")
    parser.add_argument("--mut_prob", "-m", help="Mutation probability (0.0 to 1.0)", default="0")
    parser.add_argument("--maintenance-batches", "-b", help="Maintenance batches", default="0")

    args = parser.parse_args()

    parameters = {
        "seed_value": int(args.seed),
        "dataset": args.dataset,
        "pop_size": int(args.pop_size),
        "n_gen": int(args.n_gen),
        "cross_prob": float(args.cross_prob),
        "mut_prob": float(args.mut_prob),
        "maintenance_batches": int(args.maintenance_batches),
    }

    if PERFORMANCE_DEBUGGING:
        prof = profile.Profile()
        prof.enable()

    main(parameters=parameters)

    if PERFORMANCE_DEBUGGING:
        prof.disable()
        stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
        stats.print_stats(40)
