# EdgeSimPy components
from edge_sim_py import *

# EdgeSimPy extensions
from .simulator_extensions import *

# Helper methods
from simulator.helper_methods import *

# Maintenance strategies
from .strategies import *

# Importing Python libraries
from random import seed
import argparse


VERBOSE = True


def main(seed_value: int, algorithm: str, dataset: str, parameters: dict = {}):
    # Setting a seed value to enable reproducibility
    seed(seed_value)

    # Parsing NSGA-II parameters string
    parameters_string = ""
    if algorithm == "nsgaii":
        for key, value in parameters.items():
            parameters_string += f"{key}={value};"

    # Loading EdgeSimPy extensions
    load_edgesimpy_extensions()

    # Creating a Simulator object
    simulator = Simulator(
        tick_duration=1,
        tick_unit="seconds",
        stopping_criterion=maintenance_stopping_criterion,
        resource_management_algorithm=eval(algorithm),
        resource_management_algorithm_parameters=parameters,
        dump_interval=500,
        logs_directory=f"logs/algorithm={algorithm};{parameters_string}",
        user_defined_functions=[immobile],
    )
    simulator.executing_nsgaii_runner = False

    # Loading the dataset
    simulator.initialize(input_file=dataset)

    if VERBOSE:
        print("==== INITIAL SCENARIO ====")
        for s in EdgeServer.all():
            c = [s.cpu, s.memory, s.disk]
            d = [s.cpu_demand, s.memory_demand, s.disk_demand]
            print(f"{s}. Capacity: {c}. Demand: {d}. Services: {s.services}. Registries: {s.container_registries}")
        print("\n\n\n")

    # Executing the simulation
    simulator.run_model()

    # Displaying the simulation results
    if VERBOSE:
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


if __name__ == "__main__":
    # Parsing named arguments from the command line
    parser = argparse.ArgumentParser()

    # Generic arguments
    parser.add_argument("--seed", "-s", help="Seed value for EdgeSimPy", default="1")
    parser.add_argument("--dataset", "-d", help="Dataset file")
    parser.add_argument("--algorithm", "-a", help="Algorithm that will be executed")

    # NSGA-II arguments
    parser.add_argument("--pop_size", "-p", help="Population size", default="0")
    parser.add_argument("--n_gen", "-g", help="Number of generations", default="0")
    parser.add_argument("--cross_prob", "-c", help="Crossover probability (0.0 to 1.0)", default="1")
    parser.add_argument("--mut_prob", "-m", help="Mutation probability (0.0 to 1.0)", default="0")
    parser.add_argument("--maintenance-batches", "-b", help="Maintenance batches", default="0")
    parser.add_argument("--solution", "-q", help="Predefined maintenance plan", default="[]")

    args = parser.parse_args()

    parameters = {
        "pop_size": int(args.pop_size),
        "n_gen": int(args.n_gen),
        "cross_prob": float(args.cross_prob),
        "mut_prob": float(args.mut_prob),
        "maintenance_batches": float(args.maintenance_batches),
        "solution": eval(args.solution),
    }

    main(seed_value=int(args.seed), algorithm=args.algorithm, dataset=args.dataset, parameters=parameters)
