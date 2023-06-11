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
import time
import argparse


VERBOSE = True


def main(seed_value: int, algorithm: str, dataset: str, parameters: dict = {}):
    # Setting a seed value to enable reproducibility
    seed(seed_value)

    # Parsing NSGA-II parameters string
    parameters_string = ""
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
        dump_interval=10000,
        logs_directory=f"logs/algorithm={algorithm};{int(time.time())};{parameters_string}",
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
            svs = [svs.id for svs in s.services]
            regs = [reg.id for reg in s.container_registries]
            print(f"{s}. Capacity: {c}. Demand: {d}. Services: {svs}. Registries: {regs}")
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
        ONLY_DISPLAY_BATCH_TIME_RELATED_METRICS = True
        batch_time_related_metrics = [
            {"original_name": "max_wait_times", "new_name": "WAIT"},
            {"original_name": "max_pulling_times", "new_name": "PULL"},
            {"original_name": "max_state_migration_times", "new_name": "STAT"},
        ]

        print("=== MIGRATIONS ===")
        migrations = []
        for service in Service.all():
            for migration in service._Service__migrations:
                migration_metadata = {
                    "Batch": migration["maintenance_batch"],
                    "Duration": migration["end"] - migration["start"],
                    "WAIT": migration["waiting_time"],
                    "PULL": migration["pulling_layers_time"],
                    "STAT": migration["migrating_service_state_time"],
                    "Hits": migration["cache_hits"],
                    "Misses": migration["cache_misses"],
                    "Size cached layers": sum(migration["sizes_of_cached_layers"]),
                    "Size uncached layers": sum(migration["sizes_of_uncached_layers"]),
                }
                migrations.append(migration_metadata)
        migrations = sorted(migrations, key=lambda m: (m["Batch"], m["Duration"]))
        for migration in migrations:
            print(f"\t{migration}")

        print("=== OVERALL ===")
        overall_metrics = simulator.model_metrics["overall"]
        for key, value in overall_metrics.items():
            if key in verbose_metrics_to_hide:
                continue
            print(f"{key}: {value}")

        print("\n=== PER BATCH ===")
        per_batch_metrics = simulator.model_metrics["per_batch"]
        for key, value in per_batch_metrics.items():
            if ONLY_DISPLAY_BATCH_TIME_RELATED_METRICS:
                if any(key == metric_name["original_name"] for metric_name in batch_time_related_metrics):
                    metric_name = [metric["new_name"] for metric in batch_time_related_metrics if metric["original_name"] == key][0]
                    print(f"{metric_name}: {value}")
            else:
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
