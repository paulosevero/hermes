# Importing Python libraries
from subprocess import Popen, DEVNULL, TimeoutExpired
import itertools
import os


NUMBER_OF_PARALLEL_PROCESSES = 15


def run_simulation(
    experiment_type: bool,
    dataset: str,
    algorithm: str,
    pop_size: int,
    n_gen: int,
    cross_prob: float,
    mut_prob: float,
    maintenance_batches: int,
):
    """Executes the simulation with the specified parameters.
    Args:
        experiment_type (bool): Type of experiment being executed.
        dataset (str): Dataset being read.
        algorithm (str): Algorithm being executed.
        pop_size (int): Number of chromosomes in the NSGA-II's population.
        n_gen (int): Number of generations of the NSGA-II algorithm.
        cross_prob (float): NSGA-II's crossover probability.
        mut_prob (float): NSGA-II's mutation probability.
        maintenance_batches (int): Predefined number of maintenance batches.
    """
    # Running the simulation based on the parameters and gathering its execution time
    if experiment_type == "sensitivity_analysis":
        cmd = f"python3 -B nsgaii_runner.py --dataset {dataset} --pop_size {pop_size} --n_gen {n_gen} --cross_prob {cross_prob} --mut_prob {mut_prob} --maintenance-batches {maintenance_batches}"
    elif experiment_type == "regular_execution":
        cmd = f"python3 -B -m simulator --dataset {dataset} --algorithm {algorithm}"

    return Popen(cmd.split(" "), stdout=DEVNULL, stderr=DEVNULL)


# Parameters
EXPERIMENT_TYPE = "sensitivity_analysis"

datasets = ["datasets/dataset1.json"]
algorithms = ["nsgaii"]
maintenance_batches = [5]

population_sizes = [300]
number_of_generations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
crossover_probabilities = [1]
mutation_probabilities = [0, 0.25, 0.5, 0.75, 1]

print(f"Datasets: {datasets}")
print(f"Algorithms: {algorithms}")
print(f"Maintenance batches: {maintenance_batches}")
print(f"Population sizes: {population_sizes}")
print(f"Number of generations: {number_of_generations}")
print(f"Crossover probabilities: {crossover_probabilities}")
print(f"Mutation probabilities: {mutation_probabilities}")
print()

# Generating list of combinations with the parameters specified
combinations = list(
    itertools.product(
        datasets,
        algorithms,
        maintenance_batches,
        population_sizes,
        number_of_generations,
        crossover_probabilities,
        mutation_probabilities,
    )
)

# Executing simulations and collecting results
processes = []

print(f"EXECUTING {len(combinations)} COMBINATIONS")
exit(1)
for i, parameters in enumerate(combinations, 1):
    # Parsing parameters
    dataset = parameters[0]
    algorithm = parameters[1]
    maintenance_batches = parameters[2]
    pop_size = parameters[3]
    n_gen = parameters[4]
    cross_prob = parameters[5]
    mut_prob = parameters[6]

    print(f"\t[Execution {i}]")
    print(
        f"\t\t[{algorithm}] dataset={dataset}. pop_size={pop_size}. n_gen={n_gen}. cross_prob={cross_prob}. mut_prob={mut_prob}. maintenance_batches={maintenance_batches}"
    )

    # Executing algorithm
    proc = run_simulation(
        experiment_type=EXPERIMENT_TYPE,
        dataset=dataset,
        algorithm=algorithm,
        pop_size=pop_size,
        n_gen=n_gen,
        cross_prob=cross_prob,
        mut_prob=mut_prob,
        maintenance_batches=maintenance_batches,
    )

    processes.append(proc)

    while len(processes) > NUMBER_OF_PARALLEL_PROCESSES:
        for proc in processes:
            try:
                proc.wait(timeout=1)

            except TimeoutExpired:
                pass

            else:
                processes.remove(proc)
                print(f"PID {proc.pid} finished")

    print(f"{len(processes)} processes running in parallel")
