# Importing Python libraries
from subprocess import Popen, DEVNULL, TimeoutExpired
import itertools
from time import sleep
import os


NUMBER_OF_PARALLEL_PROCESSES = max(1, os.cpu_count() - 2)


def run_simulation(dataset: str, algorithm: str, pop_size: int, n_gen: int, cross_prob: float, mut_prob: float):
    """Executes the simulation with the specified parameters.

    Args:
        dataset (str): Dataset being read.
        algorithm (str): Algorithm being executed.
        pop_size (int): Number of chromosomes in the NSGA-II's population.
        n_gen (int): Number of generations of the NSGA-II algorithm.
        cross_prob (float): NSGA-II's crossover probability.
        mut_prob (float): NSGA-II's mutation probability.
    """
    # Running the simulation based on the parameters and gathering its execution time
    cmd = f"python3 -B -m simulator --dataset {dataset} --algorithm {algorithm} --pop_size {pop_size} --n_gen {n_gen} --cross_prob {cross_prob} --mut_prob {mut_prob}"
    return Popen(cmd.split(" "), stdout=DEVNULL, stderr=DEVNULL)


# Parameters
datasets = ["datasets/dataset1.json"]
algorithms = ["nsgaii_v3"]

population_sizes = [150, 200]
number_of_generations = [1500, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
crossover_probabilities = [1]
mutation_probabilities = [0.1]

print(f"Datasets: {datasets}")
print(f"Algorithms: {algorithms}")
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
        population_sizes,
        number_of_generations,
        crossover_probabilities,
        mutation_probabilities,
    )
)

# Executing simulations and collecting results
processes = []

print(f"EXECUTING {len(combinations)} COMBINATIONS")
for i, parameters in enumerate(combinations, 1):
    # Parsing parameters
    dataset = parameters[0]
    algorithm = parameters[1]
    pop_size = parameters[2]
    n_gen = parameters[3]
    cross_prob = parameters[4]
    mut_prob = parameters[5]

    print(f"\t[Execution {i}]")
    print(f"\t\t[{algorithm}] dataset={dataset}. pop_size={pop_size}. n_gen={n_gen}. cross_prob={cross_prob}. mut_prob={mut_prob}")

    # Executing algorithm
    proc = run_simulation(
        dataset=dataset,
        algorithm=algorithm,
        pop_size=pop_size,
        n_gen=n_gen,
        cross_prob=cross_prob,
        mut_prob=mut_prob,
    )

    sleep(2)

    processes.append(proc)

    while len(processes) >= NUMBER_OF_PARALLEL_PROCESSES:
        for proc in processes:
            try:
                proc.wait(timeout=1)

            except TimeoutExpired:
                pass

            else:
                processes.remove(proc)
                print(f"PID {proc.pid} finished")

    print(f"{len(processes)} processes running in parallel")
