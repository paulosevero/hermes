# Hermes

This repository presents Hermes, a maintenance strategy designed to reduce the time needed to apply server updates in Edge Computing infrastructures, while keeping the maintenance's impact on application performance to a minimum.


## Motivation

When maintenance involves applying updates that require server reboots, affected applications must be relocated to alternative servers to ensure service continuity. During this process, deciding on target servers for applications that must be relocated involves considering several factors. Considering Edge Computing infrastructures, where applications have stringent latency requirements, it is necessary to assess the relocation's impact on the quality of service delivered to end-users. Also, it is vital to analyze how relocations contribute to maintenance progression, as prolonged relocations can delay server maintenance, which becomes even more critical in scenarios where maintenance is aimed at correcting security vulnerabilities, and patches must be applied as fast as possible.

Although existing maintenance strategies have exploited specific virtualization-based approaches during server updates at the edge, they present some limitations that motivate further research. First, the proposed strategies employ theoretical models assuming sequential relocations, overlooking the required coordination of concurrent relocations within the network, which distances them from practical implementations. Secondly, relocations occur according to the VM model. While there is a motivation for using VMs in specific scenarios, containers have taken the lead as the prime architecture for deploying applications at the edge.

Despite the similarities between VMs and containers, there are considerable differences between relocating VM-based and container-based applications. Since VM images are monolithic, VM-based applications are migrated directly from a source server to a target. On the contrary, relocating container-based applications involves pulling their container images from image repositories called container registries to the target servers and optionally transferring application information (e.g., user session data and runtime state) from the source server to the target server in cases where applications are stateful. Additionally, since co-hosted containers share common layers of their respective images, provisioning a containerized application on a server already possessing layers that constitute its container image is faster, as fewer data needs to be downloaded from container registries.

This research builds on the observation that existing maintenance strategies rely on conceptual models grounded in the VM paradigm. As such, they neglect the composition of container images stored on edge servers during the decision-making process for application relocation, thereby missing opportunities to reduce relocation times.

To fill this gap, we propose Hermes, a novel maintenance strategy that reduces maintenance time through efficient application relocations that consider the degree of container layer sharing among applications that require relocation and on the set of container layers downloaded from edge servers.


## Repository Structure

Within the repository, you'll find the following directories and files, logically grouping common assets used to simulate server updates at the edge. You'll see something like this:

```
├── pyproject.toml
├── results_overall.csv
├── results_sensitivity_analysis.csv
├── run_experiments.py
├── parse_results.py
├── container_image_analysis.py
├── create_dataset.py
├── datasets/
└── simulator/
    ├── helper_methods.py
    ├── simulator_extensions/
    └── strategies/
    └── ...
```

In the root directory, the `pyproject.toml` file organizes all project dependencies, including the minimum required version of the Python language. This file guides the execution of Poetry, a Python library that installs the dependencies securely, avoiding conflicts with external packages.

> Modifications made to the `pyproject.toml` file are automatically inserted into `poetry.lock` whenever Poetry is called.

The `results_overall.csv` and `results_sensitivity_analysis.csv` files contain the results from a comparison between Hermes and baseline strategies from the literature.

The `run_experiments.py` file makes it easy to execute the implemented strategies. For instance, with a few instructions, we can conduct a complete sensitivity analysis of the algorithms using different sets of parameters.

The `parse_results.py` file contains the code used to parse and compute the results obtained, for example, by executing the `run_experiments.py` script.

The `container_image_analysis.py` and `create_dataset.py` files comprise the source code used for creating datasets. Created dataset files are stored in the `datasets` directory.

Finally, the `simulator` directory includes the source code for the maintenance strategies, helper methods to expedite the execution of experiments, and simulator extensions that enable maintenance simulations.


## Installation Guide

This section contains information about the prerequisites of the system and about how to configure the environment to execute the simulations.

> Project dependencies are available for Linux, Windows, and macOS. However, we highly recommend using a recent version of a Debian-based Linux distribution. The installation below was validated on ***\*Ubuntu 22.04.2 LTS\****.

The first step needed to run the simulation is installing Python 3. We can do that by executing the following command:

```bash
sudo apt install python3 python3-distutils -y
```

We use a Python library called Poetry to manage project dependencies. In addition to selecting and downloading proper versions of project dependencies, Poetry automatically provisions virtual environments for the simulator, avoiding problems with external dependencies. On Linux and macOS, we can install Poetry with the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

The command above installs Poetry executable inside Poetry’s bin directory. On Unix, it is located at `$HOME/.local/bin`. We can get more information about Poetry installation from their [documentation page](https://python-poetry.org/docs/#installation).

Considering that we already downloaded the repository, we first need to install dependencies using Poetry. To do so, we access the command line in the root directory and type the following command:

```bash
poetry shell
```

The command we just ran creates a virtual Python environment that we will use to run the simulator. Notice that Poetry automatically sends us to the newly created virtual environment. Next, we need to install the project dependencies using the following command:

```bash
poetry install
```

After a few moments, Poetry will have installed all the dependencies needed by the simulator and we will be ready to run the experiments.

>  We employ an Edge Computing simulator [EdgeSimPy](https://edgesimpy.github.io/) to model the addressed edge server maintenance scenario and compare Hermes against maintenance strategies from the literature. You can know more about EdgeSimPy by reading its paper at: [https://doi.org/10.1016/j.future.2023.06.013](https://doi.org/10.1016/j.future.2023.06.013).


## Reproducing Experiments

Below are the commands executed to reproduce the experiments comparing Hermes and baseline maintenance strategies from the literature. Please notice that the commands below need to be run inside the virtual environment created by Poetry after the project's dependencies have been successfully installed.

[**Greedy Least Batch**](https://doi.org/10.1109/LCOMM.2014.2314671):

```bash
python3 -B -m simulator --dataset "datasets/dataset1.json" --algorithm "greedy_least_batch"
```

[**Salus**](https://tede2.pucrs.br/tede2/handle/tede/9522?locale=en):

```bash
python3 -B -m simulator --dataset "datasets/dataset1.json" --algorithm "salus"
```

[**Lamp**](https://doi.org/10.1109/LCOMM.2022.3150243):

```bash
python3 -B -m simulator --dataset "datasets/dataset1.json" --algorithm "lamp"
```

**Non-dominated Sorting Genetic Algorithm (NSGA-II)-Based Algorithm**:

Unlike the other maintenance strategies, NSGA-II has configurable parameters that modify the behavior of the genetic algorithm it uses to make application relocation decisions during maintenance. A description of the custom parameters adopted by this strategy is given below:

* `--pop_size`: determines how many individuals (solutions) will compose the population of the genetic algorithm.
* `--n_gen`: determines for how many generations the genetic algorithm will be executed.
* `--cross_prob`: determines the probability that individuals from the genetic algorithm's population are crossed to generate offsprings.
* `--mut_prob`: determines the probability that elements from the chromosome suffer a mutation.

During our experiments, we conducted a sensitivity analysis to determine the best parameter configuration for the NSGA-II algorithm (sensitivity analysis results can be checked in the `logs_sensitivity_analysis` directory). The best results were achieved when running NSGA-II with `pop_size=300`, `n_gen=1700`, `cross_prob=1`, and `mut_prob=0.1`.

We can run the NSGA-II algorithm with the following command:

```bash
python3 -B -m simulator --dataset "datasets/dataset1.json" --algorithm "nsgaii" --pop_size 300 --n_gen 1700 --cross_prob 1 --mut_prob 0.1
```

**Hermes**:

```bash
python3 -B -m simulator --dataset "datasets/dataset1.json" --algorithm "hermes"
```


## Manuscript

TBD.
