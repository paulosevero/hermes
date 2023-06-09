""" Contains a NSGA-II maintenance algorithm."""
# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer

# Pymoo components
from pymoo.util.display import Display
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

# Python libraries
import numpy as np

# Helper methods
from simulator.helper_methods import *


VERBOSE = False


def create_initial_population(pop_size, services_to_migrate):

    initial_population = []

    for _ in range(pop_size):
        solution = []
        edge_servers_free_cpu_capacity = [edge_server.cpu - edge_server.cpu_demand for edge_server in EdgeServer.all()]
        edge_servers_free_ram_capacity = [edge_server.memory - edge_server.memory_demand for edge_server in EdgeServer.all()]

        services = random.sample(services_to_migrate, len(services_to_migrate))

        for service in services:
            edge_servers = random.sample(EdgeServer.all(), EdgeServer.count())

            for edge_server in edge_servers:
                if service.server == edge_server:
                    solution.append(edge_server.id)
                    break
                else:
                    has_cpu_capacity = edge_servers_free_cpu_capacity[edge_server.id - 1] - service.cpu_demand > 0
                    has_ram_capacity = edge_servers_free_ram_capacity[edge_server.id - 1] - service.memory_demand > 0
                    if has_cpu_capacity and has_ram_capacity:
                        edge_servers_free_cpu_capacity[edge_server.id - 1] -= service.cpu_demand
                        edge_servers_free_ram_capacity[edge_server.id - 1] -= service.memory_demand

                        solution.append(edge_server.id)
                        break

        initial_population.append(solution)

    return initial_population


def max_min_fairness(capacity: int, demands: list) -> list:
    """Calculates network shares using the Max-Min Fairness algorithm [1].

    [1] Gebali, F. (2008). Scheduling Algorithms. In: Analysis of Computer and Communication
    Networks. Springer, Boston, MA. https://doi.org/10.1007/978-0-387-74437-7_12.

    Args:
        capacity (int): Network bandwidth to be shared.
        demands (list): List of demands (e.g.: list of demands of services that will be migrated).

    Returns:
        list: Fair network allocation scheme.
    """
    # Giving an equal slice of bandwidth to each item in the demands list
    allocated_bandwidth = [capacity / len(demands)] * len(demands)

    # Calculating leftover demand and gathering items with satisfied bandwidth
    fullfilled_items, leftover_bandwidth = get_overprovisioned_slices(demands=demands, allocated=allocated_bandwidth)

    while leftover_bandwidth > 0 and len(fullfilled_items) < len(demands):
        bandwidth_to_share = leftover_bandwidth / (len(demands) - len(fullfilled_items))

        for index, demand in enumerate(demands):
            if demand in fullfilled_items:
                # Removing overprovisioned bandwidth
                allocated_bandwidth[index] = demand
            else:
                # Giving a larger slice of bandwidth to items that are not fullfilled
                allocated_bandwidth[index] += bandwidth_to_share

        # Recalculating leftover demand and gathering items with satisfied bandwidth
        fullfilled_items, leftover_bandwidth = get_overprovisioned_slices(demands=demands, allocated=allocated_bandwidth)

    return allocated_bandwidth


def get_overprovisioned_slices(demands: list, allocated: list) -> list:
    """Calculates the leftover demand and finds items with satisfied bandwidth.
    Args:
        demands (list): List of demands (or the demand of services that will be migrated).
        allocated (list): Allocated demand for each service within the list.
    Returns:
        list, int: Flows that were overprovisioned and their leftover bandwidth, respectively.
    """
    overprovisioned_slices = []
    leftover_bandwidth = 0

    for i in range(len(demands)):
        if allocated[i] >= demands[i]:
            leftover_bandwidth += allocated[i] - demands[i]
            overprovisioned_slices.append(demands[i])

    return overprovisioned_slices, leftover_bandwidth


def find_closest_registry(server, layer):
    # Gathering the list of registries that have the layer
    registries_with_layer = []
    for registry in [reg for reg in ContainerRegistry.all() if reg.available]:
        # Checking if the registry is hosted on a valid host in the infrastructure and if it has the layer we need to pull
        if registry.server and any(layer.digest == l.digest for l in registry.server.container_layers):
            # Selecting a network path to be used to pull the layer from the registry
            path = nx.shortest_path(
                G=server.model.topology,
                source=registry.server.base_station.network_switch,
                target=server.base_station.network_switch,
            )

            registries_with_layer.append({"object": registry, "path": path})

    # Selecting the registry from which the layer will be pulled to the (target) edge server
    registries_with_layer = sorted(registries_with_layer, key=lambda r: len(r["path"]))
    registry = registries_with_layer[0]["object"]
    path = registries_with_layer[0]["path"]

    return {
        "registry": registry,
        "path": path,
    }


class Flow:
    instances = []

    def __init__(self, topology, started_at, source, target, path, data_to_transfer, metadata) -> object:
        self.id = len(Flow.instances) + 1
        self.topology = topology
        self.started_at = started_at
        self.ended_at = None
        self.status = None  # options: "active", "waiting", "finished"
        self.source = source
        self.target = target
        self.path = path
        self.bandwidth = {}
        self.data_to_transfer = data_to_transfer
        self.metadata = metadata  # e.g.: {"type": "layer", "object": layer, "container_registry": registry}

        for i in range(0, len(self.path) - 1):
            link = topology[self.path[i]][self.path[i + 1]]
            link["simulated_active_flows"].append(self)

        Flow.instances.append(self)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.id}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.id}"

    @classmethod
    def all(cls) -> list:
        return cls.instances

    @classmethod
    def count(cls) -> list:
        return len(cls.instances)

    def step(self, current_time_step):
        """Method that executes the events involving the object at each time step."""
        if self.status == "active":
            # Updating the flow progress according to the available bandwidth
            if not any([bw == None for bw in self.bandwidth.values()]):
                self.data_to_transfer -= min(self.bandwidth.values())

            if self.data_to_transfer <= 0:
                # Updating the completed flow's properties
                self.data_to_transfer = 0

                # Storing the current step as when the flow ended
                self.ended_at = current_time_step

                # Updating the flow status to "finished"
                self.status = "finished"

                # Releasing links used by the completed flow
                for i in range(0, len(self.path) - 1):
                    link = self.topology[self.path[i]][self.path[i + 1]]
                    link["simulated_active_flows"].remove(self)

                # When container layer flows finish: Adds the container layer to its target host
                if self.metadata["type"] == "layer":
                    # Removing the flow from its target host's download queue
                    self.target.simulated_download_queue.remove(self)


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
        min_outdated_capacity_being_used = int(np.min(algorithm.pop.get("F")[:, 0]))
        max_outdated_capacity_being_used = int(np.max(algorithm.pop.get("F")[:, 0]))

        min_time_spent_relocating_services = int(np.min(algorithm.pop.get("F")[:, 1]))
        max_time_spent_relocating_services = int(np.max(algorithm.pop.get("F")[:, 1]))

        min_delay_sla_violations = int(np.min(algorithm.pop.get("F")[:, 2]))
        max_delay_sla_violations = int(np.max(algorithm.pop.get("F")[:, 2]))

        # Aggregating penalties
        penalty = int(np.min(algorithm.pop.get("CV")[:, 0]))

        self.output.append("MIN_OUTD", min_outdated_capacity_being_used)
        self.output.append("MAX_OUTD", max_outdated_capacity_being_used)
        self.output.append("MIN_MIGR", min_time_spent_relocating_services)
        self.output.append("MAX_MIGR", max_time_spent_relocating_services)
        self.output.append("MIN_SLAV", min_delay_sla_violations)
        self.output.append("MAX_SLAV", max_delay_sla_violations)
        self.output.append("PENA", penalty)


class PlacementProblem(Problem):
    """Describes the application placement as an optimization problem."""

    def __init__(self, **kwargs):
        """Initializes the problem instance."""
        self.services_to_migrate = kwargs.get("services_to_migrate")

        super().__init__(
            n_var=len(self.services_to_migrate),
            n_obj=3,
            n_constr=1,
            xl=1,
            xu=EdgeServer.count(),
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

    def calculate_maximum_time_spent_relocating_services(self, solution):
        ########################################################################
        #### Creating a data structure to hold the layer provisioning times ####
        ########################################################################
        topology = EdgeServer.first().model.topology
        services_being_migrated = []
        edge_servers_download_list = {}
        target_servers = []
        for service_id, server_id in enumerate(solution):
            server = EdgeServer._instances[server_id - 1]
            service = self.services_to_migrate[service_id]

            # Checking if the solution is suggesting the service migration
            if service.server != server:
                target_servers.append(server)

                if not hasattr(service, "simulated_migrations"):
                    service.simulated_migrations = []
                service.simulated_migrations.append(
                    {
                        "origin": service.server,
                        "target": server,
                        "waiting_time": 0,
                        "pulling_layers_time": 0,
                        "migrating_service_state_time": 0,
                    }
                )
                services_being_migrated.append(service)

                # Gathering the service's container image
                service_image = ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)

                if str(server.id) not in edge_servers_download_list:
                    edge_servers_download_list[server.id] = []

                # Adding the service layers to the list of layers that must be downloaded in the target host
                for layer_digest in service_image.layers_digests:
                    if layer_digest not in edge_servers_download_list[server.id]:
                        edge_servers_download_list[server.id].append(layer_digest)

        if VERBOSE:
            print("==== SERVICES TO MIGRATE ====")
            for service in services_being_migrated:
                print(f"\t{service}")
            print("==== DOWNLOAD LIST =====")
            for server_id, download_list in edge_servers_download_list.items():
                server = EdgeServer.find_by_id(server_id)
                print(f"\t{server}. Download List: {download_list}")
            print("\n\n")

        ################################
        #### Creating network flows ####
        ################################
        for server_id, download_list in edge_servers_download_list.items():
            server = EdgeServer.find_by_id(server_id)
            server.simulated_download_queue = []
            server.simulated_waiting_queue = []

            for layer_digest in download_list:
                layer = ContainerLayer.find_by(attribute_name="digest", attribute_value=layer_digest)
                registry_metadata = find_closest_registry(server, layer)

                flow = Flow(
                    topology=topology,
                    started_at=0,
                    target=server,
                    source=registry_metadata["registry"],
                    path=registry_metadata["path"],
                    data_to_transfer=layer.size,
                    metadata={"type": "layer", "object": layer, "registry": registry_metadata["registry"]},
                )

                if VERBOSE:
                    print(
                        f"\tCreated {flow}. Path: {flow.path}. FROM-TO: {[server, registry_metadata['registry'].server]}. Size: {flow.data_to_transfer}. Metadata: {flow.metadata}"
                    )

                # If the target server hosts a registry with the layer, no download is necessary so the flow is immediately finished
                if len(flow.path) == 1:
                    flow.status = "finished"
                    flow.ended_at = 0
                else:
                    if len(server.simulated_download_queue) < server.max_concurrent_layer_downloads:
                        server.simulated_download_queue.append(flow)
                        flow.status = "active"
                    else:
                        server.simulated_waiting_queue.append(flow)
                        flow.status = "waiting"

            if VERBOSE:
                print(f"{server}. Download Queue: {server.simulated_download_queue}. Waiting Queue: {server.simulated_waiting_queue}")

        ###############################################
        #### Simulating the migration progressions ####
        ###############################################
        time_step = 0
        while sum([1 for flow in Flow.all() if flow.status == "finished"]) < Flow.count():
            time_step += 1

            if VERBOSE:
                print(f"\n=== TIME STEP {time_step} ===")

            # Gathering the links of used by flows that either started or finished or that have more bandwidth than needed
            links_to_recalculate_bandwidth = []
            for flow in Flow.all():
                if flow.status == "active":
                    flow_just_started = flow.started_at == time_step - 1
                    flow_just_ended = flow.ended_at == time_step - 1
                    flow_wasting_bandwidth = (
                        False if flow_just_started else any([flow.data_to_transfer < bw for bw in flow.bandwidth.values()])
                    )

                    if VERBOSE:
                        print(
                            f"\t{flow}. just started: {flow_just_started}. just ended: {flow_just_ended}. wasting bw: {flow_wasting_bandwidth}"
                        )

                    if flow_just_started or flow_just_ended or flow_wasting_bandwidth:
                        for i in range(0, len(flow.path) - 1):
                            # Gathering link nodes
                            link = (flow.path[i], flow.path[i + 1])

                            # Checking if the link is not already included in "links_to_recalculate_bandwidth"
                            if link not in links_to_recalculate_bandwidth:
                                links_to_recalculate_bandwidth.append(link)

            # Calculating the bandwidth shares for the active flows
            if VERBOSE:
                print("\t==== Links to Recalculate Bandwidth ====")

            for link_nodes in links_to_recalculate_bandwidth:
                link = topology[link_nodes[0]][link_nodes[1]]

                # Recalculating bandwidth shares for the flows as some of them have changed
                if VERBOSE:
                    print(f"\t\tLink: {link}. Simulated Active Flows: {link['simulated_active_flows']}")

                flow_demands = [f.data_to_transfer for f in link["simulated_active_flows"]]

                if sum(flow_demands) > 0:
                    bw_shares = max_min_fairness(capacity=link["bandwidth"], demands=flow_demands)

                    for index, affected_flow in enumerate(link["simulated_active_flows"]):
                        affected_flow.bandwidth[link["id"]] = bw_shares[index]

                        if VERBOSE:
                            print(f"\t\t\tFlow: {affected_flow}. BW: {bw_shares[index]}. New BW Values: {affected_flow.bandwidth}")

            if VERBOSE:
                print("\t==== Recalculated Flow Bandwidth ====")

            for flow in Flow.all():
                if flow.status == "active":
                    flow.step(current_time_step=time_step)

                    if VERBOSE:
                        print(f"\t\t{flow}. Status: {flow.status}. Bandwidth: {flow.bandwidth}. Left: {flow.data_to_transfer}")

            for edge_server in target_servers:
                waiting_queue = edge_server.simulated_waiting_queue
                download_queue = edge_server.simulated_download_queue
                while len(waiting_queue) > 0 and len(download_queue) < edge_server.max_concurrent_layer_downloads:
                    flow = edge_server.simulated_waiting_queue.pop(0)
                    edge_server.simulated_download_queue.append(flow)
                    flow.status = "active"

        ########################################################
        #### Calculating the time spent relocating services ####
        ########################################################
        maximum_time_spent_relocating_services = 0

        if VERBOSE:
            print("==== ALL FLOWS HAVE ENDED ====")

        for flow in Flow.all():
            flow_metadata = {
                "status": flow.status,
                "data_to_transfer": flow.data_to_transfer,
                "started_at": flow.started_at,
                "ended_at": flow.ended_at,
                "metadata": flow.metadata,
            }

            if VERBOSE:
                print(f"\t{flow}. {flow_metadata}")

            if flow.ended_at > maximum_time_spent_relocating_services:
                maximum_time_spent_relocating_services = flow.ended_at

        ###################################
        #### Removing all Flow objects ####
        ###################################
        Flow.instances = []

        for link in NetworkLink.all():
            link["simulated_active_flows"] = []

        for edge_server in target_servers:
            edge_server.simulated_waiting_queue = []
            edge_server.simulated_download_queue = []

        for service in self.services_to_migrate:
            service.simulated_migrations = []

        return maximum_time_spent_relocating_services

    def get_fitness_score_and_constraints(self, solution) -> tuple:
        """Calculates the fitness score and penalties of a solution based on the problem definition.

        Args:
            solution (list): Solution that solves the problem.

        Returns:
            tuple: Output of the evaluation function containing the fitness scores of the solution and its penalties.
        """
        ################################################################
        #### Calculating the maximum time taken relocating services ####
        ################################################################
        maximum_time_spent_relocating_services = self.calculate_maximum_time_spent_relocating_services(solution=solution)

        outdated_capacity_being_used = 0
        delay_sla_violations = 0
        edge_servers_free_cpu_capacity = [edge_server.cpu - edge_server.cpu_demand for edge_server in EdgeServer.all()]
        edge_servers_free_ram_capacity = [edge_server.memory - edge_server.memory_demand for edge_server in EdgeServer.all()]

        for service_id, server_id in enumerate(solution):
            service = self.services_to_migrate[service_id]
            server = EdgeServer._instances[server_id - 1]

            ######################################################
            #### Calculating the number of overloaded servers ####
            ######################################################
            if service.server != server:
                edge_servers_free_cpu_capacity[server.id - 1] -= service.cpu_demand
                edge_servers_free_ram_capacity[server.id - 1] -= service.memory_demand
                # edge_servers_free_cpu_capacity[service.server.id - 1] += service.cpu_demand
                # edge_servers_free_ram_capacity[service.server.id - 1] += service.memory_demand

            ####################################################################################
            #### Calculating the number of outdated servers drained with the given solution ####
            ####################################################################################
            if server.status == "outdated":
                outdated_capacity_being_used += get_normalized_demand(object=service)

            #######################################################################################
            #### Calculating the number of delay SLA violations incurred by the given solution ####
            #######################################################################################
            user = service.application.users[0]
            delay_sla = get_service_delay_sla(service=service)
            delay = get_delay(
                wireless_delay=user.base_station.wireless_delay,
                origin_switch=user.base_station.network_switch,
                target_switch=server.base_station.network_switch,
            )
            if delay > delay_sla:
                delay_sla_violations += 1

        if VERBOSE:
            print(f"SOLUTION: {solution}")
            print(f"\tOutdated hosts being used: {outdated_capacity_being_used}")
            print(f"\tMaximum time spent relocating services: {maximum_time_spent_relocating_services}")
            print(f"\tSLA violations: {delay_sla_violations}")
            print("")

        # Aggregating fitness values
        outdated_capacity_being_used = (
            outdated_capacity_being_used * 100 / sum([get_normalized_demand(s) for s in self.services_to_migrate])
        )
        fitness = (outdated_capacity_being_used, maximum_time_spent_relocating_services, delay_sla_violations)

        # Aggregating penalties
        servers_with_overloaded_cpu_capacity = sum([1 for item in edge_servers_free_cpu_capacity if item < 0])
        servers_with_overloaded_ram_capacity = sum([1 for item in edge_servers_free_ram_capacity if item < 0])
        penalty = max(servers_with_overloaded_cpu_capacity, servers_with_overloaded_ram_capacity)

        return (fitness, penalty)


def get_migration_plan(pop_size, cross_prob, mut_prob, n_gen) -> list:
    for link in NetworkLink.all():
        link["simulated_active_flows"] = []

    # Defining the initial population
    services_to_migrate = [service for service in Service.all() if service.server.status == "outdated"]

    initial_population = create_initial_population(pop_size=pop_size, services_to_migrate=services_to_migrate)

    # Defining genetic algorithm's attributes
    method = NSGA2(
        pop_size=pop_size,
        # sampling=get_sampling("int_random"),
        sampling=np.array(initial_population),
        crossover=get_crossover("int_ux", prob=cross_prob),
        mutation=get_mutation("int_pm", prob=mut_prob),
        eliminate_duplicates=False,
    )

    # Running the genetic algorithm
    problem = PlacementProblem(services_to_migrate=services_to_migrate)
    res = minimize(
        problem,
        method,
        termination=("n_gen", n_gen),
        seed=1,
        verbose=True,
        display=MyDisplay(),
    )

    # Parsing the NSGA-II output
    solutions = []
    for i in range(len(res.X)):
        solution = {
            "Migration Plan": res.X[i].tolist(),
            "Outdated Servers Used": res.F[i][0],
            "Max. Migration Time": res.F[i][1],
            "SLA Violations": res.F[i][2],
            "Penalty": res.CV[i][0].tolist(),
        }
        solutions.append(solution)

    # Gathering min and max values for each objective in the fitness function
    min_outdated_servers_used = min([solution["Outdated Servers Used"] for solution in solutions])
    max_outdated_servers_used = max([solution["Outdated Servers Used"] for solution in solutions])
    min_migration_time = min([solution["Max. Migration Time"] for solution in solutions])
    max_migration_time = max([solution["Max. Migration Time"] for solution in solutions])
    min_sla_violations = min([solution["SLA Violations"] for solution in solutions])
    max_sla_violations = max([solution["SLA Violations"] for solution in solutions])

    # Selecting the best maintenance plan found by the NSGA-II algorithm
    solutions = sorted(
        solutions,
        key=lambda s: (s["Penalty"], s["Outdated Servers Used"], s["SLA Violations"], s["Max. Migration Time"]),
    )

    print("=== SOLUTIONS FOUND:")
    for solution in solutions:
        print(f"\t{solution}")

    best_solution = solutions[0]["Migration Plan"]

    return best_solution


def nsgaii_v3(parameters: dict):
    # Patching outdated servers that were previously drained out (i.e., those that are not currently hosting any service)
    servers_to_patch = [
        server
        for server in EdgeServer.all()
        if server.status == "outdated" and len(server.services) == 0 and len(server.container_registries) == 0
    ]
    if len(servers_to_patch) > 0:
        for server in servers_to_patch:
            server.update()

    # Draining out outdated servers
    else:
        pop_size = parameters["pop_size"]
        cross_prob = parameters["cross_prob"]
        mut_prob = parameters["mut_prob"]
        n_gen = parameters["n_gen"]

        print(f"pop_size: {pop_size}")
        print(f"cross_prob: {cross_prob}")
        print(f"mut_prob: {mut_prob}")
        print(f"n_gen: {n_gen}")

        migration_plan = get_migration_plan(pop_size=pop_size, cross_prob=cross_prob, mut_prob=mut_prob, n_gen=n_gen)

        services = [service for service in Service.all() if service.server.status == "outdated"]

        for service_id, server_id in enumerate(migration_plan):
            service = services[service_id]
            server = EdgeServer._instances[server_id - 1]

            if service.server != server:
                service.provision(target_server=server)
