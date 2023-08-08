""" Contains the simulation management functionality."""
# EdgeSimPy components
from edge_sim_py.components import *

# Python libraries
import os
import csv
import time
import msgpack
import networkx as nx
from statistics import mean, stdev


ALGORITHMS_THAT_DISPLAY_METRICS_DURING_SIMULATION = ["hermes", "hermes_v2", "lamp_v2", "lamp", "salus", "greedy_least_batch"]
SKIP_AGENT_MONITORING = True
VERBOSE_METRICS_TO_HIDE = [
    "wait_times",
    "pulling_times",
    "state_migration_times",
    "sizes_of_cached_layers",
    "sizes_of_uncached_layers",
    "migration_times",
]


def immobile(user: object):
    user.coordinates_trace.extend([user.base_station.coordinates for _ in range(5000)])


def simulator_run_model(self):
    self.executed_resource_management_algorithm = False
    # Defining initial parameters
    self.invalid_solution = False
    self.maintenance_batches = 0
    self.placement_snapshots = {}

    # Calls the method that collects monitoring data about the agents
    self.monitor()

    start_time = time.time()
    while self.running:
        # Calls the method that advances the simulation time
        self.step()

        # Calls the methods that collect monitoring data
        self.monitor()

        # Checks if the simulation should end according to the stop condition
        self.running = False if self.stopping_criterion(self) else True

    self.execution_time = time.time() - start_time
    collect_simulation_metrics(simulator=self)

    # Dumps simulation data to the disk to make sure no metrics are discarded
    self.dump_data_to_disk()
    del self.agent_metrics
    self.agent_metrics = {}


def simulator_step(self):
    """Advances the model's system in one step."""
    self.executed_resource_management_algorithm = False

    servers_being_updated = [server for server in EdgeServer.all() if server.status == "being_updated"]
    services_being_provisioned = [service for service in Service.all() if service.being_provisioned == True]

    if len(servers_being_updated) == 0 and len(services_being_provisioned) == 0:
        if self.maintenance_batches > 0:
            # Collecting the simulation metrics of the last maintenance batch
            collect_simulation_metrics(simulator=self)
            if self.resource_management_algorithm.__name__ in ALGORITHMS_THAT_DISPLAY_METRICS_DURING_SIMULATION:
                print(f"\n=== BATCH {self.maintenance_batches} METRICS ===")
                per_batch_metrics = self.model_metrics["per_batch"]
                for key, value in per_batch_metrics.items():
                    print(f"{key}: {value}")
                print("=========================================================================")

        # Updating the maintenance batch count
        self.maintenance_batches += 1
        self.resource_management_algorithm_parameters["current_maintenance_batch"] = self.maintenance_batches

        # Running resource management algorithm
        self.executed_resource_management_algorithm = True
        self.resource_management_algorithm(parameters=self.resource_management_algorithm_parameters)

    # Activating agents
    self.schedule.step()

    # Updating the "current_step" attribute inside the resource management algorithm's parameters
    self.resource_management_algorithm_parameters["current_step"] = self.schedule.steps + 1


def simulator_monitor(self):
    """Monitors a set of metrics from the model and its agents."""
    # Collecting model-level metrics
    self.collect()

    # Collecting agent-level metrics
    if SKIP_AGENT_MONITORING == False:
        for agent in self.schedule._agents.values():
            metrics = agent.collect()

            if metrics != {}:
                if f"{agent.__class__.__name__}" not in self.agent_metrics:
                    self.agent_metrics[f"{agent.__class__.__name__}"] = []

                metrics = {**{"Object": f"{agent}", "Time Step": self.schedule.steps}, **metrics}
                self.agent_metrics[f"{agent.__class__.__name__}"].append(metrics)


def simulator_dump_data_to_disk(self, clean_data_in_memory: bool = True) -> None:
    """Dumps simulation metrics to the disk.

    Args:
        clean_data_in_memory (bool, optional): Purges the list of metrics stored in the memory. Defaults to True.
    """
    if not os.path.exists(f"{self.logs_directory}/"):
        os.makedirs(f"{self.logs_directory}")

    if self.dump_interval != float("inf"):
        for key, value in self.agent_metrics.items():
            with open(f"{self.logs_directory}/{key}.msgpack", "wb") as output_file:
                output_file.write(msgpack.packb(value))

            if clean_data_in_memory:
                value = []

    # Consolidating collected simulation metrics
    metrics = {}
    for metric_name, metric_value in self.model_metrics["overall"].items():
        metrics[f"overall_{metric_name}"] = metric_value
    for metric_name, metric_value in self.model_metrics["per_batch"].items():
        metrics[f"per_batch_{metric_name}"] = metric_value

    # Saving general simulation metrics in a CSV file
    with open(f"{self.logs_directory}/general_simulation_metrics.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerows([metrics])


def collect_simulation_metrics(simulator: object):
    def get_avg(metric_metadata):
        return sum(metric_metadata) / len(metric_metadata) if len(metric_metadata) > 0 else 0

    #####################################
    #### DECLARING PER BATCH METRICS ####
    #####################################
    relocations_per_batch_and_state_size = {}

    relocations_per_batch_and_sla = {}
    delay_sla_violations_per_batch_and_sla = {}
    updated_hosts_that_dont_violate_the_applications_slas = []
    delay_sla_violations_per_batch = 0

    updated_cpu_capacity_per_batch = 0
    updated_ram_capacity_per_batch = 0
    updated_disk_capacity_per_batch = 0
    updated_cpu_capacity_available_per_batch = 0
    updated_ram_capacity_available_per_batch = 0
    updated_disk_capacity_available_per_batch = 0
    outdated_cpu_capacity_per_batch = 0
    outdated_ram_capacity_per_batch = 0
    outdated_disk_capacity_per_batch = 0
    overloaded_edge_servers_per_batch = 0
    services_hosted_by_updated_servers_per_batch = 0
    services_hosted_by_outdated_servers_per_batch = 0

    ###################################
    #### DECLARING OVERALL METRICS ####
    ###################################
    services_per_host = []

    wait_times = []
    pulling_times = []
    state_migration_times = []
    migration_times = []
    cache_hits = 0
    cache_misses = 0

    sizes_of_cached_layers = []
    sizes_of_uncached_layers = []

    sizes_of_layers_downloaded = []
    sizes_of_layers_on_download_queue = []
    sizes_of_layers_on_waiting_queue = []
    number_of_layers_downloaded = []
    number_of_layers_on_download_queue = []
    number_of_layers_on_waiting_queue = []

    longest_migration_duration = 0
    longest_migration_waiting_time = 0
    longest_migration_pulling_layers_time = 0
    longest_migration_migrating_service_state_time = 0
    longest_migration_cache_hits = 0
    longest_migration_cache_misses = 0
    longest_migration_sizes_of_cached_layers = 0
    longest_migration_sizes_of_uncached_layers = 0
    longest_migration_sizes_of_layers_downloaded = []
    longest_migration_sizes_of_layers_on_download_queue = []
    longest_migration_sizes_of_layers_on_waiting_queue = []
    longest_migration_number_of_layers_downloaded = 0
    longest_migration_number_of_layers_on_download_queue = 0
    longest_migration_number_of_layers_on_waiting_queue = 0

    ######################################
    #### COLLECTING PER BATCH METRICS ####
    ######################################
    for service in Service.all():
        if service.state not in relocations_per_batch_and_state_size:
            relocations_per_batch_and_state_size[service.state] = 0

    for user in User.all():
        for app in user.applications:
            user.set_communication_path(app=app)
            delay_sla = user.delay_slas[str(app.id)]
            delay = user._compute_delay(app=app, metric="latency")

            # Calculating the number of updated edge servers that don't violate the service SLA
            updated_hosts_that_dont_violate_the_application_sla = 0
            for edge_server in EdgeServer.all():
                if edge_server.status == "updated":
                    topology = edge_server.model.topology
                    path = nx.shortest_path(G=topology, source=user.base_station.network_switch, target=edge_server.base_station.network_switch, weight="delay")
                    edge_server_delay = user.base_station.wireless_delay + topology.calculate_path_delay(path=path)

                    if delay_sla >= edge_server_delay:
                        updated_hosts_that_dont_violate_the_application_sla += 1

            updated_hosts_that_dont_violate_the_applications_slas.append(updated_hosts_that_dont_violate_the_application_sla)

            # Calculating the number of delay SLA violations
            if delay > delay_sla:
                delay_sla_violations_per_batch += 1
                if delay_sla not in delay_sla_violations_per_batch_and_sla:
                    delay_sla_violations_per_batch_and_sla[delay_sla] = 0
                delay_sla_violations_per_batch_and_sla[delay_sla] += 1

            # Gathering service-related metrics
            for service in app.services:
                if service.server.status == "updated":
                    services_hosted_by_updated_servers_per_batch += 1
                else:
                    services_hosted_by_outdated_servers_per_batch += 1

                current_batch = simulator.maintenance_batches
                if len(service._Service__migrations) > 0 and service._Service__migrations[-1]["maintenance_batch"] == current_batch:
                    migration = service._Service__migrations[-1]
                    cache_hits += migration["cache_hits"]
                    cache_misses += migration["cache_misses"]

                    wait_times.append(migration["waiting_time"])
                    pulling_times.append(migration["pulling_layers_time"])
                    state_migration_times.append(migration["migrating_service_state_time"])
                    sizes_of_cached_layers.extend(migration["sizes_of_cached_layers"])
                    sizes_of_uncached_layers.extend(migration["sizes_of_uncached_layers"])

                    relocations_per_batch_and_state_size[service.state] += 1

                    if delay_sla not in relocations_per_batch_and_sla:
                        relocations_per_batch_and_sla[delay_sla] = 0
                    relocations_per_batch_and_sla[delay_sla] += 1

                    if migration["end"] != None:
                        migration_times.append(migration["end"] - migration["start"])

                        migration_duration = migration["end"] - migration["start"]
                        if longest_migration_duration < migration_duration:
                            longest_migration_duration = migration_duration
                            longest_migration_waiting_time = migration["waiting_time"]
                            longest_migration_pulling_layers_time = migration["pulling_layers_time"]
                            longest_migration_migrating_service_state_time = migration["migrating_service_state_time"]
                            longest_migration_cache_hits = migration["cache_hits"]
                            longest_migration_cache_misses = migration["cache_misses"]
                            longest_migration_sizes_of_cached_layers = migration["sizes_of_cached_layers"]
                            longest_migration_sizes_of_uncached_layers = migration["sizes_of_uncached_layers"]

                            longest_migration_sizes_of_layers_downloaded = migration["sizes_of_layers_downloaded"]
                            longest_migration_sizes_of_layers_on_download_queue = migration["sizes_of_layers_on_download_queue"]
                            longest_migration_sizes_of_layers_on_waiting_queue = migration["sizes_of_layers_on_waiting_queue"]
                            longest_migration_number_of_layers_downloaded = migration["number_of_layers_downloaded"]
                            longest_migration_number_of_layers_on_download_queue = migration["number_of_layers_on_download_queue"]
                            longest_migration_number_of_layers_on_waiting_queue = migration["number_of_layers_on_waiting_queue"]

    # Aggregating provisioning time metrics
    min_wait_times_per_batch = min(wait_times) if len(wait_times) > 0 else 0
    max_wait_times_per_batch = max(wait_times) if len(wait_times) > 0 else 0
    avg_wait_times_per_batch = sum(wait_times) / len(wait_times) if len(wait_times) > 0 else 0

    min_pulling_times_per_batch = min(pulling_times) if len(pulling_times) > 0 else 0
    max_pulling_times_per_batch = max(pulling_times) if len(pulling_times) > 0 else 0
    avg_pulling_times_per_batch = sum(pulling_times) / len(pulling_times) if len(pulling_times) > 0 else 0

    min_state_migration_times_per_batch = min(state_migration_times) if len(state_migration_times) > 0 else 0
    max_state_migration_times_per_batch = max(state_migration_times) if len(state_migration_times) > 0 else 0
    avg_state_migration_times_per_batch = sum(state_migration_times) / len(state_migration_times) if len(state_migration_times) > 0 else 0

    min_migration_times_per_batch = min(migration_times) if len(migration_times) > 0 else 0
    max_migration_times_per_batch = max(migration_times) if len(migration_times) > 0 else 0
    avg_migration_times_per_batch = sum(migration_times) / len(migration_times) if len(migration_times) > 0 else 0

    for edge_server in EdgeServer.all():
        cpu_is_overloaded = edge_server.cpu_demand > edge_server.cpu
        memory_is_overloaded = edge_server.memory_demand > edge_server.memory
        disk_is_overloaded = edge_server.disk_demand > edge_server.disk

        if cpu_is_overloaded or memory_is_overloaded or disk_is_overloaded:
            overloaded_edge_servers_per_batch += 1

        if edge_server.status == "updated":
            updated_cpu_capacity_per_batch += edge_server.cpu
            updated_ram_capacity_per_batch += edge_server.memory
            updated_disk_capacity_per_batch += edge_server.disk
            updated_cpu_capacity_available_per_batch += edge_server.cpu - edge_server.cpu_demand
            updated_ram_capacity_available_per_batch += edge_server.memory - edge_server.memory_demand
            updated_disk_capacity_available_per_batch += edge_server.disk - edge_server.disk_demand

        else:
            outdated_cpu_capacity_per_batch += edge_server.cpu
            outdated_ram_capacity_per_batch += edge_server.memory
            outdated_disk_capacity_per_batch += edge_server.disk

    # Consolidating per batch metrics
    if "per_batch" not in simulator.model_metrics:
        simulator.model_metrics = {
            "per_batch": {
                "maintenance_times": [],
                "delay_sla_violations": [],
                "relocations_per_state_size": [],
                "delay_sla_violations_per_sla": [],
                "relocations_per_sla": [],
                "updated_hosts_that_dont_violate_the_applications_slas": [],
                "avg_updated_hosts_that_dont_violate_the_applications_slas": [],
                "std_updated_hosts_that_dont_violate_the_applications_slas": [],
                "services_hosted_by_updated_servers": [],
                "services_hosted_by_outdated_servers": [],
                "longest_migrations_duration": [],
                "longest_migrations_waiting_time": [],
                "longest_migrations_pulling_layers_time": [],
                "longest_migrations_migrating_service_state_time": [],
                "longest_migrations_cache_hits": [],
                "longest_migrations_cache_misses": [],
                "longest_migrations_sizes_of_cached_layers": [],
                "longest_migrations_sizes_of_uncached_layers": [],
                "longest_migrations_sizes_of_layers_downloaded": [],
                "longest_migrations_sizes_of_layers_on_download_queue": [],
                "longest_migrations_sizes_of_layers_on_waiting_queue": [],
                "longest_migrations_number_of_layers_downloaded": [],
                "longest_migrations_number_of_layers_on_download_queue": [],
                "longest_migrations_number_of_layers_on_waiting_queue": [],
                "min_wait_times": [],
                "min_pulling_times": [],
                "min_state_migration_times": [],
                "min_migration_times": [],
                "avg_wait_times": [],
                "avg_pulling_times": [],
                "avg_state_migration_times": [],
                "avg_migration_times": [],
                "max_wait_times": [],
                "max_pulling_times": [],
                "max_state_migration_times": [],
                "max_migration_times": [],
                "overloaded_edge_servers": [],
                "updated_cpu_capacity": [],
                "updated_ram_capacity": [],
                "updated_disk_capacity": [],
                "updated_cpu_capacity_available": [],
                "updated_ram_capacity_available": [],
                "updated_disk_capacity_available": [],
                "outdated_cpu_capacity": [],
                "outdated_ram_capacity": [],
                "outdated_disk_capacity": [],
            }
        }

    simulator.model_metrics["per_batch"]["maintenance_times"].append(simulator.schedule.steps)

    simulator.model_metrics["per_batch"]["delay_sla_violations"].append(delay_sla_violations_per_batch)
    simulator.model_metrics["per_batch"]["delay_sla_violations_per_sla"].append(delay_sla_violations_per_batch_and_sla)
    simulator.model_metrics["per_batch"]["updated_cpu_capacity"].append(updated_cpu_capacity_per_batch)
    simulator.model_metrics["per_batch"]["updated_ram_capacity"].append(updated_ram_capacity_per_batch)
    simulator.model_metrics["per_batch"]["updated_disk_capacity"].append(updated_disk_capacity_per_batch)

    simulator.model_metrics["per_batch"]["relocations_per_state_size"].append(relocations_per_batch_and_state_size)

    simulator.model_metrics["per_batch"]["relocations_per_sla"].append(relocations_per_batch_and_sla)

    simulator.model_metrics["per_batch"]["updated_hosts_that_dont_violate_the_applications_slas"].append(updated_hosts_that_dont_violate_the_applications_slas)
    simulator.model_metrics["per_batch"]["avg_updated_hosts_that_dont_violate_the_applications_slas"].append(mean(updated_hosts_that_dont_violate_the_applications_slas))
    simulator.model_metrics["per_batch"]["std_updated_hosts_that_dont_violate_the_applications_slas"].append(stdev(updated_hosts_that_dont_violate_the_applications_slas))

    simulator.model_metrics["per_batch"]["updated_cpu_capacity_available"].append(updated_cpu_capacity_available_per_batch)
    simulator.model_metrics["per_batch"]["updated_ram_capacity_available"].append(updated_ram_capacity_available_per_batch)
    simulator.model_metrics["per_batch"]["updated_disk_capacity_available"].append(updated_disk_capacity_available_per_batch)

    simulator.model_metrics["per_batch"]["outdated_cpu_capacity"].append(outdated_cpu_capacity_per_batch)
    simulator.model_metrics["per_batch"]["outdated_ram_capacity"].append(outdated_ram_capacity_per_batch)
    simulator.model_metrics["per_batch"]["outdated_disk_capacity"].append(outdated_disk_capacity_per_batch)
    simulator.model_metrics["per_batch"]["services_hosted_by_updated_servers"].append(services_hosted_by_updated_servers_per_batch)
    simulator.model_metrics["per_batch"]["services_hosted_by_outdated_servers"].append(services_hosted_by_outdated_servers_per_batch)

    simulator.model_metrics["per_batch"]["longest_migrations_duration"].append(longest_migration_duration)
    simulator.model_metrics["per_batch"]["longest_migrations_waiting_time"].append(longest_migration_waiting_time)
    simulator.model_metrics["per_batch"]["longest_migrations_pulling_layers_time"].append(longest_migration_pulling_layers_time)
    simulator.model_metrics["per_batch"]["longest_migrations_migrating_service_state_time"].append(longest_migration_migrating_service_state_time)
    simulator.model_metrics["per_batch"]["longest_migrations_cache_hits"].append(longest_migration_cache_hits)
    simulator.model_metrics["per_batch"]["longest_migrations_cache_misses"].append(longest_migration_cache_misses)
    simulator.model_metrics["per_batch"]["longest_migrations_sizes_of_cached_layers"].append(longest_migration_sizes_of_cached_layers)
    simulator.model_metrics["per_batch"]["longest_migrations_sizes_of_uncached_layers"].append(longest_migration_sizes_of_uncached_layers)

    simulator.model_metrics["per_batch"]["longest_migrations_sizes_of_layers_downloaded"].append(longest_migration_sizes_of_layers_downloaded)
    simulator.model_metrics["per_batch"]["longest_migrations_sizes_of_layers_on_download_queue"].append(longest_migration_sizes_of_layers_on_download_queue)
    simulator.model_metrics["per_batch"]["longest_migrations_sizes_of_layers_on_waiting_queue"].append(longest_migration_sizes_of_layers_on_waiting_queue)
    simulator.model_metrics["per_batch"]["longest_migrations_number_of_layers_downloaded"].append(longest_migration_number_of_layers_downloaded)
    simulator.model_metrics["per_batch"]["longest_migrations_number_of_layers_on_download_queue"].append(longest_migration_number_of_layers_on_download_queue)
    simulator.model_metrics["per_batch"]["longest_migrations_number_of_layers_on_waiting_queue"].append(longest_migration_number_of_layers_on_waiting_queue)

    simulator.model_metrics["per_batch"]["min_wait_times"].append(min_wait_times_per_batch)
    simulator.model_metrics["per_batch"]["max_wait_times"].append(max_wait_times_per_batch)
    simulator.model_metrics["per_batch"]["avg_wait_times"].append(avg_wait_times_per_batch)
    simulator.model_metrics["per_batch"]["min_pulling_times"].append(min_pulling_times_per_batch)
    simulator.model_metrics["per_batch"]["max_pulling_times"].append(max_pulling_times_per_batch)
    simulator.model_metrics["per_batch"]["avg_pulling_times"].append(avg_pulling_times_per_batch)
    simulator.model_metrics["per_batch"]["min_state_migration_times"].append(min_state_migration_times_per_batch)
    simulator.model_metrics["per_batch"]["max_state_migration_times"].append(max_state_migration_times_per_batch)
    simulator.model_metrics["per_batch"]["avg_state_migration_times"].append(avg_state_migration_times_per_batch)
    simulator.model_metrics["per_batch"]["min_migration_times"].append(min_migration_times_per_batch)
    simulator.model_metrics["per_batch"]["max_migration_times"].append(max_migration_times_per_batch)
    simulator.model_metrics["per_batch"]["avg_migration_times"].append(avg_migration_times_per_batch)

    simulator.model_metrics["per_batch"]["overloaded_edge_servers"].append(overloaded_edge_servers_per_batch)

    ####################################
    #### COLLECTING OVERALL METRICS ####
    ####################################
    if simulator.running == False:
        algorithm = str(simulator.resource_management_algorithm.__name__)
        execution_time = simulator.execution_time if hasattr(simulator, "execution_time") else None
        invalid_solution = int(simulator.invalid_solution)
        overall_overloaded_edge_servers = sum(simulator.model_metrics["per_batch"]["overloaded_edge_servers"])
        penalty = invalid_solution * EdgeServer.count() + overall_overloaded_edge_servers
        maintenance_batches = simulator.maintenance_batches
        maintenance_time = simulator.schedule.steps
        placement_snapshots = simulator.placement_snapshots

        # NSGA-II parameters
        pop_size = simulator.resource_management_algorithm_parameters["pop_size"]
        cross_prob = simulator.resource_management_algorithm_parameters["cross_prob"]
        mut_prob = simulator.resource_management_algorithm_parameters["mut_prob"]
        n_gen = simulator.resource_management_algorithm_parameters["n_gen"]

        overall_delay_sla_violations = sum(simulator.model_metrics["per_batch"]["delay_sla_violations"])
        number_of_migrations = sum([len(s._Service__migrations) for s in Service.all()])

        entire_migration_metadata = []

        for edge_server in EdgeServer.all():
            services_per_host.append(len(edge_server.services))

        for service in Service.all():
            for migration in service._Service__migrations:
                cache_hits += migration["cache_hits"]
                cache_misses += migration["cache_misses"]

                wait_times.append(migration["waiting_time"])
                pulling_times.append(migration["pulling_layers_time"])
                state_migration_times.append(migration["migrating_service_state_time"])
                entire_migration_metadata.append(
                    {
                        "waiting": migration["waiting_time"],
                        "pulling": migration["pulling_layers_time"],
                        "state_migration": migration["migrating_service_state_time"],
                        "cache_hits": migration["cache_hits"],
                        "cache_misses": migration["cache_misses"],
                        "sizes_of_cached_layers": migration["sizes_of_cached_layers"],
                        "sizes_of_uncached_layers": migration["sizes_of_uncached_layers"],
                    }
                )

                sizes_of_cached_layers.extend(migration["sizes_of_cached_layers"])
                sizes_of_uncached_layers.extend(migration["sizes_of_uncached_layers"])

                sizes_of_layers_downloaded.extend(migration["sizes_of_layers_downloaded"])
                sizes_of_layers_on_download_queue.extend(migration["sizes_of_layers_on_download_queue"])
                sizes_of_layers_on_waiting_queue.extend(migration["sizes_of_layers_on_waiting_queue"])

                number_of_layers_downloaded.append(migration["number_of_layers_downloaded"])
                number_of_layers_on_download_queue.append(migration["number_of_layers_on_download_queue"])
                number_of_layers_on_waiting_queue.append(migration["number_of_layers_on_waiting_queue"])

                if migration["end"] != None:
                    migration_times.append(migration["end"] - migration["start"])

        # Aggregating provisioning time metrics
        if len(wait_times) > 0:
            overall_wait_times = sum(wait_times) if len(wait_times) > 0 else 0
            min_wait_times = min(wait_times) if len(wait_times) > 0 else 0
            max_wait_times = max(wait_times) if len(wait_times) > 0 else 0
            avg_wait_times = sum(wait_times) / len(wait_times) if len(wait_times) > 0 else 0
        if len(pulling_times) > 0:
            overall_pulling_times = sum(pulling_times) if len(pulling_times) > 0 else 0
            min_pulling_times = min(pulling_times) if len(pulling_times) > 0 else 0
            max_pulling_times = max(pulling_times) if len(pulling_times) > 0 else 0
            avg_pulling_times = sum(pulling_times) / len(pulling_times) if len(pulling_times) > 0 else 0
        if len(state_migration_times) > 0:
            overall_state_migration_times = sum(state_migration_times) if len(state_migration_times) > 0 else 0
            min_state_migration_times = min(state_migration_times) if len(state_migration_times) > 0 else 0
            max_state_migration_times = max(state_migration_times) if len(state_migration_times) > 0 else 0
            avg_state_migration_times = sum(state_migration_times) / len(state_migration_times) if len(state_migration_times) > 0 else 0

        overall_provisioning_time = overall_wait_times + overall_pulling_times + overall_state_migration_times

        # Consolidating overall metrics
        if "overall" not in simulator.model_metrics:
            simulator.model_metrics["overall"] = {}

        simulator.model_metrics["overall"]["algorithm"] = algorithm
        simulator.model_metrics["overall"]["execution_time"] = execution_time
        simulator.model_metrics["overall"]["invalid_solution"] = invalid_solution
        simulator.model_metrics["overall"]["penalty"] = penalty
        simulator.model_metrics["overall"]["overall_overloaded_edge_servers"] = overall_overloaded_edge_servers
        simulator.model_metrics["overall"]["placement_snapshots"] = placement_snapshots

        simulator.model_metrics["overall"]["pop_size"] = pop_size
        simulator.model_metrics["overall"]["cross_prob"] = cross_prob
        simulator.model_metrics["overall"]["mut_prob"] = mut_prob
        simulator.model_metrics["overall"]["n_gen"] = n_gen

        simulator.model_metrics["overall"]["entire_migration_metadata"] = entire_migration_metadata

        simulator.model_metrics["overall"]["services_per_host"] = services_per_host
        simulator.model_metrics["overall"]["avg_services_per_host"] = mean(services_per_host)
        simulator.model_metrics["overall"]["std_services_per_host"] = stdev(services_per_host)

        simulator.model_metrics["overall"]["maintenance_batches"] = maintenance_batches
        simulator.model_metrics["overall"]["maintenance_time"] = maintenance_time
        simulator.model_metrics["overall"]["cache_hits"] = cache_hits
        simulator.model_metrics["overall"]["cache_misses"] = cache_misses
        simulator.model_metrics["overall"]["wait_times"] = wait_times
        simulator.model_metrics["overall"]["pulling_times"] = pulling_times
        simulator.model_metrics["overall"]["state_migration_times"] = state_migration_times
        simulator.model_metrics["overall"]["sizes_of_cached_layers"] = sizes_of_cached_layers
        simulator.model_metrics["overall"]["sizes_of_uncached_layers"] = sizes_of_uncached_layers
        simulator.model_metrics["overall"]["migration_times"] = migration_times
        simulator.model_metrics["overall"]["overall_delay_sla_violations"] = overall_delay_sla_violations
        simulator.model_metrics["overall"]["number_of_migrations"] = number_of_migrations
        simulator.model_metrics["overall"]["overall_wait_times"] = overall_wait_times
        simulator.model_metrics["overall"]["min_wait_times"] = min_wait_times
        simulator.model_metrics["overall"]["max_wait_times"] = max_wait_times
        simulator.model_metrics["overall"]["avg_wait_times"] = avg_wait_times
        simulator.model_metrics["overall"]["overall_pulling_times"] = overall_pulling_times
        simulator.model_metrics["overall"]["min_pulling_times"] = min_pulling_times
        simulator.model_metrics["overall"]["max_pulling_times"] = max_pulling_times
        simulator.model_metrics["overall"]["avg_pulling_times"] = avg_pulling_times
        simulator.model_metrics["overall"]["overall_state_migration_times"] = overall_state_migration_times
        simulator.model_metrics["overall"]["min_state_migration_times"] = min_state_migration_times
        simulator.model_metrics["overall"]["max_state_migration_times"] = max_state_migration_times
        simulator.model_metrics["overall"]["avg_state_migration_times"] = avg_state_migration_times
        simulator.model_metrics["overall"]["overall_provisioning_time"] = overall_provisioning_time

        simulator.model_metrics["overall"]["all_number_of_layers_downloaded"] = number_of_layers_downloaded
        simulator.model_metrics["overall"]["all_number_of_layers_on_download_queue"] = number_of_layers_on_download_queue
        simulator.model_metrics["overall"]["all_number_of_layers_on_waiting_queue"] = number_of_layers_on_waiting_queue
        simulator.model_metrics["overall"]["all_sizes_of_layers_downloaded"] = sizes_of_layers_downloaded
        simulator.model_metrics["overall"]["all_sizes_of_layers_on_download_queue"] = sizes_of_layers_on_download_queue
        simulator.model_metrics["overall"]["all_sizes_of_layers_on_waiting_queue"] = sizes_of_layers_on_waiting_queue

        simulator.model_metrics["overall"]["sum_sizes_of_layers_downloaded"] = sum(sizes_of_layers_downloaded)
        simulator.model_metrics["overall"]["sum_sizes_of_layers_on_download_queue"] = sum(sizes_of_layers_on_download_queue)
        simulator.model_metrics["overall"]["sum_sizes_of_layers_on_waiting_queue"] = sum(sizes_of_layers_on_waiting_queue)
        simulator.model_metrics["overall"]["sum_number_of_layers_downloaded"] = sum(number_of_layers_downloaded)
        simulator.model_metrics["overall"]["sum_number_of_layers_on_download_queue"] = sum(number_of_layers_on_download_queue)
        simulator.model_metrics["overall"]["sum_number_of_layers_on_waiting_queue"] = sum(number_of_layers_on_waiting_queue)

        simulator.model_metrics["overall"]["avg_number_of_layers_downloaded"] = get_avg(number_of_layers_downloaded)
        simulator.model_metrics["overall"]["avg_number_of_layers_on_download_queue"] = get_avg(number_of_layers_on_download_queue)
        simulator.model_metrics["overall"]["avg_number_of_layers_on_waiting_queue"] = get_avg(number_of_layers_on_waiting_queue)
        simulator.model_metrics["overall"]["avg_sizes_of_layers_downloaded"] = get_avg(sizes_of_layers_downloaded)
        simulator.model_metrics["overall"]["avg_sizes_of_layers_on_download_queue"] = get_avg(sizes_of_layers_on_download_queue)
        simulator.model_metrics["overall"]["avg_sizes_of_layers_on_waiting_queue"] = get_avg(sizes_of_layers_on_waiting_queue)
