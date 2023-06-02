""" Contains the simulation management functionality."""
# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.service import Service
from edge_sim_py.components.user import User

# Python library
import os
import csv
import time
import msgpack
import tabulate


CONDENSED_METRICS = True


def simulator_run_model(self):
    if self.stopping_criterion == None:
        raise Exception("Please assign the 'stopping_criterion' attribute before starting the simulation.")

    if self.resource_management_algorithm == None:
        raise Exception("Please assign the 'resource_management_algorithm' attribute before starting the simulation.")

    # Calls the method that collects monitoring data about the agents
    self.monitor()

    self.invalid_solution = False
    self.maintenance_batches = 0
    self.resource_management_algorithm_execution_timestamps = []

    start_time = time.time()
    while self.running:
        # Calls the method that advances the simulation time
        self.step()

        # Calls the method that collects monitoring data about the agents
        self.monitor()

        # Checks if the simulation should end according to the stop condition
        self.running = False if self.stopping_criterion(self) else True

    self.execution_time = time.time() - start_time

    # Collects overall metrics after the simulation ended
    collect_overall_simulation_metrics(simulator=self, replace_metrics=True)

    migrations = []
    for service in Service.all():
        if len(service._Service__migrations) > 0:
            migration = service._Service__migrations[-1]
            total_time = migration["waiting_time"] + migration["pulling_layers_time"] + migration["migrating_service_state_time"]
            migrations.append(
                {
                    "maintenance_batch": migration["maintenance_batch"],
                    "waiting": migration["waiting_time"],
                    "pulling": migration["pulling_layers_time"],
                    "svstate": migration["migrating_service_state_time"],
                    "total": total_time,
                }
            )
    print("\n")
    print("=====================================")
    print("==== SERVICE MIGRATIONS OVERVIEW ====")
    print("=====================================")
    migrations = sorted(migrations, key=lambda migration: (migration["maintenance_batch"], migration["total"]))
    header = migrations[0].keys()
    rows = [x.values() for x in migrations]
    print(tabulate.tabulate(rows, header))

    # Dumps simulation data to the disk to make sure no metrics are discarded
    self.dump_data_to_disk()


def simulator_step(self):
    """Advances the model's system in one step."""
    self.executed_resource_management_algorithm = False

    servers_being_updated = [server for server in EdgeServer.all() if server.status == "being_updated"]
    services_being_provisioned = [service for service in Service.all() if service.being_provisioned == True]

    if len(servers_being_updated) == 0 and len(services_being_provisioned) == 0:
        # Updating the maintenance batch count
        self.maintenance_batches += 1
        self.resource_management_algorithm_parameters["current_maintenance_batch"] = self.maintenance_batches

        # Running resource management algorithm if it is not invalid
        parameters = self.resource_management_algorithm_parameters
        if self.resource_management_algorithm.__name__ == "nsgaii" and self.maintenance_batches > len(parameters["solution"]):
            self.invalid_solution = True
        else:
            self.executed_resource_management_algorithm = True
            self.resource_management_algorithm(parameters=self.resource_management_algorithm_parameters)

    # Activating agents
    self.schedule.step()

    # Updating the "current_step" attribute inside the resource management algorithm's parameters
    self.resource_management_algorithm_parameters["current_step"] = self.schedule.steps + 1

    # Collecting the batch metrics
    if self.executed_resource_management_algorithm == True:
        collect_simulation_batch_metrics(simulator=self)
        metrics = collect_overall_simulation_metrics(simulator=self)
        batch_count = self.maintenance_batches
        steps = self.schedule.steps + 1
        outdated_hosts = len([s for s in EdgeServer.all() if s.status == "outdated"])
        print(f"\n==== FINISHED MAINTENANCE BATCH {batch_count}. MAINTENANCE TIME: {steps}. OUTDATED SERVERS: {outdated_hosts}")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("")


def simulator_monitor(self):
    """Monitors a set of metrics from the model and its agents."""
    if not hasattr(self, "executing_nsgaii_runner") or (
        hasattr(self, "executing_nsgaii_runner") and self.executing_nsgaii_runner == False
    ):
        # Collecting model-level metrics
        self.collect()

        # Collecting agent-level metrics
        for agent in self.schedule._agents.values():
            metrics = agent.collect()

            if metrics != {}:
                if f"{agent.__class__.__name__}" not in self.agent_metrics:
                    self.agent_metrics[f"{agent.__class__.__name__}"] = []

                metrics = {**{"Object": f"{agent}", "Time Step": self.schedule.steps}, **metrics}
                self.agent_metrics[f"{agent.__class__.__name__}"].append(metrics)

        if self.schedule.steps == self.last_dump + self.dump_interval:
            self.dump_data_to_disk()
            self.last_dump = self.schedule.steps


def simulator_dump_data_to_disk(self, clean_data_in_memory: bool = True) -> None:
    """Dumps simulation metrics to the disk.

    Args:
        clean_data_in_memory (bool, optional): Purges the list of metrics stored in the memory. Defaults to True.
    """
    if self.dump_interval != float("inf"):
        if not hasattr(self, "executing_nsgaii_runner") or (
            hasattr(self, "executing_nsgaii_runner") and self.executing_nsgaii_runner == False
        ):
            if not os.path.exists(f"{self.logs_directory}/"):
                os.makedirs(f"{self.logs_directory}")

            for key, value in self.agent_metrics.items():
                with open(f"{self.logs_directory}/{key}.msgpack", "wb") as output_file:
                    output_file.write(msgpack.packb(value))

                if clean_data_in_memory:
                    value = []

            # Saving general simulation metrics in a CSV file
            with open(f"{self.logs_directory}/general_simulation_metrics.csv", "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.model_metrics.keys())
                writer.writeheader()
                writer.writerows([self.model_metrics])


def collect_simulation_batch_metrics(simulator: object):
    """
    => Number of delay SLA violations
    => CDF of server updated capacity throughout the simulation
    => CDF of services hosted by updated servers
    """
    delay_sla_violations_in_the_batch = 0

    updated_cpu_capacity_in_the_batch = 0
    updated_ram_capacity_in_the_batch = 0
    updated_disk_capacity_in_the_batch = 0
    outdated_cpu_capacity_in_the_batch = 0
    outdated_ram_capacity_in_the_batch = 0
    outdated_disk_capacity_in_the_batch = 0

    overloaded_edge_servers_in_the_batch = 0

    services_hosted_by_updated_servers_in_the_batch = 0
    services_hosted_by_outdated_servers_in_the_batch = 0

    # Number of delay SLA violations
    for user in User.all():
        for app in user.applications:
            user.set_communication_path(app=app)
            delay_sla = user.delay_slas[str(app.id)]
            delay = user._compute_delay(app=app, metric="latency")

            # Calculating the number of delay SLA violations
            if delay > delay_sla:
                delay_sla_violations_in_the_batch += 1

            # Gathering service-related metrics
            for service in app.services:
                # Server host metrics
                if service.server.status == "updated":
                    services_hosted_by_updated_servers_in_the_batch += 1
                else:
                    services_hosted_by_outdated_servers_in_the_batch += 1

    for edge_server in EdgeServer.all():
        if (
            edge_server.cpu_demand > edge_server.cpu
            or edge_server.memory_demand > edge_server.memory
            or edge_server.disk_demand > edge_server.disk
        ):
            overloaded_edge_servers_in_the_batch += 1

        if edge_server.status == "updated":
            updated_cpu_capacity_in_the_batch += edge_server.cpu
            updated_ram_capacity_in_the_batch += edge_server.memory
            updated_disk_capacity_in_the_batch += edge_server.disk
        else:
            outdated_cpu_capacity_in_the_batch += edge_server.cpu
            outdated_ram_capacity_in_the_batch += edge_server.memory
            outdated_disk_capacity_in_the_batch += edge_server.disk

    # Creating metric keys if they do not exist
    if "delay_sla_violations" not in simulator.model_metrics:
        simulator.model_metrics["delay_sla_violations"] = []
    if "updated_cpu_capacity" not in simulator.model_metrics:
        simulator.model_metrics["updated_cpu_capacity"] = []
    if "updated_ram_capacity" not in simulator.model_metrics:
        simulator.model_metrics["updated_ram_capacity"] = []
    if "updated_disk_capacity" not in simulator.model_metrics:
        simulator.model_metrics["updated_disk_capacity"] = []
    if "outdated_cpu_capacity" not in simulator.model_metrics:
        simulator.model_metrics["outdated_cpu_capacity"] = []
    if "outdated_ram_capacity" not in simulator.model_metrics:
        simulator.model_metrics["outdated_ram_capacity"] = []
    if "outdated_disk_capacity" not in simulator.model_metrics:
        simulator.model_metrics["outdated_disk_capacity"] = []
    if "services_hosted_by_updated_servers" not in simulator.model_metrics:
        simulator.model_metrics["services_hosted_by_updated_servers"] = []
    if "services_hosted_by_outdated_servers" not in simulator.model_metrics:
        simulator.model_metrics["services_hosted_by_outdated_servers"] = []
    if "overloaded_edge_servers" not in simulator.model_metrics:
        simulator.model_metrics["overloaded_edge_servers"] = []

    # Partial metrics (which are appended at every collection)
    simulator.model_metrics["delay_sla_violations"].append(delay_sla_violations_in_the_batch)
    simulator.model_metrics["updated_cpu_capacity"].append(updated_cpu_capacity_in_the_batch)
    simulator.model_metrics["updated_ram_capacity"].append(updated_ram_capacity_in_the_batch)
    simulator.model_metrics["updated_disk_capacity"].append(updated_disk_capacity_in_the_batch)
    simulator.model_metrics["outdated_cpu_capacity"].append(outdated_cpu_capacity_in_the_batch)
    simulator.model_metrics["outdated_ram_capacity"].append(outdated_ram_capacity_in_the_batch)
    simulator.model_metrics["outdated_disk_capacity"].append(outdated_disk_capacity_in_the_batch)
    simulator.model_metrics["services_hosted_by_updated_servers"].append(services_hosted_by_updated_servers_in_the_batch)
    simulator.model_metrics["services_hosted_by_outdated_servers"].append(services_hosted_by_outdated_servers_in_the_batch)
    simulator.model_metrics["overloaded_edge_servers"].append(overloaded_edge_servers_in_the_batch)


def collect_overall_simulation_metrics(simulator: object, replace_metrics: bool = False):
    """
    => Maintenance time
    => Provisioning time (depicting time values for: waiting, pulling, state migration)
    """
    wait_times = []
    pulling_times = []
    state_migration_times = []
    cache_hits = 0
    cache_misses = 0
    sizes_of_cached_layers = []
    sizes_of_uncached_layers = []
    migration_times = []
    time_steps_on_outdated_hosts = []
    time_steps_on_updated_hosts = []

    # Service migration metrics
    for service in Service.all():
        if hasattr(service, "time_steps_on_outdated_hosts"):
            time_steps_on_outdated_hosts.append(service.time_steps_on_outdated_hosts)
        else:
            time_steps_on_outdated_hosts.append(0)

        if hasattr(service, "time_steps_on_updated_hosts"):
            time_steps_on_updated_hosts.append(service.time_steps_on_updated_hosts)
        else:
            time_steps_on_updated_hosts.append(0)

        for migration in service._Service__migrations:
            wait_times.append(migration["waiting_time"])
            pulling_times.append(migration["pulling_layers_time"])
            state_migration_times.append(migration["migrating_service_state_time"])
            cache_hits += migration["cache_hits"]
            cache_misses += migration["cache_misses"]
            sizes_of_cached_layers.extend(migration["sizes_of_cached_layers"])
            sizes_of_uncached_layers.extend(migration["sizes_of_uncached_layers"])
            if migration["end"] != None:
                migration_times.append(migration["end"] - migration["start"])

    # Storing overall metrics
    algorithm = str(simulator.resource_management_algorithm.__name__)
    execution_time = simulator.execution_time if hasattr(simulator, "execution_time") else None
    invalid_solution = int(simulator.invalid_solution)
    overall_overloaded_edge_servers = sum(simulator.model_metrics["overloaded_edge_servers"])
    penalty = invalid_solution * EdgeServer.count() + overall_overloaded_edge_servers

    overall_delay_sla_violations = sum(simulator.model_metrics["delay_sla_violations"])

    number_of_migrations = sum([len(s._Service__migrations) for s in Service.all()])

    if len(wait_times) > 0:
        overall_wait_times = sum(wait_times)
        min_wait_times = min(wait_times)
        max_wait_times = max(wait_times)
        avg_wait_times = sum(wait_times) / len(wait_times)
    else:
        overall_wait_times = 0
        min_wait_times = 0
        max_wait_times = 0
        avg_wait_times = 0

    if len(pulling_times) > 0:
        overall_pulling_times = sum(pulling_times)
        min_pulling_times = min(pulling_times)
        max_pulling_times = max(pulling_times)
        avg_pulling_times = sum(pulling_times) / len(pulling_times)
    else:
        overall_pulling_times = 0
        min_pulling_times = 0
        max_pulling_times = 0
        avg_pulling_times = 0

    if len(state_migration_times) > 0:
        overall_state_migration_times = sum(state_migration_times)
        min_state_migration_times = min(state_migration_times)
        max_state_migration_times = max(state_migration_times)
        avg_state_migration_times = sum(state_migration_times) / len(state_migration_times)
    else:
        overall_state_migration_times = 0
        min_state_migration_times = 0
        max_state_migration_times = 0
        avg_state_migration_times = 0

    overall_provisioning_time = overall_wait_times + overall_pulling_times + overall_state_migration_times

    # Aggregating the NSGA-II parameters
    pop_size = simulator.resource_management_algorithm_parameters["pop_size"]
    n_gen = simulator.resource_management_algorithm_parameters["n_gen"]
    cross_prob = simulator.resource_management_algorithm_parameters["cross_prob"]
    mut_prob = simulator.resource_management_algorithm_parameters["mut_prob"]
    solution = simulator.resource_management_algorithm_parameters["solution"]

    # Formatting the metrics order
    if CONDENSED_METRICS:
        metrics = {
            "algorithm": algorithm,
            "pop_size": pop_size,
            "n_gen": n_gen,
            "cross_prob": cross_prob,
            "mut_prob": mut_prob,
            "solution": solution,
            "overloaded_edge_servers": overall_overloaded_edge_servers,
            "invalid_solution": invalid_solution,
            "maintenance_batches": simulator.maintenance_batches,
            "maintenance_time": simulator.schedule.steps,
            "overall_delay_sla_violations": overall_delay_sla_violations,
            "delay_sla_violations_per_batch": simulator.model_metrics["delay_sla_violations"],
            "number_of_migrations": number_of_migrations,
            "overall_wait_time": overall_wait_times,
            "max_wait_time": max_wait_times,
            "overall_pulling_time": overall_pulling_times,
            "max_pulling_time": max_pulling_times,
            "overall_state_migration_time": overall_state_migration_times,
            "max_state_migration_time": max_state_migration_times,
            "overall_provisioning_time": overall_provisioning_time,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "services_hosted_by_updated_servers": simulator.model_metrics["services_hosted_by_updated_servers"],
            "services_hosted_by_outdated_servers": simulator.model_metrics["services_hosted_by_outdated_servers"],
        }
    else:
        metrics = {
            "algorithm": algorithm,
            "pop_size": pop_size,
            "n_gen": n_gen,
            "cross_prob": cross_prob,
            "mut_prob": mut_prob,
            "solution": solution,
            "execution_time": execution_time,
            "invalid_solution": invalid_solution,
            "maintenance_batches": simulator.maintenance_batches,
            "overloaded_edge_servers": overall_overloaded_edge_servers,
            "penalty": penalty,
            "maintenance_time": simulator.schedule.steps,
            "overall_delay_sla_violations": overall_delay_sla_violations,
            "delay_sla_violations_per_batch": simulator.model_metrics["delay_sla_violations"],
            "number_of_migrations": number_of_migrations,
            "overall_wait_time": overall_wait_times,
            "min_wait_time": min_wait_times,
            "max_wait_time": max_wait_times,
            "avg_wait_time": avg_wait_times,
            "overall_pulling_time": overall_pulling_times,
            "min_pulling_time": min_pulling_times,
            "max_pulling_time": max_pulling_times,
            "avg_pulling_time": avg_pulling_times,
            "overall_state_migration_time": overall_state_migration_times,
            "min_state_migration_time": min_state_migration_times,
            "max_state_migration_time": max_state_migration_times,
            "avg_state_migration_time": avg_state_migration_times,
            "overall_provisioning_time": overall_provisioning_time,
            "wait_times": wait_times,
            "pulling_times": pulling_times,
            "state_migration_times": state_migration_times,
            "migration_times": migration_times,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "sizes_of_cached_layers": sizes_of_cached_layers,
            "sizes_of_uncached_layers": sizes_of_uncached_layers,
            "updated_cpu_capacity": simulator.model_metrics["updated_cpu_capacity"],
            "updated_ram_capacity": simulator.model_metrics["updated_ram_capacity"],
            "updated_disk_capacity": simulator.model_metrics["updated_disk_capacity"],
            "outdated_cpu_capacity": simulator.model_metrics["outdated_cpu_capacity"],
            "outdated_ram_capacity": simulator.model_metrics["outdated_ram_capacity"],
            "outdated_disk_capacity": simulator.model_metrics["outdated_disk_capacity"],
            "services_hosted_by_updated_servers": simulator.model_metrics["services_hosted_by_updated_servers"],
            "services_hosted_by_outdated_servers": simulator.model_metrics["services_hosted_by_outdated_servers"],
            "time_steps_on_outdated_hosts": time_steps_on_outdated_hosts,
            "time_steps_on_updated_hosts": time_steps_on_updated_hosts,
        }
    if replace_metrics:
        simulator.model_metrics = metrics

    return metrics
