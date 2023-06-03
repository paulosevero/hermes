""" Contains the simulation management functionality."""
# EdgeSimPy components
from edge_sim_py.component_manager import ComponentManager
from edge_sim_py.components import *

# Python libraries
import os
import csv
import time
import json
import msgpack


CONDENSED_METRICS = True


def immobile(user: object):
    user.coordinates_trace.extend([user.base_station.coordinates for _ in range(5000)])


def simulator_initialize(self, input_file: str) -> None:
    """Sets up the initial values for state variables, which includes, e.g., loading components from a dataset file.

    Args:
        input_file (str): Dataset file (URL for external JSON file, path for local JSON file, Python dictionary).
    """
    # Resetting the list of instances of EdgeSimPy's component classes
    for component_class in ComponentManager.__subclasses__():
        if component_class.__name__ != "Simulator":
            component_class._object_count = 0
            component_class._instances = []

    # Declaring an empty variable that will receive the dataset metadata (if user passes valid information)
    data = None

    with open(input_file, "r", encoding="UTF-8") as read_file:
        data = json.load(read_file)

    # Raising exception if the dataset could not be loaded based on the specified arguments
    if type(data) is not dict:
        raise TypeError("EdgeSimPy could not load the dataset based on the specified arguments.")

    # Creating simulator components based on the specified input data
    missing_keys = [key for key in data.keys() if key not in globals()]
    if len(missing_keys) > 0:
        raise Exception(f"\n\nCould not find component classes named: {missing_keys}. Please check your input file.\n\n")

    # Creating a list that will store all the relationships among components
    components = []

    # Creating the topology object and storing a reference to it as an attribute of the Simulator instance
    topology = self.initialize_agent(agent=Topology())
    self.topology = topology

    # Creating simulator components
    for key in data.keys():
        if key != "Simulator" and key != "Topology":
            for object_metadata in data[key]:
                new_component = globals()[key]._from_dict(dictionary=object_metadata["attributes"])
                new_component.attributes = object_metadata["attributes"]
                new_component.relationships = object_metadata["relationships"]

                if hasattr(new_component, "model") and hasattr(new_component, "unique_id"):
                    self.initialize_agent(agent=new_component)

                components.append(new_component)

    # Defining relationships between components
    ComponentManager.components = components
    ComponentManager.topology = topology
    for component in components:
        for key, value in component.relationships.items():
            # Defining attributes referencing callables (i.e., functions and methods)
            if type(value) == str and value in globals():
                setattr(component, f"{key}", globals()[value])

            # Defining attributes referencing lists of components (e.g., lists of edge servers, users, etc.)
            elif type(value) == list:
                attribute_values = []
                for item in value:
                    obj = (
                        globals()[item["class"]].find_by_id(item["id"])
                        if type(item) == dict and "class" in item and item["class"] in globals()
                        else None
                    )

                    if obj == None:
                        raise Exception(f"List relationship '{key}' of component {component} has an invalid item: {item}.")

                    attribute_values.append(obj)

                setattr(component, f"{key}", attribute_values)

            # Defining attributes that reference a single component (e.g., an edge server, an user, etc.)
            elif type(value) == dict and "class" in value and "id" in value:
                obj = (
                    globals()[value["class"]].find_by_id(value["id"])
                    if type(value) == dict and "class" in value and value["class"] in globals()
                    else None
                )

                if obj == None:
                    raise Exception(f"Relationship '{key}' of component {component} references an invalid object: {value}.")

                setattr(component, f"{key}", obj)

            # Defining attributes that reference a a dictionary of components (e.g., {"1": {"class": "A", "id": 1}} )
            elif type(value) == dict and all(type(entry) == dict and "class" in entry and "id" in entry for entry in value.values()):
                attribute = {}
                for k, v in value.items():
                    obj = globals()[v["class"]].find_by_id(v["id"]) if "class" in v and v["class"] in globals() else None
                    if obj == None:
                        raise Exception(f"Relationship '{key}' of component {component} references an invalid object: {value}.")
                    attribute[k] = obj

                setattr(component, f"{key}", attribute)

            # Defining "None" attributes
            elif value == None:
                setattr(component, f"{key}", None)

            else:
                raise Exception(f"Couldn't add the relationship {key} with value {value}. Please check your dataset.")

    # Filling the network topology
    for link in NetworkLink.all():
        # Adding the nodes connected by the link to the topology
        topology.add_node(link.nodes[0])
        topology.add_node(link.nodes[1])

        # Replacing NetworkX's default link dictionary with the NetworkLink object
        topology.add_edge(link.nodes[0], link.nodes[1])
        topology._adj[link.nodes[0]][link.nodes[1]] = link
        topology._adj[link.nodes[1]][link.nodes[0]] = link


def simulator_run_model(self):
    self.executed_resource_management_algorithm = False
    # Defining initial parameters
    self.invalid_solution = False
    self.maintenance_batches = 0

    # Calls the method that collects monitoring data about the agents
    self.monitor()

    start_time = time.time()
    while self.running:
        # Calls the method that advances the simulation time
        self.step()

        # Calls the methods that collect monitoring data
        self.monitor()
        if self.executed_resource_management_algorithm:
            collect_simulation_metrics(simulator=self)

        # Checks if the simulation should end according to the stop condition
        self.running = False if self.stopping_criterion(self) else True

    self.execution_time = time.time() - start_time
    collect_simulation_metrics(simulator=self)

    # Dumps simulation data to the disk to make sure no metrics are discarded
    if self.executing_nsgaii_runner == False:
        self.dump_data_to_disk()
    del self.agent_metrics
    self.agent_metrics = {}


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


def simulator_monitor(self):
    """Monitors a set of metrics from the model and its agents."""
    if self.executing_nsgaii_runner == False:
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


def simulator_dump_data_to_disk(self, clean_data_in_memory: bool = True) -> None:
    """Dumps simulation metrics to the disk.

    Args:
        clean_data_in_memory (bool, optional): Purges the list of metrics stored in the memory. Defaults to True.
    """
    if self.executing_nsgaii_runner == False:
        if not os.path.exists(f"{self.logs_directory}/"):
            os.makedirs(f"{self.logs_directory}")

        if self.dump_interval != float("inf"):
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


def collect_simulation_metrics(simulator: object):
    if simulator.executing_nsgaii_runner:
        delay_sla_violations_per_batch = 0
        overloaded_edge_servers_per_batch = 0

        ######################################
        #### COLLECTING PER BATCH METRICS ####
        ######################################
        # Calculating the number of delay SLA violations per batch
        for user in User.all():
            for app in user.applications:
                user.set_communication_path(app=app)
                delay_sla = user.delay_slas[str(app.id)]
                delay = user._compute_delay(app=app, metric="latency")

                if delay > delay_sla:
                    delay_sla_violations_per_batch += 1

        # Calculating the number of overloaded edge servers per batch
        for edge_server in EdgeServer.all():
            cpu_is_overloaded = edge_server.cpu_demand > edge_server.cpu
            memory_is_overloaded = edge_server.memory_demand > edge_server.memory
            disk_is_overloaded = edge_server.disk_demand > edge_server.disk

            if cpu_is_overloaded or memory_is_overloaded or disk_is_overloaded:
                overloaded_edge_servers_per_batch += 1

        #############################
        #### AGGREGATING METRICS ####
        #############################
        if "per_batch" not in simulator.model_metrics:
            simulator.model_metrics = {
                "per_batch": {
                    "delay_sla_violations_per_batch": [],
                    "overloaded_edge_servers_per_batch": [],
                }
            }

        if simulator.running == True:
            simulator.model_metrics["per_batch"]["delay_sla_violations_per_batch"].append(delay_sla_violations_per_batch)
            simulator.model_metrics["per_batch"]["overloaded_edge_servers_per_batch"].append(overloaded_edge_servers_per_batch)
        else:
            simulator.model_metrics["per_batch"]["delay_sla_violations_per_batch"][-1] = delay_sla_violations_per_batch
            simulator.model_metrics["per_batch"]["overloaded_edge_servers_per_batch"][-1] = overloaded_edge_servers_per_batch

        algorithm = str(simulator.resource_management_algorithm.__name__)
        execution_time = simulator.execution_time if hasattr(simulator, "execution_time") else None
        invalid_solution = int(simulator.invalid_solution)
        overall_overloaded_edge_servers = sum(simulator.model_metrics["per_batch"]["overloaded_edge_servers_per_batch"])
        penalty = invalid_solution * EdgeServer.count() + overall_overloaded_edge_servers
        maintenance_batches = simulator.maintenance_batches
        maintenance_time = simulator.schedule.steps
        overall_delay_sla_violations = sum(simulator.model_metrics["per_batch"]["delay_sla_violations_per_batch"])

        if "overall" not in simulator.model_metrics:
            simulator.model_metrics["overall"] = {}

        simulator.model_metrics["overall"]["algorithm"] = algorithm
        simulator.model_metrics["overall"]["invalid_solution"] = invalid_solution
        simulator.model_metrics["overall"]["penalty"] = penalty
        simulator.model_metrics["overall"]["overall_overloaded_edge_servers"] = overall_overloaded_edge_servers
        simulator.model_metrics["overall"]["execution_time"] = execution_time
        simulator.model_metrics["overall"]["maintenance_batches"] = maintenance_batches
        simulator.model_metrics["overall"]["maintenance_time"] = maintenance_time
        simulator.model_metrics["overall"]["overall_delay_sla_violations"] = overall_delay_sla_violations

    else:
        #####################################
        #### DECLARING PER BATCH METRICS ####
        #####################################
        delay_sla_violations_per_batch = 0
        updated_cpu_capacity_per_batch = 0
        updated_ram_capacity_per_batch = 0
        updated_disk_capacity_per_batch = 0
        outdated_cpu_capacity_per_batch = 0
        outdated_ram_capacity_per_batch = 0
        outdated_disk_capacity_per_batch = 0
        overloaded_edge_servers_per_batch = 0
        services_hosted_by_updated_servers_per_batch = 0
        services_hosted_by_outdated_servers_per_batch = 0

        ###################################
        #### DECLARING OVERALL METRICS ####
        ###################################
        wait_times = []
        pulling_times = []
        state_migration_times = []
        cache_hits = 0
        cache_misses = 0
        sizes_of_cached_layers = []
        sizes_of_uncached_layers = []
        migration_times = []

        ######################################
        #### COLLECTING PER BATCH METRICS ####
        ######################################
        for user in User.all():
            for app in user.applications:
                user.set_communication_path(app=app)
                delay_sla = user.delay_slas[str(app.id)]
                delay = user._compute_delay(app=app, metric="latency")

                # Calculating the number of delay SLA violations
                if delay > delay_sla:
                    delay_sla_violations_per_batch += 1

                # Gathering service-related metrics
                for service in app.services:
                    if service.server.status == "updated":
                        services_hosted_by_updated_servers_per_batch += 1
                    else:
                        services_hosted_by_outdated_servers_per_batch += 1

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
            else:
                outdated_cpu_capacity_per_batch += edge_server.cpu
                outdated_ram_capacity_per_batch += edge_server.memory
                outdated_disk_capacity_per_batch += edge_server.disk

        # Consolidating per batch metrics
        if "per_batch" not in simulator.model_metrics:
            simulator.model_metrics = {
                "per_batch": {
                    "delay_sla_violations_per_batch": [],
                    "services_hosted_by_updated_servers_per_batch": [],
                    "services_hosted_by_outdated_servers_per_batch": [],
                    "overloaded_edge_servers_per_batch": [],
                    "updated_cpu_capacity_per_batch": [],
                    "updated_ram_capacity_per_batch": [],
                    "updated_disk_capacity_per_batch": [],
                    "outdated_cpu_capacity_per_batch": [],
                    "outdated_ram_capacity_per_batch": [],
                    "outdated_disk_capacity_per_batch": [],
                }
            }

        if simulator.running == True:
            simulator.model_metrics["per_batch"]["delay_sla_violations_per_batch"].append(delay_sla_violations_per_batch)
            simulator.model_metrics["per_batch"]["updated_cpu_capacity_per_batch"].append(updated_cpu_capacity_per_batch)
            simulator.model_metrics["per_batch"]["updated_ram_capacity_per_batch"].append(updated_ram_capacity_per_batch)
            simulator.model_metrics["per_batch"]["updated_disk_capacity_per_batch"].append(updated_disk_capacity_per_batch)
            simulator.model_metrics["per_batch"]["outdated_cpu_capacity_per_batch"].append(outdated_cpu_capacity_per_batch)
            simulator.model_metrics["per_batch"]["outdated_ram_capacity_per_batch"].append(outdated_ram_capacity_per_batch)
            simulator.model_metrics["per_batch"]["outdated_disk_capacity_per_batch"].append(outdated_disk_capacity_per_batch)
            simulator.model_metrics["per_batch"]["services_hosted_by_updated_servers_per_batch"].append(
                services_hosted_by_updated_servers_per_batch
            )
            simulator.model_metrics["per_batch"]["services_hosted_by_outdated_servers_per_batch"].append(
                services_hosted_by_outdated_servers_per_batch
            )
            simulator.model_metrics["per_batch"]["overloaded_edge_servers_per_batch"].append(overloaded_edge_servers_per_batch)
        else:
            simulator.model_metrics["per_batch"]["delay_sla_violations_per_batch"][-1] = delay_sla_violations_per_batch
            simulator.model_metrics["per_batch"]["updated_cpu_capacity_per_batch"][-1] = updated_cpu_capacity_per_batch
            simulator.model_metrics["per_batch"]["updated_ram_capacity_per_batch"][-1] = updated_ram_capacity_per_batch
            simulator.model_metrics["per_batch"]["updated_disk_capacity_per_batch"][-1] = updated_disk_capacity_per_batch
            simulator.model_metrics["per_batch"]["outdated_cpu_capacity_per_batch"][-1] = outdated_cpu_capacity_per_batch
            simulator.model_metrics["per_batch"]["outdated_ram_capacity_per_batch"][-1] = outdated_ram_capacity_per_batch
            simulator.model_metrics["per_batch"]["outdated_disk_capacity_per_batch"][-1] = outdated_disk_capacity_per_batch
            simulator.model_metrics["per_batch"]["services_hosted_by_updated_servers_per_batch"][
                -1
            ] = services_hosted_by_updated_servers_per_batch
            simulator.model_metrics["per_batch"]["services_hosted_by_outdated_servers_per_batch"][
                -1
            ] = services_hosted_by_outdated_servers_per_batch
            simulator.model_metrics["per_batch"]["overloaded_edge_servers_per_batch"][-1] = overloaded_edge_servers_per_batch

        ####################################
        #### COLLECTING OVERALL METRICS ####
        ####################################
        if simulator.running == False:
            algorithm = str(simulator.resource_management_algorithm.__name__)
            execution_time = simulator.execution_time if hasattr(simulator, "execution_time") else None
            invalid_solution = int(simulator.invalid_solution)
            overall_overloaded_edge_servers = sum(simulator.model_metrics["per_batch"]["overloaded_edge_servers_per_batch"])
            penalty = invalid_solution * EdgeServer.count() + overall_overloaded_edge_servers
            maintenance_batches = simulator.maintenance_batches
            maintenance_time = simulator.schedule.steps

            overall_delay_sla_violations = sum(simulator.model_metrics["per_batch"]["delay_sla_violations_per_batch"])
            number_of_migrations = sum([len(s._Service__migrations) for s in Service.all()])

            for service in Service.all():
                for migration in service._Service__migrations:
                    cache_hits += migration["cache_hits"]
                    cache_misses += migration["cache_misses"]

                    wait_times.append(migration["waiting_time"])
                    pulling_times.append(migration["pulling_layers_time"])
                    state_migration_times.append(migration["migrating_service_state_time"])
                    sizes_of_cached_layers.extend(migration["sizes_of_cached_layers"])
                    sizes_of_uncached_layers.extend(migration["sizes_of_uncached_layers"])

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
                avg_state_migration_times = (
                    sum(state_migration_times) / len(state_migration_times) if len(state_migration_times) > 0 else 0
                )

            overall_provisioning_time = overall_wait_times + overall_pulling_times + overall_state_migration_times

            # Consolidating overall metrics
            if "overall" not in simulator.model_metrics:
                simulator.model_metrics["overall"] = {}

            simulator.model_metrics["overall"]["algorithm"] = algorithm
            simulator.model_metrics["overall"]["invalid_solution"] = invalid_solution
            simulator.model_metrics["overall"]["penalty"] = penalty
            simulator.model_metrics["overall"]["overall_overloaded_edge_servers"] = overall_overloaded_edge_servers
            simulator.model_metrics["overall"]["execution_time"] = execution_time
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
