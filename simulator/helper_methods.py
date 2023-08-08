# EdgeSimPy components
from edge_sim_py.simulator import Simulator
from edge_sim_py.components import *


# EdgeSimPy extensions
from simulator.simulator_extensions import *

# Python libraries
import matplotlib.pyplot as plt
import networkx as nx

# Python libraries
import random


def load_edgesimpy_extensions():
    # Loading EdgeSimPy extensions
    # Simulator.initialize = simulator_initialize
    Simulator.step = simulator_step
    Simulator.run_model = simulator_run_model
    Simulator.monitor = simulator_monitor
    Simulator.dump_data_to_disk = simulator_dump_data_to_disk
    User.set_communication_path = user_set_communication_path
    EdgeServer._to_dict = edge_server_to_dict
    EdgeServer.collect = edge_server_collect
    EdgeServer.update = edge_server_update
    EdgeServer.step = edge_server_step
    NetworkFlow.step = network_flow_step
    Service.provision = service_provision
    Service.collect = service_collect
    Service.step = service_step


def maintenance_stopping_criterion(model):
    all_servers_were_updated = EdgeServer.count() == sum([1 for server in EdgeServer.all() if server.status == "updated"])
    solution_is_invalid = model.invalid_solution == True

    return all_servers_were_updated or solution_is_invalid


def display_topology(topology: object, output_filename: str = "topology"):
    # Customizing visual representation of topology
    positions = {}
    node_labels = {}
    link_labels = {}
    colors = []
    sizes = []

    for node in topology.nodes():
        positions[node] = node.coordinates
        node_labels[node] = node.id
        node_size = 400 if any(user.coordinates == node.coordinates for user in User.all()) else 100
        sizes.append(node_size)

        if len(node.base_station.edge_servers) > 0 and len(node.base_station.edge_servers[0].container_registries) > 0:
            colors.append("blue")
        elif len(node.base_station.edge_servers) > 0 and "Server 1" in node.base_station.edge_servers[0].model_name:
            colors.append("green")
        elif len(node.base_station.edge_servers) > 0 and "Server 2" in node.base_station.edge_servers[0].model_name:
            colors.append("orange")
        elif len(node.base_station.edge_servers) > 0 and "Server 3" in node.base_station.edge_servers[0].model_name:
            colors.append("red")
        elif len(node.base_station.edge_servers) > 0 and "Server 4" in node.base_station.edge_servers[0].model_name:
            colors.append("pink")
        else:
            colors.append("gray")

    for link_switches in topology.edges():
        link = topology[link_switches[0]][link_switches[1]]
        link_labels[link_switches] = f"L{link.id}"

    # Configuring drawing scheme
    nx.draw(
        topology,
        pos=positions,
        node_color=colors,
        node_size=sizes,
        labels=node_labels,
        font_size=6,
        font_weight="bold",
        font_color="whitesmoke",
    )
    nx.draw_networkx_edge_labels(
        topology,
        pos=positions,
        edge_labels=link_labels,
        font_color="black",
        font_size=5,
    )

    # Saving a topology image in the disk
    plt.savefig(f"{output_filename}.png", dpi=600)


def uniform(n_items: int, valid_values: list, shuffle_distribution: bool = True) -> list:
    """Creates a list of size "n_items" with values from "valid_values" according to the uniform distribution.
    By default, the method shuffles the created list to avoid unbalanced spread of the distribution.

    Args:
        n_items (int): Number of items that will be created.
        valid_values (list): List of valid values for the list of values.
        shuffle_distribution (bool, optional): Defines whether the distribution is shuffled or not. Defaults to True.

    Raises:
        Exception: Invalid "valid_values" argument.

    Returns:
        uniform_distribution (list): List of values arranged according to the uniform distribution.
    """
    if not isinstance(valid_values, list) or isinstance(valid_values, list) and len(valid_values) == 0:
        raise Exception("You must inform a list of valid values within the 'valid_values' attribute.")

    # Number of occurrences that will be created of each item in the "valid_values" list
    distribution = [int(n_items / len(valid_values)) for _ in range(0, len(valid_values))]

    # List with size "n_items" that will be populated with "valid_values" according to the uniform distribution
    uniform_distribution = []

    for i, value in enumerate(valid_values):
        for _ in range(0, int(distribution[i])):
            uniform_distribution.append(value)

    # Computing leftover randomly to avoid disturbing the distribution
    leftover = n_items % len(valid_values)
    for i in range(leftover):
        random_valid_value = random.choice(valid_values)
        uniform_distribution.append(random_valid_value)

    # Shuffling distribution values in case 'shuffle_distribution' parameter is True
    if shuffle_distribution:
        random.shuffle(uniform_distribution)

    return uniform_distribution


def min_max_norm(x, minimum, maximum):
    """Normalizes a given value (x) using the Min-Max Normalization method.

    Args:
        x (any): Value that must be normalized.
        min (any): Minimum value known.
        max (any): Maximum value known.

    Returns:
        (any): Normalized value.
    """
    if minimum == maximum:
        return 1
    return (x - minimum) / (maximum - minimum)


def get_normalized_capacity(object):
    return (object.cpu * object.memory * object.disk) ** (1 / 3)


def get_normalized_demand(object):
    if hasattr(object, "disk_demand"):
        return (object.cpu_demand * object.memory_demand * object.disk_demand) ** (1 / 3)
    else:
        return (object.cpu_demand * object.memory_demand) ** (1 / 2)


def get_norm(metadata: dict, attr_name: str, min: dict, max: dict) -> float:
    """Wrapper to normalize a value using the Min-Max Normalization method.

    Args:
        metadata (dict): Dictionary that contains the metadata of the object whose values are being normalized.
        attr_name (str): Name of the attribute that must be normalized.
        min (dict): Dictionary that contains the minimum values of the attributes.
        max (dict): Dictionary that contains the maximum values of the attributes.

    Returns:
        normalized_value (float): Normalized value.
    """
    normalized_value = min_max_norm(x=metadata[attr_name], minimum=min[attr_name], maximum=max[attr_name])
    return normalized_value


def find_minimum_and_maximum(metadata: list, nsgaii=False):
    """Finds the minimum and maximum values of a list of dictionaries.

    Args:
        metadata (list): List of dictionaries that contains the analyzed metadata.
        nsgaii (bool): Tells the method what's the format of the analyzed metadata (NSGA-II's metadata has a different structure).

    Returns:
        min_and_max (dict): Dictionary that contains the minimum and maximum values of the attributes.
    """
    min_and_max = {
        "minimum": {},
        "maximum": {},
    }

    if nsgaii == False:
        for metadata_item in metadata:
            for attr_name, attr_value in metadata_item.items():
                if attr_name != "object":
                    # Updating the attribute's minimum value
                    if attr_name not in min_and_max["minimum"] or attr_name in min_and_max["minimum"] and attr_value < min_and_max["minimum"][attr_name]:
                        min_and_max["minimum"][attr_name] = attr_value

                    # Updating the attribute's maximum value
                    if attr_name not in min_and_max["maximum"] or attr_name in min_and_max["maximum"] and attr_value > min_and_max["maximum"][attr_name]:
                        min_and_max["maximum"][attr_name] = attr_value
    else:
        for metadata_item in metadata:
            for attr_name, attr_value in metadata_item.items():
                if attr_name != "Migration Plan":
                    # Updating the attribute's minimum value
                    if attr_name not in min_and_max["minimum"] or attr_name in min_and_max["minimum"] and attr_value < min_and_max["minimum"][attr_name]:
                        min_and_max["minimum"][attr_name] = attr_value

                    # Updating the attribute's maximum value
                    if attr_name not in min_and_max["maximum"] or attr_name in min_and_max["maximum"] and attr_value > min_and_max["maximum"][attr_name]:
                        min_and_max["maximum"][attr_name] = attr_value

    return min_and_max


def has_capacity_to_host(server: object, service: object):
    # Calculating the edge server's free resources
    free_cpu = server.cpu - server.cpu_demand
    free_memory = server.memory - server.memory_demand

    # Checking if the host would have resources to host the service
    can_host = free_cpu >= service.cpu_demand and free_memory >= service.memory_demand

    return can_host


def can_host_services(servers: list, services: list) -> bool:
    """Checks if a set of servers have resources to host a group of services.

    Args:
        servers (list): List of edge servers.
        services (list): List of services that we want to accommodate inside the servers.

    Returns:
        bool: Boolean expression that tells us whether the set of servers did manage or not to host the services.
    """
    services_allocated = 0
    suggested_placement = []

    # Checking if all services could be hosted by the list of servers
    for service in services:
        # Sorting servers according to their demand (descending)
        servers = sorted(servers, key=lambda sv: get_normalized_capacity(object=sv) - get_normalized_demand(object=sv))
        for server in servers:
            # We assume that the disk demand is negligible based on the average size of container images and current server capacities
            if has_capacity_to_host(server=server, service=service):
                server.cpu_demand += service.cpu_demand
                server.memory_demand += service.memory_demand

                suggested_placement.append({"server": server, "service": service})

                services_allocated += 1
                break

    # Recomputing servers' demand
    for item in suggested_placement:
        server = item["server"]
        service = item["service"]

        server.cpu_demand -= service.cpu_demand
        server.memory_demand -= service.memory_demand

    return len(services) == services_allocated


def get_delay(wireless_delay: int, origin_switch: object, target_switch: object) -> int:
    """Gets the distance (in terms of delay) between two elements (origin and target).

    Args:
        wireless_delay (int): Wireless delay that must be included in the delay calculation.
        origin_switch (object): Origin switch.
        target_switch (object): Target switch.

    Returns:
        delay (int): Delay between the origin and target switches.
    """
    topology = origin_switch.model.topology

    path = nx.shortest_path(
        G=topology,
        source=origin_switch,
        target=target_switch,
        weight="delay",
    )
    delay = wireless_delay + topology.calculate_path_delay(path=path)

    return delay


def get_service_delay_sla(service: object) -> int:
    """Gets the delay SLA of a given service.

    Args:
        service (object): Service object.

    Returns:
        int: Service's delay SLA.
    """
    application = service.application
    user = application.users[0]

    return user.delay_slas[str(application.id)]
