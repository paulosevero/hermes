# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer

# Helper methods
from simulator.helper_methods import *


def hermes_can_host_services(servers: list, services: list) -> bool:
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
        # Sorting servers
        sorted_list_of_servers = sort_candidate_servers(service=service, candidate_servers=servers)
        for server in sorted_list_of_servers:
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


def calculate_layer_relevance_score(layer: object, services_that_use_the_layer: int) -> float:
    """Calculates the layer relevance score for a given layer.

    Args:
        layer (object): Layer to calculate the relevance score for.
        services_that_use_the_layer (list): List of services that use the analyzed layer.

    Returns:
        layer_relevance_score (float): Layer relevance score.
    """
    layer_relevance_score = layer.size ** len(services_that_use_the_layer)
    return layer_relevance_score


def calculate_cache_relevance_score(layers_to_analyze: list, services_of_interest: list) -> float:
    """Calculates the relevance score of a cache composed by a list of layers based on
    attributes of these layers and their relationship of a list of services of interest.

    Args:
        layers_to_analyze (list): List of layers that compose the cache being analyzed.
        services_of_interest (list): List of services of interest used to calculate the cache relevance score.

    Returns:
        cache_relevance_score (float): Calculated cache relevance score.
    """
    cache_relevance_score = 0

    for layer in layers_to_analyze:

        services_that_use_this_layer = []

        for svs in services_of_interest:
            svs_image = ContainerImage.find_by(attribute_name="digest", attribute_value=svs.image_digest)
            if layer.digest in svs_image.layers_digests:
                services_that_use_this_layer.append(svs)

        layer_relevance = calculate_layer_relevance_score(layer=layer, services_that_use_the_layer=services_that_use_this_layer)
        cache_relevance_score += layer_relevance

    return cache_relevance_score


def sort_servers_to_drain() -> list:
    """Defines the order in which outdated servers will be drained.

    ==========================
    => CONSIDERED CRITERIA ===
    ==========================
        -> Servers with a larger capacity
        -> Servers hosting services based on smaller container images and with smaller states (preferentially those that have no states at all)
        -> Servers hosting services with looser delay SLAs

    Returns:
        servers_to_drain (list): Sorted list of outdated servers to be drained.
    """
    servers_to_drain = []

    # Collecting relevant metadata from outdated servers
    for server in [server for server in EdgeServer.all() if server.status == "outdated"]:
        #########################
        #### SERVER CAPACITY ####
        #########################
        normalized_server_capacity = get_normalized_capacity(object=server)

        ##################################
        #### SIZE OF CONTAINER IMAGES ####
        ##################################
        # Gathering the size of container images that compose the services hosted by the server
        container_image_sizes = []
        for service in server.services:
            service_image_size = 0
            image = ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)
            for layer_digest in image.layers_digests:
                layer = ContainerLayer.find_by(attribute_name="digest", attribute_value=layer_digest)
                service_image_size += layer.size
            container_image_sizes.append(service_image_size + service.state)

        maximum_container_image_size = max(container_image_sizes)

        ################################
        #### SLA OF HOSTED SERVICES ####
        ################################
        slas_of_hosted_services = [get_service_delay_sla(service=service) for service in server.services]

        # Consolidating the server metadata
        server_metadata = {
            "object": server,
            "capacity": normalized_server_capacity,
            "inversed_max_container_image_size": 1 / max(1, maximum_container_image_size),
            "slas_of_hosted_services": 1 / (sum(slas_of_hosted_services) / len(slas_of_hosted_services)),
        }

        servers_to_drain.append(server_metadata)

    # Calculating the normalized scores for each outdated server
    min_and_max = find_minimum_and_maximum(metadata=servers_to_drain)
    for server_metadata in servers_to_drain:
        server_metadata["norm_capacity"] = get_norm(
            metadata=server_metadata,
            attr_name="capacity",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )
        server_metadata["norm_inversed_max_container_image_size"] = get_norm(
            metadata=server_metadata,
            attr_name="inversed_max_container_image_size",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )
        server_metadata["norm_slas_of_hosted_services"] = get_norm(
            metadata=server_metadata,
            attr_name="slas_of_hosted_services",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )

    servers_to_drain = sorted(
        servers_to_drain,
        key=lambda server: server["norm_capacity"] + server["norm_inversed_max_container_image_size"] + server["norm_slas_of_hosted_services"],
        reverse=True,
    )

    servers_to_drain = [server_metadata["object"] for server_metadata in servers_to_drain]
    return servers_to_drain


def sort_services_to_relocate(server: object) -> list:
    """Defines the order in which services from a given server being drained will be relocated.

    ==========================
    => CONSIDERED CRITERIA ===
    ==========================
        -> Services with tight delay SLA
        -> Services based on images with layers used by co-hosted services
        -> Services with low CPU/RAM demand

    Args:
        server (object): Server being drained.

    Returns:
        services_to_relocate (list): Sorted list of services to be relocated.
    """
    services_to_relocate = []

    # Collecting relevant metadata from services that must be relocated to drain out their current host
    for service in server.services:
        ###################
        #### DELAY SLA ####
        ###################
        inversed_delay_sla = 1 / get_service_delay_sla(service=service)

        ####################################################
        #### LAYER POPULARITY WITHIN CO-HOSTED SERVICES ####
        ####################################################
        # Gathering the list of layers that compose the service's image
        service_image = ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)
        service_layers = [ContainerLayer.find_by(attribute_name="digest", attribute_value=digest) for digest in service_image.layers_digests]

        # Calculating the relevance of layers used by the service based
        # on their size and on how many co-hosted services use them
        co_hosted_services = [svs for svs in server.services if svs != service]
        cache_relevance = calculate_cache_relevance_score(layers_to_analyze=service_layers, services_of_interest=co_hosted_services)

        ########################
        #### SERVICE DEMAND ####
        ########################
        service_demand = get_normalized_demand(object=service)

        # Consolidating the service metadata
        service_metadata = {
            "object": service,
            "inversed_delay_sla": inversed_delay_sla,
            "cache_relevance": cache_relevance,
            "inversed_service_demand": 1 / service_demand,
        }

        services_to_relocate.append(service_metadata)

    # Calculating the normalized scores for each outdated server
    min_and_max = find_minimum_and_maximum(metadata=services_to_relocate)
    for service_metadata in services_to_relocate:
        service_metadata["norm_inversed_delay_sla"] = get_norm(
            metadata=service_metadata,
            attr_name="inversed_delay_sla",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )
        service_metadata["norm_cache_relevance"] = get_norm(
            metadata=service_metadata,
            attr_name="cache_relevance",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )
        service_metadata["norm_inversed_service_demand"] = get_norm(
            metadata=service_metadata,
            attr_name="inversed_service_demand",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )

    services_to_relocate = sorted(
        services_to_relocate,
        key=lambda service: service["norm_inversed_delay_sla"] + service["norm_cache_relevance"] + service["norm_inversed_service_demand"],
        reverse=True,
    )

    services_to_relocate = [service_metadata["object"] for service_metadata in services_to_relocate]
    return services_to_relocate


def sort_candidate_servers(service: object, candidate_servers: list) -> list:
    """Defines the order of candidate servers for hosting a service from a given server being drained.

    ==========================
    => CONSIDERED CRITERIA ===
    ==========================
        -> 1. Servers already updated
        -> 2. Servers close enough to the user to avoid SLA violation
        -> 3. Servers with a larger amount of cached data from layers
        -> 4. Servers downloading the least amount of layers

    Args:
        service (object): Service to be relocated.
        server_being_drained (object): Server being drained.
        candidate_servers (list): List of candidate servers for hosting the service that must be relocated.

    Returns:
        sorted_candidate_servers (list): Sorted list of candidate servers for hosting a given service.
    """
    sorted_candidate_servers = []

    # Collecting relevant metadata from the candidate servers for hosting the service from the server being drained
    for server in candidate_servers:
        #######################
        #### UPDATE STATUS ####
        #######################
        update_status = 1 if server.status == "updated" else 0

        #######################
        #### SLA VIOLATION ####
        #######################
        user = service.application.users[0]
        delay_sla = get_service_delay_sla(service=service)
        delay = get_delay(
            wireless_delay=user.base_station.wireless_delay,
            origin_switch=user.base_station.network_switch,
            target_switch=server.base_station.network_switch,
        )
        avoids_sla_violation = 1 if delay_sla >= delay else 0

        #################################
        #### AMOUNT OF CACHED LAYERS ####
        #################################
        # Gathering the list of container layers used by the service's image
        service_image = ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)
        service_layers = [ContainerLayer.find_by(attribute_name="digest", attribute_value=digest) for digest in service_image.layers_digests]

        # Gathering the list of container layers cached or about to be cached by the candidate server
        layers_downloaded = [layer for layer in server.container_layers]
        layers_on_download_queue = [flow.metadata["object"] for flow in server.download_queue]
        layers_on_waiting_queue = [layer for layer in server.waiting_queue]
        server_layers = layers_downloaded + layers_on_download_queue + layers_on_waiting_queue

        # Calculating the aggregated size of container layers used by the service's image that are cached in the candidate server
        amount_of_cached_layers = 0
        for cached_layer in server_layers:
            if any(cached_layer.digest == service_layer.digest for service_layer in service_layers):
                amount_of_cached_layers += cached_layer.size

        ##########################
        #### LAYER QUEUE SIZE ####
        ##########################
        layer_queue = len(server.waiting_queue) + len(server.download_queue)

        # Consolidating the metadata of the candidate server
        candidate_server_metadata = {
            "object": server,
            "update_status": update_status,
            "avoids_sla_violation": avoids_sla_violation,
            "amount_of_cached_layers": amount_of_cached_layers,
            "inversed_layer_queue": 1 / max(1, layer_queue),
        }
        sorted_candidate_servers.append(candidate_server_metadata)

    # Calculating the normalized scores for each candidate server
    min_and_max = find_minimum_and_maximum(metadata=sorted_candidate_servers)
    for server_metadata in sorted_candidate_servers:
        server_metadata["norm_amount_of_cached_layers"] = get_norm(
            metadata=server_metadata,
            attr_name="amount_of_cached_layers",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )
        server_metadata["norm_inversed_layer_queue"] = get_norm(
            metadata=server_metadata,
            attr_name="inversed_layer_queue",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )

    sorted_candidate_servers = sorted(
        sorted_candidate_servers,
        key=lambda server: (
            server["update_status"],
            server["avoids_sla_violation"] + server["norm_amount_of_cached_layers"] + server["norm_inversed_layer_queue"],
        ),
        reverse=True,
    )

    sorted_candidate_servers = [server_metadata["object"] for server_metadata in sorted_candidate_servers]
    return sorted_candidate_servers


def hermes_v2(parameters: dict = {}):
    """Layer-aware maintenance policy for edge computing infrastructures.

    Args:
        parameters (dict, optional): Algorithm parameters. Defaults to {}.
    """
    # Patching outdated servers that were previously drained out (i.e., those that are not currently hosting any service)
    servers_to_patch = [
        server for server in EdgeServer.all() if server.status == "outdated" and len(server.services) == 0 and len(server.container_registries) == 0
    ]
    if len(servers_to_patch) > 0:
        for server in servers_to_patch:
            server.update()

    # Draining out outdated servers
    else:
        # Getting the list of servers that still need to be patched
        servers_to_empty = sort_servers_to_drain()

        servers_being_emptied = []

        for server in servers_to_empty:
            # We consider as candidate hosts for the services all EdgeServer
            # objects not being emptied in the current maintenance step
            candidate_servers = [candidate for candidate in EdgeServer.all() if candidate not in servers_being_emptied and candidate != server]

            # Defining the order in which services will be migrated from their current host
            services = sort_services_to_relocate(server=server)
            services_to_migrate = len(services)

            if hermes_can_host_services(servers=candidate_servers, services=services):
                for _ in range(len(server.services)):
                    service = services.pop(0)

                    # Sorting candidate servers to host the service
                    candidate_servers = sort_candidate_servers(service=service, candidate_servers=candidate_servers)

                    for candidate_host in candidate_servers:
                        if has_capacity_to_host(server=candidate_host, service=service):
                            # Migrating the service
                            service.provision(target_server=candidate_host)
                            # print(f"\t[{parameters['current_maintenance_batch']}] Migrating {service} to {candidate_host}")
                            services_to_migrate -= 1

                            break

            if services_to_migrate == 0:
                servers_being_emptied.append(server)
