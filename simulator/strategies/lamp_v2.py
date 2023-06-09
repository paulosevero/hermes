""" Contains Location-Aware Maintenance Policy (LAMP) maintenance algorithm."""
# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer


# Helper methods
from simulator.helper_methods import *


DISPLAY_SIMULATION_METRICS_AT_RUNTIME = False


def sort_servers_to_drain() -> list:
    """Sorts outdated servers given a set of criteria.

    Returns:
        sorted_outdated_servers [list]: Outdated servers.
    """
    outdated_servers = [server for server in EdgeServer.all() if server.status == "outdated"]

    min_capacity = min([1 / get_normalized_capacity(object=server) for server in outdated_servers])
    max_capacity = max([1 / get_normalized_capacity(object=server) for server in outdated_servers])

    min_demand = min([get_normalized_demand(object=server) for server in outdated_servers])
    max_demand = max([get_normalized_demand(object=server) for server in outdated_servers])

    min_update_duration = min([server.patch_time for server in outdated_servers])
    max_update_duration = max([server.patch_time for server in outdated_servers])

    for server in outdated_servers:
        norm_capacity_score = min_max_norm(
            x=1 / get_normalized_capacity(object=server),
            minimum=min_capacity,
            maximum=max_capacity,
        )
        norm_demand_score = min_max_norm(
            x=get_normalized_demand(object=server),
            minimum=min_demand,
            maximum=max_demand,
        )
        norm_update_duration_score = min_max_norm(
            x=server.patch_time,
            minimum=min_update_duration,
            maximum=max_update_duration,
        )

        server.drain_score = norm_capacity_score + norm_demand_score + norm_update_duration_score

    sorted_outdated_servers = sorted(outdated_servers, key=lambda server: server.drain_score)
    return sorted_outdated_servers


def sort_candidate_servers(service: object, candidate_servers: list) -> list:
    sorted_candidate_servers = []

    # Collecting relevant metadata from the candidate servers for hosting the service from the server being drained
    for server in candidate_servers:
        ########################
        #### FREE RESOURCES ####
        ########################
        free_resources = get_normalized_capacity(object=server) - get_normalized_demand(object=server)

        ###############
        #### DELAY ####
        ###############
        user = service.application.users[0]
        delay_sla = get_service_delay_sla(service=service)
        delay = get_delay(
            wireless_delay=user.base_station.wireless_delay,
            origin_switch=user.base_station.network_switch,
            target_switch=server.base_station.network_switch,
        )
        violates_delay_sla = 1 if delay > delay_sla else 0

        #################################
        #### AMOUNT OF CACHED LAYERS ####
        #################################
        # Gathering the list of container layers used by the service's image
        service_image = ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)
        service_layers = [
            ContainerLayer.find_by(attribute_name="digest", attribute_value=digest) for digest in service_image.layers_digests
        ]

        # Calculating the aggregated size of container layers used by the service's image that are cached in the candidate server
        layer_size_to_townload = 0

        layers_downloaded = [layer for layer in server.container_layers]
        layers_on_download_queue = [flow.metadata["object"] for flow in server.download_queue]
        layers_on_waiting_queue = [layer for layer in server.waiting_queue]
        server_layers = layers_downloaded + layers_on_download_queue + layers_on_waiting_queue

        for cached_layer in server_layers:
            if not any(cached_layer.digest == service_layer.digest for service_layer in service_layers):
                layer_size_to_townload += cached_layer.size

        #######################
        #### UPDATE STATUS ####
        #######################
        update_status = 0 if server.status == "updated" else 1

        # Consolidating the metadata of the candidate server
        candidate_server_metadata = {
            "object": server,
            "free_resources": 1 / max(1, free_resources),
            "violates_delay_sla": violates_delay_sla,
            "layer_size_to_townload": layer_size_to_townload,
            "update_status": update_status,
        }
        sorted_candidate_servers.append(candidate_server_metadata)

    # Calculating the normalized scores for each candidate server
    min_and_max = find_minimum_and_maximum(metadata=sorted_candidate_servers)
    for server_metadata in sorted_candidate_servers:
        server_metadata["norm_free_resources"] = get_norm(
            metadata=server_metadata,
            attr_name="free_resources",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )
        server_metadata["norm_violates_delay_sla"] = get_norm(
            metadata=server_metadata,
            attr_name="violates_delay_sla",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )
        server_metadata["norm_layer_size_to_townload"] = get_norm(
            metadata=server_metadata,
            attr_name="layer_size_to_townload",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )

    sorted_candidate_servers = sorted(
        sorted_candidate_servers,
        key=lambda server: (
            server["update_status"],
            server["norm_violates_delay_sla"] + server["norm_layer_size_to_townload"],  # + server["norm_free_resources"],
        ),
    )

    if DISPLAY_SIMULATION_METRICS_AT_RUNTIME:
        print(f"==== SERVICE TO BE MIGRATED: {service} ====")
        for candidate_server in sorted_candidate_servers:
            metadata = {
                "update_status": round(candidate_server["update_status"], 2),
                "norm_free_resources": round(candidate_server["norm_free_resources"], 2),
                "norm_violates_delay_sla": round(candidate_server["norm_violates_delay_sla"], 2),
                "norm_layer_size_to_townload": round(candidate_server["norm_layer_size_to_townload"], 2),
            }
            print(f"\t{candidate_server['object']}. {metadata}")
        print("")

    sorted_candidate_servers = [server_metadata["object"] for server_metadata in sorted_candidate_servers]
    return sorted_candidate_servers


def lamp_v2(parameters: dict = {}):
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
        servers_being_emptied = []

        # Getting the list of servers that still need to be patched
        servers_to_empty = sort_servers_to_drain()

        for server in servers_to_empty:
            # We consider as candidate hosts for the services all EdgeServer
            # objects not being emptied in the current maintenance step
            candidate_servers = [
                candidate for candidate in EdgeServer.all() if candidate not in servers_being_emptied and candidate != server
            ]

            # Sorting services by its demand (decreasing)
            services = sorted(list(server.services), key=lambda service: -get_normalized_demand(object=service))
            services_to_migrate = len(services)

            if can_host_services(servers=candidate_servers, services=services):
                for _ in range(len(server.services)):
                    service = services.pop(0)
                    application = service.application

                    # Sorting candidate servers to host the service
                    candidate_servers = sort_candidate_servers(service=service, candidate_servers=candidate_servers)

                    for candidate_host in candidate_servers:
                        if has_capacity_to_host(server=candidate_host, service=service):
                            # Migrating the service
                            service.provision(target_server=candidate_host)
                            services_to_migrate -= 1

                            break

            if services_to_migrate == 0:
                servers_being_emptied.append(server)
