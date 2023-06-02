""" Contains Location-Aware Maintenance Policy (LAMP) maintenance algorithm."""
# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer


# Helper methods
from simulator.helper_methods import *


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


def sort_candidate_servers(user: object, server_being_drained: object, candidate_servers: list) -> list:
    min_free_resources = min(
        [get_normalized_capacity(object=candidate) - get_normalized_demand(object=candidate) for candidate in candidate_servers]
    )
    max_free_resources = max(
        [get_normalized_capacity(object=candidate) - get_normalized_demand(object=candidate) for candidate in candidate_servers]
    )
    min_delay = min(
        [
            get_delay(
                wireless_delay=user.base_station.wireless_delay,
                origin_switch=user.base_station.network_switch,
                target_switch=candidate.base_station.network_switch,
            )
            for candidate in candidate_servers
        ]
    )
    max_delay = max(
        [
            get_delay(
                wireless_delay=user.base_station.wireless_delay,
                origin_switch=user.base_station.network_switch,
                target_switch=candidate.base_station.network_switch,
            )
            for candidate in candidate_servers
        ]
    )

    for candidate in candidate_servers:
        update_score = 0 if candidate.status == "updated" else 1
        norm_free_resources_score = min_max_norm(
            x=get_normalized_capacity(object=candidate) - get_normalized_demand(object=candidate),
            minimum=min_free_resources,
            maximum=max_free_resources,
        )
        norm_delay_score = min_max_norm(
            x=get_delay(
                wireless_delay=user.base_station.wireless_delay,
                origin_switch=user.base_station.network_switch,
                target_switch=candidate.base_station.network_switch,
            ),
            minimum=min_delay,
            maximum=max_delay,
        )

        candidate.score = update_score + norm_free_resources_score + norm_delay_score

    candidate_servers = sorted(candidate_servers, key=lambda candidate: candidate.score)
    return candidate_servers


def lamp(parameters: dict = {}):
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
                    user = application.users[0]

                    # Sorting candidate servers to host the service
                    candidate_servers = sort_candidate_servers(
                        user=user, server_being_drained=server, candidate_servers=candidate_servers
                    )

                    for candidate_host in candidate_servers:
                        if has_capacity_to_host(server=candidate_host, service=service):
                            # Migrating the service
                            service.provision(target_server=candidate_host)
                            # print(f"\t[{parameters['current_maintenance_batch']}] Migrating {service} to {candidate_host}")
                            services_to_migrate -= 1

                            break

            if services_to_migrate == 0:
                servers_being_emptied.append(server)
