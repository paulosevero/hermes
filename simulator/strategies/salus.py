""" Contains Salus maintenance algorithm."""
# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer

# Helper methods
from simulator.helper_methods import *


def salus(parameters: dict):
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
        servers_to_empty = [server for server in EdgeServer.all() if server.status == "outdated"]
        servers_to_empty = sorted(
            servers_to_empty, key=lambda sv: ((sv.patch_time) * (1 / (get_normalized_capacity(object=sv) + 1))) ** (1 / 2)
        )

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

                    # Prioritizing updated servers with less space remaining
                    candidate_servers = sorted(
                        candidate_servers,
                        key=lambda c: (
                            -int(c.status == "updated"),
                            get_normalized_capacity(object=c) - get_normalized_demand(object=c),
                        ),
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
