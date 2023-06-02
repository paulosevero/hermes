# EdgeSimPy components
from edge_sim_py.components.service import Service
from edge_sim_py.components.edge_server import EdgeServer


def nsgaii(parameters: dict = {}):
    """TBD

    Args:
        parameters (dict, optional): Algorithm parameters. Defaults to {}.
    """
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
        placement_scheme = parameters["solution"][parameters["current_maintenance_batch"] - 1]

        for service_id, server_id in enumerate(placement_scheme, 1):
            service = Service.find_by_id(service_id)
            server = EdgeServer.find_by_id(server_id)

            if service.server != server:
                service.provision(target_server=server)
